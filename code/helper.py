import os 
import pandas as pd
import json
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sparknlp
import pyspark
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA

def merge_files() -> None:
    """Merge all JSON files of a folder

    Sample folder structure
    ```
    └── 📁data
        └── 📁merged
            └── test.json
        └── 📁processed
        └── 📁raw
            └── test01.json
            └── test02.json
    ```
    Args:
        None
    Return:
        returns None"""
    files = os.listdir(os.path.join("..", "data", "raw"))
    text_todf = []
    for i in range(len(files)):
        with open(os.path.join("..", "data", "raw", files[i]), 'r') as file:
            data = json.load(file)
            df_metadata = pd.json_normalize(data, max_level=1)
            text = ""
            for element in df_metadata['body_text'][0]:
                text += element['text']
            text_todf.append(text)
    df = pd.DataFrame({"text": text_todf})
    df.to_json(os.path.join("..", "data", "merged", "corpus.json"))

def custom_word_tokenize(sent_tokens: list[str]) -> list[str]:
    """Tokenises words 
    
    Args:
        sent_tokens: tokens of sentences
    Returns:
        A list of word tokens"""
    word_tokens = []
    for sent in sent_tokens:
        tokens = word_tokenize(sent)
        word_tokens.extend(tokens)
    return word_tokens

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Performs the preprocessing
    
    Args:
        data: input file as pandas.DataFrame
    Returns:
        A pandas.DataFrame which as vector form of the input
    """
    stop_words = stopwords.words('english')
    lemmatiser = WordNetLemmatizer()

    #Sentence tokenisation
    data['sent_tokens'] = data['text'].apply(sent_tokenize)
    
    # Text cleaning
    data['sent_tokens'] = data['sent_tokens'].apply(lambda sentences: [re.sub(r"[^a-zA-Z\s]", " ", sent).strip() for sent in sentences])
    
    # Normalisation
    data['sent_tokens'] = data['sent_tokens'].apply(lambda sentences: [sent.lower() for sent in sentences])

    #Word tokenisation
    data['word_tokens'] = data['sent_tokens'].apply(custom_word_tokenize)

    # # Stemming
    data['word_tokens'] = data['word_tokens'].apply(lambda word_tokens: [lemmatiser.lemmatize(word) for word in word_tokens])

    # Word Embeddings (turns into vector)
    data['to_tfidf'] = data['word_tokens'].apply(lambda tokens: ' '.join(tokens))
    vectorizer = TfidfVectorizer(min_df=0.3, max_df=0.85, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(data['to_tfidf'])
    
    # Convert to DataFrame
    tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return data, tfidf

def pipeline_model(
        train_data: pd.DataFrame,
        spark_session: pyspark.sql.session.SparkSession
) -> pyspark.ml.pipeline.PipelineModel:
    """Trains a Pipeline model used to be applied for preprocessing phase

    Args:
        train_data: data as pandas.DataFrame
        spark_session: a SparkSession
    Returns:
        A trained PipelineModel
    """
    
    sparknlp_df = spark_session.createDataFrame(train_data)
    # Document Assembler: Converts input text into a suitable format for NLP processing
    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")\
        .setCleanupMode("shrink")

    # Sentence tokenisation
    sentenceDetector = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentences')

    # Word tokenisation 
    tokeniser = Tokenizer()\
        .setInputCols(["sentences"]) \
        .setOutputCol("token")

    # Text cleaning and Normalisation
    normaliser = Normalizer()\
        .setInputCols("token")\
        .setOutputCol("normalised")\
        .setLowercase(True)\
        .setCleanupPatterns(["[^a-zA-Z\s]"])

    # Stopword Removal
    stopwords_cleaner = StopWordsCleaner()\
        .setInputCols("normalised")\
        .setOutputCol("stopwords_removed")\
        .setCaseSensitive(False)

    # Lemmatisation
    lemmatizer = LemmatizerModel.pretrained()\
        .setInputCols(["stopwords_removed"])\
        .setOutputCol("lemma")

    # Finisher 
    finisher = Finisher() \
        .setInputCols("lemma") \
        .setOutputCols("finish") \
        .setIncludeMetadata(False) # set to False to remove metadata

    # Pipeline
    pipeline = Pipeline().setStages([
            documentAssembler,
            sentenceDetector,
            tokeniser,
            normaliser,
            stopwords_cleaner,
            lemmatizer,
            finisher
            ])
    result = pipeline.fit(sparknlp_df)
    return result

def preprocess_sparknlp(
        data: pd.DataFrame,
        spark_session: pyspark.sql.session.SparkSession,
        pipeline_model: pyspark.ml.pipeline.PipelineModel
) -> pyspark.sql.dataframe.DataFrame:
    """Performs the preprocessing using a trained PipelineModel
    
    Args:
        data: input file as pandas.DataFrame
        sparknlp_session: a SparkSession
        pipeline_model: an instance of pyspark.ml.pipeline.PipelineModel to perform preprocessing
    Returns:
        A pyspark.sql.dataframe.DataFrame which has vector form of the input
    """
    # Change from pandas.DataFrame to Spark DataFrame
    sparknlp_df = spark_session.createDataFrame(data)
    result = pipeline_model.transform(sparknlp_df)    
    
    # Word Embeddings (turns into vector)
    ## Term-Frequency (TF) transform
    tfizer = CountVectorizer(inputCol='finish', outputCol='tf_features')
    tf_model = tfizer.fit(result)
    tf_result = tf_model.transform(result)
    
    ## TF-IDF trasnform
    idfizer = IDF(inputCol='tf_features', outputCol='tf_idf_features')
    idf_model = idfizer.fit(tf_result)
    tfidf_result = idf_model.transform(tf_result)

    return tfidf_result