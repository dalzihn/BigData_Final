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
    stop_words = set(stopwords.words('english'))
    
    # Read data
    data = pd.read_json(os.path.join("..", "data", "test.json"))
    lemmatizer = WordNetLemmatizer()
    #Sentence tokenisation
    data['sent_tokens'] = data['text'].apply(sent_tokenize)
    
    # Text cleaning
    data['sent_tokens'] = data['sent_tokens'].apply(lambda sentences: [re.sub(r"[^a-zA-Z\s]", "", sent) for sent in sentences])
    
    # # Normalisation
    data['sent_tokens'] = data['sent_tokens'].apply(lambda sentences: [sent.lower() for sent in sentences])

    #Word tokenisation
    data['word_tokens'] = data['sent_tokens'].apply(custom_word_tokenize)

    # # Stemming
    data['word_tokens'] = data['word_tokens'].apply(lambda word_tokens: [lemmatizer.lemmatize(word) for word in word_tokens])

    # Word Embeddings (turns into vector)
    data['to_tfidf'] = data['word_tokens'].apply(lambda tokens: ' '.join(tokens))
    vectorizer = TfidfVectorizer(min_df=0.3, max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(data['to_tfidf'])
    
    # # Convert to DataFrame
    tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return data, tfidf