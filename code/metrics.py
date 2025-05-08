from collections import defaultdict
import pyspark 
import numpy as np

def puw(
        df: pyspark.sql.dataframe.DataFrame,
        termCol: str
) -> float:
    """Calculates Propotion of Unique Words (PUW), defined as 
     the propotion of unique words in the top frequent words among topics.
     
    Args:
        df: a PySpark DataFrame which has the topic and its term indices for each topic
        termCol: name of the column storing the term indices for each topic
    
    Returns:
        A number represents PUW
    """
    topic_profile = df.select("topic", "termIndices").collect()
    terms_to_topics = defaultdict(set)

    # Get mapping of terms to topics
    for topic_term in topic_profile:
        topic_id = topic_term['topic']
        term_indices = topic_term['termIndices']
        for term in term_indices:
            terms_to_topics[term].add(topic_id)

    topics_to_puw = {}
    # PUW of each topic
    for topic_term in topic_profile:
        topic_id = topic_term['topic']
        term_indices = topic_term['termIndices']

        unique_terms = [term for term in term_indices if len(terms_to_topics[term])==1]

        puw = len(unique_terms) / len(term_indices)

        topics_to_puw[str(topic_id)] = puw

    return topics_to_puw


# Reference of the pmi function: https://github.com/christianrfg/tm_metrics
def pmi(
        topic_words: list[str],
        word_frequency: dict[int],
        word_frequency_documents: dict[str, list],
        n_docs: int
) -> float:
    """Calculates PMI for a topic
    
    Args:
        topic_words: word representation of a topic
        word_frequency: frequency of each word in corpus
        word_frequency_documents: frequency of each word for each document
        n_docs: number of documents
        
    Returns:
        The PMI metric for the topic"""
    num_topwords = len(topic_words)
    pmi = 0.0

    for j in range(1, num_topwords):
        for i in range(0, j):
            ti = topic_words[i]
            tj = topic_words[j]

            freq_i = word_frequency[ti]
            freq_j = word_frequency[tj]
            freq_ij = len(word_frequency_documents[ti].intersection(word_frequency_documents[tj]))

            wij_prob = freq_ij / float(num_docs)
            wi_wj_prob = ((freq_i*freq_j) / float(num_docs) **2)

            pmi += np.log(wij_prob / wi_wj_prob)

    return pmi