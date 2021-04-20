import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import gensim

import numpy as np 
#import nltk
#nltk.download('stopwords')
def process_titles(title):
    '''
    Input:
        title: a string containing a news title
    Output:
        title_clean: a list of words containing the processed title

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    title = re.sub(r'\$\w*', '', title)
    # remove old style retweet text "RT"
    title = re.sub(r'^RT[\s]+', '', title)
    # remove hyperlinks
    title = re.sub(r'https?:\/\/.*[\r\n]*', '', title)
    # remove hashtags
    # only removing the hash # sign from the word
    title = re.sub(r'#', '', title)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    title_tokens = tokenizer.tokenize(title)

    title_clean = []
    for word in title_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            title_clean.append(stem_word)

    return title_clean

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in list(word2vec_model.key_to_index.keys()) ]
    return np.mean(word2vec_model[doc], axis=0)

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    stop = stopwords.words('english')
    stop.extend(['the','says','new','first','said','group','may','per'])
    stop_words = set(stop)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] 
    return doc

def has_vector_representation(word2vec_model, doc):
    return not all(word not in list(word2vec_model.key_to_index.keys()) for word in doc)

def filter_docs(corpus, texts, condition_on_doc):
    number_of_docs = len(corpus)
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))
    return (corpus, texts)

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(title, freqs):
    word_l = process_titles(title)
    x = np.zeros((1, 3)) 
    x[0,0] = 1 
    for word in word_l:
        x[0,1] += freqs.get((word, 1.0),0)
        x[0,2] += freqs.get((word, 0.0),0)
    assert(x.shape == (1, 3))
    return x


def build_freqs(titles, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, title in zip(yslist, titles):
        for word in process_titles(title):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
