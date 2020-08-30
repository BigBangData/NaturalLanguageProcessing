import time
import pandas as pd
import numpy as np

# NLP 
import re
import string
import urlextract
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# functions
def load_dataset(filename, col_ix, col_names):

    dataset = pd.read_csv(filename, encoding='latin-1', usecols=col_ix)
    dataset.columns = col_names

    return dataset

def cleanup_tweet(tweet):
    
    # make lower case
    tweet = tweet.lower()
    
    # remove URLs
    url_extractor = urlextract.URLExtract()
    urls = list(set(url_extractor.find_urls(tweet)))

    for url in urls:
        tweet = tweet.replace(url, "")               
        
    # remove user handles (anonymize) and the hashtag symbol (not value)
    tweet = re.sub(r'\@\w+|\#','', tweet)
        
    # remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # remove stopwords
    # NLTK's set(stopwords.words('english')) removes too many words
    # using list of 25 semantically non-selective words (Reuters-RCV1 dataset)
    stop_words = ['a','an','and','are','as','at','be','by','for','from',
                  'has','he','in','is','it','its','of','on','that','the',
                  'to','was','were','will','with']
    
    # tokenize 
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    # stem 
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    return " ".join(stemmed_words)

def vector_clean(list_):
    map_iterator = map(cleanup_tweet, list_)
    return list(map_iterator)

def test_time(df):

    test = df.loc[:999,].copy()

    start_time = time.time()
    test.loc[:, 'text'] = vector_clean(test.loc[:, 'text'])

    elapsed_time = time.time() - start_time
    seconds=(df.shape[0]/test.shape[0])*elapsed_time
    hours=seconds/60/24

    print("time to clean (hs): " + str(hours))

# run
if __name__=="__main__":
    df = load_dataset("./data/training.1600000.processed.noemoticon.csv",
                      [0, 5], 
                      ['target', 'text'])  
    
    test_time(df)