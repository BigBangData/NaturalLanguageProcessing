#!/usr/bin/env python

# Cleanup module for Twitter search API

import re
import os
import json
import time

import string
import datetime
import urlextract
import pandas as pd

from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# load data
def load_data():
    filepath = os.path.join("data","raw","tweets") 

    dfm = []
    for f in os.listdir(filepath):
        dfm.append(pd.read_csv(os.path.join(filepath,f)))
        
    df = pd.concat(dfm)
    df = df.reset_index(drop=True)
    
    return df

def is_retweet(col):

    for i in range(len(col)):
        if re.match(r'^RT', col) is not None:
            return 1
        else:
            return 0      
        
def map_is_retweet(col):
   
    bool_map = map(lambda x: is_retweet(x), col)       
    return(list(bool_map)) 

def cleanup_tweet(tweet):

    # make all lower case
    tweet = tweet.lower()

    # remove URLs
    # URL_pattern = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    # tweet = re.sub(URL_pattern, '', tweet, flags=re.MULTILINE)
    # better, albeit slower, version
    urls = list(set(url_extractor.find_urls(tweet)))
    if len(urls) > 0:
        for url in urls:
            tweet = tweet.replace(url, "")

    # unescape HTML entities like &quot
    tweet = unescape(tweet)
    
    # remove user references (including username) or hashtags, etc.     
    pattern = r'\@\w+|\#|\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\!|\?|\¿\
                |\x82|\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
    
    tweet = re.sub(pattern, '', tweet)

    # remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # remove emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet).strip()
    
    # final test, if after utf-8 encoding it's not ascii decodable, discard
    def is_ascii(text):
        try:
            text.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True
    
    if is_ascii(tweet) == False:
        return " "
    else:
        pass
    
    # remove stopwords
    # NLTK's set(stopwords.words('english')) removes too many words
    # using list of 25 semantically non-selective words (Reuters-RCV1 dataset)
    stop_words = ['a','an','and','are','as','at','be','by','for','from',
                  'has','he','in','is','it','its','of','on','that','the',
                  'to','was','were','will','with','rt'] # plus rt for retweet

    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    return " ".join(stemmed_words)


if __name__=="__main__":

    start = time.time()

    # load
    print('Loading...')
    df = load_data()
    raw_nrows = df.shape[0]

    # create retweet col
    df['Retweet'] = map_is_retweet(df['Text'].values)

    # cleanup Tweet text
    print('Cleaning...')
    url_extractor = urlextract.URLExtract()
    df['Text'] = [cleanup_tweet(tweet) for tweet in df.loc[:,'Text']]

    # create a subset with cols of interest
    df = df[['ID','Retweet','Text','Polarity']].copy()

    # dedupe (text duplicates)
    print('Deduping...')
    dupes = df[df['Text'].duplicated(keep='first')]
    df = df[~df.ID.isin(list(dupes['ID']))]
    
    deduped_nrows = df.shape[0]
    pct_retweets = round(100*sum(df['Retweet'])/df.shape[0],2)

    # save
    print('Saving...')
    filepath = os.path.join("data","clean","tweets")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = ''.join([today_prefix, "_tweets.csv"])

    df.to_csv(os.path.join(filepath, filename), index=False)
    
    end = time.time()
    elapsed = end-start
    
    print('Cleanup successful.')
    print('Total (raw) number of rows: ' + str(raw_nrows))
    print('Final (deduped) number of rows: ' + str(deduped_nrows))
    print('Percentage of retweets: ' + str(pct_retweets))
    print('Time elapsed: ' + str(elapsed) + ' secs.')
