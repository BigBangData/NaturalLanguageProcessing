#!/usr/bin/env python

# Cleanup module for Twitter search API

import os
import re
import sys
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
    filepath = os.path.join("..","data","1_raw","tweets") 

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
    """Cleans up a tweet with the following steps:
        1. make lower case
        2. remove URLs
        3. unescape HTML entities
        4. remove user references (including username) or hashtags, etc.
        5. remove punctuation
        6. remove emojis
        7. discard non-ascii decodable text after utf-8 encoding
        8. tokenize
        9. filter stop words from tokens
        10. stem filtered tokens
        
    The function returns a 3-tuple with cleaned versions 8 through 10.
    """
    # 1
    tweet = tweet.lower()

    # 2
    # URL_pattern = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    # tweet = re.sub(URL_pattern, '', tweet, flags=re.MULTILINE)
    # better, albeit slower, version
    urls = list(set(url_extractor.find_urls(tweet)))
    if len(urls) > 0:
        for url in urls:
            tweet = tweet.replace(url, "")
    # 3
    tweet = unescape(tweet)
    
    # 4
    pattern = r'\@\w+|\#|\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\!|\?|\¿\
                |\x82|\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
    tweet = re.sub(pattern, '', tweet)

    # 5
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # 6
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet).strip()

    # 7
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

    # 8 tokenized only (remove retweet prefix)
    tweet_tokens = word_tokenize(tweet)
    retweet = ['rt']
    tweet_tokens = [token for token in tweet_tokens if not token in retweet]
    
    # 9 tokenized + filtered
    # NLTK's set(stopwords.words('english')) removes too many words
    # using list of 25 semantically non-selective words (Reuters-RCV1 dataset)
    stop_words = ['a','an','and','are','as','at','be','by','for','from',
                  'has','he','in','is','it','its','of','on','that','the',
                  'to','was','were','will','with'] 
    filtered_tokens = [token for token in tweet_tokens if not token in stop_words]

    # 10 tokenized + filtered + stemmed
    ps = PorterStemmer()
    filtered_stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
        
    v8 = " ".join(tweet_tokens)
    v9 = " ".join(filtered_tokens)
    v10 = " ".join(filtered_stemmed_tokens)  
    
    return (v8, v9, v10)


if __name__=="__main__":

    # start counter 
    start = time.time()

    # get date and time 
    dt_object = datetime.datetime.fromtimestamp(start)
    dt_object = str(dt_object).split('.')[0]
    Date, Time = dt_object.split(' ')

    # setup log dir
    log_dir = os.path.join("logs")
    try:
        os.stat(log_dir)
    except:
        os.mkdir(log_dir)

    log_name = Date.replace('-', '') + '_cleanup_log'
    log_path = os.path.join(log_dir, log_name)

    # redirect stdout to log    
    stdoutOrigin = sys.stdout 
    sys.stdout = open(log_path, "w")
    print('Date: ' + Date)
    print('Time: ' + Time)
    print('\n')
    print('Tweet cleanup')
    print('-' * 45)

    # load
    print('Loading...\n')
    df = load_data()
    raw_nrows = df.shape[0]

    # create retweet col
    # Note: RT is uppercase, this has to be before cleanup
    df['Retweet'] = map_is_retweet(df['Text'].values)

    # cleanup Tweet text, return list of 3-tuples
    print('Cleaning...\n')
    url_extractor = urlextract.URLExtract()
    tuples = [cleanup_tweet(tweet) for tweet in df.loc[:,'Text']]
    
    # unpack 3-tuples
    df.loc[:, 'tokenized'], df.loc[:, 'filtered'], df.loc[:, 'stemmed'] = \
    [x[0] for x in tuples], [x[1] for x in tuples], [x[2] for x in tuples], 

    # create a subset with cols of interest
    df = df[['ID','Retweet','Polarity',
             'tokenized','filtered','stemmed']].copy()

    # dedupe (duplicates based on tokenized text)
    print('Deduping...\n')
    dupes = df[df['tokenized'].duplicated(keep='first')]
    pct_retweets_dupes = round(100*sum(dupes['Retweet'])/dupes.shape[0],2)
    df = df[~df.ID.isin(list(dupes['ID']))]
    
    deduped_nrows = df.shape[0]
    pct_retweets = round(100*sum(df['Retweet'])/df.shape[0],2)
    pct_duped = round(100*(raw_nrows-deduped_nrows)/raw_nrows,2)

    # save
    print('Saving...\n')
    filepath = os.path.join("..","data","2_clean","tweets")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = ''.join([today_prefix, "_tweets.csv"])

    df.to_csv(os.path.join(filepath, filename), index=False)
    
    end = time.time()
    elapsed = round(end-start,2)
    
    print('Cleanup successful.')
    print('Total (raw) number of rows: ' + str(raw_nrows))
    print('Final (deduplicated) number of rows: ' + str(deduped_nrows))
    print('% of duplicated tweets: ' + str(pct_duped))
    print('Overall % of retweets: ' + str(pct_retweets))
    print('% of retweets in duplicates: ' + str(pct_retweets_dupes))
    print('Time elapsed: ' + str(elapsed) + ' secs.')
    print('-' * 45)
    
    # finish log 
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    print('Script complete. See logs folder.')