#!/usr/bin/env python

# Cleanup module for Twitter search API 
import os
import re
import sys
import json
import time
import string
import logging
import datetime
import urlextract
import pandas as pd

from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def setup_logger(log_path):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    
def load_data():
    """Loads most recent deduped version.
    """
    dirpath = os.path.join("..","data","1.2_deduped","tweets") 
    filename = sorted(os.listdir(dirpath), reverse=True)[0]
    filepath = os.path.join(dirpath, filename)
    df = pd.read_csv(filepath)
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

def calc_textlen(col):
    bool_map = map(lambda x: len(x), col)
    return(list(bool_map))

def cleanup_tweet(tweet):
    """Cleans up a tweet with the following steps:
        1. make lower case
        2. remove URLs
        3. unescape HTML entities
        4. remove extraneous characters
        5. remove punctuation
        6. remove emojis
        7. discard non-ascii decodable text after utf-8 encoding
        8. tokenize
        9. filter stop words from tokens
        10. lemmatize filtered tokens
        
    The function returns the final lemmatized and filtered tokens.
    
    Note: NLTK's set(stopwords.words('english')) is too comprehensive
          so this uses the 25 semantically non-selective words from 
          the Reuters-RCV1 dataset.
    """
    tweet = tweet.lower() # 1

    urls = list(set(url_extractor.find_urls(tweet))) # 2
    if len(urls) > 0:
        for url in urls:
            tweet = tweet.replace(url, "")

    tweet = unescape(tweet) # 3
      
    pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\!|\?|\¿|\x82 \
                |\x83|\x84|\x85|\x86|\x87|\x88|\x89 \
                |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
    
    tweet = re.sub(pattern,'', tweet) # 4  
    
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)) # 5
    
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet).strip() # 6 
    
    def is_ascii(text):
        try:
            text.encode(encoding='utf-8').decode('ascii')  # 7
        except UnicodeDecodeError:
            return False
        else:
            return True
    
    if is_ascii(tweet) == False:
        return " "
    else:
        pass
        
    # 8
    tweet_tokens = word_tokenize(tweet)
    retweet = ['rt']
    tweet_tokens = [token for token in tweet_tokens if not token in retweet]
    
    # 9
    stop_words = ['a','an','and','are','as','at','be','by','for','from',
                  'has','he','in','is','it','its','of','on','that','the',
                  'to','was','were','will','with'] 
    filtered_tokens = [token for token in tweet_tokens if not token in stop_words]

    # 10
    word_lem = WordNetLemmatizer()
    filtered_lemmatized_tokens = [word_lem.lemmatize(token) for token in filtered_tokens]
    
    return " ".join(filtered_lemmatized_tokens)


if __name__=="__main__":

    # start counter 
    start_time = time.time()

    # get date and time 
    dt_object = datetime.datetime.fromtimestamp(start_time)
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
    
    setup_logger(log_path)
    
    logger1 = logging.getLogger('load')
    logger2 = logging.getLogger('cleanup')
    logger3 = logging.getLogger('save') 
    
    logging.info('Date: ' + Date)
    logging.info('Time: ' + Time)
    logging.info('Tweet cleanup')

    # load
    logging.info('Loading...')
    start_load = time.time()
    try:
        df = load_data()
    except OSError as e:
        logger1.error('Could not load data')
        logger1.error(e)
        logger1.debug('Check file permissions or extra folder')
        sys.exit(1)
        
    load_time = round(time.time() - start_load, 4)
    logger1.info('Loading time: ' + str(load_time) + ' secs.')
    logger1.info('Deduped data cols: ' + ', '.join(list(df.columns)))
    logger1.info('Deduped data nrows: ' + str(df.shape[0]))
                 
    # create retweet col
    # Note: RT is uppercase, this has to be before cleanup
    logger2.info('Creating Retweet column.')
    df['Retweet'] = map_is_retweet(df['Text'].values)
    
    # cleanup
    logger2.info('Cleaning up text column...')
    start_cleanup = time.time()
    url_extractor = urlextract.URLExtract()
    df.loc[:, 'Lemmatized'] = [cleanup_tweet(tweet) for tweet in df.loc[:,'Text']]

    cleanup_time = round(time.time() - start_cleanup, 4)
    logger2.info('Cleanup time: ' +str(cleanup_time) + ' secs.')

    # create textlen col
    logger2.info('Creating text length column...')    
    df['Textlen'] = calc_textlen(df['Lemmatized'].values)

    # create a subset with cols of interest
    logger2.info('Subsetting columns...')
    df = df[['Polarity','Lemmatized','Retweet','Textlen']].copy()

    # save
    logger3.info('Saving...')  
    logger3.info('Cleaned data cols: ' + ', '.join(list(df.columns)))
    
    filepath = os.path.join("..","data","2_clean","tweets")
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = ''.join([today_prefix, "_tweets.csv"])

    df.to_csv(os.path.join(filepath, filename), index=False)
   
    elapsed_time = round(time.time() - start_time, 4)
    
    logging.info('Cleanup successful.')
    logging.info('See ' +str(os.path.join(filepath, filename)) + ' for data.')
    logging.info('Time elapsed: ' + str(elapsed_time) + ' secs.')
 
    print('Script complete. See ' + log_path)