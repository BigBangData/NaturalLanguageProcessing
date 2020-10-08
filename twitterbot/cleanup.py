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
import concurrent.futures

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

def load_data(ix_list):
    """Loads most recent deduped version.
    """
    dirpath = os.path.join("..","data","1.2_deduped","tweets")
    filename = sorted(os.listdir(dirpath), reverse=True)[0]
    filepath = os.path.join(dirpath, filename)
    # only read relevant rows
    df = pd.read_csv(
            filepath, 
            encoding='latin-1',
            skiprows=[ix for ix in range(1600001) if ix not in ix_list]
    )
    df.columns = ['ID','Timestamp','User','Text','Polarity'] 
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

def cleanup_tweet(tweet, url_extractor):
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

def clean_training_data(params):
       
    # unpack parameters
    ix_list, num = params
	
    start_time = time.time()
	
    # get date and time 
    dt_object = datetime.datetime.fromtimestamp(start_time)
    dt_object = str(dt_object).split('.')[0]
    Date, Time = dt_object.split(' ')

    log_name = Date.replace('-', '') + '_cleanup_log'
    log_path = os.path.join('logs', log_name)
	
    setup_logger(log_path)

    logger1 = logging.getLogger('train.load')
    logger2 = logging.getLogger('train.clean')
    logger3 = logging.getLogger('train.save') 
    
    try:
        df = load_data(ix_list)
        read_ix_list = ' '.join([str(min(ix_list)+1), '-', str(max(ix_list))])
        logger1.info(''.join(['Loading subset ', str(num), ', rows: ', read_ix_list]))
    except OSError as e:
        logger1.info('Could not load data. Check file permissions.')
        logger1.debug(e)
        sys.exit(1)
        
    # create retweet col
    # Note: RT is uppercase, this has to be before cleanup
    df['Retweet'] = map_is_retweet(df['Text'].values)
    
    # cleanup
    logger2.info(''.join(['Cleaning subset ', str(num), '...']))
    # initiate url extractor
    url_extractor = urlextract.URLExtract()
    df.loc[:, 'Lemmatized'] = [cleanup_tweet(tweet, url_extractor) for tweet in df.loc[:,'Text']]
    
    # Textlen col    
    df['Textlen'] = calc_textlen(df['Lemmatized'].values)

    # subset with cols of interest
    df = df[['Polarity','Lemmatized','Retweet','Textlen']].copy()

    save_dir = os.path.join("..","data","2_clean","tweets")   
    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = "".join([today_prefix, "_train_", str(num), ".csv"])
    df.to_csv(os.path.join(save_dir, filename), index=False)

    logger3.info(''.join(["Saving clean subset: ", str(num)]))


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:

        params_list = [
                       (range(     0,    50001),  1),
                       (range(  50000,  100001),  2),
                       (range( 100000,  150001),  3),
                       (range( 150000,  200001),  4),
                       (range( 200000,  250001),  5),
                       (range( 250000,  300001),  6),
                       (range( 300000,  350001),  7),
                       (range( 350000,  400001),  8),
                       (range( 400000,  450001),  9),
                       (range( 450000,  500001), 10),
                       (range( 500000,  550001), 11),
                       (range( 550000,  600001), 12),
                       (range( 600000,  650001), 13),
                       (range( 650000,  700001), 14),
                       (range( 700000,  750001), 15)
                      ]
        
        results = [executor.submit(clean_training_data, p) for p in params_list]
            

if __name__ == '__main__':

    # start counter 
    start_time = time.time()

    # get date and time 
    dt_object = datetime.datetime.fromtimestamp(start_time)
    dt_object = str(dt_object).split('.')[0]
    Date, Time = dt_object.split(' ')
    
    # setup loggging
    try:
        os.stat('logs')
    except:
        os.mkdir('logs')

    log_name = Date.replace('-', '') + '_cleanup_log'
    log_path = os.path.join('logs', log_name)
	
    setup_logger(log_path)
    
    logging.info('Date: ' + Date)
    logging.info('Time: ' + Time)
    logging.info('Tweet cleanup')

    # save dir
    save_dir = os.path.join("..","data","2_clean","tweets")
    try:
        os.path.exists(save_dir)
    except:
        os.makedirs(save_dir)
        
    # run processes
    main()

    # finish counter
    elapsed_time = round(time.time() - start_time, 4)
    
    # print results
    logging.info('Cleanup successful.')
    logging.info(''.join(['See ', str(log_path)]))
    logging.info('See ' +str(os.path.join(save_dir)) + ' for data.')
    logging.info('Time elapsed: ' + str(elapsed_time) + ' secs.')
