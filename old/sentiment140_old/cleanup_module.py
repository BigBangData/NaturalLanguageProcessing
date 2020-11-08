#!/usr/bin/env python
import os
import re
import sys
import time

import emoji
import string
import urlextract
import pandas as pd
import concurrent.futures

from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_data(params):
    """Wrapper for cleanup_tweet multi-processing which loads 50k-row chunks 
       of a maximum 1.2M-row dataset (size of our train data) and cleans up
       chunks in asynchronous and parallel manner.
       Args
       ----
           params : (X_name, ix_list, num)
               X_name  : name of train or test set w/ cols: ID, Username, Text
               ix_list : a 50k range of row indices (ix) for data chunks
                   num : a chunk number
    """
    def load_subset(filepath, col_ix, col_names, ix_list):
        """Loads a subset of the train or test data.
           Args
           ----
               filepath  : path to X
               col_ix    : indices for cols
               col_names : names of cols
               ix_list   : indices of rows
        """
        dataset = pd.read_csv(filepath, encoding='latin-1', usecols=col_ix, 
                              skiprows=[ix for ix in range(1200001) if ix not in ix_list])
        dataset.columns = col_names
        return dataset

    def cleanup_tweet(tweet):
        """Cleans up a tweet's text with the following steps:
            1. make lower case
            2. anonymize usernames
            3. replace URLs
            4. unescape HTML entities
            5. remove extraneous characters
            6. remove punctuation
            7. replace emojis
            8. replace non-ascii decodable text (after utf-8 encoding)
            9. tokenize
            10. filter stop words from tokens
            11. lemmatize filtered tokens

        The function returns the final lemmatized and filtered tokens.

        Note: NLTK's set(stopwords.words('english')) is too comprehensive
              so this uses the 25 semantically non-selective words from 
              the Reuters-RCV1 dataset.
        """
        # lower case
        tweet = tweet.lower()
        
        # anonymize usernames (replace with 'USERNAME')
        tweet = re.sub(r'@([^\s]+)','USERNAME', tweet)
        
        # replace URLs with 'URL'
        urls = list(set(url_extractor.find_urls(tweet)))
        if len(urls) > 0:
            for url in urls:
                tweet = tweet.replace(url, 'URL')
                
        # unescape HTML
        tweet = unescape(tweet)
        
        # remove extraneous characters
        pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\!|\?|\¿|\x82\
                    |\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                    |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
        tweet = re.sub(pattern,'', tweet)
        
        # remove punctuations
        # (will count them later as part of feature engineering)
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
       
        # return text representations of emojis then replace them with 'EMOJI'
        # using emojis to label pos/neg tweets so cannot use those as predictors
        #tweet = emoji.demojize(tweet)
        #tweet = re.sub(r'(:[^:]*:)', ' EMOJI ', tweet)
        
        # better version still...
        tweet = re.sub(r'[^\x00-\x7F]+', ' EMOJI ', tweet)
        
        # replace non-ascii decodable text with 'NONASCII'
        def is_ascii(text):
            try:
                text.encode(encoding='utf-8').decode('ascii')
            except UnicodeDecodeError:
                return False
            else:
                return True

        if is_ascii(tweet) == False:
            return ' NONASCII '
        else:
            pass

        # tokenize
        tweet_tokens = word_tokenize(tweet)
        retweet = ['rt']
        tweet_tokens = [token for token in tweet_tokens \
                        if not token in retweet]
        
        # remove stop words
        stop_words = ['a','an','and','are','as','at','be','by','for','from',
                      'has','he','in','is','it','its','of','on','that','the',
                      'to','was','were','will','with'] 
        filtered_tokens = [token for token in tweet_tokens \
                           if not token in stop_words]
        
        # lemmatize
        word_lem = WordNetLemmatizer()
        lemmatized_tokens = [word_lem.lemmatize(token) \
                             for token in filtered_tokens]
        
        return " ".join(lemmatized_tokens)

    def vector_clean(list_):
        map_iterator = map(cleanup_tweet, list_)
        return list(map_iterator)

    # unpack parameters
    X_name, ix_list, num = params

    # instantiate url extractor
    url_extractor = urlextract.URLExtract()
    
    # load data
    load_dir = os.path.join("..","data","1_raw","sentiment140")
    filename = ''.join([X_name, ".csv"])
    filepath = os.path.join(load_dir, filename)
    
    df = load_subset(filepath=filepath,
                     col_ix=[1,2], 
                     col_names=['username','text'], 
                     ix_list=ix_list)

    # cleanup text
    df.loc[:, 'lemmatized'] = vector_clean(df.loc[:, 'text'])
    
    # save data
    save_dir = os.path.join("..","data","2_clean","sentiment140")
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    filename = "".join([X_name, '.', str(num), ".csv"])
    df.to_csv(os.path.join(save_dir, filename), index=False)
    
    # print out results
    result=''.join(["Saving ", '.'.join(filename.split('.')[:2])])
    return result

def main(X_name):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        if X_name == 'X_train':
            params_list = [
                           (X_name, range(     0,    50001),  1),
                           (X_name, range(  50000,  100001),  2),
                           (X_name, range( 100000,  150001),  3),
                           (X_name, range( 150000,  200001),  4),
                           (X_name, range( 200000,  250001),  5),
                           (X_name, range( 250000,  300001),  6),
                           (X_name, range( 300000,  350001),  7),
                           (X_name, range( 350000,  400001),  8),
                           (X_name, range( 400000,  450001),  9),
                           (X_name, range( 450000,  500001), 10),
                           (X_name, range( 500000,  550001), 11),
                           (X_name, range( 550000,  600001), 12),
                           (X_name, range( 600000,  650001), 13),
                           (X_name, range( 650000,  700001), 14),
                           (X_name, range( 700000,  750001), 15),
                           (X_name, range( 750000,  800001), 16),
                           (X_name, range( 800000,  850001), 17),
                           (X_name, range( 850000,  900001), 18),
                           (X_name, range( 900000,  950001), 19),
                           (X_name, range( 950000, 1000001), 20),
                           (X_name, range(1000000, 1050001), 21),
                           (X_name, range(1050000, 1100001), 22),
                           (X_name, range(1100000, 1150001), 23),
                           (X_name, range(1150000, 1200001), 24)
                          ]
        if X_name == 'X_test':
            params_list = [
                           (X_name, range(     0,    50001),  1),
                           (X_name, range(  50000,  100001),  2),
                           (X_name, range( 100000,  150001),  3),
                           (X_name, range( 150000,  200001),  4),
                           (X_name, range( 200000,  250001),  5),
                           (X_name, range( 250000,  300001),  6),
                           (X_name, range( 300000,  350001),  7),
                           (X_name, range( 350000,  400001),  8)
                          ]
            
        results = [executor.submit(clean_data, p) for p in params_list]

        # get results with the as_completed function, which gives us an iterator 
        # we loop over to yield results of our processes as they're completed
        for f in concurrent.futures.as_completed(results):
            print(f.result())
            

if __name__ == '__main__':

    start_time = time.time()
    
    # X_name should be 1st arg
    # TODO: error check
    X_name = sys.argv[1]
    
    # run main func
    main(X_name)
    
    # print out running time
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"Cleanup time: {mins:0.0f} minute(s) and {secs:0.0f} second(s).")