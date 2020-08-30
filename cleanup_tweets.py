#!/usr/bin/env python
import re
import time
import string
import urlextract
import pandas as pd
import concurrent.futures

from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def clean_training_data(params):
    """Cleans Tweets for many known issues - not a general function but specifically
    tailored to the 1.6 M row training.1600000.processed.noemoticon.csv dataset.
    
    [TODO: some folks do not,ever,use spaces after punctuation.Believe it!Fix.]
    """
    
    # unpack parameters
    ix_list, num = params

    # instantiate url extractor
    url_extractor = urlextract.URLExtract()

    def load_dataset(filepath, col_ix, col_names, ix_list):
        dataset = pd.read_csv(filepath, encoding='latin-1', usecols=col_ix, 
                              skiprows=[ix for ix in range(1600000) if ix not in ix_list])
        dataset.columns = col_names
        return dataset

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
        
        tweet = re.sub(pattern,'', tweet)

        # remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        
        # remove stopwords
        # NLTK's set(stopwords.words('english')) removes too many words
        # using list of 25 semantically non-selective words (Reuters-RCV1 dataset)
        stop_words = ['a','an','and','are','as','at','be','by','for','from',
                      'has','he','in','is','it','its','of','on','that','the',
                      'to','was','were','will','with']

        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in stop_words]

        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]

        return " ".join(stemmed_words)

    def vector_clean(list_):
        map_iterator = map(cleanup_tweet, list_)
        return list(map_iterator)

    # load data
    df = load_dataset(filepath="./data/raw/training.1600000.processed.noemoticon.csv",
                      col_ix=[0, 5], 
                      col_names=['target', 'text'], 
                      ix_list=ix_list)

    # clean data
    df.loc[:, 'text'] = vector_clean(df.loc[:, 'text'])

    # make target {0,1} 
    df.loc[df.target == 4, 'target'] = 1
    
    # save data
    df.to_csv(''.join(["./data/clean/train_", str(num), ".csv"]), index=False)
    
    # print our result
    result=''.join(["Saving cleaned up train dataset: ", str(num)])
    return result

def run_processes():
    with concurrent.futures.ProcessPoolExecutor() as executor:

        params_list = [
                       (range(     0,    50000),  1),
                       (range(  50000,  100000),  2),
                       (range( 100000,  150000),  3),
                       (range( 150000,  200000),  4),
                       (range( 200000,  250000),  5),
                       (range( 250000,  300000),  6),
                       (range( 300000,  350000),  7),
                       (range( 350000,  400000),  8),
                       (range( 400000,  450000),  9),
                       (range( 450000,  500000), 10),
                       (range( 500000,  550000), 11),
                       (range( 550000,  600000), 12),
                       (range( 600000,  650000), 13),
                       (range( 650000,  700000), 14),
                       (range( 700000,  750000), 15),
                       (range( 750000,  800000), 16),
                       (range( 800000,  850000), 17),
                       (range( 850000,  900000), 18),
                       (range( 900000,  950000), 19),
                       (range( 950000, 1000000), 20),
                       (range(1000000, 1050000), 21),
                       (range(1050000, 1100000), 22),
                       (range(1100000, 1150000), 23),
                       (range(1150000, 1200000), 24),
                       (range(1200000, 1250000), 25),
                       (range(1250000, 1300000), 26),
                       (range(1300000, 1350000), 27),
                       (range(1350000, 1400000), 28),
                       (range(1400000, 1450000), 29),
                       (range(1450000, 1500000), 30),
                       (range(1500000, 1550000), 31),
                       (range(1550000, 1600000), 32)
                      ]
        
        results = [executor.submit(clean_training_data, p) for p in params_list]

        # get results with the as_completed function, which gives us an iterator 
        # we loop over to yield results of our processes as they're completed
        for f in concurrent.futures.as_completed(results):
            print(f.result())


if __name__ == '__main__':

    start = time.perf_counter()
    run_processes()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
