#!/usr/bin/env python
import os
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

# TODO: redo cleanup function to return non-stemmed text for RSR calc.

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
                              skiprows=[ix for ix in range(1600001) if ix not in ix_list])
        
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
    load_dir = os.path.join("..","data","1_raw","sentiment140",
                            "training.1600000.processed.noemoticon.csv")
    df = load_dataset(filepath=load_dir,
                      col_ix=[0, 5], 
                      col_names=['target', 'text'], 
                      ix_list=ix_list)

    # clean data
    df.loc[:, 'text'] = vector_clean(df.loc[:, 'text'])

    # make target {0,1} 
    df.loc[df.target == 4, 'target'] = 1
    
    # save data
    save_dir = os.path.join("..","data","2_clean","sentiment140")
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    filename = "".join(["train_", str(num), ".csv"])
    df.to_csv(os.path.join(save_dir, filename), index=False)
    
    # print our result
    result=''.join(["Saving cleaned up train dataset: ", str(num)])
    return result

def run_processes():
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
                       (range( 700000,  750001), 15),
                       (range( 750000,  800001), 16),
                       (range( 800000,  850001), 17),
                       (range( 850000,  900001), 18),
                       (range( 900000,  950001), 19),
                       (range( 950000, 1000001), 20),
                       (range(1000000, 1050001), 21),
                       (range(1050000, 1100001), 22),
                       (range(1100000, 1150001), 23),
                       (range(1150000, 1200001), 24),
                       (range(1200000, 1250001), 25),
                       (range(1250000, 1300001), 26),
                       (range(1300000, 1350001), 27),
                       (range(1350000, 1400001), 28),
                       (range(1400000, 1450001), 29),
                       (range(1450000, 1500001), 30),
                       (range(1500000, 1550001), 31),
                       (range(1550000, 1600001), 32)
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
