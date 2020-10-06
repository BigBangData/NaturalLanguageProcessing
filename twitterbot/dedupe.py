#!/usr/bin/env python

# Dedupe module for Twitter search API
import os
import re
import sys
import json
import time
import logging
import datetime
import pandas as pd

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
    filepath = os.path.join("..","data","1_raw","tweets") 
    dfm = []
    for f in os.listdir(filepath):
        dfm.append(pd.read_csv(os.path.join(filepath, f)))
    df = pd.concat(dfm)
    df = df.reset_index(drop=True)
    return df


if __name__=="__main__":

    # start counter 
    start_time = time.time()

    # get date and time 
    dt_object = datetime.datetime.fromtimestamp(start_time)
    dt_object = str(dt_object).split('.')[0]
    Date, Time = dt_object.split(' ')

    # setup loggging
    log_dir = os.path.join("logs")
    try:
        os.stat(log_dir)
    except:
        os.mkdir(log_dir)

    log_name = Date.replace('-', '') + '_dedupe_log'
    log_path = os.path.join(log_dir, log_name)
    
    setup_logger(log_path)
    
    logger1 = logging.getLogger('load')
    logger2 = logging.getLogger('dedup')
    logger3 = logging.getLogger('save') 
    
    logging.info('Date: ' + Date)
    logging.info('Time: ' + Time)
    logging.info('Tweet deduplication')

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
    logger1.info('Raw data nrows: ' + str(df.shape[0]))

    # dedupe using Tweet text
    logger2.info('Deduping...')
    start_dedup = time.time()
    
    # dedupe
    dupes = df[df['Text'].duplicated(keep='first')]
    deduped_df = df[~df.ID.isin(dupes['ID'])]
    # reset index
    deduped_df.index = range(len(deduped_df.index))
    # calculating polarity class balance after deduping
    polarity_df = deduped_df[['ID','Polarity']].groupby('Polarity').count()
    target_diff = abs(polarity_df['ID'][1] - polarity_df['ID'][-1])
    tot_denom = sum(polarity_df['ID'])
    pct_target_diff = 100*round(target_diff / tot_denom, 4)
    
    dedup_time = round(time.time() - start_dedup, 4)
    logger2.info('Dedupe time: ' +str(dedup_time) + ' secs.')
    logger2.info('Deduped data nrows: ' + str(deduped_df.shape[0]))
    logger2.info("% dupes: " + str(100*round(dupes.shape[0]/df.shape[0], 4)))
    logger2.info("% diff in target (Polarity): " + str(pct_target_diff))
    
    # save
    logger3.info('Saving...')
    filepath = os.path.join("..","data","1.2_deduped","tweets")
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = ''.join([today_prefix, "_deduped_tweets.csv"])

    df.to_csv(os.path.join(filepath, filename), index=False)
    
    elapsed_time = round(time.time() - start_time, 4)
    
    logging.info('Deduplication successful.')
    logging.info('See ' +str(os.path.join(filepath, filename)) + ' for data.')
    logging.info('Time elapsed: ' + str(elapsed_time) + ' secs.')
 
    print('Script complete. See ' + log_path)
