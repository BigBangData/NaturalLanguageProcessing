#!/usr/bin/env python

# Dedupe module for Twitter search API
import os
import re
import sys
import json
import time
import datetime
import pandas as pd

def load_data():
    filepath = os.path.join("..","data","1_raw","tweets") 
    dfm = []
    for f in os.listdir(filepath):
        dfm.append(pd.read_csv(os.path.join(filepath,f)))
    df = pd.concat(dfm)
    df = df.reset_index(drop=True)
    return df


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

    log_name = Date.replace('-', '') + '_dedupe_log'
    log_path = os.path.join(log_dir, log_name)

    # redirect stdout to log 
    stdoutOrigin = sys.stdout 
    sys.stdout = open(log_path, "w")
    print('Date: ' + Date)
    print('Time: ' + Time)
    print('\n')
    print('Tweet deduplication')
    print('-' * 45)

    # load
    print('Loading...\n')
    df = load_data()
    print('Raw data nrows: ' + str(df.shape[0]))

    # dedupe using Tweet text
    print('Deduping...\n')
    dupes = df[df['Text'].duplicated(keep='first')]
    print("% dupes: " + str(100*round(dupes.shape[0]/df.shape[0], 4)))
    
    deduped_df = df[~df.ID.isin(dupes['ID'])]
    print('Deduped data nrows: ' + str(deduped_df.shape[0]))

    # reset index
    deduped_df.index = range(len(deduped_df.index))
    
    # polarity class balance after deduping
    polarity_df = deduped_df[['ID','Polarity']].groupby('Polarity').count()
    target_diff = abs(polarity_df['ID'][1] - polarity_df['ID'][-1])
    tot_denom = sum(polarity_df['ID'])
    pct_target_diff = 100*round(target_diff / tot_denom, 4)
    print("% diff in target (Polarity): " + str(pct_target_diff))
    
    # save
    print('Saving...\n')
    filepath = os.path.join("..","data","2_deduped","tweets")
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    today_prefix = datetime.datetime.now().strftime("%Y%m%d")
    filename = ''.join([today_prefix, "_tweets.csv"])

    df.to_csv(os.path.join(filepath, filename), index=False)
    
    end = time.time()
    elapsed = round(end-start, 2)
    
    print('Deduplication successful.')
    print('Time elapsed: ' + str(elapsed) + ' secs.')
    print('-' * 45)
    
    # finish log 
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    print('Script complete. See logs folder.')
