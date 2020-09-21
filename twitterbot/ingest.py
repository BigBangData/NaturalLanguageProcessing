#!/usr/bin/env python

# Ingest module for Twitter search API

import os
import sys
import json
import time
import tweepy
import datetime
import pandas as pd

def check_args():

    # error handling no args
    if len(sys.argv) < 2:
        print(
            "ERROR: No default number of runs; must supply integer between 1 and 44:\
            \nUSAGE: python ingest_tweets.py <nruns[1:44]> <ntweets=200[100:1000]>\
            ")
        sys.exit()

    # error handling first [mandatory] arg 
    try:
        if int(sys.argv[1]) not in range(1,45):
            print(
                "ERROR: Number of runs must be an integer between 1 and 44:\
                \nUSAGE: python ingest_tweets.py <nruns[1:44]> <ntweets=200[100:1000]>\
                ")
            sys.exit()
        else:
            nruns = int(sys.argv[1])
    except ValueError:
        print(
            "ERROR: Number of runs must be an integer between 1 and 44:\
            \nUSAGE: python ingest_tweets.py <nruns[1:44]> <ntweets=200[100:1000]>\
            ")
        sys.exit()
        
    # error handling second [optional] arg
    try:
        sys.argv[2]
    except IndexError:
        return (nruns, 200) # default ntweeets
    else:
        try:
            if int(sys.argv[2]) not in range(100, 1001):
                print(
                    "ERROR: Number of tweets must be an integer between 100 and 1000:\
                    \nUSAGE: python ingest_tweets.py <nruns[1:44]> <ntweets=200[100:1000]>\
                    ")
                sys.exit()
            else:
                ntweets = int(sys.argv[2])
                return (nruns, ntweets)
        except ValueError:
                print(
                    "ERROR: Number of tweets must be an integer between 100 and 1000:\
                    \nUSAGE: python ingest_tweets.py <nruns[1:44]> <ntweets=200[100:1000]>\
                    ")
                sys.exit()

def initiate_api():
    
    filepath = os.path.join("..",".conf","config.json")
    with open(filepath, 'r') as f:
        config = json.load(f)
        
    auth = tweepy.OAuthHandler(
        config["CONSUMER_KEY"], config["CONSUMER_SECRET"]
    )
    auth.set_access_token(
        config["ACCESS_KEY"], config["ACCESS_SECRET"]
    )
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def collect_tweets(api, ntweets):
    
    neg_query = '(🤬 OR 🤮 OR 😡 OR 😤 OR 🥺 OR 🤢 OR 😣 OR \
                  😟 OR 😣 OR 🤔 OR 🤥 OR 😫 OR 🤮 OR 🥵 OR \
                  😨 OR 😰 OR 😭 OR 😥 OR 🙁 OR 😩) \
            AND \
                -(😃 OR 😄 OR 😁 OR 🥰 OR 😊 OR ❤️ OR 💋 OR \
                  😍 OR 😂 OR 😎 OR 🤣 OR 😘 OR 😇 OR 🙃 OR \
                  😉 OR 😇 OR 🤩 OR 😃 OR 😄 OR 🙂) \
            AND \
                -(😭)' # unclear polarity

    pos_query = '(😃 OR 😄 OR 😁 OR 🥰 OR 😊 OR ❤️ OR 💋 OR \
                  😍 OR 😂 OR 😎 OR 🤣 OR 😘 OR 😇 OR 🙃 OR \
                  😉 OR 😇 OR 🤩 OR 😃 OR 😄 OR 🙂) \
            AND \
                -(🤬 OR 🤮 OR 😡 OR 😤 OR 🥺 OR 🤢 OR 😣 OR \
                  😟 OR 😣 OR 🤔 OR 🤥 OR 😫 OR 🤮 OR 🥵 OR \
                  😨 OR 😰 OR 😭 OR 😥 OR 🙁 OR 😩) \
            AND \
                -(😭)' # unclear polarity                  
    
    tweets = []
    # collect negative tweets
    for status in tweepy.Cursor(api.search,
                                q=neg_query,
                                include_entities=True,
                                monitor_rate_limit=True, 
                                wait_on_rate_limit=True,
                                lang="en").items(ntweets/2):
        tweets.append([status.id_str,
                       status.created_at, 
                       status.user.screen_name, 
                       status.text,
                       -1])
        
    # collect positive tweets
    for status in tweepy.Cursor(api.search,
                                q=pos_query,
                                include_entities=True,
                                monitor_rate_limit=True, 
                                wait_on_rate_limit=True,
                                lang="en").items(ntweets/2):
        tweets.append([status.id_str,
                       status.created_at, 
                       status.user.screen_name, 
                       status.text,
                       1])
    return tweets

def save_raw_tweets(tweets):

    tweets_df = pd.DataFrame(
        tweets, 
        columns=["ID", "Timestamp", "User", "Text", "Polarity"]
    )
    
    now_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("..","data","1_raw","tweets")
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    filename = ''.join([now_prefix, "_tweets.csv"])
    
    tweets_df.to_csv(os.path.join(filepath, filename), index=False)

def twitter_bot(api, ntweets):
    
    tweets = collect_tweets(api, ntweets)  
    save_raw_tweets(tweets)
    
    t = datetime.datetime.now().strftime("%H:%M:%S")
    print('New file saved at: ', str(t))

def main(nruns, ntweets):

    t1 = datetime.datetime.now().strftime("%H:%M:%S")
    print('api started at: ', str(t1))
    
    # start API
    api = initiate_api()
    
    for i in range(nruns):
        
        print('Run No.', str(i+1))
        t2 = datetime.datetime.now().strftime("%H:%M:%S")
        print('twitter bot run at: ', str(t2))
        
        # run bot
        twitter_bot(api, ntweets)
  
        # wait 15 min, except last time
        if i < nruns-1:
            print('waiting 15 mins...')
            time.sleep(915)

if __name__=="__main__":
    
    nruns, ntweets = check_args()
    
    main(nruns, ntweets)