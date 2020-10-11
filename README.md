Twitter Sentiment Analysis
======================


FINAL TODO: 
    
    - README.md should be the presentation
    - Organize projects: too many goals (Sentiment Analysis, test QWERTY Effect, TF-IDF guide)
    - Main purpose could be study of data drift, sentiment140 vs new 1.6M Tweets from twitterbot
   
NOW TODO:

    - beware of diffs; copy changes to/from sentiment140 and twitterbot (std funcs)

    - Restructured entire project: follow IntroToTextAnalytics workflow

    - FEATURE ENGINEERING IDEAS:
        - count punctuations, count ascii text, count USERNAME, count EMOJI
        - tweet starts with USERNAME
        - actual semantic stuff like count of "bad words" and count of "positive words"
    - split train/test data while raw - done 
    - then cleanup - ongoing (see sentiment140 cleanup JN)
    - then pre-process
    - careful with IDF calculation (cache IDF)
    - calculate SVD