Twitter Sentiment Analysis
===

FINAL TODO: 
    
    - README.md should be the presentation
    - Organize projects: too many goals (Sentiment Analysis, test QWERTY Effect, TF-IDF guide)
    - Main purpose could be study of data drift, sentiment140 vs new 1.6M Tweets from twitterbot
   
NOW TODO:

    - beware of diffs; copy changes to/from sentiment140 and twitterbot (std funcs)

    - Restructured entire project: follow IntroToTextAnalytics workflow

    - FEATURE ENGINEERING IDEAS:
        - count of: punctuations, ascii chars, USERNAMEs, EMOJIs, URLs
        - tweet starts with USERNAME, or EMOJI, or URL, etc.
        - semantic analysis?
			- count of swear words, or negative words, or positive words, etc. (need lists)
    - split train/test data while raw: done 
    - cleanup: done (maybe better EDA)
    - pre-process: ongoing
		- careful with IDF calculation (cache IDF)
		- Latent Semantic Analyss + SVD...