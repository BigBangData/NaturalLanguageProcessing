Twitter Sentiment Analysis
======================
    
	
DONE:
    - removed logger.info() from filelock.py 

TODO: 
    
    - README.md should be the presentation
    - Organize projects: too many goals (Sentiment Analysis, test QWERTY Effect, TF-IDF guide)
    - Main purpose could be study of data drift, sentiment140 vs new 1.6M Tweets from twitterbot
   
NEXT TODO:

	- console logging ok, not so much logs
    - multi-processing for twitterbot cleanup.py (use mod to determine remainder after div by 50k)
    - copy changes to/from sentiment140 and twitterbot
    - Restructured entire project: follow IntroToTextAnalytics workflow
    - split train/test data while raw
    - then cleanup
    - then pre-process
    - careful with IDF calculation (cache IDF)
    - calculate SVD
    - etc. 
    - standardize functions across sentiment140 and twitterbot
    - purpose could be study of data drift