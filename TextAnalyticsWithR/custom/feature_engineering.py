#!/usr/bin/env python
import re
import os
import time
import json

import numpy as np
import pandas as pd
import urlextract

from html import unescape
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# functions
def expand_contractions(text, contractions_map):
    
    pattern = re.compile('({})'.format('|'.join(contractions_map.keys())), 
                        flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_map.get(match)\
                                if contractions_map.get(match)\
                                else contractions_map.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def is_ascii(doc):
    try:
        doc.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def calc_rsr(txt):
    """Calculates the ratio of characters in 
    the right-side of the QWERTY keyboard, also
    known as RSR (Right-Side Ratio), given a 
    lower-case text object.
    """
    lside = ['q','w','e','r','t',
             'a','s','d','f','g',
             'z','x','c','v','b']
    rside = ['y','u','i','o','p',
             'h','j','k','l',
             'n','m']
    txt = str(txt)
    sub_string = [x for x in txt]
    lcount = rcount = 0
    for i in sub_string:
        if i in lside:
            lcount += 1
        elif i in rside:
            rcount += 1
        else:
            pass
    den = rcount+lcount
    if den != 0:
        return round(rcount / den, 4)
    else:
        return 0
        
# load contractions map
with open("contractions_map.json") as f:
    contractions_map = json.load(f)
    
# instantiate url extractor and lemmatizer
url_extractor = urlextract.URLExtract()
lemmatizer = WordNetLemmatizer()


class DocumentToFeaturesCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, expand_contractions=True, lower_case=True, 
                 replace_usernames=True, unescape_html=True, 
                 replace_urls=True, replace_numbers=True, 
                 remove_junk=True, remove_punctuation=True, 
                 replace_emojis=True, replace_nonascii=True, 
                 remove_stopwords=True, lemmatization=True):
        self.expand_contractions = expand_contractions
        self.lower_case = lower_case
        self.replace_usernames = replace_usernames
        self.unescape_html = unescape_html
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.remove_junk = remove_junk
        self.remove_punctuation = remove_punctuation
        self.replace_emojis = replace_emojis
        self.replace_nonascii = replace_nonascii
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # raw len
        doclen_raw = [len(doc) for doc in X]
        doclen_clean = []
        n_tokens = []
        wordlen_max = []
        wordlen_mean = []
        wordlen_std = []
        rsr_ = []
        clean_docs = []
        for doc in X:
            if self.lower_case:
                doc = doc.lower()
            if self.expand_contractions and contractions_map is not None:
                doc = expand_contractions(doc, contractions_map)
            if self.replace_usernames:
                doc = re.sub(r'^@([^\s]+)','usr', doc)
            if self.unescape_html:
                doc = unescape(doc)
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(doc)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    doc = doc.replace(url, 'url')
            if self.replace_numbers:
                doc = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'nur', doc)
            if self.remove_punctuation:
                doc = re.sub(r'\W+', ' ', doc, flags=re.M)
            if self.remove_junk:
                pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\¿|\x82\
                            |\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                            |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
                doc = re.sub(pattern, 'jek', doc)
            if self.replace_emojis:
                doc = re.sub(r'[^\x00-\x7F]+', 'emj', doc)
            if self.replace_nonascii:
                if is_ascii(doc) == False:
                    doc = 'nas'
            # clean len
            clean_docs.append(doc.strip())
            doclen_clean.append(len(doc.strip()))
            rsr_.append(calc_rsr(doc.strip()))
            # tokenize
            tokens = doc.split()
            lengths = [len(t) for t in tokens]
            n_tokens.append(len(tokens))
            # token len stats
            try:
                wordlen_max.append(max(lengths))
            except ValueError:
                wordlen_max.append(0)
            try:
                wordlen_mean.append(round(np.mean(lengths),4))
            except ValueError:
                wordlen_mean.append(0)  
            try:
                wordlen_std.append(round(np.std(lengths),4))
            except ValueError:
                wordlen_std.append(0)
        # list of lists
        X_transformed = np.array([
                                 doclen_raw,
                                 doclen_clean,
                                 n_tokens,
                                 wordlen_max,
                                 wordlen_mean,
                                 wordlen_std,
                                 rsr_   
                                ])
        return clean_docs, X_transformed.T