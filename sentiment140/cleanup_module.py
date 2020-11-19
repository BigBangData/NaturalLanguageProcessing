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

# load contractions map
with open("contractions_map.json") as f:
    contractions_map = json.load(f)

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
    
# instantiate url extractor and lemmatizer
url_extractor = urlextract.URLExtract()
lemmatizer = WordNetLemmatizer()


# classes          
class DocumentToWordCounterTransformer(BaseEstimator, TransformerMixin):
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
        X_transformed = []
        for doc in X:
            if self.lower_case:
                doc = doc.lower()
            if self.expand_contractions and contractions_map is not None:
                doc = expand_contractions(doc, contractions_map)
            if self.replace_usernames:
                doc = re.sub(r'^@([^\s]+)',' USR ', doc)
            if self.unescape_html:
                doc = unescape(doc)
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(doc)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    doc = doc.replace(url, ' URL ')
            if self.replace_numbers:
                doc = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', ' NUM ', doc)
            if self.remove_punctuation:
                doc = re.sub(r'\W+', ' ', doc, flags=re.M)
            if self.remove_junk:
                pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\¿|\x82\
                            |\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                            |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
                doc = re.sub(pattern,'', doc)
            if self.replace_emojis:
                doc = re.sub(r'[^\x00-\x7F]+', ' EMOJI ', doc)
            if self.replace_nonascii:
                if is_ascii(doc) == False:
                    doc = ' NONASCII '
            word_counts = Counter(doc.split())
            if self.remove_stopwords:
                # 25 semantically non-selective words from the Reuters-RCV1 dataset
                stop_words = ['a','an','and','are','as','at','be','by','for','from',
                              'has','he','in','is','it','its','of','on','that','the',
                              'to','was','were','will','with']
                for word in stop_words:
                    try:
                        word_counts.pop(word)
                    except KeyError:
                        continue
            if self.lemmatization and lemmatizer is not None:
                lemmatized_word_counts = Counter()
                for word, count in word_counts.items():
                    lemmatized_word = lemmatizer.lemmatize(word)
                    lemmatized_word_counts[lemmatized_word] += count
                word_counts = lemmatized_word_counts      
            X_transformed.append(word_counts)
        return np.array(X_transformed)
    
    
class DocumentToBigramCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, expand_contractions=True, lower_case=True, 
                 replace_usernames=True, unescape_html=True, 
                 replace_urls=True, replace_numbers=True, 
                 remove_junk=True, remove_punctuation=True, 
                 replace_emojis=True, replace_nonascii=True, 
                 remove_stopwords=True, lemmatization=True,
                 bigrams=True):
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
        self.bigrams = bigrams
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for doc in X:
            if self.lower_case:
                doc = doc.lower()
            if self.expand_contractions and contractions_map is not None:
                doc = expand_contractions(doc, contractions_map)
            if self.replace_usernames:
                doc = re.sub(r'^@([^\s]+)',' USR ', doc)
            if self.unescape_html:
                doc = unescape(doc)
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(doc)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    doc = doc.replace(url, ' URL ')
            if self.replace_numbers:
                doc = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', ' NUM ', doc)
            if self.remove_punctuation:
                doc = re.sub(r'\W+', ' ', doc, flags=re.M)
            if self.remove_junk:
                pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\¿|\x82\
                            |\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                            |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
                doc = re.sub(pattern,'', doc)
            if self.replace_emojis:
                doc = re.sub(r'[^\x00-\x7F]+', ' EMOJI ', doc)
            if self.replace_nonascii:
                if is_ascii(doc) == False:
                    doc = ' NONASCII '
            # tokenize
            tokens = doc.split()
            if self.remove_stopwords:
                stop_words = ['a','an','and','are','as','at','be','by','for','from',
                              'has','he','in','is','it','its','of','on','that','the',
                              'to','was','were','will','with']
                tokens = [t for t in tokens if t not in stop_words]
            if self.lemmatization and lemmatizer is not None:
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            if self.bigrams:
                bigrams = ngrams(word_tokenize(doc), 2)
                bigrams = ['_'.join(grams) for grams in bigrams]
                tokens = [*tokens, *bigrams]
            # include counts
            tokens_counts = Counter(tokens)
            # append to list
            X_transformed.append(tokens_counts)
        return np.array(X_transformed)
    

class DocumentToNgramCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, expand_contractions=True, lower_case=True, 
                 replace_usernames=True, unescape_html=True, 
                 replace_urls=True, replace_numbers=True, 
                 remove_junk=True, remove_punctuation=True, 
                 replace_emojis=True, replace_nonascii=True, 
                 remove_stopwords=True, lemmatization=True,
                 n_grams=2 # defaults to bigram
                ): 
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
        self.n_grams = n_grams
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for doc in X:
            if self.lower_case:
                doc = doc.lower()
            if self.expand_contractions and contractions_map is not None:
                doc = expand_contractions(doc, contractions_map)
            if self.replace_usernames:
                doc = re.sub(r'^@([^\s]+)',' USR ', doc)
            if self.unescape_html:
                doc = unescape(doc)
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(doc)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    doc = doc.replace(url, ' URL ')
            if self.replace_numbers:
                doc = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', ' NUM ', doc)
            if self.remove_punctuation:
                doc = re.sub(r'\W+', ' ', doc, flags=re.M)
            if self.remove_junk:
                pattern = r'\¥|\â|\«|\»|\Ñ|\Ð|\¼|\½|\¾|\¿|\x82\
                            |\x83|\x84|\x85|\x86|\x87|\x88|\x89|\
                            |\x8a|\x8b|\x8c|\x8d|\x8e|\°|\µ|\´|\º|\¹|\³'
                doc = re.sub(pattern,'', doc)
            if self.replace_emojis:
                doc = re.sub(r'[^\x00-\x7F]+', ' EMOJI ', doc)
            if self.replace_nonascii:
                if is_ascii(doc) == False:
                    doc = ' NONASCII '
            # tokenize
            tokens = doc.split()
            if self.remove_stopwords:
                stop_words = ['a','an','and','are','as','at','be','by','for','from',
                              'has','he','in','is','it','its','of','on','that','the',
                              'to','was','were','will','with']
                tokens = [t for t in tokens if t not in stop_words]
            if self.lemmatization and lemmatizer is not None:
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            if self.n_grams:
                for i in range(2, self.n_grams+1): # fix doubling of unigrams
                    grams = ngrams(word_tokenize(doc), i)
                    grams = ['_'.join(gram) for gram in grams]
                    tokens = [*tokens, *grams]
            # include counts
            tokens_counts = Counter(tokens)
            # append to list
            X_transformed.append(tokens_counts)
        return np.array(X_transformed)

    
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))