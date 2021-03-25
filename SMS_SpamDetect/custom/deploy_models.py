# #!/usr/bin/env python
import re
import os
import sys
import time
import joblib 

import numpy as np
import pandas as pd
import scipy.sparse as sp
import custom.clean_preprocess as cp

from datetime import datetime
from xgboost import XGBClassifier
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot, svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split 

class TruncatedSVD(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=800):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X = self._validate_data(X, accept_sparse=['csr', 'csc'],
                                ensure_min_features=2)
        # "arpack" algo
        U, Sigma, VT = svds(X, k=self.n_components)
        # svds doesn't abide by scipy.linalg.svd/randomized_svd
        # conventions, so reverse its outputs.
        Sigma = Sigma[::-1]
        U, VT = svd_flip(U[:, ::-1], VT[::-1])
        
        # Store:
        # eigenvalues (left singular values): terms
        self.U_ = U 
        # eigenvectors (right singular values): documents
        self.V_ = VT.T 
        # singular values
        self.sigma_ = Sigma
        
        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var

        return X_transformed

# create deployment dir
dep_dir = os.path.join("data","5_deployment")

try:
    os.stat(dep_dir)
except:
    os.mkdir(dep_dir)
    
# load transformers and model
X_train_transformer_PATH = os.path.join(dep_dir, "X_train_transformer.joblib")
with open(X_train_transformer_PATH, 'rb') as f:
    X_train_transformer = joblib.load(f)

X_train_fit_PATH = os.path.join(dep_dir, "X_train_fit.joblib")
with open(X_train_fit_PATH, 'rb') as f:
    X_train_fit = joblib.load(f) 

X_train_svd_transformer_PATH = os.path.join(dep_dir, "X_train_svd_transformer.joblib")
with open(X_train_svd_transformer_PATH, 'rb') as f:
    X_train_svd_transformer = joblib.load(f)   

X_train_svd_spam_PATH = os.path.join(dep_dir, "X_train_svd_spam.joblib")
with open(X_train_svd_spam_PATH, 'rb') as f:
    X_train_svd_spam = joblib.load(f)
    
    
def transform_newdata(new_data):
    
    # preprocess pipeline
    pipe = Pipeline([('counter', cp.DocumentToNgramCounterTransformer(n_grams=3)),
                     ('bot', cp.WordCounterToVectorTransformer(vocabulary_size=2000)),
                     ('tfidf', TfidfTransformer(sublinear_tf=True))])

    # counter
    X_test_counter = pipe['counter'].fit_transform(new_data) 
    
    # BoT
    X_test_bot = X_train_transformer.transform(X_test_counter)
    
    # Tfidf
    X_test_tfidf = X_train_fit.transform(X_test_bot)
    
    # SVD
    sigma_inverse = 1 / X_train_svd_transformer.sigma_
    U_transpose = X_train_svd_transformer.U_.T
    UT_TestTfidfT = (U_transpose @ X_test_tfidf.T)
    X_test_svd = (sigma_inverse.reshape(-1,1) * UT_TestTfidfT).T
    
    # Cosine Similarities
    test_similarities = cosine_similarity(sp.vstack((X_test_svd, X_train_svd_spam)))
    spam_cols = range(X_test_svd.shape[0], test_similarities.shape[0])
    test_mean_spam_sims = []
    for ix in range(X_test_svd.shape[0]):
        mean_spam_sim = np.mean(test_similarities[ix, spam_cols])
        test_mean_spam_sims.append(mean_spam_sim)
        
    # stack
    X_test_processed = sp.hstack((csr_matrix(test_mean_spam_sims).T, X_test_svd))
    
    return X_test_processed