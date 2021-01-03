import os
import re
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# load target
raw_path = os.path.join("..", "data","1_raw")
filename = "y_train.csv"
y = pd.read_csv(os.path.join(raw_path, filename))
y = np.array(y.iloc[:,0].ravel())
y[y=='ham'] = 0
y[y=='spam'] = 1
y = y.astype('int')

# load 12 matrices
proc_dir = os.path.join("..", "data","2_processed")
Xnames = [x for x in os.listdir(proc_dir) if re.search('.npz', x)]
Xs = []
for i, X in enumerate(Xnames):
    path_ = os.path.join(proc_dir, Xnames[i])
    Xs.append(sp.load_npz(path_))
    
# Bag-of-upto-Trigrams (2,000 terms)
X_bot = Xs[0].toarray()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_bot, y, stratify=y)

# instantiate estimator, set seed for reproducibility
clf = RandomForestClassifier(n_jobs=-1, random_state=42)

# setup shallow grid of params
param_grid_shallow = {
    'min_samples_split': [10, 25, 50], 
    'n_estimators' : [50, 100, 200],
    'max_depth': [3, 5, 10],
    'max_features': [10, 50, 100, 250]
}

# setup scorers
scorers = {
    'acc': make_scorer(accuracy_score),
    'tpr': make_scorer(recall_score, pos_label=1), # sensitivity, recall
    'tnr': make_scorer(recall_score, pos_label=0) # specificity, selectivity
}

# 5-fold CV
cv_folds = StratifiedKFold(n_splits=5)
grid_search_clf = GridSearchCV(clf, param_grid_shallow, scoring=scorers, refit='tpr', 
                               cv=cv_folds, return_train_score=True, n_jobs=-1)

if __name__=="__main__":
    start_gs = time.time()
    
    # perform gridsearch
    print(f'Starting GridSearchCV... at {start_gs:0.0f}')
    grid_search_clf.fit(X_train, y_train)
    
    # return time
    mins, secs = divmod(time.time() - start_gs, 60)
    print(f'\nElapsed: {mins:0.0f} m {secs:0.0f} s')
    
    # predict
    y_pred = grid_search_clf.predict(X_val)
    
    print('\nBest params:')
    print(grid_search_clf.best_params_)

    print('\nConfusion Matrix on validation set:')
    print(pd.DataFrame(confusion_matrix(y_val, y_pred),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))

    # print eval metrics
    def print_eval_metrics(y_val, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        print(f'accuracy: {acc:0.4f}')
        print(f'sensitivity: {tpr:0.4f}')
        print(f'specificity: {tnr:0.4f}')
    
    print('\nEvaluation metrics:')
    print_eval_metrics(y_val, y_pred)