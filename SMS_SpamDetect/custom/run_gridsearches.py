#!/usr/bin/env python
import os
import re
import time
import joblib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def print_eval_metrics(y_val, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'accuracy: {acc:0.4f}')
    print(f'sensitivity: {tpr:0.4f}')
    print(f'specificity: {tnr:0.4f}')

def gridsearch_wrapper(Xs, Xnames, test=False, k=10):
    """
    Performs grid searches and collects them in a list.
    Args:
        Xs: the numeric matrices
        Xnames: their names
        test: faster, shallower searches for testing
        k: the number of CV folds
    """
    
    start_time = time.time()
    model_dir = os.path.join("data", "3_modeling")
    
    # instantiate list of dicts to gather results
    gridsearches = []
    for ix, X_name in enumerate(Xnames):

        X_ = Xs[ix].toarray()
        X_name = X_name.split('.')[0]

        # split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_, y, stratify=y)

        # setup testing param grid
        test_param_grid = {
            'min_samples_split': [10, 15], 
            'n_estimators' : [50, 60],
            'max_depth': [4, 6],
            'max_features': [50, 60]
        }

        # setup param grid for final not-too-deep search
        param_grid = {
            'min_samples_split': [5, 10, 15],
            'n_estimators' : [100, 200],
            'max_depth': [5, 10, 20],
            'max_features': [50, 100, 250, 500]
        }

        # setup scorers
        scorers = {
            'acc': make_scorer(accuracy_score),
            'tpr': make_scorer(recall_score, pos_label=1), # sensitivity, recall
            'tnr': make_scorer(recall_score, pos_label=0) # specificity, selectivity
        }

        # instantiate estimator
        clf = RandomForestClassifier(n_jobs=4, random_state=42)

        # instantiate k-fold gridsearch
        cv_folds = StratifiedKFold(n_splits=k)
    
        if test == True:
            grid_search_clf = GridSearchCV(clf, test_param_grid, # test grid
                                           scoring=scorers, 
                                           refit='tpr', cv=cv_folds, 
                                           return_train_score=True, n_jobs=-1)
        else:
            grid_search_clf = GridSearchCV(clf, param_grid,
                                           scoring=scorers, 
                                           refit='tpr', cv=cv_folds, 
                                           return_train_score=True, n_jobs=4)           

        # train models
        print(f'\nTraining {ix+1}: {X_name}...')
        start_gs = time.time()
        grid_search_clf.fit(X_train, y_train)
        elapsed_secs = time.time() - start_gs
        print(f'Elapsed: {elapsed_secs:0.0f} s')

        # predict
        y_pred = grid_search_clf.predict(X_val)
        print(f'Best params: {grid_search_clf.best_params_}')

        # confusion matrix on validation set
        print(f'Confusion matrix on validation set:')
        print(pd.DataFrame(confusion_matrix(y_val, y_pred),
                           columns=['pred_neg', 'pred_pos'],
                           index=['neg', 'pos']))
        # eval metrics
        print('Evaluation metrics:')
        print_eval_metrics(y_val, y_pred)

        data = {'representation':X_name,
                'gridsearch_res':grid_search_clf}
        
        # save gridsearch as it finishes
        #filename = ''.join([str(ix+1), "_", X_name, "_rf_gridsearch.joblib"])
        #file_path = os.path.join(model_dir, filename)                                                    
        #joblib.dump(data, file_path)
        
        # gather results into a list of dicts
        gridsearches.append(data)
        
    mins, secs = divmod(time.time() - start_time, 60)
    print(f'\nTot elapsed: {mins:0.0f} m {secs:0.0f} s')
    return gridsearches

if __name__=="__main__":
 
    # load target
    raw_path = os.path.join("..", "data", "1_raw")
    filename = "y_train.csv"
    y = pd.read_csv(os.path.join(raw_path, filename))
    y = np.array(y.iloc[:,0].ravel())
    y[y=='ham'] = 0
    y[y=='spam'] = 1
    y = y.astype('int')

    # load 12 matrices
    proc_dir = os.path.join("..", "data", "2_processed")
    Xnames = [x for x in os.listdir(proc_dir) if re.search('.npz', x)]
    Xs = []
    for ix, X in enumerate(Xnames):
        path_ = os.path.join(proc_dir, Xnames[ix])
        Xs.append(sp.load_npz(path_))

    # uncomment for test or full run
    #results = gridsearch_wrapper(Xs[1:2], Xnames[1:2], test=True, k=3)
    
    # testing full grid with X_bot only
    results = gridsearch_wrapper(Xs=Xs[0:1], Xnames=Xnames[0:1], test=False, k=10)
    
    # persist results
    model_dir = os.path.join("..", "data", "3_modeling")
    
    # change date manually
    file_path = os.path.join(model_dir, "01062021_rf_gridsearches_3.joblib")
    joblib.dump(results, file_path)