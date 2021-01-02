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
raw_path = os.path.join("data","1_raw")
filename = "y_train.csv"
y = pd.read_csv(os.path.join(raw_path, filename))
y = np.array(y.iloc[:,0].ravel())
y[y=='ham'] = 0
y[y=='spam'] = 1
y = y.astype('int')

# load 12 matrices
proc_dir = os.path.join("data","2_processed")
Xnames = [x for x in os.listdir(proc_dir) if re.search('.npz', x)]
Xs = []
for i, X in enumerate(Xnames):
    path_ = os.path.join(proc_dir, Xnames[i])
    Xs.append(sp.load_npz(path_))
    
# 1. X_bot
X_bot = Xs[0].toarray()

X_train, X_test, y_train, y_test = train_test_split(X_bot, y, stratify=y)

# random forest
clf = RandomForestClassifier(n_jobs=-1, random_state=42)

param_grid = {
    'min_samples_split': [5, 10, 20],
    'n_estimators' : [50, 100, 200],
    'max_depth': [3, 5, 10], 
    'max_features': [10, 25, 50, 100, 200] # mtry
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

def gridsearch_wrapper(cv=5, refit_score='accuracy_score', n_jobs=-1):
    """Fits a GridSearchCV classifier using refit_score for optimization
       Prints classifier's performance metrics
    """
    T1 = time.time()
    cv_folds = StratifiedKFold(n_splits=cv)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, 
                               cv=cv_folds, return_train_score=True, n_jobs=n_jobs)
    
    print('Fitting GridSearchCV...')
    grid_search.fit(X_train, y_train)
    
    # make the predictions
    y_pred = grid_search.predict(X_test)
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)
    
    # confusion matrix on the test data
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))
    
    mins, secs = divmod(time.time() - T1, 60)
    print(f'\nElapsed: {mins:0.0f} m {secs:0.0f} s')
    return grid_search

def format_results(gridsearch, sort_by):
    """Format results, returning top 6 given a sorting score.
    """
    res_df = pd.DataFrame(gridsearch.cv_results_)
    res_df = res_df[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 
                     'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']]
    res_df = res_df.sort_values(by=sort_by, ascending=False).round(3).head()
    return res_df

if __name__=="__main__":
    gridsearch_recall = gridsearch_wrapper(refit_score='recall_score')