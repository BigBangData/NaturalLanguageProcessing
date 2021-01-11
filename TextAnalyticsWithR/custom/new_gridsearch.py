# #!/usr/bin/env python
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier

def print_eval_metrics(y_val, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'accuracy: {acc:0.4f}')
    print(f'sensitivity: {tpr:0.4f}')
    print(f'specificity: {tnr:0.4f}')
    
def gridsearch_wrapper(Xs, Xnames, y, param_grid, k=10, n_jobs=6):
    """
    Performs grid searches and collects them in a list.
    Args:
        Xs: numeric matrices
        Xnames: their names
        y: the target
        k: the number of CV folds
        n_jobs: the number of logical cores
    """
    start_time = time.time()
    
    # instantiate list of dicts to gather results
    gridsearches = []
    for ix, X_name in enumerate(Xnames):

        X_ = Xs[ix].toarray()
        X_name = X_name.split('.')[0]

        # split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_, y, stratify=y)

        # setup scorers
        scorers = {
            'acc': make_scorer(accuracy_score),
            'tpr': make_scorer(recall_score, pos_label=1), # sensitivity, recall
            'tnr': make_scorer(recall_score, pos_label=0) # specificity, selectivity
        }

        # instantiate estimator
        clf = RandomForestClassifier(n_jobs=n_jobs, random_state=42)

        # instantiate k-fold gridsearch
        cv_folds = StratifiedKFold(n_splits=k)
    
        grid_search_clf = GridSearchCV(clf, 
                                       param_grid,
                                       scoring=scorers, 
                                       refit='tpr', 
                                       cv=cv_folds, 
                                       return_train_score=True, 
                                       n_jobs=n_jobs)
        
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
        
        # gather results into a list of dicts
        gridsearches.append(data)
        
    mins, secs = divmod(time.time() - start_time, 60)
    print(f'\nElapsed: {mins:0.0f} m {secs:0.0f} s')
    
    return gridsearches