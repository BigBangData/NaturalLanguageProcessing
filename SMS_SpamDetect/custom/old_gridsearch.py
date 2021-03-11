# #!/usr/bin/env python
import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score

def scikitlearn_cv(clf, X, y, seed_, cv=10, test_size=.25):
    
    scorer_ = {
        'acc': make_scorer(accuracy_score),
        'tpr': make_scorer(recall_score, pos_label=1),
        'tnr': make_scorer(recall_score, pos_label=0)
    }
    
    acc = cross_val_score(clf, X, y, cv=cv, verbose=0, scoring=scorer_['acc'], n_jobs=-1)
    tpr = cross_val_score(clf, X, y, cv=cv, verbose=0, scoring=scorer_['tpr'], n_jobs=-1)
    tnr = cross_val_score(clf, X, y, cv=cv, verbose=0, scoring=scorer_['tnr'], n_jobs=-1)
    
    return acc.mean(), tpr.mean(), tnr.mean()

def collect_cvs(clf, Xs, Xnames, y, seed_, cv=10, test_size=.25):

    accs, tprs, tnrs, secs = [], [], [], []
    for X in Xs:
        start_cv = time.time()
        acc, tpr, tnr = scikitlearn_cv(clf, X, y, seed_=seed_, cv=cv, test_size=test_size)
        accs.append(round(acc, 4))
        tprs.append(round(tpr, 4))
        tnrs.append(round(tnr, 4))
        secs.append(round(time.time() - start_cv, 1))

    data = {'representation': Xnames,
            'mean_accuracy': accs,
            'mean_sensitivity': tprs, 
            'mean_specificity': tnrs,
            'elapsed_seconds':secs
           }
    
    return data

def build_random_forests(Xs, Xnames, y, cv_seed, rf_seed, mtry_, trees, 
                         max_leaf_nodes, cv=10, max_samples=None, n_jobs=-1):
    """Given:
           Xs: a list of X representations (training data)
           Xnames: a list their names (descriptions)
           y: the target variable
           cv_seed: random seed for cross validation
           rf_seed: random seed for rf classifier
           mtry_: a list of values for the max_features param
           trees: number of trees
           max_leaf_nodes: max number of leaf nodes
           cv: number of folds, defaults to k=5
           max_samples: max num of samples, defaults to None
           n_jobs: defaults to -1 (all cores but one)
       Return:
           A dataframe of results of cv over various mtry values
           With mean accuracy, sensitivity, specificity
    """
    list_of_dfs = []
    for mtry in mtry_:
        rf_clf = RandomForestClassifier(n_estimators=trees,
                                        max_samples=None,
                                        max_features=mtry,
                                        max_leaf_nodes=max_leaf_nodes,
                                        random_state=rf_seed,
                                        n_jobs=n_jobs,
                                        verbose=0)
        
        data = collect_cvs(rf_clf, Xs, Xnames, y, seed_=cv_seed, cv=cv)
        df = pd.DataFrame(data)
        df['mtry'] = mtry
        
        list_of_dfs.append(df)
     
    flattened_df = pd.concat(list_of_dfs)
    
    # reset index
    ix_num = len(mtry_) * len(Xs)
    flattened_df.index = range(ix_num)
    
    return flattened_df