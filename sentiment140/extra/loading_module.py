#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import pandas as pd
import concurrent.futures as cf

def load_subset(params):
    
    # unpack params
    X_name, num = params
    
    # load clean subset
    clean_path = os.path.join("..","data","2_clean","sentiment140")
    filename = "".join([X_name, '.', str(num), ".csv"])
    full_path = os.path.join(clean_path, filename)
    df = pd.read_csv(full_path)
    return df
 
def load_clean_data(X_name):
    
    with cf.ThreadPoolExecutor() as executor:
        if X_name == 'X_train':
            params_list = [
                           (X_name,  1),
                           (X_name,  2),
                           (X_name,  3),
                           (X_name,  4),
                           (X_name,  5),
                           (X_name,  6),
                           (X_name,  7),
                           (X_name,  8),
                           (X_name,  9),
                           (X_name, 10),
                           (X_name, 11),
                           (X_name, 12),
                           (X_name, 13),
                           (X_name, 14),
                           (X_name, 15),
                           (X_name, 16),
                           (X_name, 17),
                           (X_name, 18),
                           (X_name, 19),
                           (X_name, 20),
                           (X_name, 21),
                           (X_name, 22),
                           (X_name, 23),
                           (X_name, 24)
                          ]
        if X_name == 'X_test':
            params_list = [
                           (X_name, 1),
                           (X_name, 2),
                           (X_name, 3),
                           (X_name, 4),
                           (X_name, 5),
                           (X_name, 6),
                           (X_name, 7),
                           (X_name, 8)
                          ]
                
        results = [executor.submit(load_subset, p) for p in params_list]

        subset_list = []
        for f in cf.as_completed(results):
            subset_list.append(f.result())
        
        # concatenate subsets    
        X = pd.concat(subset_list)

        raw_dir = os.path.join("..","data","1_raw","sentiment140")
        
        # update: do not use train_ix, cannot properly index to compare
        #         with X_transformed, no need anyway (train_ix saved)
        if X_name == 'X_train':
            # load original training indices
            #train_ix = np.load(os.path.join(raw_dir, "train_ix.npy"))
            #X.index = list(train_ix)
            X.index = range(len(X))
        
            # load y vector and reindex
            y_filepath = os.path.join(raw_dir, "y_train.csv")
            y = pd.read_csv(y_filepath)
            #y.index = list(train_ix)
            
        if X_name == 'X_text':
            # load original text indices
            #test_ix = np.load(os.path.join(raw_dir, "test_ix.npy"))
            #X.index = list(test_ix)
            X.index = range(len(X))
            
            # load y vector and reindex
            y_filepath = os.path.join(raw_dir, "y_test.csv")
            y = pd.read_csv(y_filepath)
            #y.index = list(test_ix)      
  
        return (X, y)