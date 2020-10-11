#!/usr/bin/env python
import os
import sys
import time

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

        df = pd.concat(subset_list)
        df = df.sort_index()
        return df