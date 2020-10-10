#!/usr/bin/env python
import os
import sys
import time

import pandas as pd
import concurrent.futures as cf

def load_subset(params):
    X_name, ix_list, num = params
    filepath = os.path.join("..","data","2_clean","sentiment140")
    filename = "".join([X_name, '.', str(num), ".csv"])
    full_path = os.path.join(filepath, filename)
    df = pd.read_csv(full_path)
    #df.index = ix_list # why was this necessary?
    return df
 
def load_clean_data(X_name):
    start_time = time.time()
    with cf.ThreadPoolExecutor() as executor:
        if X_name == 'X_train':
            params_list = [
                           (X_name, range(     0,    50001),  1),
                           (X_name, range(  50000,  100001),  2),
                           (X_name, range( 100000,  150001),  3),
                           (X_name, range( 150000,  200001),  4),
                           (X_name, range( 200000,  250001),  5),
                           (X_name, range( 250000,  300001),  6),
                           (X_name, range( 300000,  350001),  7),
                           (X_name, range( 350000,  400001),  8),
                           (X_name, range( 400000,  450001),  9),
                           (X_name, range( 450000,  500001), 10),
                           (X_name, range( 500000,  550001), 11),
                           (X_name, range( 550000,  600001), 12),
                           (X_name, range( 600000,  650001), 13),
                           (X_name, range( 650000,  700001), 14),
                           (X_name, range( 700000,  750001), 15),
                           (X_name, range( 750000,  800001), 16),
                           (X_name, range( 800000,  850001), 17),
                           (X_name, range( 850000,  900001), 18),
                           (X_name, range( 900000,  950001), 19),
                           (X_name, range( 950000, 1000001), 20),
                           (X_name, range(1000000, 1050001), 21),
                           (X_name, range(1050000, 1100001), 22),
                           (X_name, range(1100000, 1150001), 23),
                           (X_name, range(1150000, 1200001), 24)
                          ]
        if X_name == 'X_test':
            params_list = [
                           (X_name, range(     0,    50001),  1),
                           (X_name, range(  50000,  100001),  2),
                           (X_name, range( 100000,  150001),  3),
                           (X_name, range( 150000,  200001),  4),
                           (X_name, range( 200000,  250001),  5),
                           (X_name, range( 250000,  300001),  6),
                           (X_name, range( 300000,  350001),  7),
                           (X_name, range( 350000,  400001),  8)
                          ]
                
        results = [executor.submit(load_subset, p) for p in params_list]

        subset_list = []
        for f in cf.as_completed(results):
            subset_list.append(f.result())

        df = pd.concat(subset_list)
        df = df.sort_index()
        return df
    
        # print out running time
        mins, secs = divmod(time.time() - start_time, 60)
        print(f"Elapsed time: {mins:0.0f} minute(s) and {secs:0.0f} second(s).")