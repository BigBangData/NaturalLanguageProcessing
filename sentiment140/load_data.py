#!/usr/bin/env python

# data-loading module for the 1.6 M sentiment140 dataset 

import os
import time
import pandas as pd
import concurrent.futures as cf

def load_training_subset(params):
    ix_list, num = params
    filepath = os.path.join("..","data","2_clean","sentiment140")
    filename = "".join(["train_", str(num), ".csv"])
    full_path = os.path.join(filepath, filename)
    
    df = pd.read_csv(full_path)
    df.index = ix_list
    return df
 
def run_processes():
    with cf.ThreadPoolExecutor() as executor:

            params_list = [
                           (range(     0,    50000),  1),
                           (range(  50000,  100000),  2),
                           (range( 100000,  150000),  3),
                           (range( 150000,  200000),  4),
                           (range( 200000,  250000),  5),
                           (range( 250000,  300000),  6),
                           (range( 300000,  350000),  7),
                           (range( 350000,  400000),  8),
                           (range( 400000,  450000),  9),
                           (range( 450000,  500000), 10),
                           (range( 500000,  550000), 11),
                           (range( 550000,  600000), 12),
                           (range( 600000,  650000), 13),
                           (range( 650000,  700000), 14),
                           (range( 700000,  750000), 15),
                           (range( 750000,  800000), 16),
                           (range( 800000,  850000), 17),
                           (range( 850000,  900000), 18),
                           (range( 900000,  950000), 19),
                           (range( 950000, 1000000), 20),
                           (range(1000000, 1050000), 21),
                           (range(1050000, 1100000), 22),
                           (range(1100000, 1150000), 23),
                           (range(1150000, 1200000), 24),
                           (range(1200000, 1250000), 25),
                           (range(1250000, 1300000), 26),
                           (range(1300000, 1350000), 27),
                           (range(1350000, 1400000), 28),
                           (range(1400000, 1450000), 29),
                           (range(1450000, 1500000), 30),
                           (range(1500000, 1550000), 31),
                           (range(1550000, 1599999), 32)
                          ]

            results = [executor.submit(load_training_subset, p) for p in params_list]

            subset_list = []
            for f in cf.as_completed(results):
                subset_list.append(f.result())

            df = pd.concat(subset_list)
            df = df.sort_index()
            return df