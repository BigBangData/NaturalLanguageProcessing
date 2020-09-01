#!/usr/bin/env python
import time
import pandas as pd
import concurrent.futures

# dfm = df master 
# df = subset 

def load_training_data(params):
 
    # unpack parameters
    ix_list, num = params
                        
    # load subset
    filepath = ''.join(["./data/clean/train_", str(num), ".csv"])
    df = pd.read_csv(filepath)
    df.index = ix_list
    dfm.append(df)

    # print our result
    result=''.join(["Appending train subseet ", str(num), " ..."])
    print(result)
    
    return dfm


def run_processes():
    # since it's mostly I/O-bound heavy, use multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
                   
        params_list = [
                       (range(     0,    50000),  1),
                       (range(  50000,  100000),  2),
                       (range( 100000,  150000),  3)
                      ]
        results = [executor.submit(load_training_data, p) for p in params_list]

        # get results with the as_completed function, which gives us an iterator 
        # we loop over to yield results of our processes as they're completed
        for f in concurrent.futures.as_completed(results):
            print(f.result())


if __name__ == '__main__':

    start = time.perf_counter()
    # instantiate dfm 
    dfm = pd.DataFrame()
    dfm = run_processes()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')