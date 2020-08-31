#!/usr/bin/env python
import time
import pandas as pd
import concurrent.futures

# dfm = df master 
# df = subset 

def load_training_data(num):
 
    # load subset
	filepath = ''.join("./data/clean/train_", str(num), ".csv")
    df = pd.read_csv(filepath)
  
	dfm.append(df, ignore_index=True) 
	# ignore index builds index... could be bad w/ async, loses mapping
	# one way would be to give it indices as params (already written for saving)						
    
    # print our result
    result=''.join(["Appending train subseet ", str(num), " ..."])
    return result


def run_processes():
	# since it's mostly I/O-bound heavy, use multithreading
    with concurrent.futures.ThreadingPoolExecutor() as executor:
                   
        results = [executor.submit(load_training_data, num) for num in range(1, 33)]

        # get results with the as_completed function, which gives us an iterator 
        # we loop over to yield results of our processes as they're completed
        for f in concurrent.futures.as_completed(results):
            print(f.result())


if __name__ == '__main__':

    start = time.perf_counter()
	# instantiate dfm 
	dfm = pd.DataFrame()
    run_processes()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')