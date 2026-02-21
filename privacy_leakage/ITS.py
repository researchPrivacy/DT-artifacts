'''
    Algorithm ITS
'''
import numpy as np
import time 

def ITS(eps, CMF):
    num_rows, num_columns = np.shape(CMF)

    if num_rows == 1:
        return eps, 0
    
    max_val = 0
    Eps = np.exp(eps)
    start_time = time.time()
    if num_rows == 1:
        return eps, 0
    max_array = np.nanmax(CMF, axis=0)
    min_array = np.nanmin(CMF, axis=0)
    for i in range(num_columns):

        max_h = max_array[i]
        min_h = min_array[i]

        L = np.log((max_h * Eps + (1 - max_h)) / (min_h * Eps + (1 - min_h)))

        if max_val < L:
            max_val = L

    leakage = max_val
    end_time = time.time()
    leakage = min(leakage, eps)
    leakage = max(0, leakage)
    return leakage, end_time - start_time
    
