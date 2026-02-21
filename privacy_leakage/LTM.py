'''
    Algorithm LTM
'''
import numpy as np
import time 

def compute_H_related(q_original, d_original, eps):
    q_plus = []
    d_plus = []
    for j in range(len(q_original)):
        q_j = q_original[j]
        d_j = d_original[j]

        if q_j > d_j:
            q_plus.append(q_j)
            d_plus.append(d_j)

    update = False

    first_run = True
    while update or first_run:
        update = False
        first_run = False

        q = sum(q_plus)
        d = sum(d_plus)
        remove_indexes = []
        for j in range(len(q_plus)):
            q_j = q_plus[j]
            d_j = d_plus[j]
            if d_j== 0:
                continue
            if (q_j/d_j) <= (q*(np.exp(eps)-1)+1)/(d*(np.exp(eps)-1)+1):
                remove_indexes.append(j)
                update = True
        q_plus = np.delete(np.array(q_plus), remove_indexes, axis=0)
        d_plus = np.delete(np.array(d_plus), remove_indexes, axis=0)

    return (1+q*(np.exp(eps)-1))/(1+d*(np.exp(eps)-1))

def LTM(eps, CMF):
    num_rows, _ = np.shape(CMF)
    max_val = 0
    start_time = time.time()
    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            G_prime = CMF[j,:]
            L = compute_H_related(G, G_prime, eps)
            if max_val < L:
                max_val = L
    leakage = np.log(max_val)
    end_time = time.time()
    leakage = min(leakage, eps)
    leakage = max(0, leakage)
    return leakage, end_time - start_time
