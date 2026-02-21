'''
    Main function DT()
    Compute Dependency Triad, enable get_parameters = True
    To compute CPL using DT, enable leakage_only = True
'''


import numpy as np
import math
import time

def ignore_nan(arr):
    return max(filter(lambda x: not (math.isinf(x) or math.isnan(x)), arr))

def CPL(eps, alpha, beta, delta):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)

    B = (Beta - 1)*(1 - delta) / (Alpha * Beta - 1)

    if (-1 + delta + Alpha - delta * Eps) < 0:
        B = 0

    A = Alpha*B + delta
    leakage = np.log((1 + A * (Eps - 1)) / (1 + B * (Eps - 1)))
    
    return leakage

def get_alpha_delta(CMF, Delta):
    num_rows, _ = np.shape(CMF)

    alpha = 0
    delta = 0

    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            delta_temp = 0
            G_prime = CMF[j,:]
            
            for k in range(len(G)):
                g = G[k]
                g_prime = G_prime[k]
                d = Delta[i, k]
                d_prime = Delta[j, k]
                alpha = max(alpha, np.log(min(1, g + d)/(g_prime - d_prime)))

                if g_prime - d_prime <= 0:
                    delta_temp += g + d

            delta = min(1, max(delta, delta_temp))
                 
    return alpha, delta

def get_beta(CMF, alpha, delta, Delta):
    num_rows, _ = np.shape(CMF)
    phi = 0

    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            G_prime = CMF[j,:]
            
            Q = G/(G_prime + 1e-6)
            sorted_indices = np.argsort(Q)[::-1]
            A = 0
            B = 0

            for k in sorted_indices:
                g = G[k]
                g_prime = G_prime[k]
                d = Delta[i, k]
                d_prime = Delta[j, k]

                a = min(1 - A, g + d)
                b = max(0, g_prime - d_prime)

                if a > b:
                    A += a
                    B += b
            phi = max(phi, A - B)
    beta = np.log((1 + phi + np.exp(alpha) * (-1 + delta) - 2 * delta) / (1 + (-1 + phi) * np.exp(alpha) - delta))
    return beta

def DT(eps, CMF, Delta = 0, leakage_only = False, DB_parameters = (0,0,0), get_parameters=False):
    if isinstance(Delta, int):
        Delta = np.zeros((np.shape(CMF)))
    if leakage_only:
        alpha, beta, delta = DB_parameters
        return CPL(alpha=alpha, beta=beta, delta=delta, eps=eps)
    
    alpha, delta = get_alpha_delta(CMF, Delta)
    
    beta = get_beta(CMF, alpha, delta, Delta)

    if get_parameters:
        return alpha, beta, delta
    return CPL(alpha=alpha, beta=beta, delta=delta, eps=eps)
