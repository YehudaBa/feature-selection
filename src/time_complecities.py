import numpy as np

def O_nd(n, d):
    return n*d

def O_nd_log_n(n, d):
    return int(n*d*np.log(n))

def O_d2(n ,d):
    return d**2

def O_nd_log_d(n, d):
    return int(n*d*np.log(d))

def O_kn_d2(n, d, k):
    return (k*n)*d**2

