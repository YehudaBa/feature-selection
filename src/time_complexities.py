import numpy as np

def O_nd(n, d):
    return n*d

def O_nd_log_n(n, d):
    return int(n*d*np.log(n))

def O_d2(n ,d):
    return d**2

def O_nd_log_n(n, d):
    return int(n*d*np.log(n))

def O_nd_log_n_T(n, d, T = 300):
    return int(T*n*d*np.log(n))


def O_ndT(n, d, T=100):
    return n * d * T



