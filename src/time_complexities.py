import numpy as np

def O_nd(n, d):
    """
    Computes the time complexity O(n * d), where:
    - n: Number of samples
    - d: Number of features

    Parameters:
        n (int): Number of samples.
        d (int): Number of features.

    Returns:
        int: Computed time complexity.
    """
    return n*d

def O_nd_log_n(n, d):
    """
    Computes the time complexity O(n * d * log(n)), where:
    - n: Number of samples
    - d: Number of features

    Parameters:
        n (int): Number of samples.
        d (int): Number of features.

    Returns:
        int: Computed time complexity.
    """
    return int(n*d*np.log(n))

def O_d2(n ,d):
    """
    Computes the time complexity O(d^2), where:
    - d: Number of features

    Parameters:
        n (int): Number of samples (not used in this function).
        d (int): Number of features.

    Returns:
        int: Computed time complexity.
    """
    return d**2

def O_nd_log_n_T(n, d, T = 300):
    """
    Computes the time complexity O(n * d * log(n) * T), where:
    - n: Number of samples
    - d: Number of features
    - T: A constant multiplier (default is 300), which can represent the number of iterations, trees, or a time factor.

    Parameters:
        n (int): Number of samples.
        d (int): Number of features.
        T (int, optional): Constant multiplier.

    Returns:
        int: Computed time complexity.
    """
    return int(T*n*d*np.log(n))

def O_ndT(n, d, T=100):
    """
    Computes the time complexity O(n * d * T), where:
    - n: Number of samples
    - d: Number of features
    - T: A constant multiplier, which can represent the number of iterations, trees, or a time factor.

    Parameters:
        n (int): Number of samples.
        d (int): Number of features.
        T (int, optional): Constant multiplier.

    Returns:
        int: Computed time complexity.
    """
    return n * d * T



