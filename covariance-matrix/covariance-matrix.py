import numpy as np

def covariance_matrix(X):
    arr = np.array(X)
    
    if arr.ndim <= 1 or arr.shape[0] <= 1:
        return None

    n = arr.shape[0]
    u = np.mean(arr, axis=0)
    arr_cen = arr - u

    cov = (arr_cen.T @ arr_cen) / (n - 1)
    
    return cov
