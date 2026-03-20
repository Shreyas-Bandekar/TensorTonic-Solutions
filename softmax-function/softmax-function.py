import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    arr = np.array(x)
    
    axis = 1 if arr.ndim > 1 else 0
    
    e_x = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
