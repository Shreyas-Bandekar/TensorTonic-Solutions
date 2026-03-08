import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function for TensorTonic.
    """
    arr = np.asarray(x, dtype=float)
    
    return np.where(
        arr >= 0, 
        1 / (1 + np.exp(-arr)),           
        np.exp(arr) / (1 + np.exp(arr))  
    )