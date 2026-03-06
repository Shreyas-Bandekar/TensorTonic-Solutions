import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function for TensorTonic.
    """
    # 1. Always convert input to a numpy array first
    arr = np.asarray(x, dtype=float)
    
    # 2. Use np.where with the ARRAY (arr), not the list (x)
    # This prevents the '>=' error and handles overflow
    return np.where(
        arr >= 0, 
        1 / (1 + np.exp(-arr)),           # Formula for positive numbers
        np.exp(arr) / (1 + np.exp(arr))   # Stable formula for negative numbers
    )