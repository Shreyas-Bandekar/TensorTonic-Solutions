import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)
    activated = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    return activated
    pass