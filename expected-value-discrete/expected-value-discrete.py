import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)

    if x.shape != p.shape:
        raise ValueError("Dimensions of outcomes (x) and probabilities (p) must match.")
    
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1.0.")

    return float(np.dot(x, p))
