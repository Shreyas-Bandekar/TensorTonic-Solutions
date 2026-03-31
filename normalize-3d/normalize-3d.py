import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.array(v, dtype=float)
    
    # Detect if input is a single vector (shape (3,))
    is_1d = v.ndim == 1
    if is_1d:
        v = v[np.newaxis, :]  # Temporarily promote to (1, 3)

    # Calculate L2 norm for each row
    norms = np.linalg.norm(v, axis=1, keepdims=True)

    # Vectorized division: only divide where norm > 0 to keep zero vectors as [0,0,0]
    v_hat = np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-6)

    # Return to original shape (flatten if it was (3,))
    return v_hat[0] if is_1d else v_hat
