import numpy as np

def robust_scaling(values):
    values = np.array(values)
    n = len(values)
    
    # Handle single element case immediately
    if n <= 1:
        return [0.0] if n == 1 else []

    values_sorted = np.sort(values)
    median = np.median(values)
    
    mid = n // 2
    if n % 2 == 0:
        lower_half = values_sorted[:mid]
        upper_half = values_sorted[mid:]
    else:
        lower_half = values_sorted[:mid]
        upper_half = values_sorted[mid+1:]
    
    # Check if halves are empty before calculating median
    q1 = np.median(lower_half) if lower_half.size > 0 else median
    q3 = np.median(upper_half) if upper_half.size > 0 else median
    iqr = q3 - q1
    
    if iqr == 0:
        return (values - median).tolist()
        
    return ((values - median) / iqr).tolist()
