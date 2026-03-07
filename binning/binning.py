import numpy as np 

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    range_val = max_val - min_val
    if range_val == 0:
        return [0] * len(values)
        
    bin_width = range_val / num_bins

    # Calculate indices, floor them, and clip to ensure max value stays in bounds
    index = (values - min_val) / bin_width
    bins = np.floor(index).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)
    
    # Convert the NumPy array back to a standard Python list
    return bins.tolist()
