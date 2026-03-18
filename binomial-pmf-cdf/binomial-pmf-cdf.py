import numpy as np
from scipy.stats import binom

def binomial_pmf_cdf(n, p, k):
    
    pmf_val = float(binom.pmf(k, n, p))
    cdf_val = float(binom.cdf(k, n, p))
    
    return (pmf_val, cdf_val)
