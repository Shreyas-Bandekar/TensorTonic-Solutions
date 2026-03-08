import numpy as np

def matrix_transpose(A):
    a = np.array(A)
    M, N = a.shape
    
    transposed = np.zeros((N, M))

    for i in range(M):        
        for j in range(N):    
            transposed[j, i] = a[i, j]
            
    return transposed