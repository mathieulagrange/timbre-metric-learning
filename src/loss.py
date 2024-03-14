import numpy as np


def compute_A(r):
    
    """
    Build the linear operator involved in the nonnegative least-square regression used to compute the warm start of the learning algorithm.
    
    Args:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
    
    
    Returns:
    
        A: component-wise squared differences between features of pairs of sounds, matrix of size s x n with s = l x (l-1) / 2
        
    """
    
    
    (D,N) = np.shape(r)
    S     = N*(N - 1)//2 
    
    # store the S x D matrix containing the squared differences between feature components
    A = np.zeros((S,D))
    
    k = 0
    for i in range(N):
        for j in range(i+1,N):
            A[k,:] = (r[:,i] - r[:,j])**2
            k +=1
    
    return A

