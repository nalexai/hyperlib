from mpmath import mp, fdiv, fsub, fdot, acosh
import numpy as np

def norm_square(x, precision=50):
    return fdot(x,x)

def poincare_dist(x, y, precision=50):
    ''' 
    Calculate the hyperbolic distance between points in the Poincare Ball (curvature=-1) with high precision. 
    Note that calculating hyperbolic distance of d requires O(d) bits of precision.
        Args:
            x, y (ndarray): length D arrays representing points in the D-dimensional ball |x| < 1
            precision (int): bits of precision to use
        Returns:
            mpmath float object. Can be converted back to regular float
    '''
    mp.dps = precision
    x2 = norm_square(x, precision=precision)
    y2 = norm_square(y, precision=precision)
    xy2 = norm_square(x-y, precision=precision)
    denom = fsub(1,x2)*fsub(1,y2)
    return acosh(1+2*fdiv(xy2,denom))

def poincare_metric(X, precision=50):
    ''' Calculate the distance matrix of points in the Poincare Ball (curvature = -1) with high precision.
        Args:
            X (ndarray): N x D matrix representing N points in the D-dimensional ball |x|< 1
            precision (int): bits of precision to use
        Returns:
            distance matrix in compressed 1D form ( see scipy.squareform )
    '''
    N = X.shape[0]
    out = np.zeros(shape=(N*(N-1)//2), dtype=np.float64)
    for i in range(N):
        idx = N*i-((i+2)*(i+2))//2
        for j in range(i+1,N):
            out[idx+j] = poincare_dist(X[i],X[j], precision=precision)
    return out
