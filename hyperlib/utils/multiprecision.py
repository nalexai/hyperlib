from gmpy2 import mpfr
import gmpy2
import numpy as np

def poincare_dist(x, y, precision=64):
    ''' Calculate the hyperbolic distance between points in the Poincare Ball (curvature=-1) with high precision. 
    Note that calculating hyperbolic distance of d requires O(d) bits of precision.
        Args:
            x, y (ndarray): length D arrays representing points in the D-dimensional ball |x| < 1
            precision (int): bits of precision to use
        Returns:
            gmpy2.mpfr object. Can be converted back to regular float'''
    gmpy2.get_context().precision = precision
    x2 = norm_square(x, precision=precision)
    y2 = norm_square(y, precision=precision)
    xy2 = norm_square(x-y, precision=precision)
    denom = gmpy2.sub(1,x2)*gmpy2.sub(1,y2)
    return gmpy2.acosh(1+2*gmpy2.div(xy2,denom))

def norm_square(x, precision=64):
    out = mpfr('0.0')
    for i in range(len(x)):
        out += gmpy2.square(x[i])
    return out

def poincare_metric(X, precision=64):
    ''' Calculate the distance matrix of points in the Poincare Ball (curvature = -1) with high precision.
        Args:
            X (ndarray): N x D matrix representing N points in the D-dimensional ball |x|< 1
            precision (int): bits of precision to use
        Returns:
            distance matrix in compressed 1D form ( see scipy.squareform )'''
    N = X.shape[0]
    out = np.zeros(shape=(N*(N-1)//2), dtype=np.float64)
    for i in range(N):
        for j in range(i+1,N):
            idx = N*i+j-((i+2)*(i+1))//2
            out[idx] = poincare_dist(X[i],X[j])
    return out
