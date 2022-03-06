'''
This package contains functions for high precision calculations of hyperbolic functions.
Note that a hyperbolic distance of d requires O(d) bits of precision to compute.
'''
import mpmath as mpm
import numpy as np

def poincare_dist(x, y, c=1.0, precision=None):
    ''' 
    The hyperbolic distance between points in the Poincare model with curvature -1/c 
        Args:
            x, y: size 1xD mpmath matrix representing point in the D-dimensional ball |x| < 1
            precision (int): bits of precision to use
        Returns:
            mpmath float object. Can be converted back to regular float
    '''
    if precision is not None:
        mpm.mp.dps = precision
    x2 = mpm.fdot(x,x) 
    y2 = mpm.fdot(y,y) 
    xy = mpm.fdot(x,y)
    sqrt_c = mpm.sqrt(c)
    denom = 1 - 2*c*xy + c**2 * x2*y2
    norm = mpm.norm( ( -(1-2*c*xy + c*y2)*x + (1.-c*x2)*y ) /denom )
    return 2/sqrt_c * mpm.atanh( sqrt_c * norm ) 

def poincare_dist0(x, c=1.0, precision=None):
    ''' Distance from 0 to x in the Poincare model with curvature -1/c'''
    if precision is not None:
        mpm.mp.dps = precision
    x_norm = mpm.norm(x)
    sqrt_c = mpm.sqrt(c)
    return 2/sqrt_c * mpm.atanh( x_norm / sqrt_c)

def poincare_metric(X, precision=None):
    ''' Calculate the distance matrix of points in the Poincare model with curvature -1/c 
        Args:
            X (ndarray): N x D matrix representing N points in the D-dimensional ball |x|< 1
            precision (int): bits of precision to use
        Returns:
            distance matrix in compressed 1D form ( see scipy.squareform )
    '''
    if precision is not None:
        mpm.mp.dps = precision
    N = X.shape[0]
    out = np.zeros(shape=(N*(N-1)//2), dtype=np.float64)
    for i in range(N):
        idx = N*i-((i+1)*(i+2))//2
        for j in range(i+1,N):
            out[idx+j] = poincare_dist(X[i],X[j],precision=precision)
    return out

def poincare_reflect(a, x, c=1.0, precision=None):
    '''
    Spherical inversion (or "Poincare reflection") of a point x about a sphere
    with center at point "a", and radius = 1/c.
    '''
    if precision is not None:
        mpm.mp.dps = precision
    a2 = mpm.fdot(a,a)
    x2 = mpm.fdot(x,x)
    xa2 = x2 + a2 - 2*mpm.fdot(x,a)
    r2 = a2 - 1./c 
    scal = mpm.fdiv(r2, xa2)
    return scal*(x-a) + a

def poincare_reflect0(z, x, c=1.0, precision=None):
    '''
    Spherical inversion (or "Poincare reflection") of a point x
    such that point z is mapped to the origin.
    '''
    if precision is not None:
        mpm.mp.dps = precision
    # the reflection is poincare_reflect(a, x) where
    # a = c * z / |z|**2
    z2 = mpm.fdot(z,z)
    zscal = c / z2
    x2 = mpm.fdot(x,x)
    a2 = c* zscal
    r2 = a2 - 1./c
    xa2 = x2 + a2 - 2*zscal*mpm.fdot(z,x)
    scal = mpm.fdiv(r2, xa2)
    return scal*( x - zscal * z) + zscal * z
