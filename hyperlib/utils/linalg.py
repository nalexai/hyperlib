'''
This package contains linear algebra utils. 
Functions with suffix "_mp" operate on mpmath matrices. 
'''
import mpmath as mpm
import numpy as np
from math import sqrt


def rotate_3D(x, y):
    '''Returns 3D rotation matrix that sends x to the direction parallel to y'''
    xnorm = np.linalg.norm(x)
    ynorm = np.linalg.norm(y)
    cos = np.dot(x,y)/(xnorm*ynorm)
    sin = sqrt(1. - cos**2)
    K = (np.outer(y,x) - np.outer(x,y))/(xnorm*ynorm*sin)
    return np.eye(3) + sin*K + (1.-cos)*(K@K)


def cross_mp(x, y):
    return mpm.matrix([
        x[1]*y[2]-x[2]*y[1],
        x[2]*y[0]-x[0]*y[2],
        x[0]*y[1]-x[1]*y[0],
    ])


def rotate_3D_mp(x, y):
    xnorm = mpm.norm(x)
    ynorm = mpm.norm(y)
    cos = mpm.fdot(x,y)/(xnorm*ynorm)
    sin = mpm.sqrt(1.-cos**2)
    K = (y.T*x-x.T*y)/(xnorm*ynorm*sin)
    return mpm.eye(3) + sin*K + (1.-cos)*(K*K)


def rotate_mp(pts, x, y):
    out = mpm.zeros(pts.rows, pts.cols)
    v = x/mpm.norm(x) 
    cos = mpm.fdot(x,y) / (mpm.norm(x), mpm.norm(y))
    sin = mpm.sqrt(1.-cos**2)
    u = y - mpm.fdot(v, y) * v 

    mat = mpm.eye(x.cols) - u.T * u - v.T * v \
        + cos * u.T * u - sin * v.T * u + sin * u.T * v + cos * v.T * v
    return mat


def from_numpy(M):
    '''Convert 2D numpy array to mpmath matrix'''
    N = mpm.matrix(*M.shape)
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            N[i,j] = M[i,j]
    return N


def to_numpy(N):
    '''Convert 2D mpmath matrix to numpy array'''
    M = np.zeros((N.rows,N.cols), dtype=np.float64)
    for i in range(N.rows):
        M[i,:] = N[i,:]
    return M
