import numpy as np
from random import randint

# there are more efficient algorithms for this
# https://people.csail.mit.edu/virgi/6.890/papers/APBP.pdf
def max_min(A, B):
    '''max-min product of two square matrices
    params: 
        A, B: NxN numpy arrays '''
    assert A.shape == B.shape
    return np.max(np.minimum(A[:, :, None], B[None, :, :]), axis=1)

def mat_gromov_prod(dists, base):
    '''Gromov products of N-point metric space relative to base point
    Args:
        dists (ndarray): NxN matrix of pairwise distances
        base (int): index of the basepoint in 0...N-1 '''
    assert dists.shape[0] == dists.shape[1] and 0 <= base < dists.shape[0]
    row = dists[base, :][None, :]
    col = dists[:, base][:, None]
    return 0.5*(row+col-dists)

def delta_rel(dists, base):
    ''' Measure the delta-hyperbolicity constant of data 
    with respect to basepoint, normalized by the diameter (max dist).
    Args:
        dists (ndarray): NxN matrix of pairwise distances
        base (int): index of basepoint in 0...N-1
    '''
    assert dists.shape[0] == dists.shape[1] and 0 <= base < dists.shape[0]
    G = mat_gromov_prod(dists, base)
    delta = np.max(max_min(G,G)-G)
    diam = np.max(dists)
    return delta/diam

def delta_sample(X, **kwargs):
    bs = kwargs.get("bs", X.shape[0]//8)
    tries = kwargs.get("tries", 4)
    dist = kwargs.get("dist", None)
    deltas = []
    for i in range(tries):
        idx = np.random.choice(X.shape[0], bs)
        batch = X[idx]
        if dist is None:
            dists = np.linalg.norm(
                        batch[None:,]-batch[:,None],
                        axis=-1)
        else:
            dists = dist(batch,batch)
        deltas.append(
                delta_rel(dists,randint(0,bs-1))
                )
    return deltas 

def compressed(dists):
    ''' Compress N x N symmetric matrix with 0s on the diagonal 
    to length N*(N-1)//2 array. Same as scipy.squareform
    Args:
        dists (ndarray): NxN matrix of pairwise distances
    Returns:
        ndarray storing dist[i,j] at index N*i + j - ((i+2)*(i+1))//2
        where i < j < N
    '''
    assert dists.shape[0] == dists.shape[1] and 0 <= base < dists.shape[0]
    N = dists.shape[0]
    out = np.zeros(N*(N-1)//2)
    for i in range(N):
        for j in range(i+1,N):
            out[N*i+j-((i+1)*(i+2))//2] = dists[i,j]
    return out

def decompressed(v, N):
    ''' Converted compressed distance matrix to 2D form. Same as scipy.squareform
    Args:
        v (ndarray): length N*(N-1)//2 compressed matrix
        N (int): dimension of original matrix
    Returns:
        N x N decompressed distancce matrix
    '''
    assert len(v) == N*(N-1)//2
    out = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            out[i,j] = v[N*i+j - ((i+2)*(i+1))//2]
    return out + out.T
