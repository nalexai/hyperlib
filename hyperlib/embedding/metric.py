import numpy as np
from scipy.spatial.distance import squareform
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

def delta_rel(dists, base=None):
    ''' Measure the delta-hyperbolicity constant of data 
    with respect to basepoint, normalized by the diameter (max dist).
    Args:
        dists (ndarray): NxN matrix of pairwise distances
        base (int): index of basepoint in 0...N-1 (default = random)
    '''
    if base is None:
        base = randint(0,dists.shape[0]-1)
    assert is_metric(dists) and 0 <= base < dists.shape[0]
    G = mat_gromov_prod(dists, base)
    delta = np.max(max_min(G,G)-G)
    diam = np.max(dists)
    return delta/diam

def delta_sample(X, **kwargs):
    bs = kwargs.get("bs", X.shape[0])
    tries = kwargs.get("tries", 10)
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

def is_metric(X, tol=1e-8):
    return len(X.shape) == 2 and \
            np.all( np.abs(X-X.T)<tol ) and\
            np.all( np.abs(np.diag(X))<tol ) and\
            np.all(X >= 0)

def avg_distortion(metric1, metric2):
    ''' Average distortion between two metrics.
        Args:
            metric1, metric2 (ndarray): N x N distance matrices, 
                    or length N*(N-1)//2 compressed distance matrices
        Returns:
            average distortion (float)
    ''' 
    assert metric1.shape == metric2.shape
    if len(metric1.shape) > 1:
        assert is_metric(metric1)
        X = squareform(metric1)
    else:
        X = metric1
    if len(metric2.shape) > 1:
        assert is_metric(metric2)
        Y = squareform(metric2)
    else:
        Y = metric2
    return np.mean( np.abs(X-Y)/Y )
