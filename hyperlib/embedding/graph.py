from math import sqrt, floor 
import numpy as np
from scipy.spatial.distance import squareform
import scipy.sparse as sp
from .metric import is_metric
import _embedding

def treerep(dists, **kwargs):
    """
    TreeRep algorithm from Sonthalia & Gilbert, 'Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding'
    takes a metric (distance matrix) and computes a weighted tree that approximates it.
    	Args:
    		metric (ndarray): size NxN distance matrix or 
                            compressed matrix of length N*(N-1)//2 ( e.g from scipy.pdist )
    		tol (double): tolerance for checking equalities (default=0.1)
    	Returns:
            A dict mapping edges (u,v), u<v to their edge weight
        
        Notes:
            - TreeRep is a randomized algorithm, so you can generate multiple trees and take the best.
            - TreeRep inserts new nodes (called Steiner nodes) labeled with ints >= N.
                They may or may not have an interpretation in terms of the data.
            - some edge weights may be 0. In that case you can either retract the edge
                or perturb it to a small positive number.
    """
    tol = kwargs.get("tol",0.1)
    if len(dists.shape) == 1:
        N = floor( (1+sqrt(1+8*len(dists)))/2 )
        assert N*(N-1)//2 == len(dists)
        W = _embedding.treerep(dists, N, tol)
    elif len(dists.shape) == 2:
        assert is_metric(dists)
        W = _embedding.treerep(squareform(dist), dists.shape[0], tol)
    else:
        raise ValueError("Invalid distance matrix")
    return W

def to_networkx(edge_weights):
    ''' Convert edge weight dict to networkx weighted graph'''
    import networkx 
    G = networkx.Graph()
    for e in edge_weights:
        G.add_edge(e[0], e[1], weight=edge_weights[e])
    return G

def to_sparse(edge_weights):
    ''' Convert edge weight dict to compressed adjacency matrix,
    stored in a scipy.sparse.dok_matrix
    '''
    N = max([e[1] for e in edge_weights])+1
    mat = sp.dok_matrix((N,N), dtype=np.float64)
    for e in edge_weights:
        mat[e[0],e[1]] = edge_weights[e]
        mat[e[1],e[0]] = edge_weights[e]
    return mat
