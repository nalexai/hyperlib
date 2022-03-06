from math import sqrt, floor
from scipy.spatial.distance import squareform
import numpy as np
import mpmath as mpm

from ..utils.graph import to_networkx
from .metric import is_metric
import __hyperlib_embedding

def treerep(dists, **kwargs):
    """
    TreeRep algorithm from Sonthalia & Gilbert, 'Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding'
    takes a metric (distance matrix) and computes a weighted tree that approximates it.
    	Args:
    		metric (ndarray): size NxN distance matrix or 
                            compressed matrix of length N*(N-1)//2 ( e.g from scipy.pdist )
    		tol (double): tolerance for checking equalities (default=0.1)
            return_networkx (bool): return a networkx.Graph instead of edges (default=False)
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
        W = __hyperlib_embedding.treerep(dists, N, tol)
    elif len(dists.shape) == 2:
        assert is_metric(dists)
        W = __hyperlib_embedding.treerep(squareform(dists), dists.shape[0], tol)
    else:
        raise ValueError("Invalid distance matrix")

    if kwargs.get("return_networkx", False):
        return to_networkx(W)
    return W
