from math import sqrt, floor, log, log2
from itertools import product
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import squareform
import numpy as np
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
from .metric import is_metric
from ..utils.multiprecision import poincare_reflect0
import mpmath as mpm
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
        W = _embedding.treerep(squareform(dists), dists.shape[0], tol)
    else:
        raise ValueError("Invalid distance matrix")
    return W

def to_networkx(edge_weights):
    ''' Convert edge weight dict to networkx weighted graph'''
    G = nx.Graph()
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

def sarkar_embedding(tree, root, **kwargs):
    ''' 
    Embed a tree in the Poincare disc using Sarkar's algorithm 
    from "Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane.
        Args:
            tree (networkx.Graph) : The tree represented with int node labels.
                  Weighted trees should have the edge attribute "weight"
            root (int): The node to use as the root of the embedding 
        Keyword Args:
            weighted (bool): True if the tree is weighted (default True)
            tau (float): the scaling factor for distances. 
                        By default it is calculated based on statistics of the tree.
            epsilon (float): parameter >0 controlling distortion bound (default 0.1).
            precision (int): number of bits of precision to use.
                            By default it is calculated based on tau and epsilon.
        Returns:
            size N x 2 mpmath.matrix containing the coordinates of embedded nodes
    '''
    eps = kwargs.get("epsilon",0.1)
    weighted = kwargs.get("weighted", True)
    tau = kwargs.get("tau",None)
    max_deg = max(tree.degree)[1]

    if tau is None:
        tau = (1+eps)/eps * mpm.log(2*max_deg/ mpm.pi)
    prc = kwargs.get("precision",None)
    if prc is None:
        dists = single_source_dijkstra_path_length(tree,root)
        l = 2*max(dists.values())
        prc = floor( (log(max_deg+1)) * l/eps +1)
    mpm.mp.dps = prc
    
    n = tree.order()
    emb = mpm.zeros(n,2)
    place = []

    # place the children of root
    for i, v in enumerate(tree[root]):
        if weighted: 
            r = mpm.tanh( tau*tree[root][v]["weight"])
        else:
            r = mpm.tanh(tau)
        theta = 2*i*mpm.pi / tree.degree[root]
        emb[v,0] = r*mpm.cos(theta)
        emb[v,1] = r*mpm.sin(theta)
        place.append((root,v))
    
    # TODO parallelize this
    while place:
        u, v = place.pop() # u is the parent of v
        p, x = emb[u,:], emb[v,:]
        rp = poincare_reflect0(x, p, precision=prc)
        arg = mpm.acos(rp[0]/mpm.norm(rp))
        if rp[1] < 0:
            arg = 2*mpm.pi - arg
            
        theta = 2*mpm.pi / tree.degree[v]
        i=0
        for w in tree[v]:
            if w == u: continue
            i+=1
            if weighted:
                r = mpm.tanh(tau*tree[v][w]["weight"])
            else:
                r = mpm.tanh(tau)
            w_emb = r * mpm.matrix([mpm.cos(arg+theta*i),mpm.sin(arg+theta*i)]).T
            w_emb = poincare_reflect0(x, w_emb, precision=prc)
            emb[w,:] = w_emb
            place.append((v,w))
    return emb

def sarkar_embedding_high_dim(tree, root, **kwargs):
    eps = kwargs.get("eps",0.1)
    weighted = kwargs.get("weighted", True)
    dim = kwargs.get("dim")
    tau = kwargs.get("tau", None)
    max_deg = max(tree.degree)[1]

    prc = kwargs.get("precision",None)
    if prc is None:
        dists = single_source_dijkstra_path_length(tree,root)
        l = 2*max(dists.values())
        prc = floor( (log(max_deg+1)) * l/eps +1)
    mpm.mp.dps = prc
    
    
    n = tree.order()
    emb = mpm.zeros(n,dim)
    place = []

    # place the children of root
    for i, v in enumerate(tree[root]):
        if weighted: 
            r = mpm.tanh( tau*tree[root][v]["weight"])
        else:
            r = mpm.tanh(tau)
        theta = 2*i*mpm.pi / tree.degree[root]
        emb[v,0] = r*mpm.cos(theta)
        emb[v,1] = r*mpm.sin(theta)
        place.append((root,v))
    
    # TODO parallelize this
    while place:
        u, v = place.pop() # u is the parent of v
        p, x = emb[u,:], emb[v,:]
        rp = poincare_reflect0(x, p, precision=prc)
        arg = mpm.acos(rp[0]/mpm.norm(rp))
        if rp[1] < 0:
            arg = 2*mpm.pi - arg
            
        theta = 2*mpm.pi / tree.degree[v]
        i=0
        for w in tree[v]:
            if w == u: continue
            i+=1
            if weighted:
                r = mpm.tanh(tau*tree[v][w]["weight"])
            else:
                r = mpm.tanh(tau)
            w_emb = r * mpm.matrix([mpm.cos(arg+theta*i),mpm.sin(arg+theta*i)]).T
            w_emb = poincare_reflect0(x, w_emb, precision=prc)
            emb[w,:] = w_emb
            place.append((v,w))

def hadamard_code(n):
    ''' 
    Generate a binary Hadamard code 
        Args:
            n (int): if n = 2**k + r < 2**(k+1), generates a [n, k, 2**(k-1)] Hadamard code
        Returns:
            2**k x n np.array where code(i) = the ith row
    '''
    k = floor( log2(n) )
    r = n - 2**k 
    gen = np.zeros((k,2**k), dtype=np.int32)
    for i, w in enumerate( product([0,1], repeat=k) ):
        gen[:,i] = w
    code = gen.T @ gen % 2
    if r > 0: 
        code = np.concatenate( (code, code[:,:3]), axis=1 )
    return code

def fib_2d_code(n, eps=None):
    if eps is None: 
        if n >= 600000:
          eps = 214
        elif n>= 400000:
          eps = 75
        elif n>= 11000:
          eps = 27
        elif n>= 890:
          eps = 10
        elif n>= 177:
          eps = 3.33 
        elif n>= 24:
          eps = 1.33 
        else:
          eps = 0.33 

    golden = (1+sqrt(5))/2
    i = np.arange(0,n)
    theta = 2*np.pi*i / golden
    phi = np.arccos(
            1 - 2*(i+eps)/(n-1+2*eps)
            )
    return np.stack(
            [np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)],
            axis=1
        )
