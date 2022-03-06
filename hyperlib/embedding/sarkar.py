from math import sqrt, floor, log, log2
from itertools import product
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
import numpy as np
import mpmath as mpm

from ..utils.multiprecision import poincare_reflect0
from ..utils.linalg import rotate_3D_mp

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
    tau = kwargs.get("tau")
    max_deg = max(tree.degree)[1]

    if tau is None:
        tau = (1+eps)/eps * mpm.log(2*max_deg/ mpm.pi)
    prc = kwargs.get("precision")
    if prc is None:
        prc = _embedding_precision(tree,root,eps)
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

def sarkar_embedding_3D(tree, root, **kwargs):
    eps = kwargs.get("eps",0.1)
    weighted = kwargs.get("weighted", True)
    tau = kwargs.get("tau")
    max_deg = max(tree.degree)[1]

    if tau is None:
        tau = (1+eps)/eps * mpm.log(2*max_deg/ mpm.pi)
    prc = kwargs.get("precision")
    if prc is None:
        prc = _embedding_precision(tree,root,eps)
    mpm.mp.dps = prc
    
    n = tree.order()
    emb = mpm.zeros(n,3)
    place = []

    # place the children of root
    fib = fib_2D_code(tree.degree[root])
    for i, v in enumerate(tree[root]):
        r = mpm.tanh(tau*tree[root][v].get("weight",1.))
        v_emb = r * mpm.matrix([[fib[i,0],fib[i,1],fib[i,2]]])
        emb[v,:]=v_emb
        place.append((root,v))
    
    while place:
        u, v = place.pop() # u is the parent of v
        u_emb, v_emb = emb[u,:], emb[v,:]

        # reflect and rotate so that embedding(v) is at (0,0,0) 
        # and embedding(u) is in the direction of (0,0,1)
        u_emb = poincare_reflect0(v_emb, u_emb, precision=prc)
        R = rotate_3D_mp(mpm.matrix([[0.,0.,1.]]), u_emb)
        #u_emb = (R.T * u_emb).T

        # place children of v 
        fib = fib_2D_code(tree.degree[v])
        i=0
        for w in tree[v]:
            if w == u: # i=0 is for u (parent of v)
                continue            
            i+=1
            r = mpm.tanh(tau*tree[w][v].get("weight",1.))
            w_emb = r * mpm.matrix([[fib[i,0],fib[i,1],fib[i,2]]])

            #undo reflection and rotation
            w_emb = (R * w_emb.T).T
            w_emb = poincare_reflect0(v_emb, w_emb, precision=prc)
            emb[w,:] = w_emb
            place.append((v,w))
    return emb

def _embedding_precision(tree, root, eps):
    dists = single_source_dijkstra_path_length(tree,root)
    max_deg = max(tree.degree)[1]
    l = 2*max(dists.values())
    prc = floor( (log(max_deg+1)) * l/eps +1)
    return prc

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

def fib_2D_code(n, eps=None):
    epsilons = [0.33, 1.33, 3.33, 10, 27, 75]
    if eps is None:
        if n < 24:
            eps = epsilons[0]
        elif n < 177:
            eps = epsilons[1]
        elif n < 890:
            eps = epsilons[2]
        elif n < 11000:
            eps = epsilons[3]
        elif n < 400000:
            eps = epsilons[4]
        else:
            eps = epsilons[5]

    golden = (1+sqrt(5))/2
    i = np.arange(0,n)
    theta = 2*np.pi*i / golden
    phi = np.arccos(1 - 2*(i+eps)/(n-1+2*eps))
    M = np.stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)], axis=1)

    # rotate so that first point is (0,0,1)
    cos = M[0,2]
    sin = sqrt(1-cos**2)
    R = np.array([
        [cos, 0., sin],
        [0. , 1., 0. ],
        [-sin,0., cos],
    ])
    return M@R
