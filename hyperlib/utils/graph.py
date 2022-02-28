import networkx as nx
import numpy as np
import scipy.sparse as sp

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

def binary_tree(depth):
    return n_ary_tree(2, depth)

def trinary_tree(depth):
    return n_ary_tree(3, depth) 

def n_ary_tree(n, depth):
    assert n >= 2 
    assert depth >= 0 
    T = nx.Graph()
    for d in range(depth):
        for i in range(n**d,n**(d+1)):
            for j in range(n):
                T.add_edge(i, n*i+j)
    return nx.relabel.convert_node_labels_to_integers(T)
