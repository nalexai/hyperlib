import pytest
from random import randint
import networkx as nx

from hyperlib.utils.graph import binary_tree, trinary_tree
from hyperlib.utils.multiprecision import poincare_metric
from hyperlib.embedding.sarkar import * 

def test_sarkar_2D_unweighted():
    T = binary_tree(4)
    M = sarkar_embedding(T, 0, weighted=False, precision=30)
    n = nx.number_of_nodes(T)
    assert M.rows == n 
    assert M.cols == 2 
    assert all([ mpm.norm(M[i,:]) < 1 for i in range(n)])

def test_sarkar_3D_unweighted():
    T = trinary_tree(4)
    M = sarkar_embedding_3D(T, 0, weighted=False, tau=0.75, precision=40)
    n = nx.number_of_nodes(T)
    assert M.rows == n 
    assert M.cols == 3
    assert all([ mpm.norm(M[i,:]) < 1 for i in range(n)])
