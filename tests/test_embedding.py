import pytest
from random import randint
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform
import networkx as nx
import numpy as np

from hyperlib.embedding.treerep import treerep
from hyperlib.embedding.metric import avg_distortion, delta_rel
from hyperlib.utils.multiprecision import poincare_metric
from hyperlib.utils.graph import * 


@pytest.fixture
def sarich_data():
    return np.array([
		32.,  48.,  51.,  50.,  48.,  98., 148.,
		26.,  34.,  29.,  33., 84., 136.,
		42.,  44.,  44.,  92., 152.,
		44.,  38.,  86., 142.,
		42.,  89., 142.,
		90., 142.,
		148.
	])

def random_data(N,dim=2):
    # random points in the poincare ball 
    rs = 1.-np.random.exponential(scale=1e-4, size=(N,))
    pts = np.random.normal(size = (N,dim))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= rs[:,None]
    return pts 

def test_treerep_sarich(sarich_data):
    stats = [] 
    for _ in range(10):
        T = treerep(sarich_data)
        G = to_networkx(T)
        assert nx.algorithms.tree.recognition.is_tree(G)
        adj_mat = to_sparse(T)
        tree_metric = shortest_path(adj_mat, directed=False)
        distortion = avg_distortion(squareform(tree_metric[:8,:8]), sarich_data)
        stats.append(distortion)
    best = min(stats)
    print(f"Sarich Data\n\tTreerep Avg Distortion ------------ {best:.5f}")
    assert best < 0.1

def test_treerep_rand():
    pts = random_data(128)
    metric = poincare_metric(pts)
    stats = []
    for _ in range(10):
        T = treerep(metric)
        G = to_networkx(T)
        assert nx.algorithms.tree.recognition.is_tree(G)
        del G
        adj_mat = to_sparse(T)
        tree_metric = shortest_path(adj_mat, directed=False)
        dstn = avg_distortion(squareform(tree_metric[:128,:128], checks=False),
                                metric)
        stats.append(dstn)
    best = min(stats)
    print(f"Random Data\n\tTreerep Avg Distortion ------------ {best:.5f}")

def test_delta_hyperbolic():
    pts = random_data(128)
    metric = squareform(poincare_metric(pts), checks=False)
    delts = []
    for _ in range(10):
        base = randint(0,127)
        delts.append(delta_rel(metric,base))
    assert np.mean(pts) < 0.5
