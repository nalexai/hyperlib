#include <algorithm>
#include <map>
#include <cmath>
#include <vector>
#include <set>
#include <random>
#include <iostream>
#include "graph.h"

#ifndef TREEREP_H
#define TREEREP_H
typedef std::vector< std::vector<int> > vecvec;

/**
 * TreeRep takes a metric and computes a weighted tree that approximates it 
 * (R. Sonthalia & A.C. Gilbert https://arxiv.org/abs/2005.03847 )
 *
 * @param D: DistMat holding the pairwise distances
 * @param tol: positive double specifying the tolerance
 * 
 * Returns: 
 * 		A pair (T,W) where T is the tree and W contains the (symmetric) edge weights
 */
std::pair<Graph,Graph::wmap> treerep(const DistMat& D, double tol=0.1); 

/**
 * Generate a random tree
 * 		@param n: number of vertices in the tree
 * 		@param seed: rng seed
 */
Graph rand_tree(unsigned n, int seed=1);

// ========== Helper functions =============
double grmv_prod(int x, int y, int z, const DistMat& W);
int _treerep_recurse(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, 
					int x, int y, int z);
vecvec _sort(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn,
			int x, int y, int z, int r, bool rtr);
void _zone1(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn,int v);
void _zone2(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, int u, int v);
std::default_random_engine& _trep_rng(int seed=1);
#endif
