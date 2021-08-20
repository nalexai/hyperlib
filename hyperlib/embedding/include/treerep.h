#include <algorithm>
#include <map>
#include <cmath>
#include <thread>
#include <functional>
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
 * Returns: weighted adjacency list, where u maps to pairs {v, w} 
 * 			such that (u,v) is an edge with weight w.
 * 			
 */
Graph::wmap treerep(const DistMat& D, double tol=0.1); 

// ========== Helper functions =============
double grmv_prod(int x, int y, int z, const DistMat& W);
int _treerep_recurse(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, 
					int x, int y, int z);
void _sort(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, vecvec& zns,
			int x, int y, int z, int r, bool rtr);
void _thread_sort(Graph& G, DistMat& W, std::vector<int>& V, vecvec& zns, std::vector<int>& stn,
			int beg, int end, int x, int y, int z, int& r, bool& rtr);
void _zone1(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn,int v);
void _zone2(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, int u, int v);
std::default_random_engine& _trep_rng(int seed=1);
#endif
