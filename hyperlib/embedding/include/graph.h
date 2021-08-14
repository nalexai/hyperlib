#include <stdexcept>
#include <limits>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <fstream>
#include <iostream>

#ifndef GRAPH_H
#define GRAPH_H
typedef std::vector<int>::iterator vitr;
typedef std::vector<int>::const_iterator const_vitr;

/**
 * Symmetric matrix with 0's on the diagonal,
 * e.g. pairwise distances of N points. 
 * 
 * Constructors:
 * 		DistMat(unsigned N)
 * 			@param N: number of points ( > 0 )
 * 			@param val: value to fill matrix with (default=0)
 * 		DistMat(const DistMat& D, unsigned N) 
 * 			@param D: A size M <= N DistMat to copy values from
 * 		DistMat(const std::vector<double>& dist, unsigned N)
 * 			@param dist: A size N*(N-1)/2 vector of distances d(i,j), 0<=i<j<N
 * 						in lexographic order
 *
 * Methods:
 * 		double operator()(int i, int j)
 *			Access the element at (i,j). If i=j it is 0
 * 			@param i,j: ints >=0 and < N
 * 		DistMat& operator *=(double d)
 * 			multiply all entries by a scalar
 *		int nearest(int i, const std::vector<int>& pts) 
 *			Find the element of pts closest to point i. 
 *			@param pts: vector of non-negative ints < N
 *		int size()
 *			Return dimension of the matrix
 *		double max()
 *			Return max value in matrix	
 */
class DistMat{
	public:
		DistMat(int N, double val=0);
		DistMat(const DistMat& D, int N); 
		DistMat(const std::vector<double>& dist, int N); 
		DistMat(const double* dist, int N);
		double operator()(int i, int j) const;
		double& operator()(int i, int j);
		DistMat& operator*=(double d);
		double max() const;
		const_vitr nearest(int i, const std::vector<int>& pts) const; 
		int size() const; 
		void print() const; 
		int to_mtx(std::string file);
		std::vector<double> data();
	private:
		int _N;
		double _zero;
		std::vector<double> _data;
};

/**
 * An undirected graph stored in adjacency list. Vertices are labeled with ints.
 * 
 * Methods:
 * 		void add_edge(int u, int v)
 * 			Add edge (u,v). 
 * 		void remove_edge(int u, int v)
 *			Remove edge (u,v). 
 * 		void remove_vertex(int v)
 * 			Remove vertex v and all its edges. 
 * 		void retract(int u, int v)
 *			Retract edge (u,v) and label the new vertex u.
 *		bool is_adj(int u, int v)
 *			return true if (u,v) is in the graph
 *		std::vector<int> neighbors(int u)
 *			Get vector of vertices adjacent to u
 *		void relabel(int u, int v)
 *			relabel u as v, if u is a vertex in the graph and v is not
 * 		DistMat metric(double tol)
 *			Calculate the shortest path distance between all vertices.
 *			Assumes vertices are 0...N and graph is connected
 *			@param tol: error tolerance for Floyd-Warshall
 *			@returns: DistMat representing symmetric matrix with distances
 * 		int size()
 * 			number of vertices in the graph	
 */
class Graph{
	public:
		typedef std::map<int, std::vector<int> > vmap; //adj map 
		typedef std::map< std::pair<int,int>, double > wmap; //edge weights
		Graph();
		void add_edge(int u, int v);
		void remove_edge(int u, int v);
		void remove_vertex(int v);
		void retract(int u, int v);
		bool is_adj(int u, int v);
		std::vector<int> neighbors(int u);
		vmap adj_list();

		DistMat metric(double tol=0.1) const;
		DistMat tree_metric() const;
		double mean_avg_precision(const DistMat& D) const;
		int size() const;
		int num_edges() const;
		void print() const;
		int to_mtx(std::string file);
		void relabel(int u, int v);
	private:
		void _rm(int u, int v);
		void _insert(int u, int v);
		vmap _adj;
};

/**
 * Calculate average distortion between N-point metrics 
 * 		@param D1, D2: DistMats representing the pairwise distances between N points. 
 * 						N = D1.size() <= D2.size() is allowed in which case 
 * 						only the first N columns of D2 are used.
 */
double avg_distortion(const DistMat& D1, const DistMat& D2);

 /* ======== MTX file utilities =========
 * For format spec see https://math.nist.gov/MatrixMarket/formats.html#MMformat 
 */
#define MTX_GRAPH_HDR "%MatrixMarket matrix coordinate pattern symmetric"
#define MTX_SYM_HDR "%MatrixMarket matrix coordinate real symmetric"

/**
 * int Graph::to_mtx(std::string file)
 * 		Write graph to mtx file as a `coordinate pattern symmetric' matrix
 * 		@param file: file name
 * 		@returns: 0 if successful, otherwise 1
 * 	int DistMat::to_mtx(std::string file)
 * 		Write matrix to mtx as `coordinate real symmetric' matrix
 * 		@returns: 0 if successful, otherwise 1 
 */

/** 
 * Load graph from mtx file
 * 		@param file: file with `coordinate pattern symmetric' matrix
 * 		@throws: runtime error if file is wrong format
 */
Graph graph_from_mtx(std::string file);

/**
 * Load DistMat from mtx file
 * 		@param file: file with `coordinate real symmetric' matrix 
 * 		@thows runtime error if file is wrong format 
 */
DistMat mat_from_mtx(std::string file);

#endif
