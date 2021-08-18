#include "treerep.h"
int TREP_N;
double TREP_TOL;

std::default_random_engine& _trep_rng(int seed){
	static std::default_random_engine rng(seed);
	return rng;
}

Graph::wmap treerep(const DistMat& D, double tol){
	TREP_N = D.size();
	TREP_TOL = tol;
	DistMat W(D, 2*TREP_N);
	std::vector<int> V(TREP_N);
	for (int i=0; i<TREP_N; ++i){
		V[i] = i;
	}

	//choose random starting vertices
	std::shuffle(V.begin(), V.end(), _trep_rng());
	int x = V.back();
	V.pop_back();
	int y = V.back();
	V.pop_back();
	int z = V.back();
	V.pop_back();

	//initialize steiner nodes
	std::vector<int> stn;
	stn.reserve(TREP_N);
	for (int i=2*TREP_N; i>=TREP_N; --i){
		stn.push_back(i);
	}
	Graph G;
	_treerep_recurse(G,W,V,stn,x,y,z);

	Graph::wmap weight;
	for(int i=0; i<W.size(); ++i){
		for( int j=i+1; j<W.size(); ++j){
			if(W(i,j)<0){
				W(i,j) = 0;
			}
			if(G.is_adj(i,j)){
				weight[std::make_pair(i,j)] = W(i,j);
			}
		}
	}
	return weight;
}

int _treerep_recurse(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn,
					int x, int y, int z){
	if(stn.empty()){
		return 1;
	}

	// form the universal tree for x,y,z
	int r = stn.back();
	stn.pop_back();
	G.add_edge(r,x);
	G.add_edge(r,y);
	G.add_edge(r,z);

	bool rtr = false;
	W(r,x) = grmv_prod(x,y,z,W); 
	if (std::abs(W(r,x)) < TREP_TOL && !rtr){// retract (r, x)
		W(r,x) = 0;
		G.retract(x,r);
		stn.push_back(r);
		r = x;
		rtr = true;
	}
	W(r,y) = grmv_prod(y,x,z,W);
	if (std::abs(W(r,y)) < TREP_TOL && !rtr){// retract (r, y)
		W(r,x) = 0;
		W(r,y) = 0;
		G.retract(y,r);
		stn.push_back(r);
		r = y;
		rtr = true;
	}
	W(r,z) = grmv_prod(z,x,y,W);
	if (std::abs(W(r,z)) < TREP_TOL && !rtr){ //retract (r, z)
		W(r,x) = 0;
		W(r,y) = 0;
		W(r,z) = 0;
		G.retract(z,r);
		stn.push_back(r);
		r = z ;
		rtr = true;
	}
	//sort rest of vertices into 7 zones
	vecvec zone(7);
	if( V.size() < 32){
		_sort(G,W,V,stn,zone,x,y,z,r,rtr);
	}else{ //multithread sort
		std::vector<vecvec> tzns(4);
		for(int i=0; i<4; ++i){
			tzns[i].resize(7);
		}
		std::thread t1(_thread_sort, std::ref(G), std::ref(W), std::ref(V), std::ref(tzns[0]),std::ref(stn),
						0, V.size()/4, x,y,z,std::ref(r),std::ref(rtr));
		std::thread t2(_thread_sort, std::ref(G), std::ref(W), std::ref(V), std::ref(tzns[1]), std::ref(stn),
						V.size()/4, V.size()/2, x,y,z,std::ref(r),std::ref(rtr));
		std::thread t3(_thread_sort, std::ref(G), std::ref(W), std::ref(V), std::ref(tzns[2]), std::ref(stn),
						V.size()/2, 3*V.size()/4, x,y,z,std::ref(r),std::ref(rtr));
		std::thread t4(_thread_sort, std::ref(G), std::ref(W), std::ref(V), std::ref(tzns[3]), std::ref(stn),
						3*V.size()/4, V.size(), x,y,z,std::ref(r),std::ref(rtr));
		t1.join();
		t2.join(); 
		t3.join();
		t4.join();
		for(int i=0; i<7; ++i){
			for( int j=0; j<4; ++j){
				std::move(
						tzns[j][i].begin(), 
						tzns[j][i].end(), 
						std::back_inserter(zone[i])
					);
			}
		}
	}
	_zone1(G,W,zone[0],stn,r);
	_zone1(G,W,zone[1],stn,z);
	_zone1(G,W,zone[3],stn,x);
	_zone1(G,W,zone[5],stn,y);
	_zone2(G,W,zone[2],stn,z,r);
	_zone2(G,W,zone[4],stn,x,r);
	_zone2(G,W,zone[6],stn,y,r);
	return 0;
}

void _thread_sort(Graph& G, DistMat& W, std::vector<int>& V, vecvec& zns, std::vector<int>& stn,
						int beg, int end, int x, int y, int z, int& r, bool& rtr){
	for (int i = beg; i< end; ++i){
		int w = V[i];
		double a = grmv_prod(w,x,y,W);
		double b = grmv_prod(w,y,z,W);
		double c = grmv_prod(w,z,x,W);
		double max = std::max({a,b,c});
		if ( std::abs(a-b)<TREP_TOL && std::abs(b-c)<TREP_TOL && std::abs(c-a)<TREP_TOL){
			if (a<TREP_TOL &&  b<TREP_TOL && c<TREP_TOL && !rtr){ //retract (r,w)
				rtr = true;
				for(int i = TREP_N; i<W.size(); ++i){
					W(w,i) = W(r,i);
				}
				W(r,x) = 0;
				W(r,y) = 0;
				W(r,z) = 0;
				G.retract(w,r);
				stn.push_back(r);
				r = w;
			}else{
				zns[0].push_back(w); //zone1(r)
				W(r,w) = (a+b+c)/3;
			}
		}else if (a == max){
			if (std::abs(W(z,w)-b)<TREP_TOL || std::abs(W(z,w)-c)<TREP_TOL){
				zns[1].push_back(w); // zone1(z)
			}else{
				zns[2].push_back(w); // zone2(z)
			}
			W(r,w) = a;
		}else if (b == max){
			if (std::abs(W(z,w)-a)<TREP_TOL || std::abs(W(z,w)-c)<TREP_TOL){
				zns[3].push_back(w); // zone1(x)
			}else{
				zns[4].push_back(w); // zone2(x)
			}
			W(r,w) = b;
		}else if (c == max){
			if (std::abs(W(z,w)-b)<TREP_TOL || std::abs(W(z,w)-a)<TREP_TOL){
				zns[5].push_back(w); // zone1(y)
			}else{
				zns[6].push_back(w); // zone2(y)
			}
			W(w,r) = c;
		}
	}
}

void _sort(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, vecvec& zns,
			int x, int y, int z, int r, bool rtr){ 
	for (int i = 0; i< V.size(); ++i){
		int w = V[i];
		double a = grmv_prod(w,x,y,W); 
		double b = grmv_prod(w,y,z,W);
		double c = grmv_prod(w,z,x,W);
		double max = std::max({a,b,c});
		if ( std::abs(a-b)<TREP_TOL && std::abs(b-c)<TREP_TOL && std::abs(c-a)<TREP_TOL){
			if (a<TREP_TOL &&  b<TREP_TOL && c<TREP_TOL && !rtr){ //retract (r,w)
				rtr = true;
				for( int i = TREP_N; i<W.size(); ++i){
					W(w,i) = W(r,i);
				}
				W(r,x) = 0;
				W(r,y) = 0;
				W(r,z) = 0;
				G.retract(w,r);
				stn.push_back(r);
				r = w;
			}else{
				zns[0].push_back(w); //zone1(r)
				W(r,w) = (a+b+c)/3;
			}
		}else if (a == max){
			if (std::abs(W(z,w)-b)<TREP_TOL || std::abs(W(z,w)-c)<TREP_TOL){
				zns[1].push_back(w); // zone1(z)
			}else{
				zns[2].push_back(w); // zone2(z)
			}
			W(r,w) = a; 
		}else if (b == max){
			if (std::abs(W(z,w)-a)<TREP_TOL || std::abs(W(z,w)-c)<TREP_TOL){
				zns[3].push_back(w); // zone1(x)
			}else{
				zns[4].push_back(w); // zone2(x)
			}
			W(r,w) = b;
		}else if (c == max){
			if (std::abs(W(z,w)-b)<TREP_TOL || std::abs(W(z,w)-a)<TREP_TOL){
				zns[5].push_back(w); // zone1(y)
			}else{
				zns[6].push_back(w); // zone2(y)
			}
			W(w,r) = c;
		}
	}
}

void _zone1(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, int v){
	int S = V.size(); 
	if (S == 1){
		int u = V.back();
		V.pop_back();
		G.add_edge(u,v);
	}else if (S>1){	
		std::shuffle(V.begin(), V.end(), _trep_rng());
		int u = V.back();
		V.pop_back();
		int z = V.back();
		V.pop_back();
		_treerep_recurse(G,W,V,stn,v,u,z);
	}
}

void _zone2(Graph& G, DistMat& W, std::vector<int>& V, std::vector<int>& stn, 
					int u, int v){
	if (!V.empty()){
		std::vector<int>::const_iterator it = W.nearest(v, V);
		int z = *it; 
		V.erase(it);
		G.remove_edge(u,v);
		_treerep_recurse(G,W,V,stn,u,v,z);
	}
}

inline double grmv_prod(int x, int y, int z, const DistMat& W){
	return 0.5*(W(x,y)+W(x,z)-W(y,z));
}
