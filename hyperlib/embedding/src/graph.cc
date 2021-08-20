#include "graph.h"

Graph::Graph(){
}

void Graph::remove_vertex(int v){
	vmap::iterator i = _adj.find(v);
	if (i != _adj.end()){
		vitr j;
		for (j = _adj[v].begin(); j!= _adj[v].end(); ++j){
			_rm(*j,v);
		}
		_adj.erase(i);
	}
}

Graph::vmap Graph::adj_list(){
	return _adj;
}

void Graph::add_edge(int u, int v){ 
	_insert(u,v);
	_insert(v,u);
}

void Graph::remove_edge(int u, int v){
	vmap::iterator fd = _adj.find(u);
	if (fd != _adj.end()){
		_rm(u,v);
	}
	fd = _adj.find(v);
	if (fd != _adj.end()){
		_rm(v,u);
	}
}

void Graph::retract(int u, int v){
	vmap::iterator fd = _adj.find(v);
	if (fd == _adj.end() || _adj[v].empty()){
		return;
	}
	for (vitr it=_adj[v].begin(); it!=_adj[v].end(); ++it){
		if( u!=*it){
			_rm(*it,v);
			add_edge(*it,u);
		}
	}
	_rm(u,v);
	_adj.erase(v);
}

inline void Graph::_rm(int u, int v){
	vitr it = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
	if (it !=_adj[u].end() && *it == v){
		_adj[u].erase(it);
	}
}

inline void Graph::_insert(int u, int v){
	if (u == v){ //don't insert loops
		return;
	}
	vitr it = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
	if (it == _adj[u].end()){
		_adj[u].push_back(v);
	}else if( *it != v){
		_adj[u].insert(it,v);
	}
}

inline std::vector<int> Graph::neighbors(int u){
	return _adj[u];
}

DistMat Graph::metric(double tol) const{
	int N = _adj.size();
	double infty = std::numeric_limits<double>::infinity();
	DistMat W(N, infty); 
	vmap::const_iterator itr,jtr,ktr;
	for (itr=_adj.begin(); itr!=_adj.end();++itr){
		for(const_vitr vit=itr->second.begin();vit!=itr->second.end(); ++vit){ 
			W(itr->first,*vit) = 1;
		}
	}
	// Floyd-Warshall
	int i,j,k;
	for (k=0,ktr=_adj.begin(); ktr!=_adj.end(); ++ktr,++k){
		for (i=0,itr=_adj.begin(); itr!=_adj.end(); ++itr,++i){
			for (j=i+1,jtr=_adj.begin(); j<N;++jtr,++j){
				if( i!=j && j!=k && k!=i
					&& (W(i,j) > W(i,k) + W(k,j) + tol ) ){
					W(i,j) = W(i,k) + W(k,j);
				}
			}
		}
	}
	return W;
}

void Graph::relabel(int u, int v){
	vmap::iterator fd = _adj.find(v);
	if( fd != _adj.end()){
		return;
	}
	fd = _adj.find(u);
	if( fd ==_adj.end()){
		return;
	}
	retract(v,u);
}


int Graph::size() const{
	return _adj.size();
}

int Graph::num_edges() const{
	int len=0;
	for (vmap::const_iterator it=_adj.begin();it!=_adj.end(); ++it){
		len += (it->second).size();
	}
	return len/2;
}

DistMat::DistMat(int N, double val): _N(N), _zero(0){
	_data.resize((N*(N-1))/2);
	for (int i=0; i<_N; ++i){
		for (int j=i+1; j<_N; ++j){
			(*this)(i,j) = val;
		}
	}
}

DistMat::DistMat(const DistMat& D, int N):_N(N), _zero(0.){
	int M = D.size();
	if (M > N){
		throw std::invalid_argument("Incompatible size");
	}
	_data.resize((N*(N-1))/2);
	for (int i = 0; i<_N; ++i){
		for( int j = i+1; j<_N; ++j){
			if( i<M && j<M){
				(*this)(i,j) = D(i,j);
			}else{
				(*this)(i,j) = 0;
			}
		}
	}
}

DistMat::DistMat(const std::vector<double>& dist, int N): _N(N), _zero(0.){
	int S = dist.size();
	if (S != (N*(N-1))/2){
		throw std::invalid_argument("Incompatible sizes "+std::to_string(S) 
									+" and "+std::to_string(N));
	}
	_data.resize((N*(N-1))/2);
	std::memcpy(_data.data(),dist.data(),S*sizeof(double));
}

DistMat::DistMat(const double* dist, int N): _N(N), _zero(0.){
	int S = (N*(N-1))/2;
	_data.reserve((N*(N-1))/2);
	std::memcpy(_data.data(),dist,S*sizeof(double));
}

double& DistMat::operator()(int i, int j){
	if(i >= _N || j >= _N || i < 0 || j < 0){
		throw std::invalid_argument("index out of bounds");
	}else if(i == j){
		return _zero;
	}else if (i > j){
		return _data[ _N*j + i - ((j+2)*(j+1))/2 ];
	}else{
		return _data[ _N*i + j - ((i+2)*(i+1))/2 ];
	}
}

double DistMat::operator()(int i, int j) const{
	if(i >= _N || j >= _N || i < 0 || j < 0){
		throw std::invalid_argument("index out of bounds");
	}else if(i == j){
		return 0;
	}else if (i >j){
		return _data[ _N*j + i - ((j+2)*(j+1))/2 ];
	}else{
		return _data[ _N*i + j - ((i+2)*(i+1))/2 ];
	}
}

DistMat& DistMat::operator*=(double d){
	for(int i=0; i< (_N*(_N-1))/2; ++i){
		_data[i] *= d;
	}
	return *this;
}

double DistMat::max() const{
	return *std::max_element(_data.begin(), _data.end());
}

std::vector<double> DistMat::data(){
	return _data;
}

const_vitr DistMat::nearest(int i, const std::vector<int>& pts) const{
	if(pts.empty()){
		throw std::invalid_argument("set of points is empty");
	}else{
		double min = (*this)(i,pts.front());
		std::vector<int>::const_iterator it, jt=pts.begin();
		for (it = pts.begin(); it != pts.end(); ++it){
			if( (*this)(i, *it) < min){
				min = (*this)(i, *it);
				jt = it;
			}
		}
		return jt;
	}
}

int DistMat::size() const{
	return _N;
}

bool Graph::is_adj(int u, int v){
	vmap::iterator fd = _adj.find(u);
	if (fd == _adj.end() || fd->second.empty()){
		return false;
	}
	vitr it = std::lower_bound(fd->second.begin(),fd->second.end(),v);
	return (it != fd->second.end() && *it == v);
}
