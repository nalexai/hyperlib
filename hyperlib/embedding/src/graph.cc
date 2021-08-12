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

void vprint(const std::vector<int>& vec){
	for (std::vector<int>::const_iterator i = vec.begin(); i!= vec.end(); ++i){
		std::cout << *i << " ";
	}
	std::cout << std::endl;
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

void Graph::print() const{
	std::cout << "======= GRAPH =======" << std::endl;
    for (vmap::const_iterator i=_adj.begin(); i != _adj.end(); ++i){
        std::cout << i->first << " :  ";
		const_vitr j;
		for (j = i->second.begin(); j != i->second.end(); ++j){
			std::cout << *j << " ";
		}
		std::cout << std::endl;
    }
    std::cout << "\n";
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

Graph graph_from_mtx(std::string file){
	Graph G;
	std::ifstream f(file, std::ios::in);
	std::string s;
	if (!f.is_open()){
		throw std::runtime_error(std::string() + "Can't open file: " + file);
	}
	std::getline(f,s);
	if (s.find(MTX_GRAPH_HDR) == std::string::npos){
		throw std::runtime_error(std::string() + "Invalid mtx file: "+ file);
	}
	//skip comments
	char c = f.peek();
	while(c=='%'){
		std::getline(f,s);
		c = f.peek();
	}
	int rows, cols, lns;
	f >> rows >> cols >> lns;
	if (rows != cols){
		throw std::runtime_error(std::string()+"Rows not equal to columns: " + file);
	}
	int u,v;
	for (int i=0; i<lns; ++i){
		f >> u >> v;
		G.add_edge(u-1,v-1); // 1-indexed labels 
	}
	f.close();
	return G;
}

DistMat mat_from_mtx(std::string file){
	std::ifstream f(file, std::ios::in);
	std::string s;
	if (!f.is_open()){
		throw std::runtime_error(std::string() + "Can't open file: " + file);
	}
	std::getline(f,s);
	if (s.find(MTX_SYM_HDR)==std::string::npos){
		throw std::runtime_error(std::string() + "Invalid mtx file: "+ file);
	}
	//skip comments
	char c = f.peek();
	while(c=='%'){
		std::getline(f,s);
		c = f.peek();
	}
	//rows, cols, entries
	int rows, cols, lns;
	f >> rows >> cols >> lns;
	if (rows != cols){
		throw std::runtime_error(std::string()+"Rows not equal to columns: " + file);
	}

	DistMat M(rows);
	int u,v;
	double val;
	for(int i=0; i<lns; ++i){
		f >> u >> v >> val;
		M(u-1,v-1) = val;
	}
	f.close();
	return M;
}

int Graph::to_mtx(std::string file){
	std::ofstream f(file,std::ios::out);
	if(!f.is_open()){
		return 1;
	}
	f << MTX_GRAPH_HDR << std::endl;
	int S = this->size();
	f << S << " " << S << " " << this->num_edges() << std::endl;
	for (int u=0; u<_adj.size(); ++u){
		std::vector<int> nbr = neighbors(u);
		for(vitr vt=nbr.begin(); vt!=nbr.end(); ++vt){
			if(*vt > u){
				f << u+1 << " " << (*vt)+1 << std::endl;
			}
		}
	}
	f.close();
	return 0;
}

int DistMat::to_mtx(std::string file){ 
	std::ofstream f(file, std::ios::out);
	if (!f.is_open()){
		return 1;
	}
	f << MTX_SYM_HDR << std::endl;
	f << _N << " " << _N << " " << (_N*(_N-1))/2 << std::endl;
	for (int i=0; i<_N; ++i){
		for (int j=i+1; j<_N; ++j){
			f << i+1 << " " << j+1 << " " << (*this)(i,j) << std::endl;
		}
	}
	return 0;
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

void DistMat::print() const{
	for (int i=0; i<_N; ++i){
		for (int j=i+1; j<_N; ++j){
			std::cout << (*this)(i,j) << " ";
		}
		std::cout << std::endl;
	}
}

bool Graph::is_adj(int u, int v){
	vmap::iterator fd = _adj.find(u);
	if (fd == _adj.end() || fd->second.empty()){
		return false;
	}
	vitr it = std::lower_bound(fd->second.begin(),fd->second.end(),v);
	return (it != fd->second.end() && *it == v);
}

double Graph::mean_avg_precision(const DistMat& D) const{
	int N = _adj.size();
	if (N > D.size()){
		throw std::invalid_argument("incompatible matrix size");
	}
	double sum=0, prc=0;
	int b, intr;
	for (vmap::const_iterator it=_adj.begin(); it!=_adj.end(); ++it){
		for(const_vitr jt=it->second.begin();jt!=it->second.end();++jt){
			int v=it->first;
			prc=0;
			b=0;
			intr=0;
			// for each u_i in neighbors(v), find the set of vertices B_i = { u: d(u,v) < d(v,u_i) }
			// and the intersection of B_i and neighbors(v). 
			// The `precision` is #(B_i intersect neighbors(v)) / #B_i
			for(int u=0; u<N; ++u){
				if (u!=v && D(u,v) <= D(v,*jt)){
					b++;
					const_vitr fd = std::lower_bound(it->second.begin(),it->second.end(),u);
					if(fd!=it->second.end() && *fd == u){
						intr++;
					}
				}
			}
			prc += (double)intr/ (double)b;
		}
		if (!it->second.empty()){
			prc /= it->second.size();
		}
		sum += prc;
	}
	return sum/N;
}

double avg_distortion(const DistMat& D1, const DistMat& D2){
	int S = D1.size();
	if (S > D2.size()){
		throw std::invalid_argument("incompatible matrix dimensions");
	}
	double sum=0;
	double d;
	for (int i=0; i<S; ++i){
		for (int j=i+1; j<S; ++j){
			d = std::abs(D1(i,j) - D2(i,j));
			if( D1(i,j) > 0){
				d /= D1(i,j);
			}
			sum += d;
		}
	}
	sum /= (S*(S-1))/2;
	return sum;
}
