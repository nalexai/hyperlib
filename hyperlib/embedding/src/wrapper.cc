#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "graph.h"
#include "treerep.h"

namespace py = pybind11;

py::array_t<double> py_metric(const Graph& G){
	DistMat D = G.metric();
	std::vector<double> metric = D.data();
	auto result = py::array_t<double>(metric.size());
	auto result_buf = result.request();
	int *result_ptr = (int *) result_buf.ptr;
	std::memcpy(result_ptr, metric.data(), metric.size()*sizeof(double));
	return result;
}

std::pair<Graph, Graph::wmap> py_graph_treerep(const Graph& G, double tol=0.1){
	DistMat D = G.metric();
	return treerep(D,tol);
}

// accepts 1D numpy array of size N*(N-1)/2 as the N-point metric 
Graph::wmap py_treerep(py::array_t<double, py::array::c_style | py::array::forcecast> metric, int N, double tol=0.1){
	if( N<2 || metric.size() != (N*(N-1))/2){
		throw std::invalid_argument("array must be size N*(N-1)/2");
	}
	DistMat M(metric.data(), N);
	return treerep(M,tol).second;
}

PYBIND11_MODULE(_embedding,m){
	py::class_<Graph>(m, "Graph")
		.def(py::init<>())
		.def("add_edge", &Graph::add_edge)
		.def("adj_list", &Graph::adj_list)
		.def("remove_edge", &Graph::remove_edge)
		.def("retract", &Graph::retract)
		.def("remove_vertex", &Graph::remove_vertex);

	m.def("graph_metric", &py_metric);
	m.def("treerep_graph", &py_graph_treerep, 
			py::arg("G"), py::arg("tol")=0.1);
	m.def("treerep", &py_treerep,
			py::arg("metric"), py::arg("N"), py::arg("tol")=0.1,
			"TreeRep algorithm from Sonthalia & Gilbert, 'Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding.'\n"
			"	Args:\n"
			"		metric (ndarray): compressed distance matrix stored in length N*(N-1)//2 array\n"
			"			dist(i,j) is stored in entry N*i + j - ((i+2)*(i+1)) // 2.\n"
			"		N (int): the number of points (dimension of distance matrix)\n"
			"		tol (double): tolerance for checking equalities (default=0.1)\n"
			"	Returns:\n"
			"		dict with keys (i,j), i<j representing edges and values representing the edge weight.\n"
			"		labels >= N are Steiner nodes inserted to form the tree.\n"
			);
					
	#ifdef VERSION_INFO
		m.attr("__version__") = VERSION_INFO;
	#else
		m.attr("__version__") = "dev";
	#endif
}

