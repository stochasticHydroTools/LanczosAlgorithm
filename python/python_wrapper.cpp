/* Raul P. Pelaez 2022. Pybind11 python wrappers for the Lanczos solver library
 */
#include "../include/utils/MatrixDot.h"
#include"lanczos_trampoline.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<stdexcept>
using real = lanczos::real;

namespace py = pybind11;
using namespace pybind11::literals;


//This class allows to inherit lanczos::MatrixDot from python
struct MatrixDotTrampoline: public lanczos::MatrixDot{  
  using MatrixDot::MatrixDot;
  
  void dot(real* v, real* Mv) override{
    pybind11::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
    // Try to look up the overridden method on the Python side.
    pybind11::function overridef = pybind11::get_override(this, "dot");
    if (overridef) {  // method is found
      //Pybind needs a dummy object to know about the lifetime of the py::arrays.
      //without it it will copy the memory.
      py::str dummy;
      // Call the Python function.
      py::array_t<real> Mvp = overridef(py::array_t<real, py::array::c_style>(this->m_size, v, dummy));
      std::copy(Mvp.data(), Mvp.data()+this->m_size, Mv);
    }
    else{
      throw std::runtime_error("[PyLanczos] The required dot function was not found in the provided functor.");
    }
  }
};

//Python wrapper class for the Lanczos solver.
class PyLanczos{
  std::shared_ptr<LanczosTrampoline> solver;
public:
  PyLanczos():
    solver(std::make_shared<LanczosTrampoline>()){}

  int run(lanczos::MatrixDot* dot, py::array_t<real> &result,  py::array_t<real> &v, real tolerance, int size){
    dot->setSize(size);
    return solver->run(dot, result.mutable_data(), v.data(), tolerance, size);
  }
  real runIterations(lanczos::MatrixDot* dot, py::array_t<real> &result,  py::array_t<real> &v, int numberIterations, int size){
    dot->setSize(size);
    return solver->runIterations(dot, result.mutable_data(), v.data(), numberIterations, size);
  }

};

#ifndef LANCZOS_PYTHON_NAME
#define LANCZOS_PYTHON_NAME Lanczos
#endif


PYBIND11_MODULE(LANCZOS_PYTHON_NAME, m){
  py::class_<lanczos::MatrixDot, MatrixDotTrampoline>(m, "MatrixDot", "The virtual class required by the Lanczos solver").
    def(py::init<>()).
    def("dot", &lanczos::MatrixDot::dot,
	"Given a result (Mv) and a vector (v), this method must write in Mv the result of multiplying the target matrix and v.",
	"v"_a, "The input vector", "Mv"_a, "The output result vector");

  py::class_<PyLanczos>(m, "Solver", "A Lanczos iterative solver. Computes sqrt(M)*v in O(N^2) operations, being N the size of the matrix, which must be square.").
    def(py::init<>()).
    //MatrixDot *dot, real* result, real* v, int size
    def("run", &PyLanczos::run, "Runs as many iterations as needed to achieve the provided tolerance.",
	"dot"_a, "A pointer to the matrix multiplication functor",
	"result"_a, "Output array storing the result of sqrt(M)*v",
	"v"_a, "The input array to be multiplied by sqrt(M)",
	"tolerance"_a, "The algorithm will run until this tolerance is achieved",
	"size"_a, "The size of the input array (and the matrix, size x size)").
    def("runIterations", &PyLanczos::runIterations, "Runs a given number of iterations and returns the residual.",
	"dot"_a, "A pointer to the matrix multiplication functor",
	"result"_a, "Output array storing the result of sqrt(M)*v",
	"v"_a, "The input array to be multiplied by sqrt(M)",
	"numberIterations"_a, "The algorithm will this many iterations",
	"size"_a, "The size of the input array (and the matrix, size x size)");

  m.def("getPrecision", [](){return (std::is_same<lanczos::real, float>::value)?"float":"double";});
}
