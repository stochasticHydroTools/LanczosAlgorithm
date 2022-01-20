# Lanczos solver,
  Computes the matrix-vector product sqrt(M)·v using a recursive algorithm.  
  For that, it requires a functor in which the () operator takes an output real* array and an input real* (both in device memory if compiled in CUDA mode or host memory otherwise) as:  
  ```c++ 
  inline void operator()(real* in_v, real * out_Mv);
  ```  
  This function must fill "out" with the result of performing the M·v dot product- > out = M·a_v.  
  If M has size NxN and the cost of the dot product is O(M). The total cost of the algorithm is O(m·M). Where m << N.  
  If M·v performs a dense M-V product, the cost of the algorithm would be O(m·N^2).  

This is a header-only library, although a shared library can be 

## Usage:  

See example.cu for an usage example that can be compiled to work in GPU or CPU mode instinctively.  
See example.cpp for a CPU only example.  

Let us go through the remaining one, a GPU-only example.  

Create the module:
```c++
	real tolerance = 1e-6;
    lanczos::Solver lanczos(tolerance);
```
Write a functor that computes the product between the original matrix and a given vector, "v":
```c++
//A functor that will return the result of multiplying a certain matrix times a given vector.
//Must inherit from lanczos::MatrixDot
struct DiagonalMatrix: public lanczos::MatrixDot{
  int size;
  DiagonalMatrix(int size): size(size){}
  
  void operator()(real* v, real* Mv){
    //An example diagonal matrix
    for(int i=0; i<size; i++){
      Mv[i] = 2*v[i];
    }
  }

};

```

Provide the solver with an instance of the functor and the target vector:  

```c++
    int size = 10;
    //A GPU vector filled with ones.
    thrust::device_vector<real> v(size);
    thrust::fill(v.begin(), v.end(), 1);
    //A vector to store the result of sqrt(M)*v
    thrust::device_vector<real> result(size);
    //A functor that multiplies by a diagonal matrix
    MatrixDot dot(size);
    //Call the solver
    real* d_result = thrust::raw_pointer_cast(result.data());
    real* d_v = thrust::raw_pointer_cast(v.data());
    int numberIterations = lanczos.solve(dot, d_result, d_v, size);
```
The solve function returns the number of iterations that were needed to achieve the requested accuracy.

## Other functions:  

After a certain number of iterations, if convergence was not achieved, the module will give up and throw an error.  
To increase this threshold you can use this function:  
```c++
lanczos::Solver::setIterationHardLimit(int newlimit);
```
## Compilation:  
This library requires lapacke and cblas (can be replaced by MKL). In GPU mode CUDA is also needed.  
Note, however, that the heavy-weight of this solver comes from the Matrix-vector multiplication that must provided by the user. The main benefit of the CUDA mode is not an increased performance of the internal library code, but the fact that the input/output arrays will live in the GPU (saving potential memory copies).  
## Optional macros:  

**CUDA_ENABLED**: Will compile a GPU enabled shared library, the solver expects input/output arrays to be in the GPU and most of the computations will be carried out in the GPU. Requires a working CUDA environment.  
**DOUBLE_PRECISION**: The library is compiled in single precision by default. This macro switches to double precision, making ```lanczos::real``` be a typedef to double.  
**USE_MKL**: Will include mkl.h instead of lapacke and cblas. You will have to modify the compilation flags accordingly.  
**SHARED_LIBRARY_COMPILATION**: The Makefile uses this macro to compile a shared library. By default, this library is header only.  

See the Makefile for further instructions.  

## References:  

  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations  J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347  
  
## Some notes:  

  From what I have seen, this algorithm converges to an error of ~1e-3 in a few steps (<5) and from that point a lot of iterations are needed to lower the error.  
  It usually achieves machine precision in under 50 iterations.  

  If the matrix does not have a sqrt (not positive definite, not symmetric...) it will usually be reflected as a nan in the current error estimation. In this case an exception will be thrown.  
