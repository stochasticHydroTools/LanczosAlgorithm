/*Raul P. Pelaez 2022. Lanczos Algotihm,
  Computes the matrix-vector product sqrt(M)·v using a recursive algorithm.
  For that, it requires a functor in which the () operator takes an output real* array and an input real* (both device memory) as:
  inline __host__ __device__ operator()(real* out_Mv, real * in_v);
  This function must fill "out" with the result of performing the M·v dot product- > out = M·a_v.
  If M has size NxN and the cost of the dot product is O(M). The total cost of the algorithm is O(m·M). Where m << N.
  If M·v performs a dense M-V product, the cost of the algorithm would be O(m·N^2).
  References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
  J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347
Some notes:

  From what I have seen, this algorithm converges to an error of ~1e-3 in a few steps (<5) and from that point a lot of iterations are needed to lower the error.
  It usually achieves machine precision in under 50 iterations.

  If the matrix does not have a sqrt (not positive definite, not symmetric...) it will usually be reflected as a nan in the current error estimation. An exception will be thrown in this case.
*/

#ifndef LANCZOSALGORITHM_H
#define LANCZOSALGORITHM_H
#include<iostream>
#include"utils/defines.h"
#include"utils/device_blas.h"
#include<vector>
#include<memory>
#include"utils/device_container.h"

#ifdef CUDA_ENABLED
#include"utils/debugTools.h"
#endif
namespace lanczos{

  struct MatrixDot{
    
    virtual void operator()(real* Mv, real*v) = 0;
    
  };
  
  struct Solver{
    Solver(real tolerance = 1e-3);

    ~Solver(){
#ifdef CUDA_ENABLED
      CublasSafeCall(cublasDestroy(cublas_handle));
#endif
    }

    //Given a Dotctor that computes a product M·v (where M is handled by Dotctor ), computes Bv = sqrt(M)·v
    //Returns the number of iterations performed
    //B = sqrt(M)
    int solve(MatrixDot *dot, real *Bv, real* v, int N);
    
    //Overload for a shared_ptr
    int solve(std::shared_ptr<MatrixDot> dot, real *Bv, real* v, int N){
      return this->solve(dot.get(), Bv, v, N);
    }

    //Overload for an instance
    template<class SomeDot>
    int solve(SomeDot &dot, real *Bv, real* v, int N){
      MatrixDot* ptr = static_cast<MatrixDot*>(&dot);
      return this->solve(ptr, Bv, v, N);
    }

    //You can use this array as input to the solve operation, which will save some memory
    real * getV(int N){
      if(N != this->N) numElementsChanged(N);
      return detail::getRawPointer(V);
    }

#ifdef CUDA_ENABLED
    //The solver will use this cuda stream when possible
    void setCudaStream(cudaStream_t st){
      this->st = st;
      CublasSafeCall(cublasSetStream(cublas_handle, st));
    }
#endif

    //Set the maximum number of iterations allowed before throwing an exception
    void setIterationHardLimit(int newlimit){
      this->iterationHardLimit = newlimit;
    }
    
  private:
    void computeCurrentResultEstimation(int iter, real *BdW, real z2);
    //Increases storage space
    void incrementMaxIterations(int inc);
    void numElementsChanged(int Nnew);
    bool checkConvergence(int current_iterations, real *Bz, real normNoise_prev);
    real computeError(real* Bz, real normNoise_prev);
    real computeNorm(real *v, int numberElements);
    void computeIteration(MatrixDot *dot, int i, real invz2);
    void registerRequiredStepsForConverge(int steps_needed);
    void resizeIfNeeded(real*z, int N);
    int N;
#ifdef CUDA_ENABLED
    cublasHandle_t cublas_handle;
    cudaStream_t st = 0;
#endif
    /*Maximum number of Lanczos iterations*/
    int max_iter; //<100 in general, increases as needed
    int iterationHardLimit = 100; //Do not perform more than this iterations
    /*Lanczos algorithm auxiliar memory*/
    device_container<real> w; //size N, v in each iteration
    device_container<real> V; //size Nxmax_iter; Krylov subspace base transformation matrix
    //Mobility Matrix in the Krylov subspace
    std::vector<real> P;    //Transformation Matrix to diagonalize H, max_iter x max_iter
    /*upper diagonal and diagonal of H*/
    std::vector<real> hdiag, hsup, htemp;
    device_container<real> htempGPU;
    device_container<real> oldBz;
    int check_convergence_steps;
    real tolerance;
  };
}

#ifndef SHARED_LIBRARY_COMPILATION
#include"LanczosAlgorithm.cu"
#endif
#endif
 
