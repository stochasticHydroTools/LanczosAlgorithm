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
  struct Solver{
    Solver(real tolerance = 1e-3);

    ~Solver(){
#ifdef CUDA_ENABLED
      CublasSafeCall(cublasDestroy(cublas_handle));
#endif
    }

    //Given a Dotctor that computes a product M·v (where M is handled by Dotctor ), computes Bv = sqrt(M)·v
    //Returns the number of iterations performed
    template<class Dotctor> //B = sqrt(M)
    int solve(Dotctor &dot, real *Bv, real* v, int N);

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
    template<class Dotctor>
    void computeIteration(Dotctor &dot, int i, real invz2);
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

  template<class Dotctor>
  int Solver::solve(Dotctor &dot, real *Bz, real*z, int N){
    //Handles the case of the number of elements changing since last call
    if(N != this->N){
      real * d_V = detail::getRawPointer(V);
      if(z == d_V){
	throw std::runtime_error("[Lanczos] Size mismatch in input");
      }
      numElementsChanged(N);
    }
    /*See algorithm I in [1]*/
    /************v[0] = z/||z||_2*****/
    /*If z is not the array provided by getV*/
    real* d_V = detail::getRawPointer(V);
    if(z != d_V){
      detail::device_copy(z, z+N, V.begin());
    }
    /*1/norm(z)*/
    real invz2 = 1.0/computeNorm(d_V, N);
    /*v[0] = v[0]*1/norm(z)*/
    device_scal(N, &invz2,  d_V, 1);
    /*Lanczos iterations for Krylov decomposition*/
    /*Will perform iterations until Error<=tolerance*/
    int i = -1;
    real normResult_prev = 1.0; //For error estimation, see eq 27 in [1]
    while(true){
      i++;
      /*Allocate more space if needed*/
      if(i == max_iter-1){
#ifdef CUDA_ENABLED
	CudaSafeCall(cudaDeviceSynchronize());
#endif
	this->incrementMaxIterations(2);
      }
      computeIteration(dot, i, invz2);
      /*Check convergence if needed*/
      if(i >= check_convergence_steps){ //Miminum of 3 iterations, will be auto tuned
	/*Compute Bz using h and z*/
	/**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
	this->computeCurrentResultEstimation(i, Bz, 1.0/invz2);
	/*The first time the result is computed it is only stored as oldBz*/
	if(i>check_convergence_steps){	  
	  if(checkConvergence(i, Bz, normResult_prev)){
	    return i;
	  }
	}
	/*Always save the current result as oldBz*/
	detail::device_copy(Bz, Bz+N, oldBz.begin());
	/*Store the norm of the result*/
	real * d_oldBz = detail::getRawPointer(oldBz);
        device_nrm2(N, d_oldBz, 1, &normResult_prev);
      }
    }
  }
  

  template<class Dotctor>
  void Solver::computeIteration(Dotctor &dot, int i, real invz2){
    real* d_V =  detail::getRawPointer(V);
    real * d_w = detail::getRawPointer(w);
    /*w = D·vi*/
    dot(d_V+N*i, d_w);
    if(i>0){
      /*w = w-h[i-1][i]·vi*/
      real alpha = -hsup[i-1];
      device_axpy(N,
		  &alpha,
		  d_V+N*(i-1), 1,
		  d_w, 1);
    }
    /*h[i][i] = dot(w, vi)*/
    device_dot(N,
	       d_w, 1,
	       d_V+N*i, 1,
	       &(hdiag[i]));
    if(i<max_iter-1){
      /*w = w-h[i][i]·vi*/
      real alpha = -hdiag[i];
      device_axpy(N,
		  &alpha,
		  d_V+N*i, 1,
		  d_w, 1);
      /*h[i+1][i] = h[i][i+1] = norm(w)*/
      device_nrm2(N, (real*)d_w, 1, &(hsup[i]));
      /*v_(i+1) = w·1/ norm(w)*/
      real tol = 1e-3*hdiag[i]*invz2;
      if(hsup[i]<tol) hsup[i] = real(0.0);
      if(hsup[i]>real(0.0)){
	real invw2 = 1.0/hsup[i];
	device_scal(N, &invw2, d_w, 1);
      }
      else{/*If norm(w) = 0 that means all elements of w are zero, so set w = e1*/	
	detail::device_fill(w.begin(), w.end(), real());
	w[0] = 1;
      }
      detail::device_copy(w.begin(), w.begin()+N, V.begin() + N*(i+1));
    }
  }
}

#ifndef SHARED_LIBRARY_COMPILATION
#include"LanczosAlgorithm.cu"
#endif
#endif
 
