/*Raul P. Pelaez 2017-2022. Lanczos algorithm

References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
  -http://dx.doi.org/10.1063/1.4742347

*/
#include"LanczosAlgorithm.h"
#include<string.h>
#include"utils/lapack_and_blas_defines.h"
#include<stdexcept>
#ifdef CUDA_ENABLED
#include"utils/debugTools.h"
#endif

namespace lanczos{

  Solver::Solver(real tolerance):
    N(0),
    max_iter(3), check_convergence_steps(3), tolerance(tolerance)
  {
    //Allocate necessary startig space
    this->incrementMaxIterations(0);
#ifdef CUDA_ENABLED
    //Init cuBLAS for Lanczos process
    CublasSafeCall(cublasCreate_v2(&cublas_handle));
#endif    
  }

  Solver::~Solver(){
#ifdef CUDA_ENABLED
    CublasSafeCall(cublasDestroy(cublas_handle));
#endif
  }

  real* Solver::getV(int N){
    if(N != this->N) numElementsChanged(N);
    return detail::getRawPointer(V);
  }

  
  void Solver::numElementsChanged(int newN){
    this-> N = newN;
    try{
      w.resize((N+1), real());
      V.resize(N*max_iter, 0);
      oldBz.resize((N+1), real());
    }
    catch(...){
      throw std::runtime_error("[Lanczos] Could not allocate memory");
    }
  }
  //Increase maximum dimension of Krylov subspace, reserve necessary memory
  void Solver::incrementMaxIterations(int inc){
    V.resize(N*(max_iter+inc),0);
    P.resize((max_iter+inc)*(max_iter+inc),0);
    hdiag.resize((max_iter+inc)+1,0);
    hsup.resize((max_iter+inc)+1,0);
    htemp.resize(2*(max_iter+inc),0);
    htempGPU.resize(2*(max_iter+inc),0);
    this->max_iter += inc;
  }

    int Solver::solve(MatrixDot *dot, real *Bz, const real*z, int N){
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

  void Solver::computeIteration(MatrixDot *dot, int i, real invz2){
    real* d_V =  detail::getRawPointer(V);
    real * d_w = detail::getRawPointer(w);
    /*w = D·vi*/
    dot->setSize(N);
    dot->dot(d_V+N*i, d_w);
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

  //Computes the current result guess sqrt(M)·v, stores in BdW
  void Solver::computeCurrentResultEstimation(int iter, real * BdW, real z2){
    iter++;
    /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
    /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
    /**************LAPACKE********************/
    /*The tridiagonal matrix is stored only with its diagonal and subdiagonal*/
    /*Store both in a temporal array*/
    for(int i=0; i<iter; i++){
      htemp[i] = hdiag[i];
      htemp[i+iter]= hsup[i];
    }
    /*P = eigenvectors must be filled with zeros, I do not know why*/
    real* h_P = P.data();
    memset(h_P, 0, iter*iter*sizeof(real));
    /*Compute eigenvalues and eigenvectors of a triangular symmetric matrix*/
    auto info = LAPACKE_steqr(LAPACK_COL_MAJOR, 'I',
			      iter, &htemp[0], &htemp[0]+iter,
			      h_P, iter);
    if(info!=0){
      throw std::runtime_error("[Lanczos] Could not diagonalize tridiagonal krylov matrix, steqr failed with code " + std::to_string(info));
    }
    /***Hdiag_temp = Hdiag·P·e1****/
    for(int j=0; j<iter; j++){
      htemp[j] = sqrt(htemp[j])*P[iter*j];
    }
    /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
    /*Compute with blas*/
    real alpha = 1.0;
    real beta = 0.0;
    cblas_gemv(CblasColMajor, CblasNoTrans,
	       iter, iter,
	       alpha,
	       h_P, iter,
	       &htemp[0], 1,
	       beta,
	       &htemp[0]+iter, 1);
    detail::device_copy(htemp.begin()+iter, htemp.begin()+2*iter, htempGPU.begin());
    /*y = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
    beta = 0.0;
    device_gemv(N, iter,
		&z2,
		detail::getRawPointer(V), N,
		detail::getRawPointer(htempGPU), 1,
		&beta,
		BdW, 1);
  }

  real Solver::computeError(real *Bz, real normNoise_prev){
    /*Compute error as in eq 27 in [1]
      Error = ||Bz_i - Bz_{i-1}||_2 / ||Bz_{i-1}||_2
    */
    /*oldBz = Bz-oldBz*/
    real * d_oldBz = detail::getRawPointer(oldBz);
    real a=-1.0;
    device_axpy(N,
		&a,
		Bz, 1,
		d_oldBz, 1);
    /*yy = ||Bz_i - Bz_{i-1}||_2*/
    real yy;
    device_nrm2(N,  d_oldBz, 1, &yy);
    //eq. 27 in [1]
    real Error = abs(yy/normNoise_prev);
    if(std::isnan(Error)){
      throw std::runtime_error("[Lanczos] Unknown error (found NaN in result guess)");
    }
    return Error;
  }

  void Solver::registerRequiredStepsForConverge(int steps_needed){
    if(steps_needed-2 > check_convergence_steps){
      check_convergence_steps += 1;
    }
    //Or check more often if I performed too many iterations
    else{
      check_convergence_steps = std::max(1, check_convergence_steps - 2);
    }
  }

  bool Solver::checkConvergence(int current_iterations, real *Bz, real normResult_prev){
    auto Error = computeError(Bz, normResult_prev);
    //Convergence achieved!
    if(Error <= tolerance){
      registerRequiredStepsForConverge(current_iterations);
      return true;
    }
    if(current_iterations>=iterationHardLimit){
      throw std::runtime_error("[Lanczos] Could not converge.");
    }
    return false;
  }

  real Solver::computeNorm(real *v, int numberElements){
    real norm2;
    device_nrm2(numberElements, v, 1, &norm2);
    return norm2;
  }
}
