/*Raul P. Pelaez 2017-2022. Lanczos algorithm

References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
  -http://dx.doi.org/10.1063/1.4742347

*/
#include"LanczosAlgorithm.h"
#include<string.h>
#include"utils/lapack_and_blas_defines.h"
#include<stdexcept>
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
