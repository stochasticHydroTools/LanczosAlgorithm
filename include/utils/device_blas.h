

#ifndef DEVICE_BLAS_H
#define DEVICE_BLAS_H
#ifdef CUDA_ENABLED
#include"defines.h"
#include"cuda_lib_defines.h"
#include"cublasDebug.h"
#define device_gemv(...) CublasSafeCall(cublasgemv(cublas_handle, CUBLAS_OP_N, __VA_ARGS__));
#define device_nrm2(...) CublasSafeCall(cublasnrm2(cublas_handle,  __VA_ARGS__));
#define device_axpy(...) CublasSafeCall(cublasaxpy(cublas_handle,  __VA_ARGS__));
#define device_dot(...) CublasSafeCall(cublasdot(cublas_handle,  __VA_ARGS__));
#define device_scal(...) CublasSafeCall(cublasscal(cublas_handle,  __VA_ARGS__));
#else

#include"lapack_and_blas_defines.h"
void device_gemv(int n, int m, lanczos::real * alpha,
		 lanczos::real* A, int inca,
		 lanczos::real* B, int incb,
		 lanczos::real *beta,
		 lanczos::real* C, int incc){
  cblas_gemv(CblasColMajor, CblasNoTrans, n, m, *alpha,A,inca, B, incb, *beta, C, incc);
}
void device_nrm2(int n,
		 lanczos::real* A, int inca, lanczos::real* res){
  *res = cblas_nrm2(n, A, inca);
}

void device_axpy(int n, lanczos::real * alpha,
		 lanczos::real* A, int inca,
		 lanczos::real* B, int incb){
  cblas_axpy(n, *alpha, A, inca, B, incb);
}

void device_dot(int n,
		lanczos::real* A, int inca,
		lanczos::real* B, int incb,
		lanczos::real * alpha){
  *alpha = cblas_dot(n, A, inca, B, incb);
}

void device_scal(int n, lanczos::real * alpha,
		 lanczos::real* A, int inca){
  cblas_scal(n, *alpha, A, inca);
}

#endif

#endif
