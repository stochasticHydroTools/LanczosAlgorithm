/*Raul P. Pelaez 2022. Lanczos solver CPU example
This code will compute and print sqrt(M)*v using the iterative Krylov solver in [1].

In this usage example, the matrix M is a diagonal matrix and v is filled with ones.

This code is equivalent to example.cu, but can only be compiled in CPU mode.

You can compile it with:
g++ -std=c++14 -Iinclude example.cpp -llapacke -lcblas

References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
  J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347
 */
#include<iostream>
#include <LanczosAlgorithm.h>

//Using this floating point type (either float or double) will make the code be compiled with the same
// precision as the library.
using real = lanczos::real;
//A functor that will return the result of multiplying a certain matrix times a given vector
struct DiagonalMatrix: public lanczos::MatrixDot{
  int size;
  DiagonalMatrix(int size): size(size){}
  
  void dot(real* v, real* Mv) override{
    //an example diagonal matrix
    for(int i=0; i<size; i++){
      Mv[i] = (2+i/10.0)*v[i]*2;
    }
  }

};


int main(){
  {
    //Initialize the solver
    lanczos::Solver lanczos;
    int size = 10;
    //A vector filled with 1.
    //Lanczos defines this type for convenience. It will be a thrust::device_vector if CUDA_ENABLED is defined and an std::vector otherwise
    std::vector<real> v(size);
    std::fill(v.begin(), v.end(), 1);
    //A vector to store the result of sqrt(M)*v
    std::vector<real> result(size);
    //A functor that multiplies by a diagonal matrix
    DiagonalMatrix dot(size);
    //Call the solver
    int numberIterations = lanczos.run(dot, result.data(), v.data(), tolerance, size);
    std::cout<<"Solved after "<<numberIterations<< " iterations"<<std::endl;
    //Now result is filled with sqrt(M)*v = sqrt(2)*[1,1,1...1]
    std::cout<<"Result: ";for(int i = 0; i<10; i++) std::cout<<result[i]<<" "; std::cout<<std::endl;
    //Compute error
    std::cout<<"Error: ";
    for(int i = 0; i<10; i++){
      real truth = sqrt(2+i/10.0);
      std::cout<<abs(result[i]-truth)/truth<<" ";
    }
    std::cout<<std::endl;
  }
  return 0;
}
