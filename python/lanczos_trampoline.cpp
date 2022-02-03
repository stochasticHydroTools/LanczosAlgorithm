/* Raul P. Pelaez 2022. Implementation for the trampoline lanczos class.
Simply defer the calls to the actual lanczos solver code.
*/
#include"lanczos_trampoline.h"
#include"../include/LanczosAlgorithm.h"

using namespace lanczos;

LanczosTrampoline::LanczosTrampoline():
  solver(std::make_shared<lanczos::Solver>()){}

int LanczosTrampoline::run(MatrixDot *dot, real* result, const real* v, real tolerance, int size){
  int numberIterations = solver->run(dot, result, v, tolerance, size);
  return numberIterations;
}

real LanczosTrampoline::runIterations(MatrixDot *dot, real* result, const real* v, int numberIterations, int size){
  real residual = solver->runIterations(dot, result, v, numberIterations, size);
  return residual;
}
