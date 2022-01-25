/* Raul P. Pelaez 2022. Implementation for the trampoline lanczos class.
Simply defer the calls to the actual lanczos solver code.
*/
#include"lanczos_trampoline.h"
#include"../include/LanczosAlgorithm.h"

using namespace lanczos;

LanczosTrampoline::LanczosTrampoline(real tolerance):
  solver(std::make_shared<lanczos::Solver>(tolerance)){}

int LanczosTrampoline::solve(MatrixDot *dot, real* result, const real* v, int size){
  int numberIterations = solver->solve(dot, result, v, size);
  return numberIterations;
}
