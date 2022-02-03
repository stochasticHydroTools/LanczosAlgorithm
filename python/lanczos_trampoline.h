/* Raul P. Pelaez 2022. A class that bridges with lanczos::Solver without actually including any of its code.
This allows to compile the lanczos::Solver library separatedly from the python wrapper.

 */
#ifndef LANCZOS_TRAMPOLINE_H
#define LANCZOS_TRAMPOLINE_H
#include<memory>
#include"../include/utils/MatrixDot.h"

namespace lanczos{
  class Solver;
}
  

class LanczosTrampoline{
  std::shared_ptr<lanczos::Solver> solver;
public:
  LanczosTrampoline();

  int run(lanczos::MatrixDot *dot, lanczos::real* result, const lanczos::real* v, lanczos::real tolerance, int size);
  lanczos::real runIterations(lanczos::MatrixDot *dot, lanczos::real* result, const lanczos::real* v, int numberIterations, int size);
  
};

#endif
