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
  LanczosTrampoline(lanczos::real tolerance);

  int solve(lanczos::MatrixDot *dot, lanczos::real* result, const lanczos::real* v, int size);
  
};

#endif
