#ifndef LANCZOS_MATRIX_DOT_H
#define LANCZOS_MATRIX_DOT_H
#include"defines.h"
namespace lanczos{
  
  struct MatrixDot{
    void setSize(int newsize){this->m_size = newsize;}
    virtual void dot(real* v, real*Mv) = 0;
  protected:
    int m_size;
  };
}
#endif
