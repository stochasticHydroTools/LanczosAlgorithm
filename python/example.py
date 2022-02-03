#Raul P. Pelaez 2022. Usage example for the Lanczos solver's Python interface
#A class that computes the dot product of a matrix, M,  and an arbitrary vector, v, must be written to use the solver (see DiagonalMatrix below).
#The class must inherit from Lanczos.MatrixDot and provide a function called "dot" that given an arbitrary vector, v, returns the product Mv.
#When provided with an instance of this class, the function "solve" in Lanczos.Solver will return the product sqrt(M)v
#IMPORTANT: Remeber to use the same numerical precision here and when compiling the library (see the Makefile for more info)
#Try help(Lanczos)

import Lanczos 
import numpy as np

#Lanczos provides the precision it was compiled in via this function.
precision = np.float32 if Lanczos.getPrecision() else np.float64;


# A simple class that computes the product of a diagonal matrix (2*I) by the input vector
class DiagonalMatrix(Lanczos.MatrixDot):

    def dot(self, v):
#        size=v.size()
        Mv = v*2.0
        return Mv

#Create the solver
solver = Lanczos.Solver()

#Let us compute the result of sqrt(2*I)*v, where v=[1,1,1....1] and I the identity matrix
#The result vector will be filled with sqrt(2)
size = 1000000
result = np.zeros(size, precision);
v = np.ones(size, precision);

dotProduct = DiagonalMatrix()
#The solve function fills the result vector with sqrt(M)*v and returns the residual at the last iteration.
numIter = 2
error = solver.runIterations(dotProduct, result,v, numIter, size)

print("Done after "+ str(numIter) + " iterations with a residual of " + str(error)+ ".")
print("Result vector (should be filled with ~sqrt(2)="+str(np.sqrt(2))+"):")
print(result)
