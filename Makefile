

NVCC=nvcc
CXX=g++

#Uncomment for a GPU enabled library
#CUDA_ENABLED=-DCUDA_ENABLED

#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION


INCLUDEFLAGS=-Iinclude/
NVCCLDFLAGS= -lcublas
LDFLAGS= -llapacke -lcblas

LIBNAME=liblanczos.so


CXXFLAGS=-fPIC -w -O3 -g -std=c++14 $(INCLUDEFLAGS) $(DOUBLE_PRECISION) $(CUDA_ENABLED)
NVCCFLAGS=-ccbin=$(CXX) -Xcompiler "$(CXXFLAGS)" -std=c++14 -O3 $(INCLUDEFLAGS) $(DOUBLE_PRECISION) $(CUDA_ENABLED)

ifndef CUDA_ENABLED
COMPILER=$(CXX)
CXXFLAGS:=$(CXXFLAGS)  -xc++
else
COMPILER=$(NVCC) 
LDFLAGS:=$(LDFLAGS) $(NVCCLDFLAGS)
CXXFLAGS:=$(NVCCFLAGS) -I$(CUDA_ROOT)/include
endif

all: shared $(patsubst %.cu, %, $(wildcard *.cu)) $(patsubst %.cpp, %, $(wildcard *.cpp))

$(LIBNAME): $(wildcard include/*.cu)
	$(COMPILER) -DSHARED_LIBRARY_COMPILATION -shared $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


shared: $(LIBNAME)


%: %.cu Makefile
	$(COMPILER) $(CXXFLAGS) $<  -o $@ $(LDFLAGS)



#clean: $(patsubst include/%.cu, %.clean, $(wildcard include/*.cu))

#%.clean:
#rm -f $(@:.clean=.so)
clean:
	rm -rf include/*.o $(LIBNAME) example
