

NVCC=nvcc
CXX=g++
#To compile the python wrapper
PYTHON3=python3
#Pybind is cloned as a submodule to this location
PYBIND_INCLUDE=extern/pybind11/include

#Uncomment for a GPU enabled library
#CUDA_ENABLED=-DCUDA_ENABLED

#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION


INCLUDEFLAGS=-Iinclude/
NVCCLDFLAGS= -lcublas
LDFLAGS= -llapacke -lcblas

LIBNAME=liblanczos.so
PYTHON_MODULE_NAME=Lanczos


CXXFLAGS=-fPIC -w -O3 -g -std=c++14 $(INCLUDEFLAGS) $(DOUBLE_PRECISION)
NVCCFLAGS=-ccbin=$(CXX) -Xcompiler "$(CXXFLAGS)" -std=c++14 -O3 $(INCLUDEFLAGS) $(DOUBLE_PRECISION) $(CUDA_ENABLED)

PYTHON_LIBRARY_NAME=python/$(PYTHON_MODULE_NAME)$(shell $(PYTHON3)-config --extension-suffix)

ifndef CUDA_ENABLED
COMPILER=$(CXX)
CXXFLAGS_BOTH:=$(CXXFLAGS)  -xc++
LDFLAGS_BOTH:=$(LDFLAGS)
else
COMPILER=$(NVCC) 
LDFLAGS_BOTH:=$(LDFLAGS) $(NVCCLDFLAGS)
CXXFLAGS_BOTH:=$(NVCCFLAGS) -I$(CUDA_ROOT)/include $(CUDA_ENABLED)
endif

all: shared  python $(patsubst %.cu, %, $(wildcard *.cu)) $(patsubst %.cpp, %, $(wildcard *.cpp)) Makefile

$(LIBNAME): $(wildcard include/*.cu)
	$(COMPILER) -DSHARED_LIBRARY_COMPILATION -shared $(CXXFLAGS_BOTH) $^ -o $@ $(LDFLAGS_BOTH)


shared: $(LIBNAME) Makefile


python: $(PYTHON_LIBRARY_NAME) Makefile
#	-DLANCZOS_PYTHON_NAME=$(PYTHON_MODULE_NAME)

$(PYTHON_LIBRARY_NAME): python/python_wrapper.cpp python/lanczos_trampoline.o
	$(CXX) $(CXXFLAGS) `$(PYTHON3)-config --includes` -I $(PYBIND_INCLUDE) -shared  $^ -o $@ $(LDFLAGS)

python/lanczos_trampoline.o: python/lanczos_trampoline.cpp Makefile
	$(CXX) $(CXXFLAGS) -c $<  -o $@

%: %.cu Makefile
	$(COMPILER) $(CXXFLAGS_BOTH) $<  -o $@ $(LDFLAGS_BOTH)



#clean: $(patsubst include/%.cu, %.clean, $(wildcard include/*.cu))

#%.clean:
#rm -f $(@:.clean=.so)
clean:
	rm -rf include/*.o python/*.o $(LIBNAME) example $(PYTHON_LIBRARY_NAME)
