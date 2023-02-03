ifeq ($(debug),1)
	CFLAGS += -O0 -g
	CXXFLAGS += -O0 -g
	CUFLAGS += -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -O2 -g -DNDEBUG
	CXXFLAGS += -O2 -g -DNDEBUG
	CUFLAGS += -O2 -g -DNDEBUG 
else
	CFLAGS += -Ofast -DNDEBUG
	CXXFLAGS += -Ofast -DNDEBUG
	CUFLAGS += -O3 -DNDEBUG 
endif

ifeq ($(avx2sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX2_SORT -Iexternal
endif

CFLAGS += -pedantic 
# CFLAGS += -std=c99 

CXXFLAGS += -std=c++11
CXXFLAGS += -fno-exceptions -fno-rtti -static
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC
CUFLAGS += --compiler-options -fPIC -std=c++17

# CUFLAGS += --compiler-options -fPIC -O0 -g -std=c++17

INCLUDES += -I$(PWD) -I$(PWD)/../matrix.io

GOALS = assemble assemble3 condense_matrix condense_vector idx_to_indicator remap_vector
DEPS = -L$(PWD)/../matrix.io/ -lmatrix.io -lstdc++

LDFLAGS += $(DEPS) -lm

MPICC ?= mpicc
CXX ?= c++
AR ?= ar
NVCC ?= nvcc

all : $(GOALS)

OBJS = \
	sortreduce.o \
	crs_graph.o \
	sortreduce.o \
	read_mesh.o  \
	mass.o \
	dirichlet.o \
	neumann.o

ifeq ($(cuda), 1)
	CUDA_OBJS = cuda_laplacian.o
	DEPS += -L/opt/cuda/lib64 -lcudart

	OBJS += $(CUDA_OBJS)
else
	SERIAL_OBJS = laplacian.o
# 	SERIAL_OBJS += neohookean.o 
	OBJS += $(SERIAL_OBJS)
endif

SIMD_OBJS = simd_neohookean.o
# SIMD_OBJS +=  simd_laplacian.o 

OBJS += $(SIMD_OBJS)

plugins: utopia_sfem.dylib

libsfem.a : $(OBJS)
	ar rcs $@ $^

assemble : assemble.o libsfem.a
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

assemble3 : assemble3.o libsfem.a
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

utopia_sfem.dylib : utopia_sfem_plugin.o  libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)  

utopia_sfem_plugin.o : plugin/utopia_sfem_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

sortreduce.o: sortreduce.cpp
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

%.o : %.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o *.a $(GOALS)

.SUFFIXES:

.PHONY: clean all
