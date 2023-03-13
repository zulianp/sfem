ifeq ($(debug),1)
	CFLAGS += -O0 -g
	CXXFLAGS += -O0 -g
	CUFLAGS += -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -O2 -g -DNDEBUG
	CXXFLAGS += -O2 -g -DNDEBUG
	CUFLAGS += -O2 -g -DNDEBUG 
else ifeq ($(asan), 1)
	ASAN_FLAGS += -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O0
	CXXFLAGS += $(ASAN_FLAGS)
	CFLAGS += $(ASAN_FLAGS)
else
	CFLAGS += -Ofast -DNDEBUG
	CXXFLAGS += -Ofast -DNDEBUG
	CUFLAGS += -O3 -DNDEBUG 
endif

ifeq ($(avx512sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX512_SORT -Iexternal/x86-simd-sort/src -march=native
	CFLAGS += -march=native
endif

ifeq ($(avx2sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX2_SORT -Iexternal -march=core-avx2
	CFLAGS += -march=core-avx2
endif

# Folder structure
VPATH = pizzastack:resampling:mesh:operators:drivers:base:algebra
INCLUDES += -Ipizzastack -Iresampling -Imesh -Ioperators -Ibase -Ialgebra


CFLAGS += -pedantic -Wextra
# CFLAGS += -std=c99 

CXXFLAGS += -std=c++11
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC
INTERNAL_CXXFLAGS += -fno-exceptions -fno-rtti 
CUFLAGS += --compiler-options -fPIC -std=c++17 -arch=native 

# CUFLAGS += --compiler-options -fPIC -O0 -g -std=c++17

INCLUDES += -I$(PWD) -I$(PWD)/.. -I$(PWD)/../matrix.io 

# Assemble systems
GOALS = assemble assemble3 assemble4 

# Mesh manipulation
GOALS += partition select_submesh refine skin surf_split

# FE post-process
GOALS += cgrad cshear projection_p0_to_p1 wss lumped_mass_inv

# BLAS
GOALS += axpy

# Algebra post process
GOALS += condense_matrix condense_vector idx_to_indicator remap_vector sgather smask set_diff soverride

# Resampling
GOALS += pizzastack_to_mesh

# Application of operators
GOALS += divergence

DEPS = -L$(PWD)/../matrix.io/ -lmatrix.io -lstdc++

LDFLAGS += $(DEPS) -lm

MPICC ?= mpicc
CXX ?= c++
MPICXX ?= mpicxx
AR ?= ar
NVCC ?= nvcc

all : $(GOALS)

OBJS = \
	sortreduce.o \
	crs_graph.o \
	sortreduce.o \
	argsort.o \
	read_mesh.o  \
	mass.o \
	dirichlet.o \
	neumann.o \
	sfem_mesh.o \
	sfem_mesh_write.o \
	isotropic_phasefield_for_fracture.o


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

plugins: isolver_sfem.dylib

libsfem.a : $(OBJS)
	ar rcs $@ $^

YAML_CPP_INCLUDES = -I$(INSTALL_DIR)/yaml-cpp/include/ 
YAML_CPP_LIBRARIES = $(INSTALL_DIR)/yaml-cpp/lib/libyaml-cpp.a
ISOLVER_INCLUDES = -I../isolver/interfaces/lsolve -I../isolver/plugin/lsolve -I../isolver/plugin/
ssolve : drivers/ssolve.cpp isolver_lsolve_frontend.o libsfem.a
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) $(YAML_CPP_INCLUDES) $(YAML_CPP_LIBRARIES) -o $@ $^ $(LDFLAGS) ; \

isolver_lsolve_frontend.o : ../isolver/plugin/lsolve/isolver_lsolve_frontend.cpp
	$(MPICXX) $(CXXFLAGS) $(INTERNAL_CXXFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) -c $<

assemble : assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble3 : assemble3.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble4 : assemble4.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

partition : partition.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

select_submesh : select_submesh.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

refine : refine.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

skin : skin.c extract_surface_graph.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

surf_split : drivers/surf_split.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

extract_surface_graph.o : extract_surface_graph.cpp
	$(CXX) $(CXXFLAGS) $(INTERNAL_CXXFLAGS) $(INCLUDES) -c $<

pizzastack_to_mesh: resampling/pizzastack_to_mesh.c pizzastack/grid.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

axpy : algebra/axpy.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

set_diff : drivers/set_diff.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

sgather : sgather.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

soverride : drivers/soverride.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

divergence : drivers/divergence.c div.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lumped_mass_inv : drivers/lumped_mass_inv.c mass.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

projection_p0_to_p1 : drivers/projection_p0_to_p1.c div.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

smask : drivers/smask.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cgrad : drivers/cgrad.c grad_p1.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cshear : drivers/cshear.c grad_p1.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

wss : drivers/wss.c grad_p1.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

div.o : operators/div.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

grad_p1.o : operators/grad_p1.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<
	
isolver_sfem.dylib : isolver_sfem_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)  

isolver_sfem_plugin.o : plugin/isolver_sfem_plugin.c 
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

sortreduce.o: sortreduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

argsort.o: argsort.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

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
