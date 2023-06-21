ifeq ($(debug),1)
	CFLAGS += -O0 -g
	CXXFLAGS += -O0 -g
	CUFLAGS += -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -O2 -g -DNDEBUG
	CXXFLAGS += -O2 -g -DNDEBUG
	CUFLAGS += -O2 -g -DNDEBUG 
else ifeq ($(asan), 1)
	ASAN_FLAGS += -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g 
	ASAN_FLAGS += -O0
# 	ASAN_FLAGS += -O1
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

ifeq ($(mpisort), 1)
	INCLUDES += -I../mpi-sort/include
	CFLAGS += -L../mpi-sort/lib -lmpi-sort -DSFEM_ENABLE_MPI_SORT
endif

ifeq ($(openmp), 1)
	CFLAGS += -fopenmp
	CXXFLAGS += -fopenmp
endif

ifeq ($(parmetis), 1)
	metis = 1
	CFLAGS += -I$(PARMETIS_DIR)/include -DSFEM_ENABLE_PARMETIS
	CXXFLAGS += -I$(PARMETIS_DIR)/include -DSFEM_ENABLE_PARMETIS
	DEPS += -L$(PARMETIS_DIR)/lib -lparmetis
endif

ifeq ($(metis), 1)
	CFLAGS += -I$(METIS_DIR)/include -DSFEM_ENABLE_METIS
	CXXFLAGS += -I$(METIS_DIR)/include -DSFEM_ENABLE_METIS
	DEPS += -L$(METIS_DIR)/lib -lmetis
	DEPS += -L$(GKLIB_DIR)/lib -lGKlib
endif

# Folder structure
VPATH = pizzastack:resampling:mesh:operators:drivers:base:algebra:matrix:operators/tet10:operators/tet4:operators/tri3:operators/tri6:operators/cvfem:graphs:parametrize:operators/phase_field_for_fracture:operators/kernels
INCLUDES += -Ipizzastack -Iresampling -Imesh -Ioperators -Ibase -Ialgebra -Imatrix -Ioperators/tet10 -Ioperators/tet4 -Ioperators/tri3 -Ioperators/tri6 -Ioperators/cvfem -Igraphs -Iparametrize -Ioperators/phase_field_for_fracture  -Ioperators/kernels


CFLAGS += -pedantic -Wextra
# CFLAGS += -std=c99 

CXXFLAGS += -std=c++11
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC
INTERNAL_CXXFLAGS += -fno-exceptions -fno-rtti

CUFLAGS += --compiler-options "-fPIC $(CXXFLAGS)" -std=c++14 -arch=sm_60  #-arch=native 

# CUFLAGS += --compiler-options -fPIC -O0 -g -std=c++17

INCLUDES += -I$(PWD) -I$(PWD)/.. -I$(PWD)/../matrix.io 

# Assemble systems
GOALS = assemble assemble3 assemble4 neohookean_assemble stokes 

# Mesh manipulation
GOALS += partition select_submesh refine skin select_surf volumes sfc
GOALS += mesh_p1_to_p2

# FE post-process
GOALS += cgrad cshear cstrain cprincipal_strains cprincipal_stresses cauchy_stress vonmises
GOALS += wss surface_outflux integrate_divergence cdiv lform_surface_outflux
GOALS += projection_p0_to_p1 surface_projection grad_and_project

# BLAS
GOALS += axpy

# Algebra post process
GOALS += condense_matrix condense_vector idx_to_indicator remap_vector sgather smask set_diff set_union soverride

# Resampling
GOALS += pizzastack_to_mesh

# Application of operators
GOALS += divergence lapl lumped_mass_inv lumped_boundary_mass_inv u_dot_grad_q

# Array utilities
GOALS += soa_to_aos roi

# CVFEM
GOALS += cvfem_assemble

# Graph analysis
GOALS += assemble_adjaciency_matrix

ifeq ($(metis), 1)
	GOALS += partition_mesh_based_on_operator
endif

DEPS += -L$(PWD)/../matrix.io/ -lmatrix.io -lstdc++

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
	spmv.o \
	read_mesh.o  \
	mass.o \
	boundary_mass.o \
	dirichlet.o \
	div.o \
	strain.o \
	principal_strains.o \
	neohookean_principal_stresses.o \
	neumann.o \
	sfem_mesh.o \
	sfem_mesh_write.o \
	mesh_aura.o \
	isotropic_phasefield_for_fracture.o \
	tet10_laplacian.o \
	adj_table.o \
	laplacian.o \
	trishell3_l2_projection_p0_p1.o \
	trishell6_l2_projection_p1_p2.o \
	surface_l2_projection.o \
	grad_p1.o 

OBJS += tri3_laplacian.o

# Tet4
OBJS += tet4_div.o \
	tet4_mass.o \
	tet4_l2_projection_p0_p1.o \
	trishell3_l2_projection_p0_p1.o

# Tet10
OBJS += tet10_grad.o \
	tet10_div.o \
	tet10_mass.o \
	tet10_l2_projection_p1_p2.o 


# CVFEM
OBJS += cvfem_tri3_diffusion.o 

# Graphs
ifeq ($(metis), 1)
	OBJS += sfem_metis.o
endif

ifeq ($(cuda), 1)
# 	CUDA_OBJS = tet4_cuda_laplacian.o
# 	CUDA_OBJS = tet4_cuda_laplacian_2.o
	CUDA_OBJS = tet4_cuda_laplacian_3.o
	CUDA_OBJS += tet4_cuda_phase_field_for_fracture.o
	
	CUDA_OBJS += cuda_crs.o
	DEPS += -L/opt/cuda/lib64 -lcudart
	DEPS += -lnvToolsExt

	OBJS += $(CUDA_OBJS)
else
	SERIAL_OBJS = tet4_laplacian.o
	OBJS += $(SERIAL_OBJS)
endif

OBJS += neohookean.o

# SIMD_OBJS = simd_neohookean.o
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

cauchy_stress : cauchy_stress.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

vonmises : vonmises.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

surface_outflux : surface_outflux.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lform_surface_outflux : lform_surface_outflux.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

integrate_divergence : integrate_divergence.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble : assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble3 : assemble3.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble4 : assemble4.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

stokes : stokes.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

neohookean_assemble : neohookean_assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble_adjaciency_matrix: assemble_adjaciency_matrix.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \
	
# CVFEM
cvfem_assemble : cvfem_assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

partition : partition.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

partition_mesh_based_on_operator : partition_mesh_based_on_operator.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

sfc : sfc.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

select_submesh : select_submesh.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

refine : refine.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

skin : skin.c extract_surface_graph.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

mesh_p1_to_p2 : mesh_p1_to_p2.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

volumes : drivers/volumes.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

select_surf : drivers/select_surf.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

extract_surface_graph.o : extract_surface_graph.cpp
	$(CXX) $(CXXFLAGS) $(INTERNAL_CXXFLAGS) $(INCLUDES) -c $<

pizzastack_to_mesh: pizzastack_to_mesh.c grid.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

axpy : axpy.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

set_diff : drivers/set_diff.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

set_union : drivers/set_union.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

sgather : sgather.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

soa_to_aos : soa_to_aos.o
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

soverride : drivers/soverride.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

divergence : drivers/divergence.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

u_dot_grad_q : drivers/u_dot_grad_q.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lapl : drivers/lapl.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lumped_mass_inv : drivers/lumped_mass_inv.c mass.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lumped_boundary_mass_inv : drivers/lumped_boundary_mass_inv.c mass.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

projection_p0_to_p1 : drivers/projection_p0_to_p1.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

grad_and_project : drivers/grad_and_project.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

surface_projection : drivers/surface_projection.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

smask : drivers/smask.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cgrad : drivers/cgrad.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cshear : drivers/cshear.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cstrain : drivers/cstrain.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cprincipal_strains : drivers/cprincipal_strains.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cprincipal_stresses : drivers/cprincipal_stresses.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

cdiv : drivers/cdiv.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \


wss : drivers/wss.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

roi : drivers/roi.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

div.o : operators/div.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

isolver_sfem.dylib : isolver_sfem_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)  

isolver_sfem_plugin.o : plugin/isolver_sfem_plugin.c 
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

sortreduce.o : sortreduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

argsort.o : argsort.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

principal_strains.o : principal_strains.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

neohookean_principal_stresses.o : neohookean_principal_stresses.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

%.o : %.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

%.o : %.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o *.a $(GOALS); rm -r *.dSYM  


.SUFFIXES:

.PHONY: clean all
