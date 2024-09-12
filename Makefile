SHELL := /bin/bash

# LDFLAGS=`mpic++ -showme:link`

MARCH ?= native
VECTOR_SIZE ?= 512

GPU_ARCH ?= sm_75
DISABLE_CUDA ?= 0

ARM ?= 0  # 1 for ARM architecture

VECTOR_WIDTH_OPT = -mprefer-vector-width

ifeq ($(ARM), 1)
	VECTOR_WIDTH_OPT = -msve-vector-bits
	# MPICXX = mpicxx
	# MPICC = mpicc
endif

$(info $$DISABLE_CUDA is [${DISABLE_CUDA}])
$(info $$GPU_ARCH is [${GPU_ARCH}])


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
# 	DEPS += -static-libsan
# 	DEPS += -static
else
	CFLAGS += -Ofast -DNDEBUG -march=${MARCH} -fno-signed-zeros -fno-trapping-math -fassociative-math ${VECTOR_WIDTH_OPT}=${VECTOR_SIZE}  -fPIC  -no-pie 
	CXXFLAGS += -Ofast -DNDEBUG -march=${MARCH} -fno-signed-zeros -fno-trapping-math -fassociative-math ${VECTOR_WIDTH_OPT}=${VECTOR_SIZE} -fPIC  -no-pie 
	CUFLAGS += -O3 -DNDEBUG
endif

ifeq ($(avx512sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX512_SORT -Iexternal/x86-simd-sort/src -march=native -DSFEM_ENABLE_EXPLICIT_VECTORIZATION
	CFLAGS += -march=native -DSFEM_ENABLE_EXPLICIT_VECTORIZATION
endif

ifeq ($(avx2sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX2_SORT -Iexternal -march=core-avx2 -DSFEM_ENABLE_EXPLICIT_VECTORIZATION
	CFLAGS += -march=core-avx2 -DSFEM_ENABLE_EXPLICIT_VECTORIZATION
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

CFLAGS += -DSFEM_MEM_DIAGNOSTICS

# Folder structure
VPATH = pizzastack:resampling:resampling/tet10:mesh:operators:operators/cuda:drivers:drivers/cuda:drivers/metis:base:algebra:matrix:operators/tet10:operators/tet4:operators/macro_tet4:operators/tri3:operators/macro_tri3:operators/trishell3:operators/tri6:operators/beam2:operators/cvfem:graphs:parametrize:operators/phase_field_for_fracture:operators/kernels:operators/navier_stokes:solver:operators/cvfem_tet4:operators/cvfem_tri3:operators/cvfem_quad4:examples:algebra/cuda:frontend:frontend/cuda:operators/hierarchical
INCLUDES += -Ipizzastack -Iresampling -Iresampling/tet10 -Imesh -Ioperators -Ibase -Ialgebra -Imatrix -Ioperators/tet10 -Ioperators/tet4 -Ioperators/macro_tet4 -Ioperators/tri3 -Ioperators/macro_tri3 -Ioperators/trishell3 -Ioperators/tri6 -Ioperators/beam2 -Ioperators/cvfem -Igraphs -Iparametrize -Ioperators/phase_field_for_fracture  -Ioperators/kernels -Ioperators/navier_stokes -Isolver -Ioperators/cvfem_tet4 -Ioperators/cvfem_tri3 -Ioperators/cvfem_quad4 -Ialgebra/cuda -Ifrontend -Ifrontend/cuda  -Ialgebra/cuda -Ioperators/hierarchical


CFLAGS += -pedantic -Wextra
CFLAGS += -fPIC
# CFLAGS += -std=c99

CXXFLAGS += -std=c++14
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC
INTERNAL_CXXFLAGS += -fno-exceptions -fno-rtti

CUFLAGS += --compiler-options "-fPIC $(CXXFLAGS)" -std=c++14 -arch=sm_60 -Xptxas=-O3,-v -use_fast_math
#-arch=native
# CUFLAGS += --compiler-options "-fPIC $(CXXFLAGS)" -std=c++14 -arch=sm_86  #-arch=native

# CUFLAGS += --compiler-options -fPIC -O0 -g -std=c++17

INCLUDES += -I$(PWD) -I$(PWD)/.. -I$(PWD)/../matrix.io

# Assemble systems
GOALS = assemble assemble3 assemble4 neohookean_assemble stokes stokes_check linear_elasticity_assemble

# Mesh manipulation
GOALS += partition select_submesh refine skin extract_sharp_edges extrude wedge6_to_tet4 mesh_self_intersect select_surf volumes sfc
GOALS += mesh_p1_to_p2 create_dual_graph create_element_adjaciency_table create_surface_from_element_adjaciency_table
GOALS += mesh_to_blocks

# FE post-process
GOALS += cgrad cshear cstrain cprincipal_strains cprincipal_stresses cauchy_stress vonmises
GOALS += wss surface_outflux integrate_divergence cdiv lform_surface_outflux
GOALS += projection_p0_to_p1 surface_projection grad_and_project

# BLAS
GOALS += axpy spmv

# Algebra post process
GOALS += condense_matrix condense_vector idx_to_indicator remap_vector sgather smask set_diff set_union soverride

# Resampling
GOALS += pizzastack_to_mesh

# Application of operators
GOALS += divergence lapl lumped_mass_inv lumped_boundary_mass_inv u_dot_grad_q
GOALS += crs_apply_dirichlet

# Array utilities
GOALS += soa_to_aos aos_to_soa roi unique

# CVFEM
GOALS += cvfem_assemble run_convection_diffusion

# Graph analysis
GOALS += assemble_adjaciency_matrix

# Contact
GOALS += gap_from_sdf geometry_aware_gap_from_sdf mesh_to_sdf grid_to_mesh

GOALS += bgs

GOALS += taylor_hood_navier_stokes heat_equation run_poisson

ifeq ($(metis), 1)
	GOALS += partition_mesh_based_on_operator
endif

DEPS += -L$(PWD)/../matrix.io/ -lmatrix.io -lstdc++

LDFLAGS += $(DEPS) -lm -fPIC

MPICC ?= mpicc
CXX ?= c++
CC ?= cc
MPICXX ?= mpicxx
AR ?= ar
NVCC ?= nvcc

export MPICH_CXX=${CXX}
export OMPI_CXX=${CXX}

export MPICH_CC=${CC}
export OMPI_CC=${CC}


all :    $(GOALS)

pre-build:
	mpicc --version



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
	neumann.o \
	sfem_mesh.o \
	sfem_mesh_write.o \
	mesh_aura.o \
	adj_table.o \
	laplacian.o \
	trishell3_l2_projection_p0_p1.o \
	trishell6_l2_projection_p1_p2.o \
	surface_l2_projection.o \
	linear_elasticity.o \
	stokes_mini.o \
	phase_field_for_fracture.o  \
	navier_stokes.o \
	boundary_condition.o \
	boundary_condition_io.o \
	constrained_gs.o \
	sfem_logger.o \
	extract_sharp_features.o \
	mesh_utils.o

# Tri3
OBJS += tri3_stokes_mini.o \
		tri3_mass.o \
		tri3_phase_field_for_fracture.o \
		tri3_linear_elasticity.o \
		tri3_laplacian.o



# Macro Tri3
OBJS += macro_tri3_laplacian.o

# Macro Tet4
OBJS += macro_tet4_laplacian.o
OBJS += macro_tet4_linear_elasticity.o
# This is bugged
# OBJS += macro_tet4_laplacian_simd.o

# TriShell3
OBJS += trishell3_mass.o

# Tri6
OBJS += tri6_mass.o tri6_laplacian.o tri6_navier_stokes.o

# Tet4
OBJS += tet4_div.o \
	tet4_mass.o \
	tet4_l2_projection_p0_p1.o \
	tet4_linear_elasticity.o \
	tet4_phase_field_for_fracture.o \
	tet4_stokes_mini.o \
	trishell3_l2_projection_p0_p1.o \
	tet4_grad.o \
	tet4_isotropic_phasefield_for_fracture.o \
	tet4_strain.o \
	tet4_principal_strains.o \
	tet4_neohookean_principal_stresses.o \
	tet4_neohookean.o

# Beam2
OBJS += beam2_mass.o

# Tet10
OBJS += tet10_grad.o \
	tet10_div.o \
	tet10_mass.o \
	tet10_laplacian.o \
	tet10_l2_projection_p1_p2.o \
	tet10_navier_stokes.o \
	tet10_linear_elasticity.o 
	
# Multilevel
OBJS += sfem_prolongation_restriction.o

# Resampling
OBJS += sfem_resample_gap.o sfem_resample_field.o sfem_resample_field_v2.o sfem_resample_field_V8.o sfem_resample_field_V4.o
OBJS += tet10_weno.o tet10_resample_field.o tet10_resample_field_V2.o

# CVFEM
OBJS += cvfem_tri3_diffusion.o cvfem_tet4_convection.o cvfem_tri3_convection.o cvfem_quad4_convection.o cvfem_quad4_laplacian.o
OBJS += cvfem_operators.o
# Graphs
ifeq ($(metis), 1)
	OBJS += sfem_metis.o
endif

ifeq ($(cuda), 1)
# 	CUDA_OBJS = tet4_cuda_laplacian.o
# 	CUDA_OBJS = tet4_cuda_laplacian_2.o
# 	CUDA_OBJS = tet4_cuda_laplacian_3.o # FIXME
	CUDA_OBJS = tet4_laplacian.o
	CUDA_OBJS += tet4_cuda_phase_field_for_fracture.o
	CUDA_OBJS += tet4_laplacian_incore_cuda.o
	CUDA_OBJS += tet10_laplacian_incore_cuda.o
	CUDA_OBJS += macro_tet4_laplacian_incore_cuda.o
	CUDA_OBJS += sfem_cuda_blas.o
	CUDA_OBJS += boundary_condition_incore_cuda.o
	CUDA_OBJS += tet4_linear_elasticity_incore_cuda.o
	CUDA_OBJS += tet10_linear_elasticity_incore_cuda.o
	CUDA_OBJS += macro_tet4_linear_elasticity_incore_cuda.o

	OBJS += linear_elasticity_incore_cuda.o
	OBJS += laplacian_incore_cuda.o
	OBJS += sfem_Function_incore_cuda.o

	INCLUDES += -Ioperators/cuda

	CUDA_OBJS += cuda_crs.o
	DEPS += -L/opt/cuda/lib64 -lcudart -lcusparse -lcusolver -lcublas
	DEPS += -lnvToolsExt
	CFLAGS += -I/opt/cuda/include -DSFEM_ENABLE_CUDA
	CXXFLAGS += -DSFEM_ENABLE_CUDA

	OBJS += $(CUDA_OBJS)
else
	SERIAL_OBJS = tet4_laplacian.o
	OBJS += $(SERIAL_OBJS)
endif

OBJS += $(SIMD_OBJS)

OBJS += sfem_Function.o 

plugins: isolver_sfem.dylib franetg_plugin.dylib hyperelasticity_plugin.dylib nse_plugin.dylib stokes_plugin.dylib

libsfem.a : $(OBJS)
	ar rcs $@ $^

YAML_CPP_INCLUDES = -I$(INSTALL_DIR)/yaml-cpp/include/
YAML_CPP_LIBRARIES = $(INSTALL_DIR)/yaml-cpp/lib/libyaml-cpp.a
ISOLVER_INCLUDES = -I../isolver/interfaces/lsolve -I../isolver/plugin/lsolve -I../isolver/plugin/

ssolve : drivers/ssolve.cpp isolver_lsolve_frontend.o libsfem.a
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) $(YAML_CPP_INCLUDES) $(YAML_CPP_LIBRARIES) -o $@ $^ $(LDFLAGS) ; \

taylor_hood_navier_stokes: drivers/taylor_hood_navier_stokes.c isolver_lsolve_frontend.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) -o $@ $^ $(LDFLAGS) ; \

heat_equation: drivers/heat_equation.c isolver_lsolve_frontend.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) -o $@ $^ $(LDFLAGS) ; \

bgs : bgs.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

isolver_lsolve_frontend.o : ../isolver/plugin/lsolve/isolver_lsolve_frontend.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(ISOLVER_INCLUDES) -c $<

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

macro_element_apply : macro_element_apply.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

stokes : stokes.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

stokes_check : stokes_check.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

crs_apply_dirichlet : crs_apply_dirichlet.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

neohookean_assemble : neohookean_assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

linear_elasticity_assemble : linear_elasticity_assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

assemble_adjaciency_matrix: assemble_adjaciency_matrix.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

# CVFEM
cvfem_assemble : cvfem_assemble.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

run_convection_diffusion : run_convection_diffusion.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

run_poisson : run_poisson.o libsfem.a 
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

run_poisson.o : run_poisson.cpp sfem_cg.hpp sfem_bcgs.hpp
	$(MPICXX) examples/run_poisson.cpp -c $(CXXFLAGS) $(INCLUDES) 

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

extract_sharp_edges : extract_sharp_edges.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

mesh_self_intersect : mesh_self_intersect.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

extrude : extrude.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

wedge6_to_tet4 : wedge6_to_tet4.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

create_surface_from_element_adjaciency_table : create_surface_from_element_adjaciency_table.o libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

gap_from_sdf : gap_from_sdf.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

# Resampling


ifeq ($(DISABLE_CUDA), 1)
    CUDA_LIBS_PATH = 
    CUDART_LINK =
    SFEM_CUDA_A = 
    LINK_SFEM_CUDA =
else
    CUDA_LIBS_PATH = -L/opt/nvidia/hpc_sdk/Linux_aarch64/23.9/cuda/12.2/targets/sbsa-linux/lib/
    CUDA_LIBS_PATH = -L/usr/local/cuda-12.3/lib64 
    CUDART_LINK =  -lcudart
    SFEM_CUDA_A = ${PWD}/resampling/cuda/libsfem_resample_field_cuda.a
    LINK_SFEM_CUDA = -L${PWD}/resampling/cuda -lsfem_resample_field_cuda
endif

$(info $$SFEM_CUDA_A is [${SFEM_CUDA_A}])

grid_to_mesh: grid_to_mesh.c libsfem.a ${SFEM_CUDA_A} ${PWD}/resampling/quadratures_rule.h
	$(MPICC) -o $@ drivers/grid_to_mesh.c libsfem.a $(CFLAGS) $(INCLUDES)  $(LDFLAGS) ${LINK_SFEM_CUDA} ${CUDA_LIBS_PATH} ${CUDART_LINK}  ${SFEM_CUDA_A}

${PWD}/resampling/cuda/libsfem_resample_field_cuda.a: ${PWD}/resampling/cuda/sfem_resample_field_cuda.cu ${PWD}/resampling/cuda/tet10_resample_field.cu ${PWD}/resampling/cuda/quadratures_rule_cuda.h ${PWD}/resampling/cuda/tet10_weno_cuda.cu ${PWD}/resampling/cuda/tet10_weno_cuda.cuh
	${MAKE} -C ${PWD}/resampling/cuda GPU_ARCH=${GPU_ARCH} ARM=${ARM}

# else
# # CPU version
# grid_to_mesh: grid_to_mesh.c libsfem.a
# 	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

# endif

geometry_aware_gap_from_sdf : geometry_aware_gap_from_sdf.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

mesh_to_sdf : mesh_to_sdf.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

mesh_p1_to_p2 : mesh_p1_to_p2.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

mesh_to_blocks : mesh_to_blocks.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

volumes : drivers/volumes.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

select_surf : drivers/select_surf.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

create_dual_graph : drivers/create_dual_graph.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES)  -o $@ $^ $(LDFLAGS) ; \

create_element_adjaciency_table : drivers/create_element_adjaciency_table.c libsfem.a
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

aos_to_soa : aos_to_soa.o
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

unique:  drivers/unique.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

div.o : operators/div.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

isolver_sfem.dylib : isolver_sfem_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)

isolver_sfem_plugin.o : plugin/isolver_sfem_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

franetg_plugin.dylib : franetg_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)

franetg_plugin.o : plugin/franetg_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

nse_plugin.dylib : nse_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)

nse_plugin.o : plugin/nse_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

stokes_plugin.dylib : stokes_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)

stokes_plugin.o : plugin/stokes_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

hyperelasticity_plugin.dylib : hyperelasticity_plugin.o libsfem.a
	$(MPICC) -shared -o $@ $^ $(LDFLAGS)

hyperelasticity_plugin.o : plugin/hyperelasticity_plugin.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

sfem_Function.o : sfem_Function.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve -c $<

sfem_Function_incore_cuda.o : sfem_Function_incore_cuda.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(CXXFLAGS) -I../isolver/interfaces/nlsolve -c $<

sortreduce.o : sortreduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

argsort.o : argsort.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

tet4_principal_strains.o : tet4_principal_strains.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

tet4_neohookean_principal_stresses.o : tet4_neohookean_principal_stresses.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTERNAL_CXXFLAGS) -c $<

cuspmv : drivers/cuda/cuda_do_spmv.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

lapl_matrix_free : drivers/cuda/lapl_matrix_free.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

linear_elasticity_matrix_free : drivers/cuda/linear_elasticity_matrix_free.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

run_poisson_cuda : examples/run_poisson_cuda.cpp libsfem.a
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

steady_state_sim  : drivers/steady_state_sim.cpp libsfem.a
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -I../isolver/interfaces/nlsolve  -o $@ $^ $(LDFLAGS) ; \

spmv : drivers/cuda/do_spmv.c libsfem.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) ; \

%.o : %.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

%.o : %.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o *.a *.dylib *.so $(GOALS); rm -r *.dSYM


.SUFFIXES:

.PHONY: clean all
