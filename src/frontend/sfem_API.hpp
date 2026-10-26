#ifndef SFEM_API_HPP
#define SFEM_API_HPP

// An umbrella over the library. What it used to define lives with the module that owns it:
//   algebra:  sfem_blas_factory.hpp, sfem_operator_factory.hpp, sfem_device.hpp, and the solver factories
//             in each solver class header (sfem_cg.hpp, sfem_mprgp.hpp, ...)
//   frontend: sfem_Transfer.hpp, sfem_LinearOperatorFactory.hpp, and the condition factories in
//             sfem_DirichletConditions.hpp, sfem_NeumannConditions.hpp, sfem_ContactConditions.hpp
//   smesh:    derefine_elements (smesh_semistructured.hpp)
// It still includes everything it used to, so existing includers build; new code includes what it uses.
#include "sfem_LinearOperatorFactory.hpp"
#include "sfem_Transfer.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_device.hpp"
#include "sfem_operator_factory.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_mprgp.hpp"
#include "sfem_parallel_cg.hpp"

// C includes
#include "matrixio_array.h"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
// #include "sfem_mask.hpp"
#include "smesh_mesh.hpp"
#include "smesh_prolongation.hpp"
#include "smesh_restrict.hpp"
#include "smesh_restriction.hpp"
#include "smesh_sideset.hpp"
#include "smesh_sshex8.hpp"
#include "smesh_sshex8_prolongation.hpp"
#include "smesh_sshex8_restriction.hpp"
#include "smesh_ssquad4.hpp"
#include "smesh_ssquad4_prolongation.hpp"
#include "smesh_sstet4.hpp"
#include "smesh_sstet4_prolongation.hpp"

// C++ includes
#include "sfem_ACRS.hpp"
#include "sfem_SCRS.hpp"
#include "sfem_SDACRS.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_ContactConditions.hpp"

#include "sfem_CooSym.hpp"
#include "sfem_Function.hpp"
#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_ParallelMatrixFreeOperator.hpp"
#include "smesh_env.hpp"
#include "smesh_exchange.hpp"

#include "sfem_sell_spmv.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_CRS.hpp"
#include "sfem_DIA.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_bcrs_sym_SpMV.hpp"
#include "sfem_cg.hpp"
#include "sfem_parallel_cg.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_mprgp.hpp"
#include "smesh_glob.hpp"
#include "smesh_semistructured.hpp"

// CUDA includes
#ifdef SFEM_ENABLE_CUDA
#include "sfem_ContactConditions_cuda.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_ShiftableBlockSymJacobiKernels_cuda.hpp"
#include "sfem_BLAS_cuda.hpp"
#include "sfem_crs_spmv_cuda.hpp"
#include "sfem_MPRGPKernels_cuda.hpp"
#include "sfem_solver_cuda.hpp"
#include "smesh_sshex8_prolongation.cuh"
#include "smesh_sshex8_restriction.cuh"
#include "smesh_tet4_prolongation_restriction.cuh"
#endif

// Externals
#include "matrixio_crs.h"

// System
#include <sys/stat.h>

#include <algorithm>
#include <vector>

#endif  // SFEM_API_HPP

