# ##############################################################################
# SFEM number selection
# ##############################################################################

set(SFEM_REAL_TYPE "float64" CACHE STRING "Set SFEM real_t type. Forwarded to SMESH.")
set(SFEM_SCALAR_TYPE "${SFEM_REAL_TYPE}" CACHE STRING "Set SFEM scalar_t type. Forwarded to SMESH.")
set(SFEM_GEOM_TYPE "float32" CACHE STRING "Set SFEM geom_t type. Forwarded to SMESH.")
set(SFEM_JACOBIAN_CPU_TYPE "float32" CACHE STRING "Set SFEM jacobian_t type. Forwarded to SMESH.")
set(SFEM_JACOBIAN_GPU_TYPE "float" CACHE STRING "Set SFEM cu_jacobian_t type. Forwarded to SMESH.")
set(SFEM_ACCUMULATOR_TYPE "${SFEM_REAL_TYPE}" CACHE STRING "Set SFEM accumulator_t type. Forwarded to SMESH.")
set(SFEM_IDX_TYPE "int32" CACHE STRING "Set SFEM idx_t type. Forwarded to SMESH.")
set(SFEM_COUNT_TYPE "${SFEM_IDX_TYPE}" CACHE STRING "Set SFEM count_t type. Forwarded to SMESH.")
set(SFEM_ELEMENT_IDX_TYPE "${SFEM_IDX_TYPE}" CACHE STRING "Set SFEM element_idx_t type. Forwarded to SMESH.")
set(SFEM_LOCAL_IDX_TYPE "int16" CACHE STRING "Set SFEM local_idx_t type. Forwarded to SMESH.")

message(STATUS
    "--------------------------------------------------------------------------------------\n"
    "\nSFEM Numbers\n"
    "--------------------------------------------------------------------------------------\n"
    "Delegating number traits to SMESHNumbers.cmake with SFEM-selected type options:\n"
    "\tSFEM_REAL_TYPE=${SFEM_REAL_TYPE}\n"
    "\tSFEM_SCALAR_TYPE=${SFEM_SCALAR_TYPE}\n"
    "\tSFEM_GEOM_TYPE=${SFEM_GEOM_TYPE}\n"
    "\tSFEM_JACOBIAN_CPU_TYPE=${SFEM_JACOBIAN_CPU_TYPE}\n"
    "\tSFEM_JACOBIAN_GPU_TYPE=${SFEM_JACOBIAN_GPU_TYPE}\n"
    "\tSFEM_ACCUMULATOR_TYPE=${SFEM_ACCUMULATOR_TYPE}\n"
    "\tSFEM_IDX_TYPE=${SFEM_IDX_TYPE}\n"
    "\tSFEM_COUNT_TYPE=${SFEM_COUNT_TYPE}\n"
    "\tSFEM_ELEMENT_IDX_TYPE=${SFEM_ELEMENT_IDX_TYPE}\n"
    "\tSFEM_LOCAL_IDX_TYPE=${SFEM_LOCAL_IDX_TYPE}\n"
    "--------------------------------------------------------------------------------------\n"
)
