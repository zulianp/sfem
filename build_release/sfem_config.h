#ifndef SFEM_CONFIG_HPP
#define SFEM_CONFIG_HPP

#define SFEM_ENABLE_AMG
/* #undef SFEM_ENABLE_AVX2 */
/* #undef SFEM_ENABLE_AVX512 */
/* #undef SFEM_ENABLE_CUBLAS */
/* #undef SFEM_ENABLE_CUDA */
/* #undef SFEM_ENABLE_CUSPARSE */
#define SFEM_ENABLE_EXPLICIT_VECTORIZATION
/* #undef SFEM_ENABLE_HXTSORT */
/* #undef SFEM_ENABLE_LAPACK */
#define SFEM_ENABLE_MEM_DIAGNOSTICS
/* #undef SFEM_ENABLE_METIS */
#define SFEM_ENABLE_OPENMP
/* #undef SFEM_ENABLE_RYAML */
#define SFEM_ENABLE_TRACE
/* #undef SFEM_USE_OCCUPANCY_MAX_POTENTIAL */
#define SFEM_ENABLE_MPI
/* #undef SFEM_ENABLE_AVX512_SORT */

#define SFEM_PRINT_THROUGHPUT 0

#ifdef SFEM_ENABLE_CUDA
#define SFEM_CUDA_ARCH 60
#endif

// It will be removed in the future (i.e., always true)
#define SFEM_ENABLE_CUSTOM_NUMBERS

#define CUDA_UNIFIED_MEMORY 0
#define CUDA_MANAGED_MEMORY 1
#define CUDA_HOST_MEMORY 2

#ifdef SFEM_ENABLE_CUSTOM_NUMBERS

#include <stdint.h>
#include <stdlib.h>

// -----------------------------
// Floating point representation
// -----------------------------

// geometric quantities / mesh
typedef float geom_t;
#define SFEM_MPI_GEOM_T MPI_FLOAT
#define SFEM_CUSPARSE_GEOM_T CUDA_R_32F
#define d_GEOM_T "f"
#define dtype_GEOM_T "float32"

// real numbers
typedef double real_t;
#define SFEM_MPI_REAL_T MPI_DOUBLE
#define SFEM_CUSPARSE_REAL_T CUDA_R_64F
#define d_REAL_T "g"
#define dtype_REAL_T "float64"
#define SIZEOF_REAL_T 8

#define SFEM_REAL_T_IS_FLOAT64 (SIZEOF_REAL_T == 8)
#define SFEM_REAL_T_IS_FLOAT32 (SIZEOF_REAL_T == 4)

// computation real numbers and explicit vectorized types
typedef double scalar_t;
#define SFEM_MPI_SCALAR_T MPI_DOUBLE
#define SFEM_CUSPARSE_SCALAR_T CUDA_R_64F
#define d_SCALAR_T "g"
#define dtype_SCALAR_T "float64"

#ifndef _WIN32
#define SFEM_VEC_SIZE 4
typedef scalar_t vec_t __attribute__((vector_size(SFEM_VEC_SIZE * sizeof(scalar_t)),
                                      aligned(SFEM_VEC_SIZE * sizeof(scalar_t))));
#else
#define SFEM_VEC_SIZE 1
typedef scalar_t vec_t;
#endif
// Jacobian
typedef float jacobian_t;
#define SFEM_MPI_JACOBIAN_CPU_T MPI_FLOAT
#define SFEM_CUSPARSE_JACOBIAN_CPU_T CUDA_R_32F
#define d_JACOBIAN_CPU_T "f"
#define dtype_JACOBIAN_CPU_T "float32"

// Jacobian (GPU)
#define cu_jacobian_t float

// Accumulator for local kernels (before local to global)
typedef double accumulator_t;
#define SFEM_MPI_ACCUMULATOR_T MPI_DOUBLE
#define SFEM_CUSPARSE_ACCUMULATOR_T 
#define d_ACCUMULATOR_T "g"
#define dtype_ACCUMULATOR_T "float64"

// -----------------------------
// Indexing representation
// -----------------------------

// node / dof indices
typedef int32_t  idx_t;
#define SFEM_MPI_IDX_T MPI_INT32_T 
#define SFEM_CUSPARSE_IDX_T CUSPARSE_INDEX_32I
#define d_IDX_T "d"
#define dtype_IDX_T "int32"
#define SFEM_IDX_INVALID -1

// nnz count
typedef int32_t  count_t;
#define SFEM_MPI_COUNT_T MPI_INT32_T 
#define SFEM_CUSPARSE_COUNT_T CUSPARSE_INDEX_32I
#define d_COUNT_T "d"
#define dtype_COUNT_T "int32"
#define SFEM_COUNT_INVALID -1

// element ids
typedef int32_t  element_idx_t;
#define SFEM_MPI_ELEMENT_IDX_T MPI_INT32_T 
#define SFEM_CUSPARSE_ELEMENT_IDX_T CUSPARSE_INDEX_32I
#define d_ELEMENT_IDX_T "d"
#define dtype_ELEMENT_IDX_T "int32"
#define SFEM_ELEMENT_IDX_INVALID -1

// Local ids for small block of elements (goal is 16 bits with max id 65'536)
typedef int16_t  local_idx_t;
#define SFEM_MPI_LOCAL_IDX_T MPI_INT16_T 
#define SFEM_CUSPARSE_LOCAL_IDX_T 
#define d_LOCAL_IDX_T "hd"
#define dtype_LOCAL_IDX_T "int16"
#define SFEM_LOCAL_IDX_INVALID -1

// -----------------------------
// Other types
#define SFEM_TET4_CUDA OFF
#define SFEM_TET10_CUDA OFF
#define SFEM_TET10_WENO ON

#define SFEM_CUDA_MEMORY_MODEL 2

#define SFEM_LOG_LEVEL 5

#endif  // SFEM_ENABLE_CUSTOM_NUMBERS

#define SFEM_WARP_SIZE 32
#define SFEM_WARP_FULL_MASK 0xffffffff

#ifdef SFEM_ENABLE_CUDA

// To enable atomicOr we need int
typedef int mask_t;
#define SFEM_MPI_MASK_T MPI_INT
#define SFEM_PTRDIFF_INVALID -1

#else

typedef char mask_t;
#define SFEM_MPI_MASK_T MPI_CHAR
#define SFEM_PTRDIFF_INVALID -1

#endif // SFEM_ENABLE_CUDA

typedef int64_t sshex_side_code_t;
typedef uint16_t block_idx_t;

#endif  // SFEM_CONFIG_HPP
