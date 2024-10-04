#ifndef SFEM_BASE_H
#define SFEM_BASE_H

#include <stdlib.h>

#define SFEM_READ_ENV(name, conversion) \
    do {                                \
        char *var = getenv(#name);      \
        if (var) {                      \
            name = conversion(var);     \
        }                               \
    } while (0)

#ifdef NDEBUG
#define SFEM_INLINE inline
#else
#define SFEM_INLINE
#endif

typedef float geom_t;
#define SFEM_MPI_GEOM_T MPI_FLOAT
#define d_GEOM_T "f"
#define dtype_GEOM_T "float32"

typedef int idx_t;
#define SFEM_MPI_IDX_T MPI_INT
#define d_IDX_T "d"
#define dtype_IDX_T "int32"

typedef double real_t;
#define SFEM_MPI_REAL_T MPI_DOUBLE
#define d_REAL_T "g"
#define dtype_REAL_T "float64"

// typedef long count_t;
// #define SFEM_MPI_COUNT_T MPI_LONG

typedef int count_t;
#define SFEM_MPI_COUNT_T MPI_INT
#define d_COUNT_T "d"
#define dtype_COUNT_T "int32"

// typedef count_t element_idx_t;
// #define SFEM_MPI_ELEMENT_IDX_T SFEM_MPI_COUNT_T

// typedef long element_idx_t;
// #define SFEM_MPI_ELEMENT_IDX_T MPI_LONG

typedef int element_idx_t;
#define SFEM_MPI_ELEMENT_IDX_T MPI_INT
#define d_ELEMENT_IDX_T "d"
#define dtype_ELEMENT_IDX_T "int32"

#define SFEM_UNUSED(var) (void)var

// #define SFEM_RESTRICT __restrict__
#define SFEM_RESTRICT

#define PRINT_CURRENT_FUNCTION printf("\033[32m\nEnter Function\033[0m: %s, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__);
#define RETURN_FROM_FUNCTION(__RET_VAL__) printf("\033[31m\nReturn from function\033[0m: %s, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__); return (__RET_VAL__);

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0

// typedef  __half2 jacobian_t;
// typedef __fp16 jacobian_t;
typedef geom_t jacobian_t;
// typedef geom_t jacobian_t;

typedef int16_t lidx_t;
#define d_ELEMENT_LIDX_T "hd"

#endif  // SFEM_BASE_H
