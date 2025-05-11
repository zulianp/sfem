#ifndef SFEM_BASE_H
#define SFEM_BASE_H

#include <stdlib.h>

// FIXME
#ifndef SFEM_MAKEFILE_COMPILATION
#include "sfem_config.h"
#endif

#define SFEM_SUCCESS 0
#define SFEM_FAILURE 1

#define PRINT_CURRENT_FUNCTION \
    printf("\033[32m\nEnter Function\033[0m: \033[33m%s\033[0m, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__);

#define RETURN_FROM_FUNCTION(__RET_VAL__)                                                                                \
    printf("\033[31m\nReturn from function\033[0m: \033[33m%s\033[0m, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
    return __RET_VAL__;

#define SFEM_READ_ENV(name, conversion) \
    do {                                \
        char* var = getenv(#name);      \
        if (var) {                      \
            name = conversion(var);     \
        }                               \
    } while (0)

#define SFEM_REQUIRE_ENV(name, conversion)                                                \
    do {                                                                                  \
        char* var = getenv(#name);                                                        \
        if (var) {                                                                        \
            name = conversion(var);                                                       \
        } else {                                                                          \
            fprintf(stderr, "[Error] %s is required (%s:%d)", #name, __FILE__, __LINE__); \
            assert(0);                                                                    \
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);                                                \
        }                                                                                 \
    } while (0)

#define SFEM_ERROR(...)                                             \
    do {                                                            \
        fprintf(stderr, __VA_ARGS__);                               \
        fprintf(stderr, "Aborting at %s:%d\n", __FILE__, __LINE__); \
        fflush(stderr);                                             \
        assert(0);                                                  \
        sfem_abort();                                               \
    } while (0)

#define SFEM_IMPLEMENT_ME() SFEM_ERROR("Implement me!\n")

#ifdef NDEBUG
#define SFEM_INLINE inline
#else
#define SFEM_INLINE
#endif

#define SFEM_UNUSED(var) (void)var
#ifndef _WIN32
#define SFEM_RESTRICT __restrict__
#else
#define SFEM_RESTRICT __restrict 
#endif

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0


#define ON 1
#define OFF 0

#ifndef SFEM_ENABLE_CUSTOM_NUMBERS

typedef float geom_t;
#define SFEM_MPI_GEOM_T MPI_FLOAT
#define d_GEOM_T "f"
#define dtype_GEOM_T "float32"

typedef int idx_t;
#define SFEM_MPI_IDX_T MPI_INT
#define SFEM_CUSPARSE_IDX_T CUSPARSE_INDEX_32I
#define d_IDX_T "d"
#define dtype_IDX_T "int32"

typedef double real_t;
#define SFEM_MPI_REAL_T MPI_DOUBLE
#define SFEM_CUSPARSE_REAL_T CUDA_R_64F
#define d_REAL_T "g"
#define dtype_REAL_T "float64"

typedef geom_t jacobian_t;

typedef long count_t;
#define SFEM_MPI_COUNT_T MPI_LONG
#define SFEM_CUSPARSE_COUNT_T CUSPARSE_INDEX_64I
#define d_COUNT_T "ld"
#define dtype_COUNT_T "int64"

typedef int element_idx_t;
#define SFEM_MPI_ELEMENT_IDX_T MPI_INT
#define d_ELEMENT_IDX_T "d"
#define dtype_ELEMENT_IDX_T "int32"

typedef int16_t local_idx_t;
#define d_LOCAL_IDX_T "hd"

// #define SFEM_RESTRICT __restrict__
#define SFEM_RESTRICT

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0

// typedef  __half2 jacobian_t;
// typedef __fp16 jacobian_t;
typedef geom_t jacobian_t;
// typedef geom_t jacobian_t;

typedef int16_t lidx_t;
#define d_ELEMENT_LIDX_T "hd"
typedef real_t scalar_t;
typedef real_t accumulator_t;
#define SFEM_VEC_SIZE 4
typedef scalar_t vec_t __attribute__((vector_size(SFEM_VEC_SIZE * sizeof(scalar_t)), aligned(SFEM_VEC_SIZE * sizeof(scalar_t))));

#endif


#ifdef __cplusplus
extern "C" {
#endif

void sfem_abort();

#ifdef __cplusplus
}
#endif
#endif  // SFEM_BASE_H
