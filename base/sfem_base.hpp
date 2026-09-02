#ifndef SFEM_BASE_H
#define SFEM_BASE_H

#include <stdio.h>
#include <stdlib.h>

// FIXME
#ifndef SFEM_MAKEFILE_COMPILATION
#include "sfem_config.h"
#endif

#define SFEM_SUCCESS 0
#define SFEM_FAILURE 1

// #define SFEM_LOG_LEVEL 4

#if SFEM_LOG_LEVEL >= 5

#define PRINT_CURRENT_FUNCTION \
    { printf("\033[32m\nEnter Function\033[0m: \033[33m%s\033[0m, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__); }

#define RETURN_FROM_FUNCTION(__RET_VAL__)                                                                                    \
    {                                                                                                                        \
        printf("\033[31m\nReturn from function\033[0m: \033[33m%s\033[0m, file: %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return __RET_VAL__;                                                                                                  \
    }

#else

#define PRINT_CURRENT_FUNCTION

#define RETURN_FROM_FUNCTION(__RET_VAL__) \
    { return __RET_VAL__; }

#endif


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
#define SFEM_FORCE_INLINE inline __attribute__((always_inline))
#else
#define SFEM_INLINE
#define SFEM_FORCE_INLINE
#endif

// Marks a function as callable from both host and device. Expands to nothing on
// a host-only compiler, so the same header serves the CPU and CUDA/HIP builds.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SFEM_HOST_DEVICE __host__ __device__
#else
#define SFEM_HOST_DEVICE
#endif

// Beware: SFEM_INLINE above expands to *nothing* unless NDEBUG is set, so it
// cannot by itself give a header function internal linkage or inline semantics.
// Device-callable functions defined in headers must therefore be spelled either
//     static SFEM_INLINE SFEM_HOST_DEVICE ...   (the `static` does the work)
// or
//     SFEM_DEVICE_INLINE ...                    (carries a real `inline`)
// Never rely on `SFEM_INLINE SFEM_HOST_DEVICE` alone.
#define SFEM_DEVICE_INLINE SFEM_HOST_DEVICE inline

#define SFEM_UNUSED(var) (void)var
#ifndef _WIN32
#define SFEM_RESTRICT __restrict__
#else
#define SFEM_RESTRICT __restrict 
#endif

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0

#ifdef __cplusplus
extern "C" {
#endif

void sfem_abort();

#ifdef __cplusplus
}
#endif
#endif  // SFEM_BASE_H
