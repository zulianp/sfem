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

#define SFEM_UNUSED(var) (void)var
#ifndef _WIN32
#define SFEM_RESTRICT __restrict__
#else
#define SFEM_RESTRICT __restrict 
#endif

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0


// #define ON 1
// #define OFF 0

#ifdef __cplusplus
extern "C" {
#endif

void sfem_abort();

#ifdef __cplusplus
}
#endif
#endif  // SFEM_BASE_H
