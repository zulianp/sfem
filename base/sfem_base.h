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

typedef int idx_t;
#define SFEM_MPI_IDX_T MPI_INT

typedef double real_t;
#define SFEM_MPI_REAL_T MPI_DOUBLE

// typedef long count_t;
// #define SFEM_MPI_COUNT_T MPI_LONG

typedef int count_t;
#define SFEM_MPI_COUNT_T MPI_INT

// typedef count_t element_idx_t;
// #define SFEM_MPI_ELEMENT_IDX_T SFEM_MPI_COUNT_T

// typedef long element_idx_t;
// #define SFEM_MPI_ELEMENT_IDX_T MPI_LONG

typedef int element_idx_t;
#define SFEM_MPI_ELEMENT_IDX_T MPI_INT

#define SFEM_UNUSED(var) (void)var

#define SFEM_RESTRICT __restrict__

#define SFEM_MAX_PATH_LENGTH 2056
#define SFEM_OK 0

#endif  // SFEM_BASE_H
