#ifndef SFEM_BASE_H
#define SFEM_BASE_H

#include <stdlib.h>

#define SFEM_READ_ENV(name, conversion) \
    do {                               \
        char *var = getenv(#name);     \
        if (var) {                     \
            name = conversion(var);    \
        }                              \
    } while (0)

#ifdef NDEBUG
#define SFEM_INLINE
#else
#define SFEM_INLINE
#endif

typedef float geom_t;
typedef int idx_t;
typedef double real_t;

#endif //SFEM_BASE_H
