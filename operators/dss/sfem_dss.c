#include "sfem_dss.h"

#define SFEM_DSS_COPY_IMPL(name, scalar_t)                                      \
    int name(const ptrdiff_t n, const scalar_t *const input, scalar_t *const output) { \
        if (!input || !output || n < 0) {                                       \
            return SFEM_FAILURE;                                                \
        }                                                                       \
                                                                                \
        _Pragma("omp parallel for")                                             \
        for (ptrdiff_t i = 0; i < n; ++i) {                                     \
            output[i] = input[i];                                               \
        }                                                                       \
                                                                                \
        return SFEM_SUCCESS;                                                    \
    }

SFEM_DSS_COPY_IMPL(sfem_dss_apply_real32, float)
SFEM_DSS_COPY_IMPL(sfem_dss_apply_real64, double)

#undef SFEM_DSS_COPY_IMPL
