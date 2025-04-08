#include "edgeshell2_integrate_values.h"

#include "sfem_macros.h"

#include <assert.h>
#include <math.h>

static SFEM_INLINE void edgeshell2_integrate(const real_t                px0,
                                             const real_t                px1,
                                             const real_t                py0,
                                             const real_t                py1,
                                             const real_t                value,
                                             real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 =
            (1.0 / 2.0) * value * sqrt(pow(px0, 2) - 2 * px0 * px1 + pow(px1, 2) + pow(py0, 2) - 2 * py0 * py1 + pow(py1, 2));
    element_vector[0] = x0;
    element_vector[1] = x0;
}

int edgeshell2_integrate_value(const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 value,
                               const int                    block_size,
                               const int                    component,
                               real_t *const SFEM_RESTRICT  out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        real_t      element_vector[2];
        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];

        edgeshell2_integrate(x[i0], x[i1], y[i0], y[i1], value, element_vector);

// Only edge dofs
#pragma omp atomic update
        out[i0 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i1 * block_size + component] += element_vector[1];
    }

    return SFEM_SUCCESS;
}

int edgeshell2_integrate_values(const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t **const SFEM_RESTRICT       elements,
                                geom_t **const SFEM_RESTRICT      points,
                                const real_t                      scale_factor,
                                const real_t *const SFEM_RESTRICT values,
                                const int                         block_size,
                                const int                         component,
                                real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        real_t      element_vector[2];
        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];

        const scalar_t value = scale_factor * values[e];
        edgeshell2_integrate(x[i0], x[i1], y[i0], y[i1], value, element_vector);

// Only edge dofs
#pragma omp atomic update
        out[i0 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i1 * block_size + component] += element_vector[1];
    }

    return SFEM_SUCCESS;
}
