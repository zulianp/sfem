#include "edgeshell3_integrate_values.hpp"

#include "sfem_macros.hpp"

#include <assert.h>
#include <math.h>

static SFEM_INLINE void edgeshell3_integrate(const real_t                px0,
                                             const real_t                px1,
                                             const real_t                px2,
                                             const real_t                py0,
                                             const real_t                py1,
                                             const real_t                py2,
                                             const real_t                value,
                                             real_t *const SFEM_RESTRICT element_vector) {
    static const real_t qp[3] = {-0.77459666924148337704, 0.0, 0.77459666924148337704};
    static const real_t qw[3] = {0.55555555555555555556, 0.88888888888888888889, 0.55555555555555555556};

    real_t e0 = 0;
    real_t e1 = 0;
    real_t e2 = 0;

    for (int q = 0; q < 3; ++q) {
        const real_t r = qp[q];

        const real_t n0 = (0.5 * r) * (r - 1);
        const real_t n1 = (0.5 * r) * (r + 1);
        const real_t n2 = 1 - r * r;

        const real_t dn0 = r - 0.5;
        const real_t dn1 = r + 0.5;
        const real_t dn2 = -2 * r;

        const real_t dx = dn0 * px0 + dn1 * px1 + dn2 * px2;
        const real_t dy = dn0 * py0 + dn1 * py1 + dn2 * py2;
        const real_t dS = qw[q] * value * sqrt(dx * dx + dy * dy);

        e0 += n0 * dS;
        e1 += n1 * dS;
        e2 += n2 * dS;
    }

    element_vector[0] = e0;
    element_vector[1] = e1;
    element_vector[2] = e2;
}

int edgeshell3_integrate_value(const ptrdiff_t              nelements,
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
        real_t      element_vector[3];
        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];
        const idx_t i2 = elements[2][e];

        edgeshell3_integrate(x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], value, element_vector);

#pragma omp atomic update
        out[i0 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i1 * block_size + component] += element_vector[1];

#pragma omp atomic update
        out[i2 * block_size + component] += element_vector[2];
    }

    return SFEM_SUCCESS;
}

int edgeshell3_integrate_values(const ptrdiff_t                   nelements,
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
        real_t      element_vector[3];
        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];
        const idx_t i2 = elements[2][e];

        const scalar_t value = scale_factor * values[e];
        edgeshell3_integrate(x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], value, element_vector);

#pragma omp atomic update
        out[i0 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i1 * block_size + component] += element_vector[1];

#pragma omp atomic update
        out[i2 * block_size + component] += element_vector[2];
    }

    return SFEM_SUCCESS;
}
