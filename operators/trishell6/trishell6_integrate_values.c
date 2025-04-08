#include "trishell6_integrate_values.h"

#include "sfem_macros.h"

#include <assert.h>
#include <math.h>

static SFEM_INLINE void tri_shell_6_integrate(const scalar_t          px0,
                                              const scalar_t          px1,
                                              const scalar_t          px2,
                                              const scalar_t          py0,
                                              const scalar_t          py1,
                                              const scalar_t          py2,
                                              const scalar_t          pz0,
                                              const scalar_t          pz1,
                                              const scalar_t          pz2,
                                              const scalar_t          value,
                                              scalar_t *SFEM_RESTRICT element_vector) {
    const scalar_t x0  = 2 * px0;
    const scalar_t x1  = px1 * x0;
    const scalar_t x2  = py0 * x1;
    const scalar_t x3  = py1 * py2;
    const scalar_t x4  = pz0 * pz1;
    const scalar_t x5  = pz0 * pz2;
    const scalar_t x6  = pz1 * pz2;
    const scalar_t x7  = px2 * x0;
    const scalar_t x8  = py0 * x7;
    const scalar_t x9  = px1 * px2;
    const scalar_t x10 = 2 * py0;
    const scalar_t x11 = py1 * x10;
    const scalar_t x12 = py2 * x10;
    const scalar_t x13 = 2 * x9;
    const scalar_t x14 = 2 * pz0;
    const scalar_t x15 = pz1 * x14;
    const scalar_t x16 = pz2 * x14;
    const scalar_t x17 = POW2(px0);
    const scalar_t x18 = POW2(py1);
    const scalar_t x19 = POW2(py2);
    const scalar_t x20 = POW2(pz1);
    const scalar_t x21 = POW2(pz2);
    const scalar_t x22 = POW2(px1);
    const scalar_t x23 = POW2(py0);
    const scalar_t x24 = POW2(pz0);
    const scalar_t x25 = POW2(px2);
    const scalar_t x26 = 2 * x17;
    const scalar_t x27 = 2 * x23;
    const scalar_t x28 = 2 * x24;
    const scalar_t x29 =
            (1.0 / 6.0) * value *
            sqrt(-py1 * x2 + py1 * x8 + py2 * x2 - py2 * x8 - x1 * x19 - x1 * x21 + x1 * x3 - x1 * x4 + x1 * x5 + x1 * x6 -
                 x11 * x21 - x11 * x25 - x11 * x4 + x11 * x5 + x11 * x6 + x11 * x9 - x12 * x20 - x12 * x22 + x12 * x4 - x12 * x5 +
                 x12 * x6 + x12 * x9 - x13 * x3 - x13 * x6 - x15 * x19 - x15 * x25 + x15 * x3 + x15 * x9 - x16 * x18 - x16 * x22 +
                 x16 * x3 + x16 * x9 + x17 * x18 + x17 * x19 + x17 * x20 + x17 * x21 + x18 * x21 + x18 * x24 + x18 * x25 -
                 x18 * x7 + x19 * x20 + x19 * x22 + x19 * x24 + x20 * x23 + x20 * x25 - x20 * x7 + x21 * x22 + x21 * x23 +
                 x22 * x23 + x22 * x24 + x23 * x25 + x24 * x25 - x26 * x3 - x26 * x6 - x27 * x6 - x27 * x9 - x28 * x3 - x28 * x9 -
                 2 * x3 * x6 + x3 * x7 + x4 * x7 - x5 * x7 + x6 * x7);

    assert(x29 == x29);
    element_vector[0] = x29;
    element_vector[1] = x29;
    element_vector[2] = x29;
}

int trishell6_integrate_value(const ptrdiff_t              nelements,
                              const ptrdiff_t              nnodes,
                              idx_t **const SFEM_RESTRICT  elements,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t                 value,
                              const int                    block_size,
                              const int                    component,
                              real_t *const SFEM_RESTRICT  out) {
    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        scalar_t element_vector[3] = {0};

        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];
        const idx_t i2 = elements[2][e];

        tri_shell_6_integrate(x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], z[i0], z[i1], z[i2], value, element_vector);

        assert(!isnan(element_vector[0]));
        assert(!isnan(element_vector[1]));
        assert(!isnan(element_vector[2]));

        const idx_t i3 = elements[3][e];
        const idx_t i4 = elements[4][e];
        const idx_t i5 = elements[5][e];

// Only edge dofs
#pragma omp atomic update
        out[i3 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i4 * block_size + component] += element_vector[1];

#pragma omp atomic update
        out[i5 * block_size + component] += element_vector[2];
    }

    return SFEM_SUCCESS;
}

int trishell6_integrate_values(const ptrdiff_t                   nelements,
                               const ptrdiff_t                   nnodes,
                               idx_t **const SFEM_RESTRICT       elements,
                               geom_t **const SFEM_RESTRICT      xyz,
                               const real_t                      scale_factor,
                               const real_t *const SFEM_RESTRICT values,
                               const int                         block_size,
                               const int                         component,
                               real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        scalar_t element_vector[3] = {0};

        const idx_t i0 = elements[0][e];
        const idx_t i1 = elements[1][e];
        const idx_t i2 = elements[2][e];

        const scalar_t value = scale_factor * values[e];
        tri_shell_6_integrate(x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], z[i0], z[i1], z[i2], value, element_vector);

        assert(!isnan(element_vector[0]));
        assert(!isnan(element_vector[1]));
        assert(!isnan(element_vector[2]));

        const idx_t i3 = elements[3][e];
        const idx_t i4 = elements[4][e];
        const idx_t i5 = elements[5][e];

// Only edge dofs
#pragma omp atomic update
        out[i3 * block_size + component] += element_vector[0];

#pragma omp atomic update
        out[i4 * block_size + component] += element_vector[1];

#pragma omp atomic update
        out[i5 * block_size + component] += element_vector[2];
    }

    return SFEM_SUCCESS;
}
