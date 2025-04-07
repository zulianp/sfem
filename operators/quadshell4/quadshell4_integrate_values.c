#include "quadshell4_integrate_values.h"

#include "sfem_macros.h"

#include <assert.h>
#include <math.h>

static SFEM_INLINE void quad_shell_4_integrate(const scalar_t  px0,
                                               const scalar_t  px1,
                                               const scalar_t  px2,
                                               const scalar_t  px3,
                                               const scalar_t  py0,
                                               const scalar_t  py1,
                                               const scalar_t  py2,
                                               const scalar_t  py3,
                                               const scalar_t  pz0,
                                               const scalar_t  pz1,
                                               const scalar_t  pz2,
                                               const scalar_t  pz3,
                                               const scalar_t  val,
                                               scalar_t *const element_vector) {
    static const scalar_t rule_qx[4] = {0.211324865405187, 0.788675134594813, 0.211324865405187, 0.788675134594813};
    static const scalar_t rule_qy[4] = {0.211324865405187, 0.211324865405187, 0.788675134594813, 0.788675134594813};
    static const scalar_t rule_qw[4] = {0.25, 0.25, 0.25, 0.25};
    static const int      rule_n_qp  = 4;

    element_vector[0] = 0;
    element_vector[1] = 0;
    element_vector[2] = 0;
    element_vector[3] = 0;

    for (int q = 0; q < rule_n_qp; q++) {
        const scalar_t qx = rule_qx[q];
        const scalar_t qy = rule_qy[q];
        const scalar_t qw = rule_qw[q];

        const scalar_t x0 = qx - 1;
        const scalar_t x1 = -x0;
        const scalar_t x2 = qy - 1;
        const scalar_t x3 = -x2;
        const scalar_t x4 = px0 * x0 - px1 * qx + px2 * qx + px3 * x1;
        const scalar_t x5 = px0 * x2 + px1 * x3 + px2 * qy - px3 * qy;
        const scalar_t x6 = py0 * x0 - py1 * qx + py2 * qx + py3 * x1;
        const scalar_t x7 = py0 * x2 + py1 * x3 + py2 * qy - py3 * qy;
        const scalar_t x8 = pz0 * x0 - pz1 * qx + pz2 * qx + pz3 * x1;
        const scalar_t x9 = pz0 * x2 + pz1 * x3 + pz2 * qy - pz3 * qy;
        const scalar_t x10 =
                qw * val *
                sqrt((POW2(x4) + POW2(x6) + POW2(x8)) * (POW2(x5) + POW2(x7) + POW2(x9)) - POW2(x4 * x5 + x6 * x7 + x8 * x9));
        const scalar_t x11 = x10 * x3;
        const scalar_t x12 = qy * x10;
        element_vector[0] += x1 * x11;
        element_vector[1] += qx * x11;
        element_vector[2] += qx * x12;
        element_vector[3] += x1 * x12;
    }
}

int quadshell4_integrate_value(const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 value,
                               const int                    block_size,
                               const int                    component,
                               real_t *const SFEM_RESTRICT  out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        scalar_t element_vector[4] = {0};

        const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};

        quad_shell_4_integrate(x[ev[0]],
                               x[ev[1]],
                               x[ev[2]],
                               x[ev[3]],
                               y[ev[0]],
                               y[ev[1]],
                               y[ev[2]],
                               y[ev[3]],
                               z[ev[0]],
                               z[ev[1]],
                               z[ev[2]],
                               z[ev[3]],
                               value,
                               element_vector);

        assert(element_vector[0] == element_vector[0]);
        assert(element_vector[1] == element_vector[1]);
        assert(element_vector[2] == element_vector[2]);
        assert(element_vector[3] == element_vector[3]);

        for (int d = 0; d < 4; d++) {
#pragma omp atomic update
            out[ev[d] * block_size + component] += element_vector[d];
        }
    }

    return SFEM_SUCCESS;
}

int quadshell4_integrate_values(const ptrdiff_t                   nelements,
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
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (idx_t e = 0; e < nelements; ++e) {
        scalar_t element_vector[4] = {0};

        const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};

        const scalar_t value = scale_factor * values[e];
        quad_shell_4_integrate(x[ev[0]],
                               x[ev[1]],
                               x[ev[2]],
                               x[ev[3]],
                               y[ev[0]],
                               y[ev[1]],
                               y[ev[2]],
                               y[ev[3]],
                               z[ev[0]],
                               z[ev[1]],
                               z[ev[2]],
                               z[ev[3]],
                               value,
                               element_vector);

        assert(element_vector[0] == element_vector[0]);
        assert(element_vector[1] == element_vector[1]);
        assert(element_vector[2] == element_vector[2]);
        assert(element_vector[3] == element_vector[3]);

        for (int d = 0; d < 4; d++) {
#pragma omp atomic update
            out[ev[d] * block_size + component] += element_vector[d];
        }
    }

    return SFEM_SUCCESS;
}
