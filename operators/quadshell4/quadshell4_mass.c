#include "quadshell4_mass.h"
#include "sfem_defs.h"

#include <math.h>

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

static SFEM_INLINE void quadshell4_apply_mass_kernel(const scalar_t px0,
                                                     const scalar_t px1,
                                                     const scalar_t px2,
                                                     const scalar_t px3,
                                                     const scalar_t py0,
                                                     const scalar_t py1,
                                                     const scalar_t py2,
                                                     const scalar_t py3,
                                                     const scalar_t pz0,
                                                     const scalar_t pz1,
                                                     const scalar_t pz2,
                                                     const scalar_t pz3,
                                                     const scalar_t *const SFEM_RESTRICT u,
                                                     accumulator_t *const SFEM_RESTRICT element_vector) {
    static const scalar_t rule_qx[6] = {0.5,
                                        0.98304589153964795245728880523899,
                                        0.72780186391809642112479237299488,
                                        0.72780186391809642112479237299488,
                                        0.13418502421343273531598225407969,
                                        0.13418502421343273531598225407969};

    static const scalar_t rule_qy[6] = {0.5,
                                        0.5,
                                        0.074042673347699754349082179816666,
                                        0.92595732665230024565091782018333,
                                        0.18454360551162298687829339850317,
                                        0.81545639448837701312170660149683};

    static const scalar_t rule_qw[6] = {0.28571428571428571428571428571428,
                                        0.10989010989010989010989010989011,
                                        0.14151805175188302631601261486295,
                                        0.14151805175188302631601261486295,
                                        0.16067975044591917148618518733485,
                                        0.16067975044591917148618518733485};
    static const int rule_n_qp = 6;

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
                qw *
                sqrt((POW2(x4) + POW2(x6) + POW2(x8)) * (POW2(x5) + POW2(x7) + POW2(x9)) -
                     POW2(x4 * x5 + x6 * x7 + x8 * x9)) *
                (qx * u[1] + qy * u[2] + u[0] * (-qy - x0));
        const scalar_t x11 = x10 * x3;
        const scalar_t x12 = qy * x10;
        element_vector[0] += x1 * x11;
        element_vector[1] += qx * x11;
        element_vector[2] += qx * x12;
        element_vector[3] += x1 * x12;
    }
}

void quadshell4_apply_mass(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elements,
                           geom_t **const SFEM_RESTRICT points,
                           const ptrdiff_t stride_u,
                           const real_t *const u,
                           const ptrdiff_t stride_values,
                           real_t *const values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        idx_t ks[4];
        scalar_t element_u[4];
        accumulator_t element_vector[4];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int enode = 0; enode < 4; ++enode) {
            element_u[enode] = u[ev[enode] * stride_u];
        }

        quadshell4_apply_mass_kernel(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[4]],
                element_u,
                // output vector
                element_vector);

#pragma unroll(4)
        for (int edof_i = 0; edof_i < 4; edof_i++) {
#pragma omp atomic update
            values[ev[edof_i] * stride_values] += element_vector[edof_i];
        }
    }
}
