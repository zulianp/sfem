#include "macro_tri3_laplacian.h"

#define POW2(a) ((a) * (a))

static SFEM_INLINE void fff_micro_kernel(const geom_t px0,
                                         const geom_t px1,
                                         const geom_t px2,
                                         const geom_t py0,
                                         const geom_t py1,
                                         const geom_t py2,
                                         geom_t *fff) {
    const geom_t x0 = -px0 + px1;
    const geom_t x1 = -py0 + py2;
    const geom_t x2 = px0 - px2;
    const geom_t x3 = py0 - py1;
    const geom_t x4 = x0 * x1 - x2 * x3;
    const geom_t x5 = 1 / POW2(x4);
    fff[0] = x4 * (POW2(x1) * x5 + POW2(x2) * x5);
    fff[1] = x4 * (x0 * x2 * x5 + x1 * x3 * x5);
    fff[2] = x4 * (POW2(x0) * x5 + POW2(x3) * x5);
}

static SFEM_INLINE void sub_fff_0(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const real_t x0 = (1.0 / 4.0) * fff[1];
    sub_fff[0] = (1.0 / 4.0) * fff[0];
    sub_fff[1] = x0;
    sub_fff[2] = x0;
    sub_fff[3] = (1.0 / 4.0) * fff[3];
}

static SFEM_INLINE void sub_fff_1(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const real_t x0 = (1.0 / 4.0) * fff[3];
    const real_t x1 = -1.0 / 4.0 * fff[1] + (1.0 / 4.0) * fff[3];
    sub_fff[0] = x0;
    sub_fff[1] = x1;
    sub_fff[2] = x1;
    sub_fff[3] = (1.0 / 4.0) * fff[0] - 1.0 / 2.0 * fff[1] + x0;
}

static SFEM_INLINE void lapl_apply_micro_kernel(const geom_t *const SFEM_RESTRICT fff,
                                                const real_t u0,
                                                const real_t u1,
                                                const real_t u2,
                                                real_t *const SFEM_RESTRICT e0,
                                                real_t *const SFEM_RESTRICT e1,
                                                real_t *const SFEM_RESTRICT e2) {
    const real_t x0 = (1.0 / 2.0) * u0;
    const real_t x1 = fff[0] * x0;
    const real_t x2 = (1.0 / 2.0) * u1;
    const real_t x3 = fff[0] * x2;
    const real_t x4 = fff[1] * x2;
    const real_t x5 = (1.0 / 2.0) * u2;
    const real_t x6 = fff[1] * x5;
    const real_t x7 = fff[2] * x0;
    const real_t x8 = fff[2] * x5;
    const real_t x9 = (1.0 / 2.0) * fff[1] * u0;
    e0[0] += fff[1] * u0 + x1 - x3 - x4 - x6 + x7 - x8;
    e1[0] += -x1 + x3 + x6 - x9;
    e2[0] += x4 - x7 + x8 - x9;
}

void macro_tri3_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            geom_t fff[3];
            geom_t sub_fff[3];
            real_t element_u[6];
            real_t element_vector[6] = {0};

#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 6; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            fff_micro_kernel(
                xyz[0][i0], xyz[0][i1], xyz[0][i2], xyz[1][i0], xyz[1][i1], xyz[1][i2], fff);

            {
                // Corner FFFs (Same fff)
                sub_fff_0(fff, sub_fff);

                lapl_apply_micro_kernel(sub_fff,
                                        element_u[0],
                                        element_u[3],
                                        element_u[5],
                                        &element_vector[0],
                                        &element_vector[3],
                                        &element_vector[5]);

                lapl_apply_micro_kernel(sub_fff,
                                        element_u[3],
                                        element_u[1],
                                        element_u[4],
                                        &element_vector[3],
                                        &element_vector[1],
                                        &element_vector[4]);

                lapl_apply_micro_kernel(sub_fff,
                                        element_u[5],
                                        element_u[4],
                                        element_u[2],
                                        &element_vector[5],
                                        &element_vector[4],
                                        &element_vector[2]);
            }

            {  // Central FFF
                sub_fff_1(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[3],
                                        element_u[4],
                                        element_u[5],
                                        &element_vector[3],
                                        &element_vector[4],
                                        &element_vector[5]);
            }

#pragma unroll(6)
            for (int v = 0; v < 6; v++) {
#pragma omp atomic update
                values[ev[v]] += element_vector[v];
            }
        }
    }
}
