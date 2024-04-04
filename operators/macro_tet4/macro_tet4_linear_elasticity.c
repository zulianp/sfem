#include "macro_tet4_linear_elasticity.h"
#include <stddef.h>
#include "sfem_base.h"

static SFEM_INLINE void jacobian_micro_kernel(const geom_t px0,
                                              const geom_t px1,
                                              const geom_t px2,
                                              const geom_t px3,
                                              const geom_t py0,
                                              const geom_t py1,
                                              const geom_t py2,
                                              const geom_t py3,
                                              const geom_t pz0,
                                              const geom_t pz1,
                                              const geom_t pz2,
                                              const geom_t pz3,
                                              jacobian_t *jacobian) {
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;
}

static SFEM_INLINE void adjugate_and_det_micro_kernel(const geom_t px0,
                                                      const geom_t px1,
                                                      const geom_t px2,
                                                      const geom_t px3,
                                                      const geom_t py0,
                                                      const geom_t py1,
                                                      const geom_t py2,
                                                      const geom_t py3,
                                                      const geom_t pz0,
                                                      const geom_t pz1,
                                                      const geom_t pz2,
                                                      const geom_t pz3,
                                                      jacobian_t *adjugate,
                                                      jacobian_t *jacobian_determinant) {
    // Compute jacobian in high precision
    real_t jacobian[9];
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;

    const real_t x0 = jacobian[4] * jacobian[8];
    const real_t x1 = jacobian[5] * jacobian[7];
    const real_t x2 = jacobian[1] * jacobian[8];
    const real_t x3 = jacobian[1] * jacobian[5];
    const real_t x4 = jacobian[2] * jacobian[4];

    // Store adjugate in lower precision
    adjugate[0] = x0 - x1;
    adjugate[1] = jacobian[2] * jacobian[7] - x2;
    adjugate[2] = x3 - x4;
    adjugate[3] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];

    // Store determinant in lower precision
    jacobian_determinant[0] = jacobian[0] * x0 - jacobian[0] * x1 +
                              jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                              jacobian[6] * x3 - jacobian[6] * x4;
}

static SFEM_INLINE void sub_adj_0(const jacobian_t *const SFEM_RESTRICT adjugate,
                                  const ptrdiff_t stride,
                                  jacobian_t *const SFEM_RESTRICT sub_adjugate) {
    sub_adjugate[0] = 2 * adjugate[0 * stride];
    sub_adjugate[1] = 2 * adjugate[1 * stride];
    sub_adjugate[2] = 2 * adjugate[2 * stride];
    sub_adjugate[3] = 2 * adjugate[3 * stride];
    sub_adjugate[4] = 2 * adjugate[4 * stride];
    sub_adjugate[5] = 2 * adjugate[5 * stride];
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}
static SFEM_INLINE void sub_adj_4(const jacobian_t *const SFEM_RESTRICT adjugate,
                                  const ptrdiff_t stride,
                                  jacobian_t *const SFEM_RESTRICT sub_adjugate) {
    const real_t x0 = 2 * adjugate[0 * stride];
    const real_t x1 = 2 * adjugate[1 * stride];
    const real_t x2 = 2 * adjugate[2 * stride];
    sub_adjugate[0] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[3] = -x0;
    sub_adjugate[4] = -x1;
    sub_adjugate[5] = -x2;
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}

static SFEM_INLINE void sub_adj_5(const jacobian_t *const SFEM_RESTRICT adjugate,
                                  const ptrdiff_t stride,
                                  jacobian_t *const SFEM_RESTRICT sub_adjugate) {
    const real_t x0 = 2 * adjugate[3 * stride];
    const real_t x1 = 2 * adjugate[6 * stride] + x0;
    const real_t x2 = 2 * adjugate[4 * stride];
    const real_t x3 = 2 * adjugate[7 * stride] + x2;
    const real_t x4 = 2 * adjugate[5 * stride];
    const real_t x5 = 2 * adjugate[8 * stride] + x4;
    sub_adjugate[0] = -x1;
    sub_adjugate[1] = -x3;
    sub_adjugate[2] = -x5;
    sub_adjugate[3] = x0;
    sub_adjugate[4] = x2;
    sub_adjugate[5] = x4;
    sub_adjugate[6] = 2 * adjugate[0 * stride] + x1;
    sub_adjugate[7] = 2 * adjugate[1 * stride] + x3;
    sub_adjugate[8] = 2 * adjugate[2 * stride] + x5;
}

static SFEM_INLINE void sub_adj_6(const jacobian_t *const SFEM_RESTRICT adjugate,
                                  const ptrdiff_t stride,
                                  jacobian_t *const SFEM_RESTRICT sub_adjugate) {
    const real_t x0 = 2 * adjugate[3 * stride];
    const real_t x1 = 2 * adjugate[4 * stride];
    const real_t x2 = 2 * adjugate[5 * stride];
    sub_adjugate[0] = 2 * adjugate[0 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[1 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[2 * stride] + x2;
    sub_adjugate[3] = 2 * adjugate[6 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[7 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[8 * stride] + x2;
    sub_adjugate[6] = -x0;
    sub_adjugate[7] = -x1;
    sub_adjugate[8] = -x2;
}

static SFEM_INLINE void sub_adj_7(const jacobian_t *const SFEM_RESTRICT adjugate,
                                  const ptrdiff_t stride,
                                  jacobian_t *const SFEM_RESTRICT sub_adjugate) {
    const real_t x0 = 2 * adjugate[6 * stride];
    const real_t x1 = 2 * adjugate[7 * stride];
    const real_t x2 = 2 * adjugate[8 * stride];
    sub_adjugate[0] = -x0;
    sub_adjugate[1] = -x1;
    sub_adjugate[2] = -x2;
    sub_adjugate[3] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[6] = 2 * adjugate[0 * stride];
    sub_adjugate[7] = 2 * adjugate[1 * stride];
    sub_adjugate[8] = 2 * adjugate[2 * stride];
}

static SFEM_INLINE void tet4_linear_elasticity_apply_kernel_opt(
    const real_t mu,
    const real_t lambda,
    const jacobian_t *const SFEM_RESTRICT adjugate,
    const jacobian_t jacobian_determinant,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    real_t *const SFEM_RESTRICT outx,
    real_t *const SFEM_RESTRICT outy,
    real_t *const SFEM_RESTRICT outz) {
    // Evaluation of displacement gradient
    real_t disp_grad[9];
    {
        const real_t x0 = 1.0 / jacobian_determinant;
        const real_t x1 = adjugate[0] * x0;
        const real_t x2 = adjugate[3] * x0;
        const real_t x3 = adjugate[6] * x0;
        const real_t x4 = -x1 - x2 - x3;
        const real_t x5 = adjugate[1] * x0;
        const real_t x6 = adjugate[4] * x0;
        const real_t x7 = adjugate[7] * x0;
        const real_t x8 = -x5 - x6 - x7;
        const real_t x9 = adjugate[2] * x0;
        const real_t x10 = adjugate[5] * x0;
        const real_t x11 = adjugate[8] * x0;
        const real_t x12 = -x10 - x11 - x9;
        // X
        disp_grad[0] = ux[0] * x4 + ux[1] * x1 + ux[2] * x2 + ux[3] * x3;
        disp_grad[1] = ux[0] * x8 + ux[1] * x5 + ux[2] * x6 + ux[3] * x7;
        disp_grad[2] = ux[0] * x12 + ux[1] * x9 + ux[2] * x10 + ux[3] * x11;

        // Y
        disp_grad[3] = uy[0] * x4 + uy[1] * x1 + uy[2] * x2 + uy[3] * x3;
        disp_grad[4] = uy[0] * x8 + uy[1] * x5 + uy[2] * x6 + uy[3] * x7;
        disp_grad[5] = uy[0] * x12 + uy[1] * x9 + uy[2] * x10 + uy[3] * x11;

        // Z
        disp_grad[6] = uz[2] * x2 + uz[3] * x3 + uz[0] * x4 + uz[1] * x1;
        disp_grad[7] = uz[2] * x6 + uz[3] * x7 + uz[0] * x8 + uz[1] * x5;
        disp_grad[8] = uz[2] * x10 + uz[3] * x11 + uz[0] * x12 + uz[1] * x9;
    }

    // We can reuse the buffer to avoid additional register usage
    real_t *P = disp_grad;
    {
        const real_t x0 = (1.0 / 3.0) * mu;
        const real_t x1 =
            (1.0 / 12.0) * lambda * (2 * disp_grad[0] + 2 * disp_grad[4] + 2 * disp_grad[8]);
        const real_t x2 = (1.0 / 6.0) * mu;
        const real_t x3 = x2 * (disp_grad[1] + disp_grad[3]);
        const real_t x4 = x2 * (disp_grad[2] + disp_grad[6]);
        const real_t x5 = x2 * (disp_grad[5] + disp_grad[7]);
        P[0] = disp_grad[0] * x0 + x1;
        P[1] = x3;
        P[2] = x4;
        P[3] = x3;
        P[4] = disp_grad[4] * x0 + x1;
        P[5] = x5;
        P[6] = x4;
        P[7] = x5;
        P[8] = disp_grad[8] * x0 + x1;
    }

    // Bilinear form
    {
        const real_t x0 = adjugate[0] + adjugate[3] + adjugate[6];
        const real_t x1 = adjugate[1] + adjugate[4] + adjugate[7];
        const real_t x2 = adjugate[2] + adjugate[5] + adjugate[8];
        // X
        outx[0] = -P[0] * x0 - P[1] * x1 - P[2] * x2;
        outx[1] = P[0] * adjugate[0] + P[1] * adjugate[1] + P[2] * adjugate[2];
        outx[2] = P[0] * adjugate[3] + P[1] * adjugate[4] + P[2] * adjugate[5];
        outx[3] = P[0] * adjugate[6] + P[1] * adjugate[7] + P[2] * adjugate[8];
        // Y
        outy[0] = -P[3] * x0 - P[4] * x1 - P[5] * x2;
        outy[1] = P[3] * adjugate[0] + P[4] * adjugate[1] + P[5] * adjugate[2];
        outy[2] = P[3] * adjugate[3] + P[4] * adjugate[4] + P[5] * adjugate[5];
        outy[3] = P[3] * adjugate[6] + P[4] * adjugate[7] + P[5] * adjugate[8];
        // Z
        outz[0] = -P[6] * x0 - P[7] * x1 - P[8] * x2;
        outz[1] = P[6] * adjugate[0] + P[7] * adjugate[1] + P[8] * adjugate[2];
        outz[2] = P[6] * adjugate[3] + P[7] * adjugate[4] + P[8] * adjugate[5];
        outz[3] = P[6] * adjugate[6] + P[7] * adjugate[7] + P[8] * adjugate[8];
    }
}

void macro_tet4_linear_elasticity_init(linear_elasticity_t *const ctx,
                                       const real_t mu,
                                       const real_t lambda,
                                       const ptrdiff_t nelements,
                                       idx_t **const SFEM_RESTRICT elements,
                                       geom_t **const SFEM_RESTRICT points) {
    jacobian_t *jacobian_adjugate = (jacobian_t *)calloc(9 * nelements, sizeof(jacobian_t));
    jacobian_t *jacobian_determinant = (jacobian_t *)calloc(nelements, sizeof(jacobian_t));

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            adjugate_and_det_micro_kernel(points[0][elements[0][e]],
                                          points[0][elements[1][e]],
                                          points[0][elements[2][e]],
                                          points[0][elements[3][e]],
                                          points[1][elements[0][e]],
                                          points[1][elements[1][e]],
                                          points[1][elements[2][e]],
                                          points[1][elements[3][e]],
                                          points[2][elements[0][e]],
                                          points[2][elements[1][e]],
                                          points[2][elements[2][e]],
                                          points[2][elements[3][e]],
                                          &jacobian_adjugate[e * 9],
                                          &jacobian_determinant[e]);
        }
    }

    ctx->mu = mu;
    ctx->lambda = lambda;
    ctx->jacobian_adjugate = jacobian_adjugate;
    ctx->jacobian_determinant = jacobian_determinant;
    ctx->elements = elements;
    ctx->nelements = nelements;
    ctx->element_type = MACRO_TET4;
}

void macro_tet4_linear_elasticity_destroy(linear_elasticity_t *const ctx) {
    free(ctx->jacobian_adjugate);
    free(ctx->jacobian_determinant);

    ctx->jacobian_adjugate = 0;
    ctx->jacobian_determinant = 0;

    ctx->elements = 0;
    ctx->nelements = 0;
    ctx->element_type = INVALID;
}

static const int sub_tets[8][4] = {{0, 4, 6, 7},
                                   {4, 1, 5, 8},
                                   {6, 5, 2, 9},
                                   {7, 8, 9, 3},
                                   {4, 5, 6, 8},
                                   {7, 4, 6, 8},
                                   {6, 5, 9, 8},
                                   {7, 6, 9, 8}};

typedef void (*SubAdjFun)(const jacobian_t *const SFEM_RESTRICT,
                          const ptrdiff_t,
                          jacobian_t *const SFEM_RESTRICT);

static SubAdjFun octahedron_adj_fun[4] = {&sub_adj_4, &sub_adj_5, &sub_adj_6, &sub_adj_7};

static SFEM_INLINE void subtet_gather(const int i,
                                      const real_t *const SFEM_RESTRICT in,
                                      real_t *const SFEM_RESTRICT out) {
    const int *g = sub_tets[i];
    for (int v = 0; v < 4; ++v) {
        out[v] = in[g[v]];
    }
}

static SFEM_INLINE void subtet_scatter_add(const int i,
                                           const real_t *const SFEM_RESTRICT in,
                                           real_t *const SFEM_RESTRICT out) {
    const int *s = sub_tets[i];
    for (int v = 0; v < 4; ++v) {
        out[s[v]] += in[v];
    }
}

void macro_tet4_linear_elasticity_apply_opt(const linear_elasticity_t *const ctx,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT values) {
    const real_t mu = ctx->mu;
    const real_t lambda = ctx->lambda;

    const jacobian_t *const g_jacobian_adjugate = (jacobian_t *)ctx->jacobian_adjugate;
    const jacobian_t *const g_jacobian_determinant = (jacobian_t *)ctx->jacobian_determinant;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < ctx->nelements; ++i) {
            idx_t ev[10];

            // Sub-geometry
            jacobian_t sub_adjugate[9];

            // Is it sensibile to have so many "registers" here?
            real_t ux[10];
            real_t uy[10];
            real_t uz[10];

            real_t outx[10] = {0};
            real_t outy[10] = {0};
            real_t outz[10] = {0};

            // Sub-buffers
            real_t sub_ux[4];
            real_t sub_uy[4];
            real_t sub_uz[4];

            real_t sub_outx[4];
            real_t sub_outy[4];
            real_t sub_outz[4];

            const jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[i * 9];
            const jacobian_t jacobian_determinant = g_jacobian_determinant[i];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = ctx->elements[v][i];
            }

            for (int v = 0; v < 10; ++v) {
                ux[v] = u[ev[v] * 3];
                uy[v] = u[ev[v] * 3 + 1];
                uz[v] = u[ev[v] * 3 + 2];
            }

            // All cached from here

            {  // Corner tests
                sub_adj_0(jacobian_adjugate, 1, sub_adjugate);

                for (int i = 0; i < 4; i++) {
                    subtet_gather(i, ux, sub_ux);
                    subtet_gather(i, uy, sub_uy);
                    subtet_gather(i, uz, sub_uz);

                    tet4_linear_elasticity_apply_kernel_opt(mu,
                                                            lambda,
                                                            sub_adjugate,
                                                            jacobian_determinant,
                                                            sub_ux,
                                                            sub_uy,
                                                            sub_uz,
                                                            sub_outx,
                                                            sub_outy,
                                                            sub_outz);

                    subtet_scatter_add(i, sub_outx, outx);
                    subtet_scatter_add(i, sub_outy, outy);
                    subtet_scatter_add(i, sub_outz, outz);
                }
            }

            {  // Octahedron tets
#if 1
                for (int i = 0; i < 4; i++) {
                    SubAdjFun sub_adj_fun = octahedron_adj_fun[i];

                    (*sub_adj_fun)(jacobian_adjugate, 1, sub_adjugate);

                    subtet_gather(4 + i, ux, sub_ux);
                    subtet_gather(4 + i, uy, sub_uy);
                    subtet_gather(4 + i, uz, sub_uz);

                    tet4_linear_elasticity_apply_kernel_opt(mu,
                                                            lambda,
                                                            sub_adjugate,
                                                            jacobian_determinant,
                                                            sub_ux,
                                                            sub_uy,
                                                            sub_uz,
                                                            sub_outx,
                                                            sub_outy,
                                                            sub_outz);

                    subtet_scatter_add(4 + i, sub_outx, outx);
                    subtet_scatter_add(4 + i, sub_outy, outy);
                    subtet_scatter_add(4 + i, sub_outz, outz);
                }

#else
               // For cuda use this

                // 4)
                sub_adj_4(jacobian_adjugate, 1, sub_adjugate);

                subtet_gather(4, ux, sub_ux);
                subtet_gather(4, uy, sub_uy);
                subtet_gather(4, uz, sub_uz);

                tet4_linear_elasticity_apply_kernel_opt(mu,
                                                        lambda,
                                                        sub_adjugate,
                                                        jacobian_determinant,
                                                        sub_ux,
                                                        sub_uy,
                                                        sub_uz,
                                                        sub_outx,
                                                        sub_outy,
                                                        sub_outz);

                subtet_scatter_add(4, sub_outx, outx);
                subtet_scatter_add(4, sub_outy, outy);
                subtet_scatter_add(4, sub_outz, outz);

                // 5)
                sub_adj_5(jacobian_adjugate, 1, sub_adjugate);

                subtet_gather(5, ux, sub_ux);
                subtet_gather(5, uy, sub_uy);
                subtet_gather(5, uz, sub_uz);

                tet4_linear_elasticity_apply_kernel_opt(mu,
                                                        lambda,
                                                        sub_adjugate,
                                                        jacobian_determinant,
                                                        sub_ux,
                                                        sub_uy,
                                                        sub_uz,
                                                        sub_outx,
                                                        sub_outy,
                                                        sub_outz);

                subtet_scatter_add(5, sub_outx, outx);
                subtet_scatter_add(5, sub_outy, outy);
                subtet_scatter_add(5, sub_outz, outz);

                // 6)
                sub_adj_6(jacobian_adjugate, 1, sub_adjugate);

                subtet_gather(6, ux, sub_ux);
                subtet_gather(6, uy, sub_uy);
                subtet_gather(6, uz, sub_uz);

                tet4_linear_elasticity_apply_kernel_opt(mu,
                                                        lambda,
                                                        sub_adjugate,
                                                        jacobian_determinant,
                                                        sub_ux,
                                                        sub_uy,
                                                        sub_uz,
                                                        sub_outx,
                                                        sub_outy,
                                                        sub_outz);

                subtet_scatter_add(6, sub_outx, outx);
                subtet_scatter_add(6, sub_outy, outy);
                subtet_scatter_add(6, sub_outz, outz);

                // 7)
                subtet_gather(7, ux, sub_ux);
                subtet_gather(7, uy, sub_uy);
                subtet_gather(7, uz, sub_uz);

                sub_adj_7(jacobian_adjugate, 1, sub_adjugate);
                tet4_linear_elasticity_apply_kernel_opt(mu,
                                                        lambda,
                                                        sub_adjugate,
                                                        jacobian_determinant,
                                                        sub_ux,
                                                        sub_uy,
                                                        sub_uz,
                                                        sub_outx,
                                                        sub_outy,
                                                        sub_outz);

                subtet_scatter_add(7, sub_outx, outx);
                subtet_scatter_add(7, sub_outy, outy);
                subtet_scatter_add(7, sub_outz, outz);
#endif
            }

            // up to here

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3] += outx[v];
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3 + 1] += outy[v];
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3 + 2] += outz[v];
            }
        }
    }
}

void macro_tet4_linear_elasticity_diag(const linear_elasticity_t *const ctx,
                                       real_t *const SFEM_RESTRICT diag) {
    //
    assert(0);
}



void macro_tet4_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elements,
                                            geom_t **const SFEM_RESTRICT points,
                                            const real_t mu,
                                            const real_t lambda,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT values) {

#if 0
    linear_elasticity_t ctx;
    macro_tet4_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
    macro_tet4_linear_elasticity_apply_opt(&ctx, u, values);
    macro_tet4_linear_elasticity_destroy(&ctx);
#else
    static linear_elasticity_t ctx;
    static int initialized = 0;

    if(!initialized) {
        macro_tet4_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
        initialized = 1;
    }

    macro_tet4_linear_elasticity_apply_opt(&ctx, u, values);
#endif
}
