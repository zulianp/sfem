#include "tet10_linear_elasticity.h"
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

static SFEM_INLINE void tet10_linear_elasticity_apply_kernel_opt(
    const real_t mu,
    const real_t lambda,
    const jacobian_t *const SFEM_RESTRICT adjugate,
    const jacobian_t jacobian_determinant,
    const real_t qx,
    const real_t qy,
    const real_t qz,
    const real_t qw,
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_vector) {
    real_t disp_grad[9];
    {
        const real_t x0 = 1.0 / jacobian_determinant;
        const real_t x1 = 4 * qx;
        const real_t x2 = x1 - 1;
        const real_t x3 = 4 * qy;
        const real_t x4 = -u[6] * x3;
        const real_t x5 = qz - 1;
        const real_t x6 = 8 * qx + 4 * qy + 4 * x5;
        const real_t x7 = 4 * qz;
        const real_t x8 = x1 + x3 + x7 - 3;
        const real_t x9 = u[0] * x8;
        const real_t x10 = -u[7] * x7 + x9;
        const real_t x11 = u[1] * x2 - u[4] * x6 + u[5] * x3 + u[8] * x7 + x10 + x4;
        const real_t x12 = x3 - 1;
        const real_t x13 = -u[4] * x1;
        const real_t x14 = 4 * qx + 8 * qy + 4 * x5;
        const real_t x15 = u[2] * x12 + u[5] * x1 - u[6] * x14 + u[9] * x7 + x10 + x13;
        const real_t x16 = x7 - 1;
        const real_t x17 = 4 * qx + 4 * qy + 8 * qz - 4;
        const real_t x18 = u[3] * x16 - u[7] * x17 + u[8] * x1 + u[9] * x3 + x13 + x4 + x9;
        const real_t x19 = -u[16] * x3;
        const real_t x20 = u[10] * x8;
        const real_t x21 = -u[17] * x7 + x20;
        const real_t x22 = u[11] * x2 - u[14] * x6 + u[15] * x3 + u[18] * x7 + x19 + x21;
        const real_t x23 = -u[14] * x1;
        const real_t x24 = u[12] * x12 + u[15] * x1 - u[16] * x14 + u[19] * x7 + x21 + x23;
        const real_t x25 = u[13] * x16 - u[17] * x17 + u[18] * x1 + u[19] * x3 + x19 + x20 + x23;
        const real_t x26 = -u[26] * x3;
        const real_t x27 = u[20] * x8;
        const real_t x28 = -u[27] * x7 + x27;
        const real_t x29 = u[21] * x2 - u[24] * x6 + u[25] * x3 + u[28] * x7 + x26 + x28;
        const real_t x30 = -u[24] * x1;
        const real_t x31 = u[22] * x12 + u[25] * x1 - u[26] * x14 + u[29] * x7 + x28 + x30;
        const real_t x32 = u[23] * x16 - u[27] * x17 + u[28] * x1 + u[29] * x3 + x26 + x27 + x30;
        disp_grad[0] = x0 * (adjugate[0] * x11 + adjugate[3] * x15 + adjugate[6] * x18);
        disp_grad[1] = x0 * (adjugate[1] * x11 + adjugate[4] * x15 + adjugate[7] * x18);
        disp_grad[2] = x0 * (adjugate[2] * x11 + adjugate[5] * x15 + adjugate[8] * x18);
        disp_grad[3] = x0 * (adjugate[0] * x22 + adjugate[3] * x24 + adjugate[6] * x25);
        disp_grad[4] = x0 * (adjugate[1] * x22 + adjugate[4] * x24 + adjugate[7] * x25);
        disp_grad[5] = x0 * (adjugate[2] * x22 + adjugate[5] * x24 + adjugate[8] * x25);
        disp_grad[6] = x0 * (adjugate[0] * x29 + adjugate[3] * x31 + adjugate[6] * x32);
        disp_grad[7] = x0 * (adjugate[1] * x29 + adjugate[4] * x31 + adjugate[7] * x32);
        disp_grad[8] = x0 * (adjugate[2] * x29 + adjugate[5] * x31 + adjugate[8] * x32);
    }

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

    for (int i = 0; i < 9; i++) {
        P[i] *= qw;
    }

    {
        const real_t x0 = 4 * qx;
        const real_t x1 = 4 * qy;
        const real_t x2 = 4 * qz;
        const real_t x3 = x0 + x1 + x2 - 3;
        const real_t x4 = adjugate[0] + adjugate[3] + adjugate[6];
        const real_t x5 = adjugate[1] + adjugate[4] + adjugate[7];
        const real_t x6 = adjugate[2] + adjugate[5] + adjugate[8];
        const real_t x7 = x0 - 1;
        const real_t x8 = x1 - 1;
        const real_t x9 = x2 - 1;
        const real_t x10 = adjugate[3] * qx;
        const real_t x11 = adjugate[6] * qx;
        const real_t x12 = qz - 1;
        const real_t x13 = 2 * qx + qy + x12;
        const real_t x14 = adjugate[0] * x13 + x10 + x11;
        const real_t x15 = 4 * P[0];
        const real_t x16 = adjugate[4] * qx;
        const real_t x17 = adjugate[7] * qx;
        const real_t x18 = adjugate[1] * x13 + x16 + x17;
        const real_t x19 = 4 * P[1];
        const real_t x20 = adjugate[5] * qx;
        const real_t x21 = adjugate[8] * qx;
        const real_t x22 = adjugate[2] * x13 + x20 + x21;
        const real_t x23 = 4 * P[2];
        const real_t x24 = adjugate[0] * qy;
        const real_t x25 = x10 + x24;
        const real_t x26 = adjugate[1] * qy;
        const real_t x27 = x16 + x26;
        const real_t x28 = adjugate[2] * qy;
        const real_t x29 = x20 + x28;
        const real_t x30 = adjugate[6] * qy;
        const real_t x31 = qx + 2 * qy + x12;
        const real_t x32 = adjugate[3] * x31 + x24 + x30;
        const real_t x33 = adjugate[7] * qy;
        const real_t x34 = adjugate[4] * x31 + x26 + x33;
        const real_t x35 = adjugate[8] * qy;
        const real_t x36 = adjugate[5] * x31 + x28 + x35;
        const real_t x37 = adjugate[0] * qz;
        const real_t x38 = adjugate[3] * qz;
        const real_t x39 = qx + qy + 2 * qz - 1;
        const real_t x40 = adjugate[6] * x39 + x37 + x38;
        const real_t x41 = adjugate[1] * qz;
        const real_t x42 = adjugate[4] * qz;
        const real_t x43 = adjugate[7] * x39 + x41 + x42;
        const real_t x44 = adjugate[2] * qz;
        const real_t x45 = adjugate[5] * qz;
        const real_t x46 = adjugate[8] * x39 + x44 + x45;
        const real_t x47 = x11 + x37;
        const real_t x48 = x17 + x41;
        const real_t x49 = x21 + x44;
        const real_t x50 = x30 + x38;
        const real_t x51 = x33 + x42;
        const real_t x52 = x35 + x45;
        const real_t x53 = 4 * P[3];
        const real_t x54 = 4 * P[4];
        const real_t x55 = 4 * P[5];
        const real_t x56 = 4 * P[6];
        const real_t x57 = 4 * P[7];
        const real_t x58 = 4 * P[8];

        element_vector[0] += x3 * (P[0] * x4 + P[1] * x5 + P[2] * x6);
        element_vector[1] += x7 * (P[0] * adjugate[0] + P[1] * adjugate[1] + P[2] * adjugate[2]);
        element_vector[2] += x8 * (P[0] * adjugate[3] + P[1] * adjugate[4] + P[2] * adjugate[5]);
        element_vector[3] += x9 * (P[0] * adjugate[6] + P[1] * adjugate[7] + P[2] * adjugate[8]);
        element_vector[4] += -x14 * x15 - x18 * x19 - x22 * x23;
        element_vector[5] += x15 * x25 + x19 * x27 + x23 * x29;
        element_vector[6] += -x15 * x32 - x19 * x34 - x23 * x36;
        element_vector[7] += -x15 * x40 - x19 * x43 - x23 * x46;
        element_vector[8] += x15 * x47 + x19 * x48 + x23 * x49;
        element_vector[9] += x15 * x50 + x19 * x51 + x23 * x52;
        element_vector[10] += x3 * (P[3] * x4 + P[4] * x5 + P[5] * x6);
        element_vector[11] += x7 * (P[3] * adjugate[0] + P[4] * adjugate[1] + P[5] * adjugate[2]);
        element_vector[12] += x8 * (P[3] * adjugate[3] + P[4] * adjugate[4] + P[5] * adjugate[5]);
        element_vector[13] += x9 * (P[3] * adjugate[6] + P[4] * adjugate[7] + P[5] * adjugate[8]);
        element_vector[14] += -x14 * x53 - x18 * x54 - x22 * x55;
        element_vector[15] += x25 * x53 + x27 * x54 + x29 * x55;
        element_vector[16] += -x32 * x53 - x34 * x54 - x36 * x55;
        element_vector[17] += -x40 * x53 - x43 * x54 - x46 * x55;
        element_vector[18] += x47 * x53 + x48 * x54 + x49 * x55;
        element_vector[19] += x50 * x53 + x51 * x54 + x52 * x55;
        element_vector[20] += x3 * (P[6] * x4 + P[7] * x5 + P[8] * x6);
        element_vector[21] += x7 * (P[6] * adjugate[0] + P[7] * adjugate[1] + P[8] * adjugate[2]);
        element_vector[22] += x8 * (P[6] * adjugate[3] + P[7] * adjugate[4] + P[8] * adjugate[5]);
        element_vector[23] += x9 * (P[6] * adjugate[6] + P[7] * adjugate[7] + P[8] * adjugate[8]);
        element_vector[24] += -x14 * x56 - x18 * x57 - x22 * x58;
        element_vector[25] += x25 * x56 + x27 * x57 + x29 * x58;
        element_vector[26] += -x32 * x56 - x34 * x57 - x36 * x58;
        element_vector[27] += -x40 * x56 - x43 * x57 - x46 * x58;
        element_vector[28] += x47 * x56 + x48 * x57 + x49 * x58;
        element_vector[29] += x50 * x56 + x51 * x57 + x52 * x58;
    }

}

static const int n_qp = 8;
static const real_t qx[8] =
    {0.0, 1.0, 0.0, 0.0, 0.333333333333, 0.333333333333, 0.0, 0.333333333333};

static const real_t qy[8] =
    {0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.0, 0.333333333333, 0.333333333333};

static const real_t qz[8] =
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.333333333333, 0.333333333333};

static const real_t qw[8] = {0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225};

void tet10_linear_elasticity_init(linear_elasticity_t *const ctx,
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
    ctx->element_type = TET10;
}

void tet10_linear_elasticity_destroy(linear_elasticity_t *const ctx) {
    free(ctx->jacobian_adjugate);
    free(ctx->jacobian_determinant);

    ctx->jacobian_adjugate = 0;
    ctx->jacobian_determinant = 0;

    ctx->elements = 0;
    ctx->nelements = 0;
    ctx->element_type = INVALID;
}

void tet10_linear_elasticity_apply_opt(const linear_elasticity_t *const ctx,
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

            real_t element_u[30];
            real_t element_vector[30];

            const jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[i * 9];
            const jacobian_t jacobian_determinant = g_jacobian_determinant[i];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = ctx->elements[v][i];
            }

            for (int v = 0; v < 10; ++v) {
                element_u[v] = u[ev[v] * 3];
                element_u[10 + v] = u[ev[v] * 3 + 1];
                element_u[20 + v] = u[ev[v] * 3 + 2];
            }

            for (int k = 0; k < n_qp; k++) {
                tet10_linear_elasticity_apply_kernel_opt(mu,
                                                         lambda,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         qx[k],
                                                         qy[k],
                                                         qz[k],
                                                         qw[k],
                                                         element_u,
                                                         element_vector);
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3] += element_vector[v];
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3 + 1] += element_vector[10 + v];
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v] * 3 + 2] += element_vector[20 + v];
            }
        }
    }
}

void tet10_linear_elasticity_diag(const linear_elasticity_t *const ctx,
                                  real_t *const SFEM_RESTRICT diag) {
    //
    assert(0);
}

void tet10_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t mu,
                                       const real_t lambda,
                                       const real_t *const SFEM_RESTRICT u,
                                       real_t *const SFEM_RESTRICT values) {
#if 0
    linear_elasticity_t ctx;
    tet10_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
    tet10_linear_elasticity_apply_opt(&ctx, u, values);
    tet10_linear_elasticity_destroy(&ctx);
#else
    static linear_elasticity_t ctx;
    static int initialized = 0;

    if (!initialized) {
        tet10_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
        initialized = 1;
    }

    tet10_linear_elasticity_apply_opt(&ctx, u, values);
#endif
}
