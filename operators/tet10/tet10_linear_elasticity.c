#include "tet10_linear_elasticity.h"
#include "sfem_base.h"

#include "tet10_inline_cpu.h"
#include "tet10_linear_elasticity_inline_cpu.h"

#include <stddef.h>

static SFEM_INLINE void tet10_linear_elasticity_apply_adj(const scalar_t mu,
                                                          const scalar_t lambda,
                                                          const scalar_t *const SFEM_RESTRICT
                                                                  adjugate,
                                                          const scalar_t jacobian_determinant,
                                                          const scalar_t qx,
                                                          const scalar_t qy,
                                                          const scalar_t qz,
                                                          const scalar_t qw,
                                                          const scalar_t *const SFEM_RESTRICT ux,
                                                          const scalar_t *const SFEM_RESTRICT uy,
                                                          const scalar_t *const SFEM_RESTRICT uz,
                                                          accumulator_t *const SFEM_RESTRICT outx,
                                                          accumulator_t *const SFEM_RESTRICT outy,
                                                          accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t disp_grad[9] = {0};

#define MICRO_KERNEL_USE_CODEGEN 1

#if MICRO_KERNEL_USE_CODEGEN
    // Code-gen way

    const scalar_t denom = 1;
    {
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = 4 * qx;
        const scalar_t x2 = x1 - 1;
        const scalar_t x3 = 4 * qy;
        const scalar_t x4 = -ux[6] * x3;
        const scalar_t x5 = qz - 1;
        const scalar_t x6 = 8 * qx + 4 * qy + 4 * x5;
        const scalar_t x7 = 4 * qz;
        const scalar_t x8 = x1 + x3 + x7 - 3;
        const scalar_t x9 = ux[0] * x8;
        const scalar_t x10 = -ux[7] * x7 + x9;
        const scalar_t x11 = ux[1] * x2 - ux[4] * x6 + ux[5] * x3 + ux[8] * x7 + x10 + x4;
        const scalar_t x12 = x3 - 1;
        const scalar_t x13 = -ux[4] * x1;
        const scalar_t x14 = 4 * qx + 8 * qy + 4 * x5;
        const scalar_t x15 = ux[2] * x12 + ux[5] * x1 - ux[6] * x14 + ux[9] * x7 + x10 + x13;
        const scalar_t x16 = x7 - 1;
        const scalar_t x17 = 4 * qx + 4 * qy + 8 * qz - 4;
        const scalar_t x18 = ux[3] * x16 - ux[7] * x17 + ux[8] * x1 + ux[9] * x3 + x13 + x4 + x9;
        const scalar_t x19 = -uy[6] * x3;
        const scalar_t x20 = uy[0] * x8;
        const scalar_t x21 = -uy[7] * x7 + x20;
        const scalar_t x22 = uy[1] * x2 - uy[4] * x6 + uy[5] * x3 + uy[8] * x7 + x19 + x21;
        const scalar_t x23 = -uy[4] * x1;
        const scalar_t x24 = uy[2] * x12 + uy[5] * x1 - uy[6] * x14 + uy[9] * x7 + x21 + x23;
        const scalar_t x25 = uy[3] * x16 - uy[7] * x17 + uy[8] * x1 + uy[9] * x3 + x19 + x20 + x23;
        const scalar_t x26 = -uz[6] * x3;
        const scalar_t x27 = uz[0] * x8;
        const scalar_t x28 = -uz[7] * x7 + x27;
        const scalar_t x29 = uz[1] * x2 - uz[4] * x6 + uz[5] * x3 + uz[8] * x7 + x26 + x28;
        const scalar_t x30 = -uz[4] * x1;
        const scalar_t x31 = uz[2] * x12 + uz[5] * x1 - uz[6] * x14 + uz[9] * x7 + x28 + x30;
        const scalar_t x32 = uz[3] * x16 - uz[7] * x17 + uz[8] * x1 + uz[9] * x3 + x26 + x27 + x30;
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
#else
    // Programmatic way

    const scalar_t denom = jacobian_determinant;
    {
        scalar_t temp[9] = {0};
        scalar_t grad[10];

        tet10_ref_shape_grad_x(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[0] += u[i] * g;
            temp[3] += u[10 + i] * g;
            temp[6] += u[20 + i] * g;
        }

        tet10_ref_shape_grad_y(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[1] += u[i] * g;
            temp[4] += u[10 + i] * g;
            temp[7] += u[20 + i] * g;
        }

        tet10_ref_shape_grad_z(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[2] += u[i] * g;
            temp[5] += u[10 + i] * g;
            temp[8] += u[20 + i] * g;
        }

        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
#pragma unroll
                for (int k = 0; k < 3; k++) {
                    disp_grad[i * 3 + j] += temp[i * 3 + k] * adjugate[k * 3 + j];
                }
            }
        }
    }

#endif
    // Includes first Piola-Kirchoff stress: P^T * J^-T * det(J)

    scalar_t *P_tXJinv_t = disp_grad;
    {
        const scalar_t x0 = (1.0 / 6.0) * mu;
        const scalar_t x1 = x0 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x2 = x0 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x3 = 2 * mu;
        const scalar_t x4 = lambda * (disp_grad[0] + disp_grad[4] + disp_grad[8]);
        const scalar_t x5 = (1.0 / 6.0) * disp_grad[0] * x3 + (1.0 / 6.0) * x4;
        const scalar_t x6 = x0 * (disp_grad[5] + disp_grad[7]);
        const scalar_t x7 = (1.0 / 6.0) * disp_grad[4] * x3 + (1.0 / 6.0) * x4;
        const scalar_t x8 = (1.0 / 6.0) * disp_grad[8] * x3 + (1.0 / 6.0) * x4;
        P_tXJinv_t[0] = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
        P_tXJinv_t[1] = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
        P_tXJinv_t[2] = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
        P_tXJinv_t[3] = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
        P_tXJinv_t[4] = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
        P_tXJinv_t[5] = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
        P_tXJinv_t[6] = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
        P_tXJinv_t[7] = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
        P_tXJinv_t[8] = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
    }

    // Scale by quadrature weight
    for (int i = 0; i < 9; i++) {
        P_tXJinv_t[i] *= qw / denom;
    }

// On CPU both versions are equivalent
#if MICRO_KERNEL_USE_CODEGEN
    {
        const scalar_t x0 = 4 * qx;
        const scalar_t x1 = 4 * qy;
        const scalar_t x2 = 4 * qz;
        const scalar_t x3 = x0 + x1 + x2 - 3;
        const scalar_t x4 = x0 - 1;
        const scalar_t x5 = x1 - 1;
        const scalar_t x6 = x2 - 1;
        const scalar_t x7 = P_tXJinv_t[1] * x0;
        const scalar_t x8 = P_tXJinv_t[2] * x0;
        const scalar_t x9 = qz - 1;
        const scalar_t x10 = 8 * qx + 4 * qy + 4 * x9;
        const scalar_t x11 = P_tXJinv_t[0] * x1;
        const scalar_t x12 = P_tXJinv_t[2] * x1;
        const scalar_t x13 = 4 * qx + 8 * qy + 4 * x9;
        const scalar_t x14 = P_tXJinv_t[0] * x2;
        const scalar_t x15 = P_tXJinv_t[1] * x2;
        const scalar_t x16 = 4 * qx + 4 * qy + 8 * qz - 4;
        const scalar_t x17 = P_tXJinv_t[4] * x0;
        const scalar_t x18 = P_tXJinv_t[5] * x0;
        const scalar_t x19 = P_tXJinv_t[3] * x1;
        const scalar_t x20 = P_tXJinv_t[5] * x1;
        const scalar_t x21 = P_tXJinv_t[3] * x2;
        const scalar_t x22 = P_tXJinv_t[4] * x2;
        const scalar_t x23 = P_tXJinv_t[7] * x0;
        const scalar_t x24 = P_tXJinv_t[8] * x0;
        const scalar_t x25 = P_tXJinv_t[6] * x1;
        const scalar_t x26 = P_tXJinv_t[8] * x1;
        const scalar_t x27 = P_tXJinv_t[6] * x2;
        const scalar_t x28 = P_tXJinv_t[7] * x2;
        outx[0] += x3 * (P_tXJinv_t[0] + P_tXJinv_t[1] + P_tXJinv_t[2]);
        outx[1] += P_tXJinv_t[0] * x4;
        outx[2] += P_tXJinv_t[1] * x5;
        outx[3] += P_tXJinv_t[2] * x6;
        outx[4] += -P_tXJinv_t[0] * x10 - x7 - x8;
        outx[5] += x11 + x7;
        outx[6] += -P_tXJinv_t[1] * x13 - x11 - x12;
        outx[7] += -P_tXJinv_t[2] * x16 - x14 - x15;
        outx[8] += x14 + x8;
        outx[9] += x12 + x15;

        outy[0] += x3 * (P_tXJinv_t[3] + P_tXJinv_t[4] + P_tXJinv_t[5]);
        outy[1] += P_tXJinv_t[3] * x4;
        outy[2] += P_tXJinv_t[4] * x5;
        outy[3] += P_tXJinv_t[5] * x6;
        outy[4] += -P_tXJinv_t[3] * x10 - x17 - x18;
        outy[5] += x17 + x19;
        outy[6] += -P_tXJinv_t[4] * x13 - x19 - x20;
        outy[7] += -P_tXJinv_t[5] * x16 - x21 - x22;
        outy[8] += x18 + x21;
        outy[9] += x20 + x22;

        outz[0] += x3 * (P_tXJinv_t[6] + P_tXJinv_t[7] + P_tXJinv_t[8]);
        outz[1] += P_tXJinv_t[6] * x4;
        outz[2] += P_tXJinv_t[7] * x5;
        outz[3] += P_tXJinv_t[8] * x6;
        outz[4] += -P_tXJinv_t[6] * x10 - x23 - x24;
        outz[5] += x23 + x25;
        outz[6] += -P_tXJinv_t[7] * x13 - x25 - x26;
        outz[7] += -P_tXJinv_t[8] * x16 - x27 - x28;
        outz[8] += x24 + x27;
        outz[9] += x26 + x28;
    }

#else

    {
        scalar_t grad[10];
        tet10_ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[0] * g;
            outy[i] += P_tXJinv_t[3] * g;
            outz[i] += P_tXJinv_t[6] * g;
        }

        tet10_ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[1] * g;
            outy[i] += P_tXJinv_t[4] * g;
            outz[i] += P_tXJinv_t[7] * g;
        }

        tet10_ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[2] * g;
            outy[i] += P_tXJinv_t[5] * g;
            outz[i] += P_tXJinv_t[8] * g;
        }
    }

#endif

#undef MICRO_KERNEL_USE_CODEGEN
}

static const int n_qp = 8;
static const scalar_t qx[8] =
        {0.0, 1.0, 0.0, 0.0, 0.333333333333, 0.333333333333, 0.0, 0.333333333333};

static const scalar_t qy[8] =
        {0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.0, 0.333333333333, 0.333333333333};

static const scalar_t qz[8] =
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.333333333333, 0.333333333333};

static const scalar_t qw[8] = {0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225};

int tet10_linear_elasticity_apply(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t mu,
                                  const real_t lambda,
                                  const ptrdiff_t u_stride,
                                  const real_t *const SFEM_RESTRICT ux,
                                  const real_t *const SFEM_RESTRICT uy,
                                  const real_t *const SFEM_RESTRICT uz,
                                  const ptrdiff_t out_stride,
                                  real_t *const SFEM_RESTRICT outx,
                                  real_t *const SFEM_RESTRICT outy,
                                  real_t *const SFEM_RESTRICT outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
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
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_apply_adj(mu,
                                              lambda,
                                              jacobian_adjugate,
                                              jacobian_determinant,
                                              qx[k],
                                              qy[k],
                                              qz[k],
                                              qw[k],
                                              element_ux,
                                              element_uy,
                                              element_uz,
                                              element_outx,
                                              element_outy,
                                              element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_diag(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t out_stride,
                                 real_t *const SFEM_RESTRICT outx,
                                 real_t *const SFEM_RESTRICT outy,
                                 real_t *const SFEM_RESTRICT outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
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
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_diag_adj(jacobian_adjugate,
                                             jacobian_determinant,
                                             mu,
                                             lambda,
                                             qx[k],
                                             qy[k],
                                             qz[k],
                                             qw[k],
                                             element_outx,
                                             element_outy,
                                             element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }
    return 0;
}

int tet10_linear_elasticity_apply_opt(const ptrdiff_t nelements,
                                      idx_t **const SFEM_RESTRICT elements,
                                      const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                      const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                      const real_t mu,
                                      const real_t lambda,
                                      const ptrdiff_t u_stride,
                                      const real_t *const SFEM_RESTRICT ux,
                                      const real_t *const SFEM_RESTRICT uy,
                                      const real_t *const SFEM_RESTRICT uz,
                                      const ptrdiff_t out_stride,
                                      real_t *const SFEM_RESTRICT outx,
                                      real_t *const SFEM_RESTRICT outy,
                                      real_t *const SFEM_RESTRICT outz) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        const scalar_t jacobian_determinant = g_jacobian_determinant[i];
        scalar_t jacobian_adjugate[9];
        for(int k = 0; k < 9; k++) {
            jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_apply_adj(mu,
                                              lambda,
                                              jacobian_adjugate,
                                              jacobian_determinant,
                                              qx[k],
                                              qy[k],
                                              qz[k],
                                              qw[k],
                                              element_ux,
                                              element_uy,
                                              element_uz,
                                              element_outx,
                                              element_outy,
                                              element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_diag_opt(const ptrdiff_t nelements,
                                     idx_t **const SFEM_RESTRICT elements,
                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                     const real_t mu,
                                     const real_t lambda,
                                     const ptrdiff_t out_stride,
                                     real_t *const SFEM_RESTRICT outx,
                                     real_t *const SFEM_RESTRICT outy,
                                     real_t *const SFEM_RESTRICT outz) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        const scalar_t jacobian_determinant = g_jacobian_determinant[i];
        scalar_t jacobian_adjugate[9];
        for(int k = 0; k < 9; k++) {
            jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_diag_adj(jacobian_adjugate,
                                             jacobian_determinant,
                                             mu,
                                             lambda,
                                             qx[k],
                                             qy[k],
                                             qz[k],
                                             qw[k],
                                             element_outx,
                                             element_outy,
                                             element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_hessian(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    const real_t mu,
                                    const real_t lambda,
                                    const count_t *const SFEM_RESTRICT rowptr,
                                    const idx_t *const SFEM_RESTRICT colidx,
                                    real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        idx_t ks[10];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        accumulator_t element_matrix[(10 * 3) * (10 * 3)] = {0};

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
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
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_hessian_adj(qx[k],
                                                qy[k],
                                                qz[k],
                                                qw[k],
                                                jacobian_adjugate,
                                                jacobian_determinant,
                                                mu,
                                                lambda,
                                                element_matrix);
        }

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = elements[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            {
                const idx_t *row = &colidx[rowptr[dof_i]];
                tet10_find_cols(ev, row, lenrow, ks);
            }

            // Blocks for row
            real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < 10; ++edof_j) {
                const idx_t offset_j = ks[edof_j] * block_size;

                for (int bi = 0; bi < block_size; ++bi) {
                    const int ii = bi * 10 + edof_i;

                    // Jump rows (including the block-size for the columns)
                    real_t *row = &block_start[bi * lenrow * block_size];

                    for (int bj = 0; bj < block_size; ++bj) {
                        const int jj = bj * 10 + edof_j;
                        const real_t val = element_matrix[ii * 30 + jj];

#pragma omp atomic update
                        row[offset_j + bj] += val;
                    }
                }
            }
        }
    }

    return 0;
}
