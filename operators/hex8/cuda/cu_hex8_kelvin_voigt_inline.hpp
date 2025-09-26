#ifndef CU_HEX8_KELVIN_VOIGT_INLINE_HPP
#define CU_HEX8_KELVIN_VOIGT_INLINE_HPP

#include "cu_hex8_inline.hpp"

/*

//--------------------------
// Koperand
//--------------------------

// mundane ops: 71 divs: 0 sqrts: 0
// total ops: 71

const scalar_t x0 = (1.0/2.0)*k;
const scalar_t x1 = x0*(disp_grad[1] + disp_grad[3]);
const scalar_t x2 = x0*(disp_grad[2] + disp_grad[6]);
const scalar_t x3 = 3*k;
const scalar_t x4 = (3*K - k)*(disp_grad[0] + disp_grad[4] + disp_grad[8]);
const scalar_t x5 = (1.0/3.0)*disp_grad[0]*x3 + (1.0/3.0)*x4;
const scalar_t x6 = x0*(disp_grad[5] + disp_grad[7]);
const scalar_t x7 = (1.0/3.0)*disp_grad[4]*x3 + (1.0/3.0)*x4;
const scalar_t x8 = (1.0/3.0)*disp_grad[8]*x3 + (1.0/3.0)*x4;
K_tXJinv_t[0] = adjugate[0]*x5 + adjugate[1]*x1 + adjugate[2]*x2;
K_tXJinv_t[1] = adjugate[3]*x5 + adjugate[4]*x1 + adjugate[5]*x2;
K_tXJinv_t[2] = adjugate[6]*x5 + adjugate[7]*x1 + adjugate[8]*x2;
K_tXJinv_t[3] = adjugate[0]*x1 + adjugate[1]*x7 + adjugate[2]*x6;
K_tXJinv_t[4] = adjugate[3]*x1 + adjugate[4]*x7 + adjugate[5]*x6;
K_tXJinv_t[5] = adjugate[6]*x1 + adjugate[7]*x7 + adjugate[8]*x6;
K_tXJinv_t[6] = adjugate[0]*x2 + adjugate[1]*x6 + adjugate[2]*x8;
K_tXJinv_t[7] = adjugate[3]*x2 + adjugate[4]*x6 + adjugate[5]*x8;
K_tXJinv_t[8] = adjugate[6]*x2 + adjugate[7]*x6 + adjugate[8]*x8;
//--------------------------
// Coperand
//--------------------------

// mundane ops: 84 divs: 0 sqrts: 0
// total ops: 84

const scalar_t x0 = (1.0/2.0)*velo_grad[1] + (1.0/2.0)*velo_grad[3];
const scalar_t x1 = (1.0/2.0)*velo_grad[2] + (1.0/2.0)*velo_grad[6];
const scalar_t x2 = 0.33333333333333331*velo_grad[4];
const scalar_t x3 = 0.33333333333333331*velo_grad[8];
const scalar_t x4 = -0.66666666666666674*velo_grad[0] + x2 + x3;
const scalar_t x5 = (1.0/2.0)*velo_grad[5] + (1.0/2.0)*velo_grad[7];
const scalar_t x6 = 0.33333333333333331*velo_grad[0];
const scalar_t x7 = -0.66666666666666674*velo_grad[4] + x3 + x6;
const scalar_t x8 = -0.66666666666666674*velo_grad[8] + x2 + x6;
C_tXJinv_t[0] = eta*(-adjugate[0]*x4 + adjugate[1]*x0 + adjugate[2]*x1);
C_tXJinv_t[1] = eta*(-adjugate[3]*x4 + adjugate[4]*x0 + adjugate[5]*x1);
C_tXJinv_t[2] = eta*(-adjugate[6]*x4 + adjugate[7]*x0 + adjugate[8]*x1);
C_tXJinv_t[3] = eta*(adjugate[0]*x0 - adjugate[1]*x7 + adjugate[2]*x5);
C_tXJinv_t[4] = eta*(adjugate[3]*x0 - adjugate[4]*x7 + adjugate[5]*x5);
C_tXJinv_t[5] = eta*(adjugate[6]*x0 - adjugate[7]*x7 + adjugate[8]*x5);
C_tXJinv_t[6] = eta*(adjugate[0]*x1 + adjugate[1]*x5 - adjugate[2]*x8);
C_tXJinv_t[7] = eta*(adjugate[3]*x1 + adjugate[4]*x5 - adjugate[5]*x8);
C_tXJinv_t[8] = eta*(adjugate[6]*x1 + adjugate[7]*x5 - adjugate[8]*x8);
//--------------------------
// Moperand
//--------------------------

// mundane ops: 4 divs: 0 sqrts: 0
// total ops: 4

const scalar_t x0 = jacobian_determinant*rho;
M_load[0] = acce_vec[0]*x0;
M_load[1] = acce_vec[1]*x0;
M_load[2] = acce_vec[2]*x0;



*/

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_x(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qy;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qy * x1;
    const scalar_t x4 = qz * x0;
    const scalar_t x5 = qy * qz;
    out[0]            = -x2;
    out[1]            = x2;
    out[2]            = x3;
    out[3]            = -x3;
    out[4]            = -x4;
    out[5]            = x4;
    out[6]            = x5;
    out[7]            = -x5;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_y(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qx * x1;
    const scalar_t x4 = qz * x0;
    const scalar_t x5 = qx * qz;
    out[0]            = -x2;
    out[1]            = -x3;
    out[2]            = x3;
    out[3]            = x2;
    out[4]            = -x4;
    out[5]            = -x5;
    out[6]            = x5;
    out[7]            = x4;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_z(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qx * x1;
    const scalar_t x4 = qx * qy;
    const scalar_t x5 = qy * x0;
    out[0]            = -x2;
    out[1]            = -x3;
    out[2]            = -x4;
    out[3]            = -x5;
    out[4]            = x2;
    out[5]            = x3;
    out[6]            = x4;
    out[7]            = x5;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_fun(const scalar_t  qx,
                                                                const scalar_t  qy,
                                                                const scalar_t  qz,
                                                                scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = 1 - qz;
    const scalar_t x3 = x0 * x1;
    const scalar_t x4 = qx * x1;
    const scalar_t x5 = qx * qy;
    const scalar_t x6 = qy * x0;

    out[0] = x3 * x2;  // (1-qx)(1-qy)(1-qz)
    out[1] = x4 * x2;  // qx(1-qy)(1-qz)
    out[2] = x5 * x2;  // qx*qy*(1-qz)
    out[3] = x6 * x2;  // (1-qx)*qy*(1-qz)
    out[4] = x3 * qz;  // (1-qx)(1-qy)*qz
    out[5] = x4 * qz;  // qx(1-qy)*qz
    out[6] = x5 * qz;  // qx*qy*qz
    out[7] = x6 * qz;  // (1-qx)*qy*qz
}

template <typename scalar_t, typename accumulator_t>
static __host__ __device__ void cu_hex8_kelvin_voigt_apply_adj(const scalar_t                      k,
                                                               const scalar_t                      K,
                                                               const scalar_t                      eta,
                                                               const scalar_t                      rho,
                                                               const scalar_t *const SFEM_RESTRICT adjugate,
                                                               const scalar_t                      jacobian_determinant,
                                                               const scalar_t                      qx,
                                                               const scalar_t                      qy,
                                                               const scalar_t                      qz,
                                                               const scalar_t                      qw,
                                                               const scalar_t *const SFEM_RESTRICT ux,
                                                               const scalar_t *const SFEM_RESTRICT uy,
                                                               const scalar_t *const SFEM_RESTRICT uz,
                                                               const scalar_t *const SFEM_RESTRICT vx,
                                                               const scalar_t *const SFEM_RESTRICT vy,
                                                               const scalar_t *const SFEM_RESTRICT vz,
                                                               const scalar_t *const SFEM_RESTRICT ax,
                                                               const scalar_t *const SFEM_RESTRICT ay,
                                                               const scalar_t *const SFEM_RESTRICT az,
                                                               accumulator_t *const SFEM_RESTRICT  outx,
                                                               accumulator_t *const SFEM_RESTRICT  outy,
                                                               accumulator_t *const SFEM_RESTRICT  outz) {
    const scalar_t denom        = jacobian_determinant;
    scalar_t       disp_grad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    scalar_t       velo_grad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    scalar_t       acce_vec[3]  = {0, 0, 0};
    assert(denom > 0);
    {
        scalar_t temp_u[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        scalar_t grad[8];

        cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[0] += ux[i] * g;
            temp_u[3] += uy[i] * g;
            temp_u[6] += uz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[1] += ux[i] * g;
            temp_u[4] += uy[i] * g;
            temp_u[7] += uz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[2] += ux[i] * g;
            temp_u[5] += uy[i] * g;
            temp_u[8] += uz[i] * g;
        }
        for (int i = 0; i < 3; i++) {
            #pragma unroll
                    for (int j = 0; j < 3; j++) {
            #pragma unroll
                        for (int k = 0; k < 3; k++) {
                            disp_grad[i * 3 + j] += temp_u[i * 3 + k] * adjugate[k * 3 + j];
                            assert(disp_grad[i * 3 + j] == disp_grad[i * 3 + j]);
                        }
                    }
                }
    }

    {
        scalar_t temp_v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        scalar_t grad[8];

        cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[0] += vx[i] * g;
            temp_v[3] += vy[i] * g;
            temp_v[6] += vz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[1] += vx[i] * g;
            temp_v[4] += vy[i] * g;
            temp_v[7] += vz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[2] += vx[i] * g;
            temp_v[5] += vy[i] * g;
            temp_v[8] += vz[i] * g;
        }
        for (int i = 0; i < 3; i++) {
            #pragma unroll
                    for (int j = 0; j < 3; j++) {
            #pragma unroll
                        for (int k = 0; k < 3; k++) {
                            velo_grad[i * 3 + j] += temp_v[i * 3 + k] * adjugate[k * 3 + j];
                            assert(velo_grad[i * 3 + j] == velo_grad[i * 3 + j]);
                        }
                    }
                }
    }

scalar_t *K_tXJinv_t = disp_grad;
{
    const scalar_t x0 = (1.0 / 2.0) * k;
    const scalar_t x1 = x0 * (disp_grad[1] + disp_grad[3]);
    const scalar_t x2 = x0 * (disp_grad[2] + disp_grad[6]);
    const scalar_t x3 = 3 * k;
    const scalar_t x4 = (3 * K - k) * (disp_grad[0] + disp_grad[4] + disp_grad[8]);
    const scalar_t x5 = (1.0 / 3.0) * disp_grad[0] * x3 + (1.0 / 3.0) * x4;
    const scalar_t x6 = x0 * (disp_grad[5] + disp_grad[7]);
    const scalar_t x7 = (1.0 / 3.0) * disp_grad[4] * x3 + (1.0 / 3.0) * x4;
    const scalar_t x8 = (1.0 / 3.0) * disp_grad[8] * x3 + (1.0 / 3.0) * x4;
    K_tXJinv_t[0]     = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
    K_tXJinv_t[1]     = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
    K_tXJinv_t[2]     = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
    K_tXJinv_t[3]     = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
    K_tXJinv_t[4]     = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
    K_tXJinv_t[5]     = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
    K_tXJinv_t[6]     = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
    K_tXJinv_t[7]     = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
    K_tXJinv_t[8]     = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
}

scalar_t *C_tXJinv_t = velo_grad;
{
    const scalar_t x0 = (1.0 / 2.0) * velo_grad[1] + (1.0 / 2.0) * velo_grad[3];
    const scalar_t x1 = (1.0 / 2.0) * velo_grad[2] + (1.0 / 2.0) * velo_grad[6];
    const scalar_t x2 = 0.33333333333333331 * velo_grad[4];
    const scalar_t x3 = 0.33333333333333331 * velo_grad[8];
    const scalar_t x4 = -0.66666666666666674 * velo_grad[0] + x2 + x3;
    const scalar_t x5 = (1.0 / 2.0) * velo_grad[5] + (1.0 / 2.0) * velo_grad[7];
    const scalar_t x6 = 0.33333333333333331 * velo_grad[0];
    const scalar_t x7 = -0.66666666666666674 * velo_grad[4] + x3 + x6;
    const scalar_t x8 = -0.66666666666666674 * velo_grad[8] + x2 + x6;
    C_tXJinv_t[0]     = eta * (-adjugate[0] * x4 + adjugate[1] * x0 + adjugate[2] * x1);
    C_tXJinv_t[1]     = eta * (-adjugate[3] * x4 + adjugate[4] * x0 + adjugate[5] * x1);
    C_tXJinv_t[2]     = eta * (-adjugate[6] * x4 + adjugate[7] * x0 + adjugate[8] * x1);
    C_tXJinv_t[3]     = eta * (adjugate[0] * x0 - adjugate[1] * x7 + adjugate[2] * x5);
    C_tXJinv_t[4]     = eta * (adjugate[3] * x0 - adjugate[4] * x7 + adjugate[5] * x5);
    C_tXJinv_t[5]     = eta * (adjugate[6] * x0 - adjugate[7] * x7 + adjugate[8] * x5);
    C_tXJinv_t[6]     = eta * (adjugate[0] * x1 + adjugate[1] * x5 - adjugate[2] * x8);
    C_tXJinv_t[7]     = eta * (adjugate[3] * x1 + adjugate[4] * x5 - adjugate[5] * x8);
    C_tXJinv_t[8]     = eta * (adjugate[6] * x1 + adjugate[7] * x5 - adjugate[8] * x8);
}

// Scale by quadrature weight and combine K and C
scalar_t P_tXJinv_t[9];
for (int i = 0; i < 9; i++) {
    P_tXJinv_t[i] = (K_tXJinv_t[i] + C_tXJinv_t[i]) * (qw / denom);
    assert(P_tXJinv_t[i] == P_tXJinv_t[i]);
}

{
    scalar_t grad[8];
    cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        scalar_t g = grad[i];
        outx[i] += P_tXJinv_t[0] * g;
        outy[i] += P_tXJinv_t[3] * g;
        outz[i] += P_tXJinv_t[6] * g;
    }

    cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        scalar_t g = grad[i];
        outx[i] += P_tXJinv_t[1] * g;
        outy[i] += P_tXJinv_t[4] * g;
        outz[i] += P_tXJinv_t[7] * g;
    }

    cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        scalar_t g = grad[i];
        outx[i] += P_tXJinv_t[2] * g;
        outy[i] += P_tXJinv_t[5] * g;
        outz[i] += P_tXJinv_t[8] * g;
    }
}

#pragma unroll
{
    // Inertia contribution: interpolate acceleration, then distribute with shape values
    scalar_t shape[8];
    cu_hex8_kv_ref_shape_fun(qx, qy, qz, shape);

    for (int i = 0; i < 8; i++) {
        const scalar_t Ni = shape[i];
        acce_vec[0] += ax[i] * Ni;
        acce_vec[1] += ay[i] * Ni;
        acce_vec[2] += az[i] * Ni;
    }

    const scalar_t mscale = rho * denom * qw;  // rho * detJ * qw

    for (int i = 0; i < 8; i++) {
        const scalar_t Ni = shape[i];
        outx[i] += mscale * acce_vec[0] * Ni;
        outy[i] += mscale * acce_vec[1] * Ni;
        outz[i] += mscale * acce_vec[2] * Ni;
    }
}

#ifndef NDEBUG
for (int i = 0; i < 8; i++) {
    assert(outx[i] == outx[i]);
    assert(outy[i] == outy[i]);
    assert(outz[i] == outz[i]);
}
#endif
}

#endif  // CU_HEX8_KELVIN_VOIGT_INLINE_HPP