#ifndef TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H
#define TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H

#include "tet4_linear_elasticity_inline_cpu.h"  // tet4_gradient_3

// This can be used with any element type
static SFEM_INLINE void neohookean_P_tXJinv_t_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                                  const scalar_t jacobian_determinant,
                                                  const scalar_t mu,
                                                  const scalar_t lambda,
                                                  scalar_t *const SFEM_RESTRICT
                                                          F_in_P_tXJinv_t_out) {
    const scalar_t *const F = F_in_P_tXJinv_t_out;

    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = F[4] * F[8];
    const scalar_t x2 = F[5] * F[7];
    const scalar_t x3 = x1 - x2;
    const scalar_t x4 = F[1] * F[5];
    const scalar_t x5 = F[1] * F[8];
    const scalar_t x6 = F[2] * F[4];
    const scalar_t x7 =
            F[0] * x1 - F[0] * x2 + F[2] * F[3] * F[7] - F[3] * x5 + F[6] * x4 - F[6] * x6;
    const scalar_t x8 = 1.0 / x7;
    const scalar_t x9 = mu * x8;
    const scalar_t x10 = lambda * x8 * log(x7);
    const scalar_t x11 = x0 * (F[0] * mu + x10 * x3 - x3 * x9);
    const scalar_t x12 = F[2] * F[7] - x5;
    const scalar_t x13 = x0 * (F[3] * mu + x10 * x12 - x12 * x9);
    const scalar_t x14 = x4 - x6;
    const scalar_t x15 = x0 * (F[6] * mu + x10 * x14 - x14 * x9);
    const scalar_t x16 = (1.0 / 6.0) * jacobian_determinant;
    const scalar_t x17 = -F[3] * F[8] + F[5] * F[6];
    const scalar_t x18 = x0 * (F[1] * mu + x10 * x17 - x17 * x9);
    const scalar_t x19 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x20 = x0 * (F[4] * mu + x10 * x19 - x19 * x9);
    const scalar_t x21 = -F[0] * F[5] + F[2] * F[3];
    const scalar_t x22 = x0 * (F[7] * mu + x10 * x21 - x21 * x9);
    const scalar_t x23 = F[3] * F[7] - F[4] * F[6];
    const scalar_t x24 = x0 * (F[2] * mu + x10 * x23 - x23 * x9);
    const scalar_t x25 = -F[0] * F[7] + F[1] * F[6];
    const scalar_t x26 = x0 * (F[5] * mu + x10 * x25 - x25 * x9);
    const scalar_t x27 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x28 = x0 * (F[8] * mu + x10 * x27 - x27 * x9);
    F_in_P_tXJinv_t_out[0] = x16 * (adjugate[0] * x11 + adjugate[1] * x13 + adjugate[2] * x15);
    F_in_P_tXJinv_t_out[1] = x16 * (adjugate[3] * x11 + adjugate[4] * x13 + adjugate[5] * x15);
    F_in_P_tXJinv_t_out[2] = x16 * (adjugate[6] * x11 + adjugate[7] * x13 + adjugate[8] * x15);
    F_in_P_tXJinv_t_out[3] = x16 * (adjugate[0] * x18 + adjugate[1] * x20 + adjugate[2] * x22);
    F_in_P_tXJinv_t_out[4] = x16 * (adjugate[3] * x18 + adjugate[4] * x20 + adjugate[5] * x22);
    F_in_P_tXJinv_t_out[5] = x16 * (adjugate[6] * x18 + adjugate[7] * x20 + adjugate[8] * x22);
    F_in_P_tXJinv_t_out[6] = x16 * (adjugate[0] * x24 + adjugate[1] * x26 + adjugate[2] * x28);
    F_in_P_tXJinv_t_out[7] = x16 * (adjugate[3] * x24 + adjugate[4] * x26 + adjugate[5] * x28);
    F_in_P_tXJinv_t_out[8] = x16 * (adjugate[6] * x24 + adjugate[7] * x26 + adjugate[8] * x28);
}

static SFEM_INLINE void tet4_neohookean_gradient_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                                     const scalar_t jacobian_determinant,
                                                     const scalar_t mu,
                                                     const scalar_t lambda,
                                                     const scalar_t *const SFEM_RESTRICT ux,
                                                     const scalar_t *const SFEM_RESTRICT uy,
                                                     const scalar_t *const SFEM_RESTRICT uz,
                                                     // Output
                                                     scalar_t *const SFEM_RESTRICT outx,
                                                     scalar_t *const SFEM_RESTRICT outy,
                                                     scalar_t *const SFEM_RESTRICT outz) {
    scalar_t buff[9];

    // Displacement gradient
    tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, buff);

    // Deformation gradient
    buff[0] += (scalar_t)1;
    buff[4] += (scalar_t)1;
    buff[8] += (scalar_t)1;

    // P.T * J^(-T) * det(J)/6
    neohookean_P_tXJinv_t_adj(adjugate, jacobian_determinant, mu, lambda, buff);

    // Inner product
    outx[0] = -buff[0] - buff[1] - buff[2];
    outx[1] = buff[0];
    outx[2] = buff[1];
    outx[3] = buff[2];

    outy[0] = -buff[3] - buff[4] - buff[5];
    outy[1] = buff[3];
    outy[2] = buff[4];
    outy[3] = buff[5];

    outz[0] = -buff[6] - buff[7] - buff[8];
    outz[1] = buff[6];
    outz[2] = buff[7];
    outz[3] = buff[8];
}

static SFEM_INLINE void neohookean_lin_stress_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                                  const scalar_t jacobian_determinant,
                                                  const scalar_t mu,
                                                  const scalar_t lambda,
                                                  const scalar_t *const inc_grad,
                                                  scalar_t *const SFEM_RESTRICT
                                                          F_in_lin_stress_out) {
    const scalar_t *const F = F_in_lin_stress_out;
    scalar_t *const lin_stress = F_in_lin_stress_out;

    // Linearized stress
    const scalar_t x0 = F[4] * F[8];
    const scalar_t x1 = F[5] * F[7];
    const scalar_t x2 = x0 - x1;
    const scalar_t x3 = F[3] * F[7];
    const scalar_t x4 = F[3] * F[8];
    const scalar_t x5 = F[4] * F[6];
    const scalar_t x6 =
            F[0] * x0 - F[0] * x1 + F[1] * F[5] * F[6] - F[1] * x4 + F[2] * x3 - F[2] * x5;
    const scalar_t x7 = (1 / POW2(x6));
    const scalar_t x8 = lambda * x7;
    const scalar_t x9 = mu * x7;
    const scalar_t x10 = -x2;
    const scalar_t x11 = x10 * x2;
    const scalar_t x12 = log(x6);
    const scalar_t x13 = x12 * x8;
    const scalar_t x14 = -F[5] * F[6] + x4;
    const scalar_t x15 = -x14;
    const scalar_t x16 = x2 * x8;
    const scalar_t x17 = x15 * x16;
    const scalar_t x18 = x10 * x9;
    const scalar_t x19 = x15 * x8;
    const scalar_t x20 = x10 * x12;
    const scalar_t x21 = x3 - x5;
    const scalar_t x22 = x16 * x21;
    const scalar_t x23 = x21 * x8;
    const scalar_t x24 = F[1] * F[8] - F[2] * F[7];
    const scalar_t x25 = -x24;
    const scalar_t x26 = x16 * x25;
    const scalar_t x27 = x25 * x8;
    const scalar_t x28 = F[1] * F[5] - F[2] * F[4];
    const scalar_t x29 = x16 * x28;
    const scalar_t x30 = x28 * x8;
    const scalar_t x31 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x32 = x31 * x8;
    const scalar_t x33 = 1.0 / x6;
    const scalar_t x34 = mu * x33;
    const scalar_t x35 = F[8] * x34;
    const scalar_t x36 = lambda * x12 * x33;
    const scalar_t x37 = F[8] * x36;
    const scalar_t x38 = x16 * x31 - x35 + x37;
    const scalar_t x39 = F[0] * F[7] - F[1] * F[6];
    const scalar_t x40 = -x39;
    const scalar_t x41 = x40 * x8;
    const scalar_t x42 = F[7] * x34;
    const scalar_t x43 = F[7] * x36;
    const scalar_t x44 = x16 * x40 + x42 - x43;
    const scalar_t x45 = F[0] * F[5] - F[2] * F[3];
    const scalar_t x46 = -x45;
    const scalar_t x47 = x46 * x8;
    const scalar_t x48 = F[5] * x34;
    const scalar_t x49 = F[5] * x36;
    const scalar_t x50 = x16 * x46 + x48 - x49;
    const scalar_t x51 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x52 = x13 * x51;
    const scalar_t x53 = F[4] * x34;
    const scalar_t x54 = F[4] * x36;
    const scalar_t x55 = x16 * x51 - x53 + x54;
    const scalar_t x56 = x14 * x9;
    const scalar_t x57 = x12 * x14;
    const scalar_t x58 = x19 * x21;
    const scalar_t x59 = x19 * x31;
    const scalar_t x60 = x19 * x46;
    const scalar_t x61 = x19 * x25 + x35 - x37;
    const scalar_t x62 = F[6] * x34;
    const scalar_t x63 = F[6] * x36;
    const scalar_t x64 = x19 * x40 - x62 + x63;
    const scalar_t x65 = x19 * x28 - x48 + x49;
    const scalar_t x66 = F[3] * x34;
    const scalar_t x67 = F[3] * x36;
    const scalar_t x68 = x19 * x51 + x66 - x67;
    const scalar_t x69 = -x21;
    const scalar_t x70 = x69 * x9;
    const scalar_t x71 = x12 * x69;
    const scalar_t x72 = x23 * x40;
    const scalar_t x73 = x23 * x51;
    const scalar_t x74 = x23 * x25 - x42 + x43;
    const scalar_t x75 = x23 * x31 + x62 - x63;
    const scalar_t x76 = x23 * x28 + x53 - x54;
    const scalar_t x77 = x23 * x46 - x66 + x67;
    const scalar_t x78 = x24 * x9;
    const scalar_t x79 = x12 * x24;
    const scalar_t x80 = x27 * x31;
    const scalar_t x81 = x27 * x40;
    const scalar_t x82 = x27 * x28;
    const scalar_t x83 = F[2] * x34;
    const scalar_t x84 = F[2] * x36;
    const scalar_t x85 = x27 * x46 - x83 + x84;
    const scalar_t x86 = F[1] * x34;
    const scalar_t x87 = F[1] * x36;
    const scalar_t x88 = x27 * x51 + x86 - x87;
    const scalar_t x89 = -x31;
    const scalar_t x90 = x89 * x9;
    const scalar_t x91 = x12 * x89;
    const scalar_t x92 = x32 * x40;
    const scalar_t x93 = x32 * x46;
    const scalar_t x94 = x28 * x32 + x83 - x84;
    const scalar_t x95 = F[0] * x34;
    const scalar_t x96 = F[0] * x36;
    const scalar_t x97 = x32 * x51 - x95 + x96;
    const scalar_t x98 = x39 * x9;
    const scalar_t x99 = x12 * x39;
    const scalar_t x100 = x41 * x51;
    const scalar_t x101 = x28 * x41 - x86 + x87;
    const scalar_t x102 = x41 * x46 + x95 - x96;
    const scalar_t x103 = -x28;
    const scalar_t x104 = x103 * x9;
    const scalar_t x105 = x103 * x12;
    const scalar_t x106 = x30 * x46;
    const scalar_t x107 = x30 * x51;
    const scalar_t x108 = x45 * x9;
    const scalar_t x109 = x12 * x45;
    const scalar_t x110 = x47 * x51;
    const scalar_t x111 = -x51;
    const scalar_t x112 = x111 * x9;
    const scalar_t x113 = x111 * x12;
    lin_stress[0] = inc_grad[0] * (mu + x11 * x13 - x11 * x9 + POW2(x2) * x8) +
                    inc_grad[1] * (-x15 * x18 + x17 + x19 * x20) +
                    inc_grad[2] * (-x18 * x21 + x20 * x23 + x22) +
                    inc_grad[3] * (-x18 * x25 + x20 * x27 + x26) +
                    inc_grad[4] * (-x18 * x31 + x20 * x32 + x38) +
                    inc_grad[5] * (-x18 * x40 + x20 * x41 + x44) +
                    inc_grad[6] * (-x18 * x28 + x20 * x30 + x29) +
                    inc_grad[7] * (-x18 * x46 + x20 * x47 + x50) +
                    inc_grad[8] * (x10 * x52 - x18 * x51 + x55);
    lin_stress[1] = inc_grad[0] * (x16 * x57 + x17 - x2 * x56) +
                    inc_grad[1] * (mu + POW2(x15) * x8 - x15 * x56 + x19 * x57) +
                    inc_grad[2] * (-x21 * x56 + x23 * x57 + x58) +
                    inc_grad[3] * (-x25 * x56 + x27 * x57 + x61) +
                    inc_grad[4] * (-x31 * x56 + x32 * x57 + x59) +
                    inc_grad[5] * (-x40 * x56 + x41 * x57 + x64) +
                    inc_grad[6] * (-x28 * x56 + x30 * x57 + x65) +
                    inc_grad[7] * (-x46 * x56 + x47 * x57 + x60) +
                    inc_grad[8] * (x14 * x52 - x51 * x56 + x68);
    lin_stress[2] = inc_grad[0] * (x16 * x71 - x2 * x70 + x22) +
                    inc_grad[1] * (-x15 * x70 + x19 * x71 + x58) +
                    inc_grad[2] * (mu + POW2(x21) * x8 - x21 * x70 + x23 * x71) +
                    inc_grad[3] * (-x25 * x70 + x27 * x71 + x74) +
                    inc_grad[4] * (-x31 * x70 + x32 * x71 + x75) +
                    inc_grad[5] * (-x40 * x70 + x41 * x71 + x72) +
                    inc_grad[6] * (-x28 * x70 + x30 * x71 + x76) +
                    inc_grad[7] * (-x46 * x70 + x47 * x71 + x77) +
                    inc_grad[8] * (-x51 * x70 + x52 * x69 + x73);
    lin_stress[3] = inc_grad[0] * (x16 * x79 - x2 * x78 + x26) +
                    inc_grad[1] * (-x15 * x78 + x19 * x79 + x61) +
                    inc_grad[2] * (-x21 * x78 + x23 * x79 + x74) +
                    inc_grad[3] * (mu + POW2(x25) * x8 - x25 * x78 + x27 * x79) +
                    inc_grad[4] * (-x31 * x78 + x32 * x79 + x80) +
                    inc_grad[5] * (-x40 * x78 + x41 * x79 + x81) +
                    inc_grad[6] * (-x28 * x78 + x30 * x79 + x82) +
                    inc_grad[7] * (-x46 * x78 + x47 * x79 + x85) +
                    inc_grad[8] * (x24 * x52 - x51 * x78 + x88);
    lin_stress[4] = inc_grad[0] * (x16 * x91 - x2 * x90 + x38) +
                    inc_grad[1] * (-x15 * x90 + x19 * x91 + x59) +
                    inc_grad[2] * (-x21 * x90 + x23 * x91 + x75) +
                    inc_grad[3] * (-x25 * x90 + x27 * x91 + x80) +
                    inc_grad[4] * (mu + POW2(x31) * x8 - x31 * x90 + x32 * x91) +
                    inc_grad[5] * (-x40 * x90 + x41 * x91 + x92) +
                    inc_grad[6] * (-x28 * x90 + x30 * x91 + x94) +
                    inc_grad[7] * (-x46 * x90 + x47 * x91 + x93) +
                    inc_grad[8] * (-x51 * x90 + x52 * x89 + x97);
    lin_stress[5] = inc_grad[0] * (x16 * x99 - x2 * x98 + x44) +
                    inc_grad[1] * (-x15 * x98 + x19 * x99 + x64) +
                    inc_grad[2] * (-x21 * x98 + x23 * x99 + x72) +
                    inc_grad[3] * (-x25 * x98 + x27 * x99 + x81) +
                    inc_grad[4] * (-x31 * x98 + x32 * x99 + x92) +
                    inc_grad[5] * (mu + POW2(x40) * x8 - x40 * x98 + x41 * x99) +
                    inc_grad[6] * (x101 - x28 * x98 + x30 * x99) +
                    inc_grad[7] * (x102 - x46 * x98 + x47 * x99) +
                    inc_grad[8] * (x100 + x39 * x52 - x51 * x98);
    lin_stress[6] = inc_grad[0] * (-x104 * x2 + x105 * x16 + x29) +
                    inc_grad[1] * (-x104 * x15 + x105 * x19 + x65) +
                    inc_grad[2] * (-x104 * x21 + x105 * x23 + x76) +
                    inc_grad[3] * (-x104 * x25 + x105 * x27 + x82) +
                    inc_grad[4] * (-x104 * x31 + x105 * x32 + x94) +
                    inc_grad[5] * (x101 - x104 * x40 + x105 * x41) +
                    inc_grad[6] * (mu - x104 * x28 + x105 * x30 + POW2(x28) * x8) +
                    inc_grad[7] * (-x104 * x46 + x105 * x47 + x106) +
                    inc_grad[8] * (x103 * x52 - x104 * x51 + x107);
    lin_stress[7] = inc_grad[0] * (-x108 * x2 + x109 * x16 + x50) +
                    inc_grad[1] * (-x108 * x15 + x109 * x19 + x60) +
                    inc_grad[2] * (-x108 * x21 + x109 * x23 + x77) +
                    inc_grad[3] * (-x108 * x25 + x109 * x27 + x85) +
                    inc_grad[4] * (-x108 * x31 + x109 * x32 + x93) +
                    inc_grad[5] * (x102 - x108 * x40 + x109 * x41) +
                    inc_grad[6] * (x106 - x108 * x28 + x109 * x30) +
                    inc_grad[7] * (mu - x108 * x46 + x109 * x47 + POW2(x46) * x8) +
                    inc_grad[8] * (-x108 * x51 + x110 + x45 * x52);
    lin_stress[8] = inc_grad[0] * (-x112 * x2 + x113 * x16 + x55) +
                    inc_grad[1] * (-x112 * x15 + x113 * x19 + x68) +
                    inc_grad[2] * (-x112 * x21 + x113 * x23 + x73) +
                    inc_grad[3] * (-x112 * x25 + x113 * x27 + x88) +
                    inc_grad[4] * (-x112 * x31 + x113 * x32 + x97) +
                    inc_grad[5] * (x100 - x112 * x40 + x113 * x41) +
                    inc_grad[6] * (x107 - x112 * x28 + x113 * x30) +
                    inc_grad[7] * (x110 - x112 * x46 + x113 * x47) +
                    inc_grad[8] * (mu + x111 * x52 - x112 * x51 + POW2(x51) * x8);
}

static SFEM_INLINE void tet4_neohookean_hessian_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                                    const scalar_t jacobian_determinant,
                                                    const scalar_t mu,
                                                    const scalar_t lambda,
                                                    // Displacement
                                                    const scalar_t *const SFEM_RESTRICT ux,
                                                    const scalar_t *const SFEM_RESTRICT uy,
                                                    const scalar_t *const SFEM_RESTRICT uz,
                                                    // Increment
                                                    const scalar_t *const SFEM_RESTRICT hx,
                                                    const scalar_t *const SFEM_RESTRICT hy,
                                                    const scalar_t *const SFEM_RESTRICT hz,
                                                    // Output
                                                    scalar_t *const SFEM_RESTRICT outx,
                                                    scalar_t *const SFEM_RESTRICT outy,
                                                    scalar_t *const SFEM_RESTRICT outz)

{
    scalar_t buff[9];

    // Displacement gradient
    tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, buff);

    // Deformation gradient
    buff[0] += (scalar_t)1;
    buff[4] += (scalar_t)1;
    buff[8] += (scalar_t)1;

    // Increment gradient
    scalar_t inc_grad[9];
    tet4_gradient_3(adjugate, jacobian_determinant, hx, hy, hz, inc_grad);

    // 2nd Order linearization (lin_stress)
    neohookean_lin_stress_adj(adjugate, jacobian_determinant, mu, lambda, inc_grad, buff);

    // lin_stress^T * (jac_inv^(-T) * det(J)/6)
    const scalar_t x0 = (1.0 / 6.0);
    const scalar_t x1 = buff[0] * x0;
    const scalar_t x2 = buff[3] * x0;
    const scalar_t x3 = buff[6] * x0;
    const scalar_t x4 = buff[1] * x0;
    const scalar_t x5 = buff[4] * x0;
    const scalar_t x6 = buff[7] * x0;
    const scalar_t x7 = buff[2] * x0;
    const scalar_t x8 = buff[5] * x0;
    const scalar_t x9 = buff[8] * x0;
    buff[0] = adjugate[0] * x1 + adjugate[1] * x2 + adjugate[2] * x3;
    buff[1] = adjugate[3] * x1 + adjugate[4] * x2 + adjugate[5] * x3;
    buff[2] = adjugate[6] * x1 + adjugate[7] * x2 + adjugate[8] * x3;
    buff[3] = adjugate[0] * x4 + adjugate[1] * x5 + adjugate[2] * x6;
    buff[4] = adjugate[3] * x4 + adjugate[4] * x5 + adjugate[5] * x6;
    buff[5] = adjugate[6] * x4 + adjugate[7] * x5 + adjugate[8] * x6;
    buff[6] = adjugate[0] * x7 + adjugate[1] * x8 + adjugate[2] * x9;
    buff[7] = adjugate[3] * x7 + adjugate[4] * x8 + adjugate[5] * x9;
    buff[8] = adjugate[6] * x7 + adjugate[7] * x8 + adjugate[8] * x9;

    // Contraction with gradient of reference test functions
    outx[0] = -buff[0] - buff[1] - buff[2];
    outx[1] = buff[0];
    outx[2] = buff[1];
    outx[3] = buff[2];

    outy[0] = -buff[3] - buff[4] - buff[5];
    outy[1] = buff[3];
    outy[2] = buff[4];
    outy[3] = buff[5];

    outz[0] = -buff[6] - buff[7] - buff[8];
    outz[1] = buff[6];
    outz[2] = buff[7];
    outz[3] = buff[8];
}

#endif  // TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H
