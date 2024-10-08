#ifndef TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H
#define TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H

#include "tet4_linear_elasticity_inline_cpu.h"  // tet4_gradient_3

// This can be used with any element type
static SFEM_INLINE void neohookean_PxJinv_t_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                                const scalar_t jacobian_determinant,
                                                const scalar_t mu,
                                                const scalar_t lambda,
                                                scalar_t *const SFEM_RESTRICT F_in_PxJinv_t_out) {
    const scalar_t *const F = F_in_PxJinv_t_out;
    scalar_t *const PxJinv_t = F_in_PxJinv_t_out;

    const scalar_t x0 = F[4] * F[8];
    const scalar_t x1 = F[5] * F[6];
    const scalar_t x2 = F[3] * F[7];
    const scalar_t x3 = F[5] * F[7];
    const scalar_t x4 = F[3] * F[8];
    const scalar_t x5 = F[4] * F[6];
    const scalar_t x6 = F[0] * x0 - F[0] * x3 + F[1] * x1 - F[1] * x4 + F[2] * x2 - F[2] * x5;
    const scalar_t x7 = 1.0 / x6;
    const scalar_t x8 = x0 - x3;
    const scalar_t x9 = mu * x6;
    const scalar_t x10 = lambda * log(x6);
    const scalar_t x11 = (1.0 / 6.0) * F[0] * x9 - 1.0 / 6.0 * mu * x8 + (1.0 / 6.0) * x10 * x8;
    const scalar_t x12 = -x1 + x4;
    const scalar_t x13 = (1.0 / 6.0) * F[1] * x9 + (1.0 / 6.0) * mu * x12 - 1.0 / 6.0 * x10 * x12;
    const scalar_t x14 = x2 - x5;
    const scalar_t x15 = (1.0 / 6.0) * F[2] * x9 - 1.0 / 6.0 * mu * x14 + (1.0 / 6.0) * x10 * x14;
    const scalar_t x16 = F[1] * F[8] - F[2] * F[7];
    const scalar_t x17 = (1.0 / 6.0) * F[3] * x9 + (1.0 / 6.0) * mu * x16 - 1.0 / 6.0 * x10 * x16;
    const scalar_t x18 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x19 = (1.0 / 6.0) * F[4] * x9 - 1.0 / 6.0 * mu * x18 + (1.0 / 6.0) * x10 * x18;
    const scalar_t x20 = F[0] * F[7] - F[1] * F[6];
    const scalar_t x21 = (1.0 / 6.0) * F[5] * x9 + (1.0 / 6.0) * mu * x20 - 1.0 / 6.0 * x10 * x20;
    const scalar_t x22 = F[1] * F[5] - F[2] * F[4];
    const scalar_t x23 = (1.0 / 6.0) * F[6] * x9 - 1.0 / 6.0 * mu * x22 + (1.0 / 6.0) * x10 * x22;
    const scalar_t x24 = F[0] * F[5] - F[2] * F[3];
    const scalar_t x25 = (1.0 / 6.0) * F[7] * x9 + (1.0 / 6.0) * mu * x24 - 1.0 / 6.0 * x10 * x24;
    const scalar_t x26 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x27 = (1.0 / 6.0) * F[8] * x9 - 1.0 / 6.0 * mu * x26 + (1.0 / 6.0) * x10 * x26;
    PxJinv_t[0] = x7 * (adjugate[0] * x11 + adjugate[1] * x13 + adjugate[2] * x15);
    PxJinv_t[1] = x7 * (adjugate[3] * x11 + adjugate[4] * x13 + adjugate[5] * x15);
    PxJinv_t[2] = x7 * (adjugate[6] * x11 + adjugate[7] * x13 + adjugate[8] * x15);
    PxJinv_t[3] = x7 * (adjugate[0] * x17 + adjugate[1] * x19 + adjugate[2] * x21);
    PxJinv_t[4] = x7 * (adjugate[3] * x17 + adjugate[4] * x19 + adjugate[5] * x21);
    PxJinv_t[5] = x7 * (adjugate[6] * x17 + adjugate[7] * x19 + adjugate[8] * x21);
    PxJinv_t[6] = x7 * (adjugate[0] * x23 + adjugate[1] * x25 + adjugate[2] * x27);
    PxJinv_t[7] = x7 * (adjugate[3] * x23 + adjugate[4] * x25 + adjugate[5] * x27);
    PxJinv_t[8] = x7 * (adjugate[6] * x23 + adjugate[7] * x25 + adjugate[8] * x27);
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
#if 1
    scalar_t buff[9];

    // Displacement gradient
    // tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, buff);

    {
        scalar_t *const disp_grad = buff;

        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = x0 * (-ux[0] + ux[1]);
        const scalar_t x2 = x0 * (-ux[0] + ux[2]);
        const scalar_t x3 = x0 * (-ux[0] + ux[3]);
        const scalar_t x4 = x0 * (-uy[0] + uy[1]);
        const scalar_t x5 = x0 * (-uy[0] + uy[2]);
        const scalar_t x6 = x0 * (-uy[0] + uy[3]);
        const scalar_t x7 = x0 * (-uz[0] + uz[1]);
        const scalar_t x8 = x0 * (-uz[0] + uz[2]);
        const scalar_t x9 = x0 * (-uz[0] + uz[3]);
        disp_grad[0] = adjugate[0] * x1 + adjugate[3] * x2 + adjugate[6] * x3;
        disp_grad[1] = adjugate[1] * x1 + adjugate[4] * x2 + adjugate[7] * x3;
        disp_grad[2] = adjugate[2] * x1 + adjugate[5] * x2 + adjugate[8] * x3;
        disp_grad[3] = adjugate[0] * x4 + adjugate[3] * x5 + adjugate[6] * x6;
        disp_grad[4] = adjugate[1] * x4 + adjugate[4] * x5 + adjugate[7] * x6;
        disp_grad[5] = adjugate[2] * x4 + adjugate[5] * x5 + adjugate[8] * x6;
        disp_grad[6] = adjugate[0] * x7 + adjugate[3] * x8 + adjugate[6] * x9;
        disp_grad[7] = adjugate[1] * x7 + adjugate[4] * x8 + adjugate[7] * x9;
        disp_grad[8] = adjugate[2] * x7 + adjugate[5] * x8 + adjugate[8] * x9;
    }

    // Deformation gradient
    buff[0] += (scalar_t)1;
    buff[4] += (scalar_t)1;
    buff[8] += (scalar_t)1;

    {
        // // P.T * J^(-T) * det(J)/6
        neohookean_PxJinv_t_adj(adjugate, jacobian_determinant, mu, lambda, buff);
        const scalar_t *const PxJinv_t = buff;

        // Inner product
        outx[0] = -PxJinv_t[0] - PxJinv_t[1] - PxJinv_t[2];
        outx[1] = PxJinv_t[0];
        outx[2] = PxJinv_t[1];
        outx[3] = PxJinv_t[2];
        outy[0] = -PxJinv_t[3] - PxJinv_t[4] - PxJinv_t[5];
        outy[1] = PxJinv_t[3];
        outy[2] = PxJinv_t[4];
        outy[3] = PxJinv_t[5];
        outz[0] = -PxJinv_t[6] - PxJinv_t[7] - PxJinv_t[8];
        outz[1] = PxJinv_t[6];
        outz[2] = PxJinv_t[7];
        outz[3] = PxJinv_t[8];
    }

#else
    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = adjugate[1] * x0;
    const scalar_t x2 = adjugate[4] * x0;
    const scalar_t x3 = adjugate[7] * x0;
    const scalar_t x4 = -x1 - x2 - x3;
    const scalar_t x5 = -ux[0] + ux[1];
    const scalar_t x6 = -ux[0] + ux[2];
    const scalar_t x7 = -ux[0] + ux[3];
    const scalar_t x8 = x1 * x5 + x2 * x6 + x3 * x7;
    const scalar_t x9 = -uy[0] + uy[1];
    const scalar_t x10 = adjugate[0] * x0;
    const scalar_t x11 = -uy[0] + uy[2];
    const scalar_t x12 = adjugate[3] * x0;
    const scalar_t x13 = -uy[0] + uy[3];
    const scalar_t x14 = adjugate[6] * x0;
    const scalar_t x15 = x10 * x9 + x11 * x12 + x13 * x14;
    const scalar_t x16 = -uz[0] + uz[1];
    const scalar_t x17 = adjugate[2] * x0;
    const scalar_t x18 = -uz[0] + uz[2];
    const scalar_t x19 = adjugate[5] * x0;
    const scalar_t x20 = -uz[0] + uz[3];
    const scalar_t x21 = adjugate[8] * x0;
    const scalar_t x22 = x16 * x17 + x18 * x19 + x20 * x21 + 1;
    const scalar_t x23 = x15 * x22;
    const scalar_t x24 = x10 * x16 + x12 * x18 + x14 * x20;
    const scalar_t x25 = x11 * x19 + x13 * x21 + x17 * x9;
    const scalar_t x26 = -x23 + x24 * x25;
    const scalar_t x27 = x1 * x16 + x18 * x2 + x20 * x3;
    const scalar_t x28 = x17 * x5 + x19 * x6 + x21 * x7;
    const scalar_t x29 = x1 * x9 + x11 * x2 + x13 * x3 + 1;
    const scalar_t x30 = x24 * x29;
    const scalar_t x31 = x10 * x5 + x12 * x6 + x14 * x7 + 1;
    const scalar_t x32 = x25 * x27;
    const scalar_t x33 =
            x15 * x27 * x28 + x22 * x29 * x31 - x23 * x8 + x24 * x25 * x8 - x28 * x30 - x31 * x32;
    const scalar_t x34 = 1.0 / x33;
    const scalar_t x35 = mu * x34;
    const scalar_t x36 = lambda * x34 * log(x33);
    const scalar_t x37 = mu * x8 - x26 * x35 + x26 * x36;
    const scalar_t x38 = -x17 - x19 - x21;
    const scalar_t x39 = x15 * x27 - x30;
    const scalar_t x40 = mu * x28 - x35 * x39 + x36 * x39;
    const scalar_t x41 = -x10 - x12 - x14;
    const scalar_t x42 = x22 * x29 - x32;
    const scalar_t x43 = mu * x31 - x35 * x42 + x36 * x42;
    const scalar_t x44 = (1.0 / 6.0) * jacobian_determinant;
    const scalar_t x45 = -x22 * x8 + x27 * x28;
    const scalar_t x46 = mu * x15 - x35 * x45 + x36 * x45;
    const scalar_t x47 = x24 * x8 - x27 * x31;
    const scalar_t x48 = mu * x25 - x35 * x47 + x36 * x47;
    const scalar_t x49 = x22 * x31 - x24 * x28;
    const scalar_t x50 = mu * x29 - x35 * x49 + x36 * x49;
    const scalar_t x51 = x25 * x8 - x28 * x29;
    const scalar_t x52 = mu * x24 - x35 * x51 + x36 * x51;
    const scalar_t x53 = x15 * x28 - x25 * x31;
    const scalar_t x54 = mu * x27 - x35 * x53 + x36 * x53;
    const scalar_t x55 = -x15 * x8 + x29 * x31;
    const scalar_t x56 = mu * x22 - x35 * x55 + x36 * x55;
    outx[0] = x44 * (x37 * x4 + x38 * x40 + x41 * x43);
    outx[1] = x44 * (x1 * x37 + x10 * x43 + x17 * x40);
    outx[2] = x44 * (x12 * x43 + x19 * x40 + x2 * x37);
    outx[3] = x44 * (x14 * x43 + x21 * x40 + x3 * x37);
    outy[0] = x44 * (x38 * x48 + x4 * x50 + x41 * x46);
    outy[1] = x44 * (x1 * x50 + x10 * x46 + x17 * x48);
    outy[2] = x44 * (x12 * x46 + x19 * x48 + x2 * x50);
    outy[3] = x44 * (x14 * x46 + x21 * x48 + x3 * x50);
    outz[0] = x44 * (x38 * x56 + x4 * x54 + x41 * x52);
    outz[1] = x44 * (x1 * x54 + x10 * x52 + x17 * x56);
    outz[2] = x44 * (x12 * x52 + x19 * x56 + x2 * x54);
    outz[3] = x44 * (x14 * x52 + x21 * x56 + x3 * x54);
#endif
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

static SFEM_INLINE void tet4_neohookean_hessian_apply_adj(const scalar_t *const SFEM_RESTRICT
                                                                  adjugate,
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

    // lin_stress * (jac_inv^(-T) * det(J)/6)
    const scalar_t x0 = (1.0 / 6.0);
    const scalar_t x1 = buff[0] * x0;
    const scalar_t x2 = buff[1] * x0;
    const scalar_t x3 = buff[2] * x0;
    const scalar_t x4 = buff[3] * x0;
    const scalar_t x5 = buff[4] * x0;
    const scalar_t x6 = buff[5] * x0;
    const scalar_t x7 = buff[6] * x0;
    const scalar_t x8 = buff[7] * x0;
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


// FIXME (make HPC version / adjugate version)
static SFEM_INLINE void neohookean_ogden_hessian_points(const scalar_t px0,
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
                                                        const scalar_t mu,
                                                        const scalar_t lambda,
                                                        const scalar_t *const SFEM_RESTRICT ux,
                                                        const scalar_t *const SFEM_RESTRICT uy,
                                                        const scalar_t *const SFEM_RESTRICT uz,
                                                        accumulator_t *const SFEM_RESTRICT
                                                                element_matrix) {
    const scalar_t x0 = pz0 - pz3;
    const scalar_t x1 = -x0;
    const scalar_t x2 = py0 - py2;
    const scalar_t x3 = -x2;
    const scalar_t x4 = px0 - px1;
    const scalar_t x5 = -1.0 / 6.0 * x4;
    const scalar_t x6 = py0 - py3;
    const scalar_t x7 = -x6;
    const scalar_t x8 = pz0 - pz2;
    const scalar_t x9 = -x8;
    const scalar_t x10 = py0 - py1;
    const scalar_t x11 = -x10;
    const scalar_t x12 = px0 - px2;
    const scalar_t x13 = -1.0 / 6.0 * x12;
    const scalar_t x14 = pz0 - pz1;
    const scalar_t x15 = -x14;
    const scalar_t x16 = px0 - px3;
    const scalar_t x17 = -1.0 / 6.0 * x16;
    const scalar_t x18 = -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 -
                         x15 * x17 * x3 - x5 * x7 * x9;
    const scalar_t x19 = x2 * x4;
    const scalar_t x20 = x10 * x12;
    const scalar_t x21 = x19 - x20;
    const scalar_t x22 = -x21;
    const scalar_t x23 = x12 * x6;
    const scalar_t x24 = x10 * x16;
    const scalar_t x25 = x4 * x6;
    const scalar_t x26 = x16 * x2;
    const scalar_t x27 = 1.0 / (x0 * x19 - x0 * x20 + x14 * x23 - x14 * x26 + x24 * x8 - x25 * x8);
    const scalar_t x28 = uz[2] * x27;
    const scalar_t x29 = x23 - x26;
    const scalar_t x30 = -x29;
    const scalar_t x31 = uy[0] * x27;
    const scalar_t x32 = -x24 + x25;
    const scalar_t x33 = uy[3] * x27;
    const scalar_t x34 = x21 + x24 - x25 + x29;
    const scalar_t x35 = ux[1] * x27;
    const scalar_t x36 = x22 * x28 + x30 * x31 + x32 * x33 + x34 * x35;
    const scalar_t x37 = x10 * x8 - x14 * x2;
    const scalar_t x38 = -x37;
    const scalar_t x39 = uz[3] * x27;
    const scalar_t x40 = x0 * x2 - x6 * x8;
    const scalar_t x41 = -x40;
    const scalar_t x42 = uy[1] * x27;
    const scalar_t x43 = x0 * x10;
    const scalar_t x44 = x14 * x6;
    const scalar_t x45 = x43 - x44;
    const scalar_t x46 = uz[0] * x27;
    const scalar_t x47 = x37 + x40 - x43 + x44;
    const scalar_t x48 = ux[2] * x27;
    const scalar_t x49 = x38 * x39 + x41 * x42 + x45 * x46 + x47 * x48;
    const scalar_t x50 = x36 * x49;
    const scalar_t x51 = x28 * x38 + x31 * x41 + x33 * x45 + x35 * x47;
    const scalar_t x52 = x22 * x39 + x30 * x42 + x32 * x46 + x34 * x48 + 1;
    const scalar_t x53 = x51 * x52;
    const scalar_t x54 = x50 - x53;
    const scalar_t x55 = ux[3] * x27;
    const scalar_t x56 = uy[2] * x27;
    const scalar_t x57 = uz[1] * x27;
    const scalar_t x58 = ux[0] * x27;
    const scalar_t x59 = x22 * x57 + x30 * x55 + x32 * x56 + x34 * x58;
    const scalar_t x60 = -x12 * x14 + x4 * x8;
    const scalar_t x61 = x0 * x12 - x16 * x8;
    const scalar_t x62 = x0 * x4;
    const scalar_t x63 = x14 * x16;
    const scalar_t x64 = -x62 + x63;
    const scalar_t x65 = -x60 - x61 + x62 - x63;
    const scalar_t x66 = x39 * x60 + x42 * x61 + x46 * x64 + x48 * x65;
    const scalar_t x67 = x51 * x66;
    const scalar_t x68 = x55 * x61 + x56 * x64 + x57 * x60 + x58 * x65;
    const scalar_t x69 = x28 * x60 + x31 * x61 + x33 * x64 + x35 * x65 + 1;
    const scalar_t x70 = x49 * x69;
    const scalar_t x71 = x38 * x57 + x41 * x55 + x45 * x56 + x47 * x58 + 1;
    const scalar_t x72 = x36 * x66;
    const scalar_t x73 =
            x50 * x68 + x52 * x69 * x71 - x53 * x68 + x59 * x67 - x59 * x70 - x71 * x72;
    const scalar_t x74 = pow(x73, -2);
    const scalar_t x75 = lambda * x74;
    const scalar_t x76 = -x54;
    const scalar_t x77 = mu * x74;
    const scalar_t x78 = x76 * x77;
    const scalar_t x79 = x54 * x75;
    const scalar_t x80 = log(x73);
    const scalar_t x81 = x76 * x80;
    const scalar_t x82 = mu + pow(x54, 2) * x75 - x54 * x78 + x79 * x81;
    const scalar_t x83 = x27 * x65;
    const scalar_t x84 = x67 - x70;
    const scalar_t x85 = x75 * x84;
    const scalar_t x86 = x54 * x85;
    const scalar_t x87 = -x78 * x84 + x81 * x85 + x86;
    const scalar_t x88 = x27 * x34;
    const scalar_t x89 = -x52 * x69 + x72;
    const scalar_t x90 = -x89;
    const scalar_t x91 = x75 * x90;
    const scalar_t x92 = x54 * x91;
    const scalar_t x93 = -x78 * x90 + x81 * x91 + x92;
    const scalar_t x94 = x27 * x47;
    const scalar_t x95 = x82 * x83 + x87 * x88 + x93 * x94;
    const scalar_t x96 = -x84;
    const scalar_t x97 = x77 * x96;
    const scalar_t x98 = x80 * x96;
    const scalar_t x99 = mu + x75 * pow(x84, 2) - x84 * x97 + x85 * x98;
    const scalar_t x100 = x54 * x96;
    const scalar_t x101 = x75 * x80;
    const scalar_t x102 = x100 * x101 - x100 * x77 + x86;
    const scalar_t x103 = x85 * x90;
    const scalar_t x104 = x103 - x90 * x97 + x91 * x98;
    const scalar_t x105 = x102 * x83 + x104 * x94 + x88 * x99;
    const scalar_t x106 = x77 * x89;
    const scalar_t x107 = x80 * x89;
    const scalar_t x108 = mu - x106 * x90 + x107 * x91 + x75 * pow(x90, 2);
    const scalar_t x109 = -x106 * x54 + x107 * x79 + x92;
    const scalar_t x110 = x103 - x106 * x84 + x107 * x85;
    const scalar_t x111 = x108 * x94 + x109 * x83 + x110 * x88;
    const scalar_t x112 = x49 * x68 - x66 * x71;
    const scalar_t x113 = x112 * x85;
    const scalar_t x114 = -x112;
    const scalar_t x115 = x114 * x77;
    const scalar_t x116 = x114 * x80;
    const scalar_t x117 = x113 - x115 * x84 + x116 * x85;
    const scalar_t x118 = 1.0 / x73;
    const scalar_t x119 = mu * x118;
    const scalar_t x120 = x119 * x49;
    const scalar_t x121 = lambda * x118 * x80;
    const scalar_t x122 = x121 * x49;
    const scalar_t x123 = x112 * x79 - x120 + x122;
    const scalar_t x124 = -x115 * x54 + x116 * x79 + x123;
    const scalar_t x125 = x119 * x66;
    const scalar_t x126 = x121 * x66;
    const scalar_t x127 = x112 * x91 + x125 - x126;
    const scalar_t x128 = -x115 * x90 + x116 * x91 + x127;
    const scalar_t x129 = x117 * x88 + x124 * x83 + x128 * x94;
    const scalar_t x130 = -x52 * x68 + x59 * x66;
    const scalar_t x131 = x130 * x91;
    const scalar_t x132 = -x130;
    const scalar_t x133 = x132 * x77;
    const scalar_t x134 = x132 * x80;
    const scalar_t x135 = x131 - x133 * x90 + x134 * x91;
    const scalar_t x136 = -x125 + x126 + x130 * x85;
    const scalar_t x137 = -x133 * x84 + x134 * x85 + x136;
    const scalar_t x138 = x119 * x52;
    const scalar_t x139 = x121 * x52;
    const scalar_t x140 = x130 * x79 + x138 - x139;
    const scalar_t x141 = -x133 * x54 + x134 * x79 + x140;
    const scalar_t x142 = x135 * x94 + x137 * x88 + x141 * x83;
    const scalar_t x143 = x49 * x59 - x52 * x71;
    const scalar_t x144 = -x143;
    const scalar_t x145 = x144 * x79;
    const scalar_t x146 = x143 * x77;
    const scalar_t x147 = x143 * x80;
    const scalar_t x148 = x145 - x146 * x54 + x147 * x79;
    const scalar_t x149 = x120 - x122 + x144 * x85;
    const scalar_t x150 = -x146 * x84 + x147 * x85 + x149;
    const scalar_t x151 = -x138 + x139 + x144 * x91;
    const scalar_t x152 = -x146 * x90 + x147 * x91 + x151;
    const scalar_t x153 = x148 * x83 + x150 * x88 + x152 * x94;
    const scalar_t x154 = x18 * (x129 * x88 + x142 * x94 + x153 * x83);
    const scalar_t x155 = -x36 * x71 + x51 * x59;
    const scalar_t x156 = x155 * x79;
    const scalar_t x157 = -x155;
    const scalar_t x158 = x157 * x77;
    const scalar_t x159 = x157 * x80;
    const scalar_t x160 = x156 - x158 * x54 + x159 * x79;
    const scalar_t x161 = x119 * x51;
    const scalar_t x162 = x121 * x51;
    const scalar_t x163 = x155 * x85 - x161 + x162;
    const scalar_t x164 = -x158 * x84 + x159 * x85 + x163;
    const scalar_t x165 = x119 * x36;
    const scalar_t x166 = x121 * x36;
    const scalar_t x167 = x155 * x91 + x165 - x166;
    const scalar_t x168 = -x158 * x90 + x159 * x91 + x167;
    const scalar_t x169 = x160 * x83 + x164 * x88 + x168 * x94;
    const scalar_t x170 = -x36 * x68 + x59 * x69;
    const scalar_t x171 = -x170;
    const scalar_t x172 = x171 * x91;
    const scalar_t x173 = x170 * x77;
    const scalar_t x174 = x170 * x80;
    const scalar_t x175 = x172 - x173 * x90 + x174 * x91;
    const scalar_t x176 = -x165 + x166 + x171 * x79;
    const scalar_t x177 = -x173 * x54 + x174 * x79 + x176;
    const scalar_t x178 = x119 * x69;
    const scalar_t x179 = x121 * x69;
    const scalar_t x180 = x171 * x85 + x178 - x179;
    const scalar_t x181 = -x173 * x84 + x174 * x85 + x180;
    const scalar_t x182 = x175 * x94 + x177 * x83 + x181 * x88;
    const scalar_t x183 = x51 * x68 - x69 * x71;
    const scalar_t x184 = -x183;
    const scalar_t x185 = x184 * x85;
    const scalar_t x186 = x183 * x77;
    const scalar_t x187 = x183 * x80;
    const scalar_t x188 = x185 - x186 * x84 + x187 * x85;
    const scalar_t x189 = x161 - x162 + x184 * x79;
    const scalar_t x190 = -x186 * x54 + x187 * x79 + x189;
    const scalar_t x191 = -x178 + x179 + x184 * x91;
    const scalar_t x192 = -x186 * x90 + x187 * x91 + x191;
    const scalar_t x193 = x188 * x88 + x190 * x83 + x192 * x94;
    const scalar_t x194 = x18 * (x169 * x83 + x182 * x94 + x193 * x88);
    const scalar_t x195 = x27 * x30;
    const scalar_t x196 = x27 * x61;
    const scalar_t x197 = x27 * x41;
    const scalar_t x198 = x18 * (x105 * x195 + x111 * x197 + x196 * x95);
    const scalar_t x199 = x18 * (x129 * x195 + x142 * x197 + x153 * x196);
    const scalar_t x200 = x18 * (x169 * x196 + x182 * x197 + x193 * x195);
    const scalar_t x201 = x27 * x32;
    const scalar_t x202 = x27 * x64;
    const scalar_t x203 = x27 * x45;
    const scalar_t x204 = x18 * (x105 * x201 + x111 * x203 + x202 * x95);
    const scalar_t x205 = x18 * (x129 * x201 + x142 * x203 + x153 * x202);
    const scalar_t x206 = x18 * (x169 * x202 + x182 * x203 + x193 * x201);
    const scalar_t x207 = x22 * x27;
    const scalar_t x208 = x27 * x60;
    const scalar_t x209 = x27 * x38;
    const scalar_t x210 = x18 * (x105 * x207 + x111 * x209 + x208 * x95);
    const scalar_t x211 = x18 * (x129 * x207 + x142 * x209 + x153 * x208);
    const scalar_t x212 = x18 * (x169 * x208 + x182 * x209 + x193 * x207);
    const scalar_t x213 = x130 * x75;
    const scalar_t x214 = mu + pow(x130, 2) * x75 - x130 * x133 + x134 * x213;
    const scalar_t x215 = x112 * x75;
    const scalar_t x216 = x130 * x215;
    const scalar_t x217 = -x112 * x133 + x134 * x215 + x216;
    const scalar_t x218 = x144 * x75;
    const scalar_t x219 = x130 * x218;
    const scalar_t x220 = -x133 * x144 + x134 * x218 + x219;
    const scalar_t x221 = x214 * x94 + x217 * x88 + x220 * x83;
    const scalar_t x222 = mu + pow(x112, 2) * x75 - x112 * x115 + x116 * x215;
    const scalar_t x223 = -x115 * x130 + x116 * x213 + x216;
    const scalar_t x224 = x144 * x215;
    const scalar_t x225 = -x115 * x144 + x116 * x218 + x224;
    const scalar_t x226 = x222 * x88 + x223 * x94 + x225 * x83;
    const scalar_t x227 = mu + pow(x144, 2) * x75 - x144 * x146 + x147 * x218;
    const scalar_t x228 = -x130 * x146 + x147 * x213 + x219;
    const scalar_t x229 = -x112 * x146 + x147 * x215 + x224;
    const scalar_t x230 = x227 * x83 + x228 * x94 + x229 * x88;
    const scalar_t x231 = x171 * x213;
    const scalar_t x232 = -x130 * x173 + x174 * x213 + x231;
    const scalar_t x233 = x119 * x68;
    const scalar_t x234 = x121 * x68;
    const scalar_t x235 = x171 * x215 - x233 + x234;
    const scalar_t x236 = -x112 * x173 + x174 * x215 + x235;
    const scalar_t x237 = x119 * x59;
    const scalar_t x238 = x121 * x59;
    const scalar_t x239 = x171 * x218 + x237 - x238;
    const scalar_t x240 = -x144 * x173 + x174 * x218 + x239;
    const scalar_t x241 = x232 * x94 + x236 * x88 + x240 * x83;
    const scalar_t x242 = x155 * x218;
    const scalar_t x243 = -x144 * x158 + x159 * x218 + x242;
    const scalar_t x244 = x155 * x213 - x237 + x238;
    const scalar_t x245 = -x130 * x158 + x159 * x213 + x244;
    const scalar_t x246 = x119 * x71;
    const scalar_t x247 = x121 * x71;
    const scalar_t x248 = x155 * x215 + x246 - x247;
    const scalar_t x249 = -x112 * x158 + x159 * x215 + x248;
    const scalar_t x250 = x243 * x83 + x245 * x94 + x249 * x88;
    const scalar_t x251 = x184 * x215;
    const scalar_t x252 = -x112 * x186 + x187 * x215 + x251;
    const scalar_t x253 = x184 * x213 + x233 - x234;
    const scalar_t x254 = -x130 * x186 + x187 * x213 + x253;
    const scalar_t x255 = x184 * x218 - x246 + x247;
    const scalar_t x256 = -x144 * x186 + x187 * x218 + x255;
    const scalar_t x257 = x252 * x88 + x254 * x94 + x256 * x83;
    const scalar_t x258 = x18 * (x241 * x94 + x250 * x83 + x257 * x88);
    const scalar_t x259 = -x112 * x97 + x113 + x215 * x98;
    const scalar_t x260 = -x130 * x97 + x136 + x213 * x98;
    const scalar_t x261 = -x144 * x97 + x149 + x218 * x98;
    const scalar_t x262 = x259 * x88 + x260 * x94 + x261 * x83;
    const scalar_t x263 = -x144 * x78 + x145 + x218 * x81;
    const scalar_t x264 = -x112 * x78 + x123 + x215 * x81;
    const scalar_t x265 = -x130 * x78 + x140 + x213 * x81;
    const scalar_t x266 = x263 * x83 + x264 * x88 + x265 * x94;
    const scalar_t x267 = -x106 * x130 + x107 * x213 + x131;
    const scalar_t x268 = -x106 * x112 + x107 * x215 + x127;
    const scalar_t x269 = -x106 * x144 + x107 * x218 + x151;
    const scalar_t x270 = x267 * x94 + x268 * x88 + x269 * x83;
    const scalar_t x271 = x18 * (x195 * x262 + x196 * x266 + x197 * x270);
    const scalar_t x272 = x18 * (x195 * x226 + x196 * x230 + x197 * x221);
    const scalar_t x273 = x18 * (x195 * x257 + x196 * x250 + x197 * x241);
    const scalar_t x274 = x18 * (x201 * x262 + x202 * x266 + x203 * x270);
    const scalar_t x275 = x18 * (x201 * x226 + x202 * x230 + x203 * x221);
    const scalar_t x276 = x18 * (x201 * x257 + x202 * x250 + x203 * x241);
    const scalar_t x277 = x18 * (x207 * x262 + x208 * x266 + x209 * x270);
    const scalar_t x278 = x18 * (x207 * x226 + x208 * x230 + x209 * x221);
    const scalar_t x279 = x18 * (x207 * x257 + x208 * x250 + x209 * x241);
    const scalar_t x280 = x171 * x75;
    const scalar_t x281 = mu + pow(x171, 2) * x75 - x171 * x173 + x174 * x280;
    const scalar_t x282 = x155 * x280;
    const scalar_t x283 = x155 * x75;
    const scalar_t x284 = -x155 * x173 + x174 * x283 + x282;
    const scalar_t x285 = x184 * x280;
    const scalar_t x286 = x101 * x184;
    const scalar_t x287 = x170 * x286 - x173 * x184 + x285;
    const scalar_t x288 = x281 * x94 + x284 * x83 + x287 * x88;
    const scalar_t x289 = mu + pow(x155, 2) * x75 - x155 * x158 + x159 * x283;
    const scalar_t x290 = -x158 * x171 + x159 * x280 + x282;
    const scalar_t x291 = x184 * x283;
    const scalar_t x292 = x157 * x286 - x158 * x184 + x291;
    const scalar_t x293 = x289 * x83 + x290 * x94 + x292 * x88;
    const scalar_t x294 = mu + x183 * x286 + pow(x184, 2) * x75 - x184 * x186;
    const scalar_t x295 = -x155 * x186 + x187 * x283 + x291;
    const scalar_t x296 = -x171 * x186 + x187 * x280 + x285;
    const scalar_t x297 = x294 * x88 + x295 * x83 + x296 * x94;
    const scalar_t x298 = -x155 * x78 + x156 + x283 * x81;
    const scalar_t x299 = -x171 * x78 + x176 + x280 * x81;
    const scalar_t x300 = -x184 * x78 + x189 + x286 * x76;
    const scalar_t x301 = x298 * x83 + x299 * x94 + x300 * x88;
    const scalar_t x302 = -x184 * x97 + x185 + x286 * x96;
    const scalar_t x303 = -x155 * x97 + x163 + x283 * x98;
    const scalar_t x304 = -x171 * x97 + x180 + x280 * x98;
    const scalar_t x305 = x302 * x88 + x303 * x83 + x304 * x94;
    const scalar_t x306 = -x106 * x171 + x107 * x280 + x172;
    const scalar_t x307 = -x106 * x155 + x107 * x283 + x167;
    const scalar_t x308 = -x106 * x184 + x191 + x286 * x89;
    const scalar_t x309 = x306 * x94 + x307 * x83 + x308 * x88;
    const scalar_t x310 = x18 * (x195 * x305 + x196 * x301 + x197 * x309);
    const scalar_t x311 = -x133 * x171 + x134 * x280 + x231;
    const scalar_t x312 = -x133 * x155 + x134 * x283 + x244;
    const scalar_t x313 = x132 * x286 - x133 * x184 + x253;
    const scalar_t x314 = x311 * x94 + x312 * x83 + x313 * x88;
    const scalar_t x315 = x114 * x286 - x115 * x184 + x251;
    const scalar_t x316 = -x115 * x171 + x116 * x280 + x235;
    const scalar_t x317 = -x115 * x155 + x116 * x283 + x248;
    const scalar_t x318 = x315 * x88 + x316 * x94 + x317 * x83;
    const scalar_t x319 = -x146 * x155 + x147 * x283 + x242;
    const scalar_t x320 = -x146 * x171 + x147 * x280 + x239;
    const scalar_t x321 = x143 * x286 - x146 * x184 + x255;
    const scalar_t x322 = x319 * x83 + x320 * x94 + x321 * x88;
    const scalar_t x323 = x18 * (x195 * x318 + x196 * x322 + x197 * x314);
    const scalar_t x324 = x18 * (x195 * x297 + x196 * x293 + x197 * x288);
    const scalar_t x325 = x18 * (x201 * x305 + x202 * x301 + x203 * x309);
    const scalar_t x326 = x18 * (x201 * x318 + x202 * x322 + x203 * x314);
    const scalar_t x327 = x18 * (x201 * x297 + x202 * x293 + x203 * x288);
    const scalar_t x328 = x18 * (x207 * x305 + x208 * x301 + x209 * x309);
    const scalar_t x329 = x18 * (x207 * x318 + x208 * x322 + x209 * x314);
    const scalar_t x330 = x18 * (x207 * x297 + x208 * x293 + x209 * x288);
    const scalar_t x331 = x102 * x196 + x104 * x197 + x195 * x99;
    const scalar_t x332 = x195 * x87 + x196 * x82 + x197 * x93;
    const scalar_t x333 = x108 * x197 + x109 * x196 + x110 * x195;
    const scalar_t x334 = x117 * x195 + x124 * x196 + x128 * x197;
    const scalar_t x335 = x135 * x197 + x137 * x195 + x141 * x196;
    const scalar_t x336 = x148 * x196 + x150 * x195 + x152 * x197;
    const scalar_t x337 = x18 * (x195 * x334 + x196 * x336 + x197 * x335);
    const scalar_t x338 = x160 * x196 + x164 * x195 + x168 * x197;
    const scalar_t x339 = x175 * x197 + x177 * x196 + x181 * x195;
    const scalar_t x340 = x188 * x195 + x190 * x196 + x192 * x197;
    const scalar_t x341 = x18 * (x195 * x340 + x196 * x338 + x197 * x339);
    const scalar_t x342 = x18 * (x201 * x331 + x202 * x332 + x203 * x333);
    const scalar_t x343 = x18 * (x201 * x334 + x202 * x336 + x203 * x335);
    const scalar_t x344 = x18 * (x201 * x340 + x202 * x338 + x203 * x339);
    const scalar_t x345 = x18 * (x207 * x331 + x208 * x332 + x209 * x333);
    const scalar_t x346 = x18 * (x207 * x334 + x208 * x336 + x209 * x335);
    const scalar_t x347 = x18 * (x207 * x340 + x208 * x338 + x209 * x339);
    const scalar_t x348 = x195 * x222 + x196 * x225 + x197 * x223;
    const scalar_t x349 = x195 * x217 + x196 * x220 + x197 * x214;
    const scalar_t x350 = x195 * x229 + x196 * x227 + x197 * x228;
    const scalar_t x351 = x195 * x236 + x196 * x240 + x197 * x232;
    const scalar_t x352 = x195 * x249 + x196 * x243 + x197 * x245;
    const scalar_t x353 = x195 * x252 + x196 * x256 + x197 * x254;
    const scalar_t x354 = x18 * (x195 * x353 + x196 * x352 + x197 * x351);
    const scalar_t x355 = x195 * x259 + x196 * x261 + x197 * x260;
    const scalar_t x356 = x195 * x264 + x196 * x263 + x197 * x265;
    const scalar_t x357 = x195 * x268 + x196 * x269 + x197 * x267;
    const scalar_t x358 = x18 * (x201 * x355 + x202 * x356 + x203 * x357);
    const scalar_t x359 = x18 * (x201 * x348 + x202 * x350 + x203 * x349);
    const scalar_t x360 = x18 * (x201 * x353 + x202 * x352 + x203 * x351);
    const scalar_t x361 = x18 * (x207 * x355 + x208 * x356 + x209 * x357);
    const scalar_t x362 = x18 * (x207 * x348 + x208 * x350 + x209 * x349);
    const scalar_t x363 = x18 * (x207 * x353 + x208 * x352 + x209 * x351);
    const scalar_t x364 = x195 * x292 + x196 * x289 + x197 * x290;
    const scalar_t x365 = x195 * x287 + x196 * x284 + x197 * x281;
    const scalar_t x366 = x195 * x294 + x196 * x295 + x197 * x296;
    const scalar_t x367 = x195 * x300 + x196 * x298 + x197 * x299;
    const scalar_t x368 = x195 * x302 + x196 * x303 + x197 * x304;
    const scalar_t x369 = x195 * x308 + x196 * x307 + x197 * x306;
    const scalar_t x370 = x18 * (x201 * x368 + x202 * x367 + x203 * x369);
    const scalar_t x371 = x195 * x313 + x196 * x312 + x197 * x311;
    const scalar_t x372 = x195 * x315 + x196 * x317 + x197 * x316;
    const scalar_t x373 = x195 * x321 + x196 * x319 + x197 * x320;
    const scalar_t x374 = x18 * (x201 * x372 + x202 * x373 + x203 * x371);
    const scalar_t x375 = x18 * (x201 * x366 + x202 * x364 + x203 * x365);
    const scalar_t x376 = x18 * (x207 * x368 + x208 * x367 + x209 * x369);
    const scalar_t x377 = x18 * (x207 * x372 + x208 * x373 + x209 * x371);
    const scalar_t x378 = x18 * (x207 * x366 + x208 * x364 + x209 * x365);
    const scalar_t x379 = x102 * x202 + x104 * x203 + x201 * x99;
    const scalar_t x380 = x201 * x87 + x202 * x82 + x203 * x93;
    const scalar_t x381 = x108 * x203 + x109 * x202 + x110 * x201;
    const scalar_t x382 = x117 * x201 + x124 * x202 + x128 * x203;
    const scalar_t x383 = x135 * x203 + x137 * x201 + x141 * x202;
    const scalar_t x384 = x148 * x202 + x150 * x201 + x152 * x203;
    const scalar_t x385 = x18 * (x201 * x382 + x202 * x384 + x203 * x383);
    const scalar_t x386 = x160 * x202 + x164 * x201 + x168 * x203;
    const scalar_t x387 = x175 * x203 + x177 * x202 + x181 * x201;
    const scalar_t x388 = x188 * x201 + x190 * x202 + x192 * x203;
    const scalar_t x389 = x18 * (x201 * x388 + x202 * x386 + x203 * x387);
    const scalar_t x390 = x18 * (x207 * x379 + x208 * x380 + x209 * x381);
    const scalar_t x391 = x18 * (x207 * x382 + x208 * x384 + x209 * x383);
    const scalar_t x392 = x18 * (x207 * x388 + x208 * x386 + x209 * x387);
    const scalar_t x393 = x201 * x222 + x202 * x225 + x203 * x223;
    const scalar_t x394 = x201 * x217 + x202 * x220 + x203 * x214;
    const scalar_t x395 = x201 * x229 + x202 * x227 + x203 * x228;
    const scalar_t x396 = x201 * x236 + x202 * x240 + x203 * x232;
    const scalar_t x397 = x201 * x249 + x202 * x243 + x203 * x245;
    const scalar_t x398 = x201 * x252 + x202 * x256 + x203 * x254;
    const scalar_t x399 = x18 * (x201 * x398 + x202 * x397 + x203 * x396);
    const scalar_t x400 = x18 * (x207 * (x201 * x259 + x202 * x261 + x203 * x260) +
                                 x208 * (x201 * x264 + x202 * x263 + x203 * x265) +
                                 x209 * (x201 * x268 + x202 * x269 + x203 * x267));
    const scalar_t x401 = x18 * (x207 * x393 + x208 * x395 + x209 * x394);
    const scalar_t x402 = x18 * (x207 * x398 + x208 * x397 + x209 * x396);
    const scalar_t x403 = x201 * x292 + x202 * x289 + x203 * x290;
    const scalar_t x404 = x201 * x287 + x202 * x284 + x203 * x281;
    const scalar_t x405 = x201 * x294 + x202 * x295 + x203 * x296;
    const scalar_t x406 = x18 * (x207 * (x201 * x302 + x202 * x303 + x203 * x304) +
                                 x208 * (x201 * x300 + x202 * x298 + x203 * x299) +
                                 x209 * (x201 * x308 + x202 * x307 + x203 * x306));
    const scalar_t x407 = x18 * (x207 * (x201 * x315 + x202 * x317 + x203 * x316) +
                                 x208 * (x201 * x321 + x202 * x319 + x203 * x320) +
                                 x209 * (x201 * x313 + x202 * x312 + x203 * x311));
    const scalar_t x408 = x18 * (x207 * x405 + x208 * x403 + x209 * x404);
    const scalar_t x409 = x18 * (x207 * (x117 * x207 + x124 * x208 + x128 * x209) +
                                 x208 * (x148 * x208 + x150 * x207 + x152 * x209) +
                                 x209 * (x135 * x209 + x137 * x207 + x141 * x208));
    const scalar_t x410 = x18 * (x207 * (x188 * x207 + x190 * x208 + x192 * x209) +
                                 x208 * (x160 * x208 + x164 * x207 + x168 * x209) +
                                 x209 * (x175 * x209 + x177 * x208 + x181 * x207));
    const scalar_t x411 = x18 * (x207 * (x207 * x252 + x208 * x256 + x209 * x254) +
                                 x208 * (x207 * x249 + x208 * x243 + x209 * x245) +
                                 x209 * (x207 * x236 + x208 * x240 + x209 * x232));
    element_matrix[0] = x18 * (x105 * x88 + x111 * x94 + x83 * x95);
    element_matrix[1] = x154;
    element_matrix[2] = x194;
    element_matrix[3] = x198;
    element_matrix[4] = x199;
    element_matrix[5] = x200;
    element_matrix[6] = x204;
    element_matrix[7] = x205;
    element_matrix[8] = x206;
    element_matrix[9] = x210;
    element_matrix[10] = x211;
    element_matrix[11] = x212;
    element_matrix[12] = x154;
    element_matrix[13] = x18 * (x221 * x94 + x226 * x88 + x230 * x83);
    element_matrix[14] = x258;
    element_matrix[15] = x271;
    element_matrix[16] = x272;
    element_matrix[17] = x273;
    element_matrix[18] = x274;
    element_matrix[19] = x275;
    element_matrix[20] = x276;
    element_matrix[21] = x277;
    element_matrix[22] = x278;
    element_matrix[23] = x279;
    element_matrix[24] = x194;
    element_matrix[25] = x258;
    element_matrix[26] = x18 * (x288 * x94 + x293 * x83 + x297 * x88);
    element_matrix[27] = x310;
    element_matrix[28] = x323;
    element_matrix[29] = x324;
    element_matrix[30] = x325;
    element_matrix[31] = x326;
    element_matrix[32] = x327;
    element_matrix[33] = x328;
    element_matrix[34] = x329;
    element_matrix[35] = x330;
    element_matrix[36] = x198;
    element_matrix[37] = x271;
    element_matrix[38] = x310;
    element_matrix[39] = x18 * (x195 * x331 + x196 * x332 + x197 * x333);
    element_matrix[40] = x337;
    element_matrix[41] = x341;
    element_matrix[42] = x342;
    element_matrix[43] = x343;
    element_matrix[44] = x344;
    element_matrix[45] = x345;
    element_matrix[46] = x346;
    element_matrix[47] = x347;
    element_matrix[48] = x199;
    element_matrix[49] = x272;
    element_matrix[50] = x323;
    element_matrix[51] = x337;
    element_matrix[52] = x18 * (x195 * x348 + x196 * x350 + x197 * x349);
    element_matrix[53] = x354;
    element_matrix[54] = x358;
    element_matrix[55] = x359;
    element_matrix[56] = x360;
    element_matrix[57] = x361;
    element_matrix[58] = x362;
    element_matrix[59] = x363;
    element_matrix[60] = x200;
    element_matrix[61] = x273;
    element_matrix[62] = x324;
    element_matrix[63] = x341;
    element_matrix[64] = x354;
    element_matrix[65] = x18 * (x195 * x366 + x196 * x364 + x197 * x365);
    element_matrix[66] = x370;
    element_matrix[67] = x374;
    element_matrix[68] = x375;
    element_matrix[69] = x376;
    element_matrix[70] = x377;
    element_matrix[71] = x378;
    element_matrix[72] = x204;
    element_matrix[73] = x274;
    element_matrix[74] = x325;
    element_matrix[75] = x342;
    element_matrix[76] = x358;
    element_matrix[77] = x370;
    element_matrix[78] = x18 * (x201 * x379 + x202 * x380 + x203 * x381);
    element_matrix[79] = x385;
    element_matrix[80] = x389;
    element_matrix[81] = x390;
    element_matrix[82] = x391;
    element_matrix[83] = x392;
    element_matrix[84] = x205;
    element_matrix[85] = x275;
    element_matrix[86] = x326;
    element_matrix[87] = x343;
    element_matrix[88] = x359;
    element_matrix[89] = x374;
    element_matrix[90] = x385;
    element_matrix[91] = x18 * (x201 * x393 + x202 * x395 + x203 * x394);
    element_matrix[92] = x399;
    element_matrix[93] = x400;
    element_matrix[94] = x401;
    element_matrix[95] = x402;
    element_matrix[96] = x206;
    element_matrix[97] = x276;
    element_matrix[98] = x327;
    element_matrix[99] = x344;
    element_matrix[100] = x360;
    element_matrix[101] = x375;
    element_matrix[102] = x389;
    element_matrix[103] = x399;
    element_matrix[104] = x18 * (x201 * x405 + x202 * x403 + x203 * x404);
    element_matrix[105] = x406;
    element_matrix[106] = x407;
    element_matrix[107] = x408;
    element_matrix[108] = x210;
    element_matrix[109] = x277;
    element_matrix[110] = x328;
    element_matrix[111] = x345;
    element_matrix[112] = x361;
    element_matrix[113] = x376;
    element_matrix[114] = x390;
    element_matrix[115] = x400;
    element_matrix[116] = x406;
    element_matrix[117] = x18 * (x207 * (x102 * x208 + x104 * x209 + x207 * x99) +
                                 x208 * (x207 * x87 + x208 * x82 + x209 * x93) +
                                 x209 * (x108 * x209 + x109 * x208 + x110 * x207));
    element_matrix[118] = x409;
    element_matrix[119] = x410;
    element_matrix[120] = x211;
    element_matrix[121] = x278;
    element_matrix[122] = x329;
    element_matrix[123] = x346;
    element_matrix[124] = x362;
    element_matrix[125] = x377;
    element_matrix[126] = x391;
    element_matrix[127] = x401;
    element_matrix[128] = x407;
    element_matrix[129] = x409;
    element_matrix[130] = x18 * (x207 * (x207 * x222 + x208 * x225 + x209 * x223) +
                                 x208 * (x207 * x229 + x208 * x227 + x209 * x228) +
                                 x209 * (x207 * x217 + x208 * x220 + x209 * x214));
    element_matrix[131] = x411;
    element_matrix[132] = x212;
    element_matrix[133] = x279;
    element_matrix[134] = x330;
    element_matrix[135] = x347;
    element_matrix[136] = x363;
    element_matrix[137] = x378;
    element_matrix[138] = x392;
    element_matrix[139] = x402;
    element_matrix[140] = x408;
    element_matrix[141] = x410;
    element_matrix[142] = x411;
    element_matrix[143] = x18 * (x207 * (x207 * x294 + x208 * x295 + x209 * x296) +
                                 x208 * (x207 * x292 + x208 * x289 + x209 * x290) +
                                 x209 * (x207 * x287 + x208 * x284 + x209 * x281));
}

#endif  // TET4_NEOHOOKEAN_OGDEN_INLINE_CPU_H
