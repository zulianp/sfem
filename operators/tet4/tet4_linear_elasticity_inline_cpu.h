#ifndef TET4_LINEAR_ELASTICITY_INLINE_CPU_H
#define TET4_LINEAR_ELASTICITY_INLINE_CPU_H

#include "tet4_inline_cpu.h"

#define RPOW2(l) (1. / ((l) * (l)))

static SFEM_INLINE void tet4_gradient_3(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const scalar_t jacobian_determinant,
                                        const scalar_t *const SFEM_RESTRICT ux,
                                        const scalar_t *const SFEM_RESTRICT uy,
                                        const scalar_t *const SFEM_RESTRICT uz,
                                        scalar_t *const SFEM_RESTRICT disp_grad) {
    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = ux[0] - ux[1];
    const scalar_t x2 = ux[0] - ux[2];
    const scalar_t x3 = ux[0] - ux[3];
    const scalar_t x4 = uy[0] - uy[1];
    const scalar_t x5 = uy[0] - uy[2];
    const scalar_t x6 = uy[0] - uy[3];
    const scalar_t x7 = uz[0] - uz[1];
    const scalar_t x8 = uz[0] - uz[2];
    const scalar_t x9 = uz[0] - uz[3];
    disp_grad[0] = x0 * (-adjugate[0] * x1 - adjugate[3] * x2 - adjugate[6] * x3);
    disp_grad[1] = x0 * (-adjugate[1] * x1 - adjugate[4] * x2 - adjugate[7] * x3);
    disp_grad[2] = x0 * (-adjugate[2] * x1 - adjugate[5] * x2 - adjugate[8] * x3);
    disp_grad[3] = x0 * (-adjugate[0] * x4 - adjugate[3] * x5 - adjugate[6] * x6);
    disp_grad[4] = x0 * (-adjugate[1] * x4 - adjugate[4] * x5 - adjugate[7] * x6);
    disp_grad[5] = x0 * (-adjugate[2] * x4 - adjugate[5] * x5 - adjugate[8] * x6);
    disp_grad[6] = x0 * (-adjugate[0] * x7 - adjugate[3] * x8 - adjugate[6] * x9);
    disp_grad[7] = x0 * (-adjugate[1] * x7 - adjugate[4] * x8 - adjugate[7] * x9);
    disp_grad[8] = x0 * (-adjugate[2] * x7 - adjugate[5] * x8 - adjugate[8] * x9);
}

static SFEM_INLINE void tet4_linear_elasticity_loperand(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT adjugate,
        scalar_t *const SFEM_RESTRICT in_disp_grad_out_PxJinv_t) {
    // Shorter name
    scalar_t *const buff = in_disp_grad_out_PxJinv_t;

    const scalar_t x0 = (1.0 / 6.0) * mu;
    const scalar_t x1 = x0 * (buff[1] + buff[3]);
    const scalar_t x2 = x0 * (buff[2] + buff[6]);
    const scalar_t x3 = 2 * mu;
    const scalar_t x4 = lambda * (buff[0] + buff[4] + buff[8]);
    const scalar_t x5 = (1.0 / 6.0) * buff[0] * x3 + (1.0 / 6.0) * x4;
    const scalar_t x6 = x0 * (buff[5] + buff[7]);
    const scalar_t x7 = (1.0 / 6.0) * buff[4] * x3 + (1.0 / 6.0) * x4;
    const scalar_t x8 = (1.0 / 6.0) * buff[8] * x3 + (1.0 / 6.0) * x4;
    buff[0] = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
    buff[1] = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
    buff[2] = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
    buff[3] = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
    buff[4] = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
    buff[5] = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
    buff[6] = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
    buff[7] = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
    buff[8] = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
}

static SFEM_INLINE void tet4_linear_elasticity_apply_adj(const scalar_t *const SFEM_RESTRICT
                                                                 adjugate,
                                                         const scalar_t jacobian_determinant,
                                                         const scalar_t mu,
                                                         const scalar_t lambda,
                                                         const scalar_t *const SFEM_RESTRICT ux,
                                                         const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz,
                                                         accumulator_t *const SFEM_RESTRICT outx,
                                                         accumulator_t *const SFEM_RESTRICT outy,
                                                         accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t PxJinv_t[9];
    tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, PxJinv_t);
    tet4_linear_elasticity_loperand(mu, lambda, adjugate, PxJinv_t);

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

static SFEM_INLINE void tet4_linear_elasticity_apply_add_adj(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT adjugate,
        const scalar_t jacobian_determinant,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        accumulator_t *const SFEM_RESTRICT outx,
        accumulator_t *const SFEM_RESTRICT outy,
        accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t PxJinv_t[9];
    tet4_gradient_3(adjugate, jacobian_determinant, ux, uy, uz, PxJinv_t);
    tet4_linear_elasticity_loperand(mu, lambda, adjugate, PxJinv_t);

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

static SFEM_INLINE void tet4_linear_elasticity_diag_adj(const scalar_t mu,
                                                        const scalar_t lambda,
                                                        const scalar_t *const SFEM_RESTRICT
                                                                jacobian_adjugate,
                                                        const scalar_t jacobian_determinant,
                                                        accumulator_t *const SFEM_RESTRICT diagx,
                                                        accumulator_t *const SFEM_RESTRICT diagy,
                                                        accumulator_t *const SFEM_RESTRICT diagz) {
    const scalar_t x0 = lambda + 2 * mu;
    const scalar_t x1 = jacobian_adjugate[0] + jacobian_adjugate[3] + jacobian_adjugate[6];
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = jacobian_adjugate[2] + jacobian_adjugate[5] + jacobian_adjugate[8];
    const scalar_t x4 = mu * x3;
    const scalar_t x5 =
            jacobian_adjugate[2] * x4 + jacobian_adjugate[5] * x4 + jacobian_adjugate[8] * x4;
    const scalar_t x6 = jacobian_adjugate[1] + jacobian_adjugate[4] + jacobian_adjugate[7];
    const scalar_t x7 = mu * x6;
    const scalar_t x8 =
            jacobian_adjugate[1] * x7 + jacobian_adjugate[4] * x7 + jacobian_adjugate[7] * x7;
    const scalar_t x9 = (1.0 / 6.0) / jacobian_determinant;
    const scalar_t x10 = POW2(jacobian_adjugate[1]);
    const scalar_t x11 = mu * x10;
    const scalar_t x12 = POW2(jacobian_adjugate[2]);
    const scalar_t x13 = mu * x12;
    const scalar_t x14 = POW2(jacobian_adjugate[0]);
    const scalar_t x15 = POW2(jacobian_adjugate[4]);
    const scalar_t x16 = mu * x15;
    const scalar_t x17 = POW2(jacobian_adjugate[5]);
    const scalar_t x18 = mu * x17;
    const scalar_t x19 = POW2(jacobian_adjugate[3]);
    const scalar_t x20 = POW2(jacobian_adjugate[7]);
    const scalar_t x21 = mu * x20;
    const scalar_t x22 = POW2(jacobian_adjugate[8]);
    const scalar_t x23 = mu * x22;
    const scalar_t x24 = POW2(jacobian_adjugate[6]);
    const scalar_t x25 = x0 * x6;
    const scalar_t x26 = mu * x1;
    const scalar_t x27 =
            jacobian_adjugate[0] * x26 + jacobian_adjugate[3] * x26 + jacobian_adjugate[6] * x26;
    const scalar_t x28 = mu * x14;
    const scalar_t x29 = mu * x19;
    const scalar_t x30 = mu * x24;
    const scalar_t x31 = x0 * x3;
    diagx[0] = x9 * (jacobian_adjugate[0] * x2 + jacobian_adjugate[3] * x2 +
                     jacobian_adjugate[6] * x2 + x5 + x8);
    diagx[1] = x9 * (x0 * x14 + x11 + x13);
    diagx[2] = x9 * (x0 * x19 + x16 + x18);
    diagx[3] = x9 * (x0 * x24 + x21 + x23);
    diagy[0] = x9 * (jacobian_adjugate[1] * x25 + jacobian_adjugate[4] * x25 +
                     jacobian_adjugate[7] * x25 + x27 + x5);
    diagy[1] = x9 * (x0 * x10 + x13 + x28);
    diagy[2] = x9 * (x0 * x15 + x18 + x29);
    diagy[3] = x9 * (x0 * x20 + x23 + x30);
    diagz[0] = x9 * (jacobian_adjugate[2] * x31 + jacobian_adjugate[5] * x31 +
                     jacobian_adjugate[8] * x31 + x27 + x8);
    diagz[1] = x9 * (x0 * x12 + x11 + x28);
    diagz[2] = x9 * (x0 * x17 + x16 + x29);
    diagz[3] = x9 * (x0 * x22 + x21 + x30);
}

static SFEM_INLINE void tet4_linear_elasticity_value_points(const scalar_t mu,
                                                            const scalar_t lambda,
                                                            const scalar_t px0,
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
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            accumulator_t *const SFEM_RESTRICT
                                                                    element_scalar) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = -pz0 + pz3;
    const scalar_t x3 = x1 * x2;
    const scalar_t x4 = -px0 + px2;
    const scalar_t x5 = -py0 + py3;
    const scalar_t x6 = -pz0 + pz1;
    const scalar_t x7 = -px0 + px3;
    const scalar_t x8 = -py0 + py1;
    const scalar_t x9 = -pz0 + pz2;
    const scalar_t x10 = x8 * x9;
    const scalar_t x11 = x5 * x9;
    const scalar_t x12 = x2 * x8;
    const scalar_t x13 = x1 * x6;
    const scalar_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const scalar_t x15 = 1.0 / x14;
    const scalar_t x16 = x15 * (-x11 + x3);
    const scalar_t x17 = x15 * (-x12 + x5 * x6);
    const scalar_t x18 = x15 * (x10 - x13);
    const scalar_t x19 = -x16 - x17 - x18;
    const scalar_t x20 = ux[0] * x19 + ux[1] * x16 + ux[2] * x17 + ux[3] * x18;
    const scalar_t x21 = POW2(x20);
    const scalar_t x22 = (1.0 / 12.0) * lambda;
    const scalar_t x23 = x15 * (-x2 * x4 + x7 * x9);
    const scalar_t x24 = x15 * (x0 * x2 - x6 * x7);
    const scalar_t x25 = x15 * (-x0 * x9 + x4 * x6);
    const scalar_t x26 = -x23 - x24 - x25;
    const scalar_t x27 = uy[0] * x26 + uy[1] * x23 + uy[2] * x24 + uy[3] * x25;
    const scalar_t x28 = POW2(x27);
    const scalar_t x29 = x15 * (-x0 * x5 + x7 * x8);
    const scalar_t x30 = x15 * (x0 * x1 - x4 * x8);
    const scalar_t x31 = x15 * (-x1 * x7 + x4 * x5);
    const scalar_t x32 = -x29 - x30 - x31;
    const scalar_t x33 = uz[2] * x29 + uz[3] * x30 + uz[0] * x32 + uz[1] * x31;
    const scalar_t x34 = POW2(x33);
    const scalar_t x35 = ux[0] * x32 + ux[1] * x31 + ux[2] * x29 + ux[3] * x30;
    const scalar_t x36 = (1.0 / 12.0) * mu;
    const scalar_t x37 = ux[0] * x26 + ux[1] * x23 + ux[2] * x24 + ux[3] * x25;
    const scalar_t x38 = (1.0 / 6.0) * mu;
    const scalar_t x39 = uy[0] * x32 + uy[1] * x31 + uy[2] * x29 + uy[3] * x30;
    const scalar_t x40 = uy[0] * x19 + uy[1] * x16 + uy[2] * x17 + uy[3] * x18;
    const scalar_t x41 = uz[2] * x24 + uz[3] * x25 + uz[0] * x26 + uz[1] * x23;
    const scalar_t x42 = uz[2] * x17 + uz[3] * x18 + uz[0] * x19 + uz[1] * x16;
    const scalar_t x43 = (1.0 / 6.0) * lambda * x20;
    element_scalar[0] =
            x14 * ((1.0 / 6.0) * lambda * x27 * x33 + x21 * x22 + x21 * x38 + x22 * x28 +
                   x22 * x34 + x27 * x43 + x28 * x38 + x33 * x43 + x34 * x38 + POW2(x35) * x36 +
                   x35 * x38 * x42 + x36 * POW2(x37) + x36 * POW2(x39) + x36 * POW2(x40) +
                   x36 * POW2(x41) + x36 * POW2(x42) + x37 * x38 * x40 + x38 * x39 * x41);
}

static SFEM_INLINE void tet4_linear_elasticity_crs_points(const scalar_t mu,
                                                              const scalar_t lambda,
                                                              const scalar_t px0,
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
                                                              accumulator_t *const SFEM_RESTRICT
                                                                      element_matrix) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = -pz0 + pz3;
    const scalar_t x3 = x1 * x2;
    const scalar_t x4 = -px0 + px2;
    const scalar_t x5 = -py0 + py3;
    const scalar_t x6 = -pz0 + pz1;
    const scalar_t x7 = -px0 + px3;
    const scalar_t x8 = -py0 + py1;
    const scalar_t x9 = -pz0 + pz2;
    const scalar_t x10 = x8 * x9;
    const scalar_t x11 = x5 * x9;
    const scalar_t x12 = x2 * x8;
    const scalar_t x13 = x1 * x6;
    const scalar_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const scalar_t x15 = -x11 + x3;
    const scalar_t x16 = RPOW2(x14);
    const scalar_t x17 = x10 - x13;
    const scalar_t x18 = x16 * x17;
    const scalar_t x19 = x15 * x18;
    const scalar_t x20 = (1.0 / 3.0) * lambda;
    const scalar_t x21 = -x12 + x5 * x6;
    const scalar_t x22 = x20 * x21;
    const scalar_t x23 = x15 * x16;
    const scalar_t x24 = (2.0 / 3.0) * mu;
    const scalar_t x25 = x21 * x24;
    const scalar_t x26 = POW2(x15);
    const scalar_t x27 = (1.0 / 6.0) * lambda;
    const scalar_t x28 = x16 * x27;
    const scalar_t x29 = -x1 * x7 + x4 * x5;
    const scalar_t x30 = POW2(x29);
    const scalar_t x31 = (1.0 / 6.0) * mu;
    const scalar_t x32 = x16 * x31;
    const scalar_t x33 = x30 * x32;
    const scalar_t x34 = -x2 * x4 + x7 * x9;
    const scalar_t x35 = POW2(x34);
    const scalar_t x36 = x32 * x35;
    const scalar_t x37 = (1.0 / 3.0) * mu;
    const scalar_t x38 = x16 * x37;
    const scalar_t x39 = x26 * x28 + x26 * x38 + x33 + x36;
    const scalar_t x40 = POW2(x21);
    const scalar_t x41 = -x0 * x5 + x7 * x8;
    const scalar_t x42 = POW2(x41);
    const scalar_t x43 = x32 * x42;
    const scalar_t x44 = x0 * x2 - x6 * x7;
    const scalar_t x45 = POW2(x44);
    const scalar_t x46 = x32 * x45;
    const scalar_t x47 = x28 * x40 + x38 * x40 + x43 + x46;
    const scalar_t x48 = x16 * POW2(x17);
    const scalar_t x49 = x0 * x1 - x4 * x8;
    const scalar_t x50 = POW2(x49);
    const scalar_t x51 = x32 * x50;
    const scalar_t x52 = -x0 * x9 + x4 * x6;
    const scalar_t x53 = POW2(x52);
    const scalar_t x54 = x32 * x53;
    const scalar_t x55 = x27 * x48 + x37 * x48 + x51 + x54;
    const scalar_t x56 = x29 * x41;
    const scalar_t x57 = x38 * x56;
    const scalar_t x58 = x38 * x49;
    const scalar_t x59 = x29 * x58;
    const scalar_t x60 = x41 * x58;
    const scalar_t x61 = x57 + x59 + x60;
    const scalar_t x62 = x38 * x44;
    const scalar_t x63 = x34 * x62;
    const scalar_t x64 = x34 * x52;
    const scalar_t x65 = x38 * x64;
    const scalar_t x66 = x52 * x62;
    const scalar_t x67 = x63 + x65 + x66;
    const scalar_t x68 = x15 * x28;
    const scalar_t x69 = x32 * x56;
    const scalar_t x70 = x32 * x44;
    const scalar_t x71 = x34 * x70;
    const scalar_t x72 = x15 * x38;
    const scalar_t x73 = x21 * x72;
    const scalar_t x74 = x21 * x68 + x69 + x71 + x73;
    const scalar_t x75 = x32 * x49;
    const scalar_t x76 = x29 * x75;
    const scalar_t x77 = x32 * x64;
    const scalar_t x78 = x17 * x72;
    const scalar_t x79 = x17 * x68 + x76 + x77 + x78;
    const scalar_t x80 = x14 * (-x39 - x74 - x79);
    const scalar_t x81 = x17 * x21;
    const scalar_t x82 = x41 * x75;
    const scalar_t x83 = x52 * x70;
    const scalar_t x84 = x38 * x81;
    const scalar_t x85 = x28 * x81 + x82 + x83 + x84;
    const scalar_t x86 = x14 * (-x47 - x74 - x85);
    const scalar_t x87 = x14 * (-x55 - x79 - x85);
    const scalar_t x88 = x32 * x34;
    const scalar_t x89 = x15 * x88 + x34 * x68;
    const scalar_t x90 = x28 * x34;
    const scalar_t x91 = x15 * x70 + x21 * x90;
    const scalar_t x92 = x32 * x52;
    const scalar_t x93 = x15 * x92 + x17 * x90;
    const scalar_t x94 = x89 + x91 + x93;
    const scalar_t x95 = x21 * x88 + x44 * x68;
    const scalar_t x96 = x28 * x44;
    const scalar_t x97 = x21 * x70 + x21 * x96;
    const scalar_t x98 = x17 * x96 + x21 * x92;
    const scalar_t x99 = x95 + x97 + x98;
    const scalar_t x100 = x17 * x88 + x52 * x68;
    const scalar_t x101 = x28 * x52;
    const scalar_t x102 = x101 * x21 + x17 * x70;
    const scalar_t x103 = x101 * x17 + x17 * x92;
    const scalar_t x104 = x100 + x102 + x103;
    const scalar_t x105 = x14 * (x104 + x94 + x99);
    const scalar_t x106 = -x14 * x94;
    const scalar_t x107 = -x14 * x99;
    const scalar_t x108 = -x104 * x14;
    const scalar_t x109 = x29 * x32;
    const scalar_t x110 = x109 * x15 + x29 * x68;
    const scalar_t x111 = x28 * x29;
    const scalar_t x112 = x32 * x41;
    const scalar_t x113 = x111 * x21 + x112 * x15;
    const scalar_t x114 = x111 * x17 + x15 * x75;
    const scalar_t x115 = x110 + x113 + x114;
    const scalar_t x116 = x109 * x21 + x41 * x68;
    const scalar_t x117 = x28 * x41;
    const scalar_t x118 = x112 * x21 + x117 * x21;
    const scalar_t x119 = x117 * x17 + x21 * x75;
    const scalar_t x120 = x116 + x118 + x119;
    const scalar_t x121 = x109 * x17 + x49 * x68;
    const scalar_t x122 = x28 * x49;
    const scalar_t x123 = x112 * x17 + x122 * x21;
    const scalar_t x124 = x122 * x17 + x17 * x75;
    const scalar_t x125 = x121 + x123 + x124;
    const scalar_t x126 = x14 * (x115 + x120 + x125);
    const scalar_t x127 = -x115 * x14;
    const scalar_t x128 = -x120 * x14;
    const scalar_t x129 = -x125 * x14;
    const scalar_t x130 = x14 * x74;
    const scalar_t x131 = x14 * x79;
    const scalar_t x132 = x14 * (-x100 - x89 - x95);
    const scalar_t x133 = x14 * x89;
    const scalar_t x134 = x14 * x95;
    const scalar_t x135 = x100 * x14;
    const scalar_t x136 = x14 * (-x110 - x116 - x121);
    const scalar_t x137 = x110 * x14;
    const scalar_t x138 = x116 * x14;
    const scalar_t x139 = x121 * x14;
    const scalar_t x140 = x14 * x85;
    const scalar_t x141 = x14 * (-x102 - x91 - x97);
    const scalar_t x142 = x14 * x91;
    const scalar_t x143 = x14 * x97;
    const scalar_t x144 = x102 * x14;
    const scalar_t x145 = x14 * (-x113 - x118 - x123);
    const scalar_t x146 = x113 * x14;
    const scalar_t x147 = x118 * x14;
    const scalar_t x148 = x123 * x14;
    const scalar_t x149 = x14 * (-x103 - x93 - x98);
    const scalar_t x150 = x14 * x93;
    const scalar_t x151 = x14 * x98;
    const scalar_t x152 = x103 * x14;
    const scalar_t x153 = x14 * (-x114 - x119 - x124);
    const scalar_t x154 = x114 * x14;
    const scalar_t x155 = x119 * x14;
    const scalar_t x156 = x124 * x14;
    const scalar_t x157 = x16 * x20;
    const scalar_t x158 = x157 * x44;
    const scalar_t x159 = x16 * x24;
    const scalar_t x160 = x159 * x44;
    const scalar_t x161 = x26 * x32;
    const scalar_t x162 = x161 + x28 * x35 + x33 + x35 * x38;
    const scalar_t x163 = x32 * x40;
    const scalar_t x164 = x163 + x28 * x45 + x38 * x45 + x43;
    const scalar_t x165 = x31 * x48;
    const scalar_t x166 = x165 + x28 * x53 + x38 * x53 + x51;
    const scalar_t x167 = x73 + x78 + x84;
    const scalar_t x168 = x15 * x32;
    const scalar_t x169 = x168 * x21;
    const scalar_t x170 = x169 + x34 * x96 + x63 + x69;
    const scalar_t x171 = x168 * x17;
    const scalar_t x172 = x171 + x28 * x64 + x65 + x76;
    const scalar_t x173 = x14 * (-x162 - x170 - x172);
    const scalar_t x174 = x32 * x81;
    const scalar_t x175 = x174 + x52 * x96 + x66 + x82;
    const scalar_t x176 = x14 * (-x164 - x170 - x175);
    const scalar_t x177 = x14 * (-x166 - x172 - x175);
    const scalar_t x178 = x29 * x88 + x29 * x90;
    const scalar_t x179 = x29 * x96 + x41 * x88;
    const scalar_t x180 = x101 * x29 + x34 * x75;
    const scalar_t x181 = x178 + x179 + x180;
    const scalar_t x182 = x29 * x70 + x41 * x90;
    const scalar_t x183 = x41 * x70 + x41 * x96;
    const scalar_t x184 = x101 * x41 + x44 * x75;
    const scalar_t x185 = x182 + x183 + x184;
    const scalar_t x186 = x29 * x92 + x49 * x90;
    const scalar_t x187 = x41 * x92 + x49 * x96;
    const scalar_t x188 = x101 * x49 + x52 * x75;
    const scalar_t x189 = x186 + x187 + x188;
    const scalar_t x190 = x14 * (x181 + x185 + x189);
    const scalar_t x191 = -x14 * x181;
    const scalar_t x192 = -x14 * x185;
    const scalar_t x193 = -x14 * x189;
    const scalar_t x194 = x14 * x170;
    const scalar_t x195 = x14 * x172;
    const scalar_t x196 = x14 * (-x178 - x182 - x186);
    const scalar_t x197 = x14 * x178;
    const scalar_t x198 = x14 * x182;
    const scalar_t x199 = x14 * x186;
    const scalar_t x200 = x14 * x175;
    const scalar_t x201 = x14 * (-x179 - x183 - x187);
    const scalar_t x202 = x14 * x179;
    const scalar_t x203 = x14 * x183;
    const scalar_t x204 = x14 * x187;
    const scalar_t x205 = x14 * (-x180 - x184 - x188);
    const scalar_t x206 = x14 * x180;
    const scalar_t x207 = x14 * x184;
    const scalar_t x208 = x14 * x188;
    const scalar_t x209 = x157 * x49;
    const scalar_t x210 = x159 * x49;
    const scalar_t x211 = x161 + x28 * x30 + x30 * x38 + x36;
    const scalar_t x212 = x163 + x28 * x42 + x38 * x42 + x46;
    const scalar_t x213 = x165 + x28 * x50 + x38 * x50 + x54;
    const scalar_t x214 = x169 + x28 * x56 + x57 + x71;
    const scalar_t x215 = x122 * x29 + x171 + x59 + x77;
    const scalar_t x216 = x14 * (-x211 - x214 - x215);
    const scalar_t x217 = x122 * x41 + x174 + x60 + x83;
    const scalar_t x218 = x14 * (-x212 - x214 - x217);
    const scalar_t x219 = x14 * (-x213 - x215 - x217);
    const scalar_t x220 = x14 * x214;
    const scalar_t x221 = x14 * x215;
    const scalar_t x222 = x14 * x217;
    element_matrix[0] = x14 * (x18 * x22 + x18 * x25 + x19 * x20 + x19 * x24 + x22 * x23 +
                               x23 * x25 + x39 + x47 + x55 + x61 + x67);
    element_matrix[1] = x80;
    element_matrix[2] = x86;
    element_matrix[3] = x87;
    element_matrix[4] = x105;
    element_matrix[5] = x106;
    element_matrix[6] = x107;
    element_matrix[7] = x108;
    element_matrix[8] = x126;
    element_matrix[9] = x127;
    element_matrix[10] = x128;
    element_matrix[11] = x129;
    element_matrix[12] = x80;
    element_matrix[13] = x14 * x39;
    element_matrix[14] = x130;
    element_matrix[15] = x131;
    element_matrix[16] = x132;
    element_matrix[17] = x133;
    element_matrix[18] = x134;
    element_matrix[19] = x135;
    element_matrix[20] = x136;
    element_matrix[21] = x137;
    element_matrix[22] = x138;
    element_matrix[23] = x139;
    element_matrix[24] = x86;
    element_matrix[25] = x130;
    element_matrix[26] = x14 * x47;
    element_matrix[27] = x140;
    element_matrix[28] = x141;
    element_matrix[29] = x142;
    element_matrix[30] = x143;
    element_matrix[31] = x144;
    element_matrix[32] = x145;
    element_matrix[33] = x146;
    element_matrix[34] = x147;
    element_matrix[35] = x148;
    element_matrix[36] = x87;
    element_matrix[37] = x131;
    element_matrix[38] = x140;
    element_matrix[39] = x14 * x55;
    element_matrix[40] = x149;
    element_matrix[41] = x150;
    element_matrix[42] = x151;
    element_matrix[43] = x152;
    element_matrix[44] = x153;
    element_matrix[45] = x154;
    element_matrix[46] = x155;
    element_matrix[47] = x156;
    element_matrix[48] = x105;
    element_matrix[49] = x132;
    element_matrix[50] = x141;
    element_matrix[51] = x149;
    element_matrix[52] = x14 * (x157 * x64 + x158 * x34 + x158 * x52 + x159 * x64 + x160 * x34 +
                                x160 * x52 + x162 + x164 + x166 + x167 + x61);
    element_matrix[53] = x173;
    element_matrix[54] = x176;
    element_matrix[55] = x177;
    element_matrix[56] = x190;
    element_matrix[57] = x191;
    element_matrix[58] = x192;
    element_matrix[59] = x193;
    element_matrix[60] = x106;
    element_matrix[61] = x133;
    element_matrix[62] = x142;
    element_matrix[63] = x150;
    element_matrix[64] = x173;
    element_matrix[65] = x14 * x162;
    element_matrix[66] = x194;
    element_matrix[67] = x195;
    element_matrix[68] = x196;
    element_matrix[69] = x197;
    element_matrix[70] = x198;
    element_matrix[71] = x199;
    element_matrix[72] = x107;
    element_matrix[73] = x134;
    element_matrix[74] = x143;
    element_matrix[75] = x151;
    element_matrix[76] = x176;
    element_matrix[77] = x194;
    element_matrix[78] = x14 * x164;
    element_matrix[79] = x200;
    element_matrix[80] = x201;
    element_matrix[81] = x202;
    element_matrix[82] = x203;
    element_matrix[83] = x204;
    element_matrix[84] = x108;
    element_matrix[85] = x135;
    element_matrix[86] = x144;
    element_matrix[87] = x152;
    element_matrix[88] = x177;
    element_matrix[89] = x195;
    element_matrix[90] = x200;
    element_matrix[91] = x14 * x166;
    element_matrix[92] = x205;
    element_matrix[93] = x206;
    element_matrix[94] = x207;
    element_matrix[95] = x208;
    element_matrix[96] = x126;
    element_matrix[97] = x136;
    element_matrix[98] = x145;
    element_matrix[99] = x153;
    element_matrix[100] = x190;
    element_matrix[101] = x196;
    element_matrix[102] = x201;
    element_matrix[103] = x205;
    element_matrix[104] = x14 * (x157 * x56 + x159 * x56 + x167 + x209 * x29 + x209 * x41 +
                                 x210 * x29 + x210 * x41 + x211 + x212 + x213 + x67);
    element_matrix[105] = x216;
    element_matrix[106] = x218;
    element_matrix[107] = x219;
    element_matrix[108] = x127;
    element_matrix[109] = x137;
    element_matrix[110] = x146;
    element_matrix[111] = x154;
    element_matrix[112] = x191;
    element_matrix[113] = x197;
    element_matrix[114] = x202;
    element_matrix[115] = x206;
    element_matrix[116] = x216;
    element_matrix[117] = x14 * x211;
    element_matrix[118] = x220;
    element_matrix[119] = x221;
    element_matrix[120] = x128;
    element_matrix[121] = x138;
    element_matrix[122] = x147;
    element_matrix[123] = x155;
    element_matrix[124] = x192;
    element_matrix[125] = x198;
    element_matrix[126] = x203;
    element_matrix[127] = x207;
    element_matrix[128] = x218;
    element_matrix[129] = x220;
    element_matrix[130] = x14 * x212;
    element_matrix[131] = x222;
    element_matrix[132] = x129;
    element_matrix[133] = x139;
    element_matrix[134] = x148;
    element_matrix[135] = x156;
    element_matrix[136] = x193;
    element_matrix[137] = x199;
    element_matrix[138] = x204;
    element_matrix[139] = x208;
    element_matrix[140] = x219;
    element_matrix[141] = x221;
    element_matrix[142] = x222;
    element_matrix[143] = x14 * x213;
}

static SFEM_INLINE void tet4_linear_elasticity_crs_adj(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t jacobian_determinant,
        accumulator_t *const SFEM_RESTRICT element_matrix) {
    const scalar_t x0 = jacobian_adjugate[1] + jacobian_adjugate[4] + jacobian_adjugate[7];
    const scalar_t x1 = mu * x0;
    const scalar_t x2 = jacobian_adjugate[1] * x1;
    const scalar_t x3 = jacobian_adjugate[2] + jacobian_adjugate[5] + jacobian_adjugate[8];
    const scalar_t x4 = mu * x3;
    const scalar_t x5 = jacobian_adjugate[2] * x4;
    const scalar_t x6 = lambda + 2 * mu;
    const scalar_t x7 = jacobian_adjugate[0] + jacobian_adjugate[3] + jacobian_adjugate[6];
    const scalar_t x8 = x6 * x7;
    const scalar_t x9 = jacobian_adjugate[0] * x8 + x2 + x5;
    const scalar_t x10 = jacobian_adjugate[4] * x1;
    const scalar_t x11 = jacobian_adjugate[5] * x4;
    const scalar_t x12 = jacobian_adjugate[3] * x8 + x10 + x11;
    const scalar_t x13 = jacobian_adjugate[7] * x1;
    const scalar_t x14 = jacobian_adjugate[8] * x4;
    const scalar_t x15 = jacobian_adjugate[6] * x8 + x13 + x14;
    const scalar_t x16 = (1.0 / 6.0) / jacobian_determinant;
    const scalar_t x17 = POW2(jacobian_adjugate[1]);
    const scalar_t x18 = mu * x17;
    const scalar_t x19 = POW2(jacobian_adjugate[2]);
    const scalar_t x20 = mu * x19;
    const scalar_t x21 = POW2(jacobian_adjugate[0]);
    const scalar_t x22 = x18 + x20 + x21 * x6;
    const scalar_t x23 = jacobian_adjugate[1] * mu;
    const scalar_t x24 = jacobian_adjugate[4] * x23;
    const scalar_t x25 = jacobian_adjugate[2] * mu;
    const scalar_t x26 = jacobian_adjugate[5] * x25;
    const scalar_t x27 = jacobian_adjugate[0] * x6;
    const scalar_t x28 = jacobian_adjugate[3] * x27 + x24 + x26;
    const scalar_t x29 = jacobian_adjugate[7] * x23;
    const scalar_t x30 = jacobian_adjugate[8] * x25;
    const scalar_t x31 = jacobian_adjugate[6] * x27 + x29 + x30;
    const scalar_t x32 = POW2(jacobian_adjugate[4]);
    const scalar_t x33 = mu * x32;
    const scalar_t x34 = POW2(jacobian_adjugate[5]);
    const scalar_t x35 = mu * x34;
    const scalar_t x36 = POW2(jacobian_adjugate[3]);
    const scalar_t x37 = x33 + x35 + x36 * x6;
    const scalar_t x38 = jacobian_adjugate[4] * mu;
    const scalar_t x39 = jacobian_adjugate[7] * x38;
    const scalar_t x40 = jacobian_adjugate[5] * mu;
    const scalar_t x41 = jacobian_adjugate[8] * x40;
    const scalar_t x42 = jacobian_adjugate[3] * jacobian_adjugate[6];
    const scalar_t x43 = x39 + x41 + x42 * x6;
    const scalar_t x44 = POW2(jacobian_adjugate[7]);
    const scalar_t x45 = mu * x44;
    const scalar_t x46 = POW2(jacobian_adjugate[8]);
    const scalar_t x47 = mu * x46;
    const scalar_t x48 = POW2(jacobian_adjugate[6]);
    const scalar_t x49 = x45 + x47 + x48 * x6;
    const scalar_t x50 = lambda * x0;
    const scalar_t x51 = jacobian_adjugate[0] * x50 + x23 * x7;
    const scalar_t x52 = jacobian_adjugate[3] * x50 + x38 * x7;
    const scalar_t x53 = jacobian_adjugate[7] * mu;
    const scalar_t x54 = jacobian_adjugate[6] * x50 + x53 * x7;
    const scalar_t x55 = jacobian_adjugate[1] * lambda;
    const scalar_t x56 = jacobian_adjugate[0] * x38 + jacobian_adjugate[3] * x55;
    const scalar_t x57 = jacobian_adjugate[0] * x53 + jacobian_adjugate[6] * x55;
    const scalar_t x58 = jacobian_adjugate[0] * x23 + jacobian_adjugate[0] * x55;
    const scalar_t x59 = jacobian_adjugate[4] * lambda;
    const scalar_t x60 = jacobian_adjugate[0] * x59 + jacobian_adjugate[3] * x23;
    const scalar_t x61 = jacobian_adjugate[3] * x53 + jacobian_adjugate[6] * x59;
    const scalar_t x62 = jacobian_adjugate[3] * x38 + jacobian_adjugate[3] * x59;
    const scalar_t x63 = jacobian_adjugate[7] * lambda;
    const scalar_t x64 = jacobian_adjugate[0] * x63 + jacobian_adjugate[6] * x23;
    const scalar_t x65 = jacobian_adjugate[3] * x63 + jacobian_adjugate[6] * x38;
    const scalar_t x66 = jacobian_adjugate[6] * x53 + jacobian_adjugate[6] * x63;
    const scalar_t x67 = lambda * x3;
    const scalar_t x68 = jacobian_adjugate[0] * x67 + x25 * x7;
    const scalar_t x69 = jacobian_adjugate[3] * x67 + x40 * x7;
    const scalar_t x70 = jacobian_adjugate[8] * mu;
    const scalar_t x71 = jacobian_adjugate[6] * x67 + x7 * x70;
    const scalar_t x72 = jacobian_adjugate[2] * lambda;
    const scalar_t x73 = jacobian_adjugate[0] * x40 + jacobian_adjugate[3] * x72;
    const scalar_t x74 = jacobian_adjugate[0] * x70 + jacobian_adjugate[6] * x72;
    const scalar_t x75 = jacobian_adjugate[0] * x25 + jacobian_adjugate[0] * x72;
    const scalar_t x76 = jacobian_adjugate[5] * lambda;
    const scalar_t x77 = jacobian_adjugate[0] * x76 + jacobian_adjugate[3] * x25;
    const scalar_t x78 = jacobian_adjugate[3] * x70 + jacobian_adjugate[6] * x76;
    const scalar_t x79 = jacobian_adjugate[3] * x40 + jacobian_adjugate[3] * x76;
    const scalar_t x80 = jacobian_adjugate[8] * lambda;
    const scalar_t x81 = jacobian_adjugate[0] * x80 + jacobian_adjugate[6] * x25;
    const scalar_t x82 = jacobian_adjugate[3] * x80 + jacobian_adjugate[6] * x40;
    const scalar_t x83 = jacobian_adjugate[6] * x70 + jacobian_adjugate[6] * x80;
    const scalar_t x84 = x16 * x28;
    const scalar_t x85 = x16 * x31;
    const scalar_t x86 = x16 * (lambda + mu);
    const scalar_t x87 = jacobian_adjugate[0] * x86;
    const scalar_t x88 = jacobian_adjugate[1] * x87;
    const scalar_t x89 = x16 * x60;
    const scalar_t x90 = x16 * x64;
    const scalar_t x91 = jacobian_adjugate[2] * x87;
    const scalar_t x92 = x16 * x77;
    const scalar_t x93 = x16 * x81;
    const scalar_t x94 = x16 * x43;
    const scalar_t x95 = x16 * x56;
    const scalar_t x96 = jacobian_adjugate[3] * x86;
    const scalar_t x97 = jacobian_adjugate[4] * x96;
    const scalar_t x98 = x16 * x65;
    const scalar_t x99 = x16 * x73;
    const scalar_t x100 = jacobian_adjugate[5] * x96;
    const scalar_t x101 = x16 * x82;
    const scalar_t x102 = x16 * x57;
    const scalar_t x103 = x16 * x61;
    const scalar_t x104 = jacobian_adjugate[6] * x86;
    const scalar_t x105 = jacobian_adjugate[7] * x104;
    const scalar_t x106 = x16 * x74;
    const scalar_t x107 = x16 * x78;
    const scalar_t x108 = jacobian_adjugate[8] * x104;
    const scalar_t x109 = jacobian_adjugate[0] * x1 + x55 * x7;
    const scalar_t x110 = jacobian_adjugate[3] * x1 + x59 * x7;
    const scalar_t x111 = jacobian_adjugate[6] * x1 + x63 * x7;
    const scalar_t x112 = mu * x7;
    const scalar_t x113 = jacobian_adjugate[0] * x112;
    const scalar_t x114 = x0 * x6;
    const scalar_t x115 = jacobian_adjugate[1] * x114 + x113 + x5;
    const scalar_t x116 = jacobian_adjugate[3] * x112;
    const scalar_t x117 = jacobian_adjugate[4] * x114 + x11 + x116;
    const scalar_t x118 = jacobian_adjugate[6] * x112;
    const scalar_t x119 = jacobian_adjugate[7] * x114 + x118 + x14;
    const scalar_t x120 = mu * x21;
    const scalar_t x121 = x120 + x17 * x6 + x20;
    const scalar_t x122 = jacobian_adjugate[0] * mu;
    const scalar_t x123 = jacobian_adjugate[3] * x122;
    const scalar_t x124 = jacobian_adjugate[1] * x6;
    const scalar_t x125 = jacobian_adjugate[4] * x124 + x123 + x26;
    const scalar_t x126 = jacobian_adjugate[6] * x122;
    const scalar_t x127 = jacobian_adjugate[7] * x124 + x126 + x30;
    const scalar_t x128 = mu * x36;
    const scalar_t x129 = x128 + x32 * x6 + x35;
    const scalar_t x130 = mu * x42;
    const scalar_t x131 = jacobian_adjugate[4] * jacobian_adjugate[7] * x6 + x130 + x41;
    const scalar_t x132 = mu * x48;
    const scalar_t x133 = x132 + x44 * x6 + x47;
    const scalar_t x134 = jacobian_adjugate[2] * x1 + x3 * x55;
    const scalar_t x135 = jacobian_adjugate[5] * x1 + x3 * x59;
    const scalar_t x136 = jacobian_adjugate[8] * x1 + x3 * x63;
    const scalar_t x137 = jacobian_adjugate[2] * x59 + jacobian_adjugate[5] * x23;
    const scalar_t x138 = jacobian_adjugate[2] * x63 + jacobian_adjugate[8] * x23;
    const scalar_t x139 = jacobian_adjugate[2] * x23 + jacobian_adjugate[2] * x55;
    const scalar_t x140 = jacobian_adjugate[4] * x25 + jacobian_adjugate[5] * x55;
    const scalar_t x141 = jacobian_adjugate[5] * x63 + jacobian_adjugate[8] * x38;
    const scalar_t x142 = jacobian_adjugate[5] * x38 + jacobian_adjugate[5] * x59;
    const scalar_t x143 = jacobian_adjugate[7] * x25 + jacobian_adjugate[8] * x55;
    const scalar_t x144 = jacobian_adjugate[7] * x40 + jacobian_adjugate[8] * x59;
    const scalar_t x145 = jacobian_adjugate[8] * x53 + jacobian_adjugate[8] * x63;
    const scalar_t x146 = x125 * x16;
    const scalar_t x147 = x127 * x16;
    const scalar_t x148 = jacobian_adjugate[1] * jacobian_adjugate[2] * x86;
    const scalar_t x149 = x140 * x16;
    const scalar_t x150 = x143 * x16;
    const scalar_t x151 = x131 * x16;
    const scalar_t x152 = x137 * x16;
    const scalar_t x153 = jacobian_adjugate[4] * jacobian_adjugate[5] * x86;
    const scalar_t x154 = x144 * x16;
    const scalar_t x155 = x138 * x16;
    const scalar_t x156 = x141 * x16;
    const scalar_t x157 = jacobian_adjugate[7] * jacobian_adjugate[8] * x86;
    const scalar_t x158 = jacobian_adjugate[0] * x4 + x7 * x72;
    const scalar_t x159 = jacobian_adjugate[3] * x4 + x7 * x76;
    const scalar_t x160 = jacobian_adjugate[6] * x4 + x7 * x80;
    const scalar_t x161 = jacobian_adjugate[1] * x4 + jacobian_adjugate[2] * x50;
    const scalar_t x162 = jacobian_adjugate[4] * x4 + jacobian_adjugate[5] * x50;
    const scalar_t x163 = jacobian_adjugate[7] * x4 + jacobian_adjugate[8] * x50;
    const scalar_t x164 = x3 * x6;
    const scalar_t x165 = jacobian_adjugate[2] * x164 + x113 + x2;
    const scalar_t x166 = jacobian_adjugate[5] * x164 + x10 + x116;
    const scalar_t x167 = jacobian_adjugate[8] * x164 + x118 + x13;
    const scalar_t x168 = x120 + x18 + x19 * x6;
    const scalar_t x169 = jacobian_adjugate[2] * x6;
    const scalar_t x170 = jacobian_adjugate[5] * x169 + x123 + x24;
    const scalar_t x171 = jacobian_adjugate[8] * x169 + x126 + x29;
    const scalar_t x172 = x128 + x33 + x34 * x6;
    const scalar_t x173 = jacobian_adjugate[5] * jacobian_adjugate[8] * x6 + x130 + x39;
    const scalar_t x174 = x132 + x45 + x46 * x6;
    const scalar_t x175 = x16 * x170;
    const scalar_t x176 = x16 * x171;
    const scalar_t x177 = x16 * x173;
    element_matrix[0] = x16 * (x12 + x15 + x9);
    element_matrix[1] = x16 * (-x22 - x28 - x31);
    element_matrix[2] = x16 * (-x28 - x37 - x43);
    element_matrix[3] = x16 * (-x31 - x43 - x49);
    element_matrix[4] = x16 * (x51 + x52 + x54);
    element_matrix[5] = x16 * (-x56 - x57 - x58);
    element_matrix[6] = x16 * (-x60 - x61 - x62);
    element_matrix[7] = x16 * (-x64 - x65 - x66);
    element_matrix[8] = x16 * (x68 + x69 + x71);
    element_matrix[9] = x16 * (-x73 - x74 - x75);
    element_matrix[10] = x16 * (-x77 - x78 - x79);
    element_matrix[11] = x16 * (-x81 - x82 - x83);
    element_matrix[12] = -x16 * x9;
    element_matrix[13] = x16 * x22;
    element_matrix[14] = x84;
    element_matrix[15] = x85;
    element_matrix[16] = -x16 * x51;
    element_matrix[17] = x88;
    element_matrix[18] = x89;
    element_matrix[19] = x90;
    element_matrix[20] = -x16 * x68;
    element_matrix[21] = x91;
    element_matrix[22] = x92;
    element_matrix[23] = x93;
    element_matrix[24] = -x12 * x16;
    element_matrix[25] = x84;
    element_matrix[26] = x16 * x37;
    element_matrix[27] = x94;
    element_matrix[28] = -x16 * x52;
    element_matrix[29] = x95;
    element_matrix[30] = x97;
    element_matrix[31] = x98;
    element_matrix[32] = -x16 * x69;
    element_matrix[33] = x99;
    element_matrix[34] = x100;
    element_matrix[35] = x101;
    element_matrix[36] = -x15 * x16;
    element_matrix[37] = x85;
    element_matrix[38] = x94;
    element_matrix[39] = x16 * x49;
    element_matrix[40] = -x16 * x54;
    element_matrix[41] = x102;
    element_matrix[42] = x103;
    element_matrix[43] = x105;
    element_matrix[44] = -x16 * x71;
    element_matrix[45] = x106;
    element_matrix[46] = x107;
    element_matrix[47] = x108;
    element_matrix[48] = x16 * (x109 + x110 + x111);
    element_matrix[49] = x16 * (-x58 - x60 - x64);
    element_matrix[50] = x16 * (-x56 - x62 - x65);
    element_matrix[51] = x16 * (-x57 - x61 - x66);
    element_matrix[52] = x16 * (x115 + x117 + x119);
    element_matrix[53] = x16 * (-x121 - x125 - x127);
    element_matrix[54] = x16 * (-x125 - x129 - x131);
    element_matrix[55] = x16 * (-x127 - x131 - x133);
    element_matrix[56] = x16 * (x134 + x135 + x136);
    element_matrix[57] = x16 * (-x137 - x138 - x139);
    element_matrix[58] = x16 * (-x140 - x141 - x142);
    element_matrix[59] = x16 * (-x143 - x144 - x145);
    element_matrix[60] = -x109 * x16;
    element_matrix[61] = x88;
    element_matrix[62] = x95;
    element_matrix[63] = x102;
    element_matrix[64] = -x115 * x16;
    element_matrix[65] = x121 * x16;
    element_matrix[66] = x146;
    element_matrix[67] = x147;
    element_matrix[68] = -x134 * x16;
    element_matrix[69] = x148;
    element_matrix[70] = x149;
    element_matrix[71] = x150;
    element_matrix[72] = -x110 * x16;
    element_matrix[73] = x89;
    element_matrix[74] = x97;
    element_matrix[75] = x103;
    element_matrix[76] = -x117 * x16;
    element_matrix[77] = x146;
    element_matrix[78] = x129 * x16;
    element_matrix[79] = x151;
    element_matrix[80] = -x135 * x16;
    element_matrix[81] = x152;
    element_matrix[82] = x153;
    element_matrix[83] = x154;
    element_matrix[84] = -x111 * x16;
    element_matrix[85] = x90;
    element_matrix[86] = x98;
    element_matrix[87] = x105;
    element_matrix[88] = -x119 * x16;
    element_matrix[89] = x147;
    element_matrix[90] = x151;
    element_matrix[91] = x133 * x16;
    element_matrix[92] = -x136 * x16;
    element_matrix[93] = x155;
    element_matrix[94] = x156;
    element_matrix[95] = x157;
    element_matrix[96] = x16 * (x158 + x159 + x160);
    element_matrix[97] = x16 * (-x75 - x77 - x81);
    element_matrix[98] = x16 * (-x73 - x79 - x82);
    element_matrix[99] = x16 * (-x74 - x78 - x83);
    element_matrix[100] = x16 * (x161 + x162 + x163);
    element_matrix[101] = x16 * (-x139 - x140 - x143);
    element_matrix[102] = x16 * (-x137 - x142 - x144);
    element_matrix[103] = x16 * (-x138 - x141 - x145);
    element_matrix[104] = x16 * (x165 + x166 + x167);
    element_matrix[105] = x16 * (-x168 - x170 - x171);
    element_matrix[106] = x16 * (-x170 - x172 - x173);
    element_matrix[107] = x16 * (-x171 - x173 - x174);
    element_matrix[108] = -x158 * x16;
    element_matrix[109] = x91;
    element_matrix[110] = x99;
    element_matrix[111] = x106;
    element_matrix[112] = -x16 * x161;
    element_matrix[113] = x148;
    element_matrix[114] = x152;
    element_matrix[115] = x155;
    element_matrix[116] = -x16 * x165;
    element_matrix[117] = x16 * x168;
    element_matrix[118] = x175;
    element_matrix[119] = x176;
    element_matrix[120] = -x159 * x16;
    element_matrix[121] = x92;
    element_matrix[122] = x100;
    element_matrix[123] = x107;
    element_matrix[124] = -x16 * x162;
    element_matrix[125] = x149;
    element_matrix[126] = x153;
    element_matrix[127] = x156;
    element_matrix[128] = -x16 * x166;
    element_matrix[129] = x175;
    element_matrix[130] = x16 * x172;
    element_matrix[131] = x177;
    element_matrix[132] = -x16 * x160;
    element_matrix[133] = x93;
    element_matrix[134] = x101;
    element_matrix[135] = x108;
    element_matrix[136] = -x16 * x163;
    element_matrix[137] = x150;
    element_matrix[138] = x154;
    element_matrix[139] = x157;
    element_matrix[140] = -x16 * x167;
    element_matrix[141] = x176;
    element_matrix[142] = x177;
    element_matrix[143] = x16 * x174;
}

static SFEM_INLINE void aux_tet4_linear_elasticity_crs_adj_less_registers(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT adjugate,
        const scalar_t jacobian_determinant,
        const int test_idx,
        const scalar_t *const SFEM_RESTRICT gtest,
        scalar_t *const SFEM_RESTRICT element_matrix) {
    // loperand common expressions
    const scalar_t x0 = adjugate[1] * gtest[0];
    const scalar_t x1 = adjugate[4] * gtest[1];
    const scalar_t x2 = adjugate[7] * gtest[2];
    const scalar_t x3 = (1.0 / 6.0) / jacobian_determinant;
    const scalar_t x4 = mu * x3;
    const scalar_t x5 = x4 * (x0 + x1 + x2);
    const scalar_t x6 = adjugate[1] * x5;
    const scalar_t x7 = adjugate[2] * gtest[0];
    const scalar_t x8 = adjugate[5] * gtest[1];
    const scalar_t x9 = adjugate[8] * gtest[2];
    const scalar_t x10 = x4 * (x7 + x8 + x9);
    const scalar_t x11 = adjugate[2] * x10;
    const scalar_t x12 = lambda + 2 * mu;
    const scalar_t x13 = adjugate[0] * gtest[0];
    const scalar_t x14 = adjugate[3] * gtest[1];
    const scalar_t x15 = adjugate[6] * gtest[2];
    const scalar_t x16 = x3 * (x12 * x13 + x12 * x14 + x12 * x15);
    const scalar_t x17 = adjugate[4] * x5;
    const scalar_t x18 = adjugate[5] * x10;
    const scalar_t x19 = adjugate[7] * x5;
    const scalar_t x20 = adjugate[8] * x10;
    const scalar_t x21 = x3 * (lambda * x13 + lambda * x14 + lambda * x15);
    const scalar_t x22 = x4 * (x13 + x14 + x15);
    const scalar_t x23 = x3 * (lambda * x0 + lambda * x1 + lambda * x2);
    const scalar_t x24 = adjugate[0] * x22;
    const scalar_t x25 = x3 * (x0 * x12 + x1 * x12 + x12 * x2);
    const scalar_t x26 = adjugate[3] * x22;
    const scalar_t x27 = adjugate[6] * x22;
    const scalar_t x28 = x3 * (lambda * x7 + lambda * x8 + lambda * x9);
    const scalar_t x29 = x3 * (x12 * x7 + x12 * x8 + x12 * x9);

    scalar_t loperand[9];
    scalar_t gtrial[3];

    // x (0)
    {
        loperand[0] = adjugate[0] * x16 + x11 + x6;
        loperand[1] = adjugate[3] * x16 + x17 + x18;
        loperand[2] = adjugate[6] * x16 + x19 + x20;
        loperand[3] = adjugate[0] * x5 + adjugate[1] * x21;
        loperand[4] = adjugate[3] * x5 + adjugate[4] * x21;
        loperand[5] = adjugate[6] * x5 + adjugate[7] * x21;
        loperand[6] = adjugate[0] * x10 + adjugate[2] * x21;
        loperand[7] = adjugate[3] * x10 + adjugate[5] * x21;
        loperand[8] = adjugate[6] * x10 + adjugate[8] * x21;

        for (int d = 0; d < 3; d++) {
            const scalar_t *const row = &loperand[d * 3];
            const int offset = test_idx * 12 + d * 4;
            element_matrix[offset + 0] = -row[0] - row[1] - row[2];
            element_matrix[offset + 1] = row[0];
            element_matrix[offset + 2] = row[1];
            element_matrix[offset + 3] = row[2];
        }
    }

    // y (1)
    {
        loperand[0] = adjugate[0] * x23 + adjugate[1] * x22;
        loperand[1] = adjugate[3] * x23 + adjugate[4] * x22;
        loperand[2] = adjugate[6] * x23 + adjugate[7] * x22;
        loperand[3] = adjugate[1] * x25 + x11 + x24;
        loperand[4] = adjugate[4] * x25 + x18 + x26;
        loperand[5] = adjugate[7] * x25 + x20 + x27;
        loperand[6] = adjugate[1] * x10 + adjugate[2] * x23;
        loperand[7] = adjugate[4] * x10 + adjugate[5] * x23;
        loperand[8] = adjugate[7] * x10 + adjugate[8] * x23;

        for (int d = 0; d < 3; d++) {
            const scalar_t *const row = &loperand[d * 3];
            const int offset = 4 * 12 + test_idx * 12 + d * 4;
            element_matrix[offset + 0] = -row[0] - row[1] - row[2];
            element_matrix[offset + 1] = row[0];
            element_matrix[offset + 2] = row[1];
            element_matrix[offset + 3] = row[2];
        }
    }

    // z (2)
    {
        loperand[0] = adjugate[0] * x28 + adjugate[2] * x22;
        loperand[1] = adjugate[3] * x28 + adjugate[5] * x22;
        loperand[2] = adjugate[6] * x28 + adjugate[8] * x22;
        loperand[3] = adjugate[1] * x28 + adjugate[2] * x5;
        loperand[4] = adjugate[4] * x28 + adjugate[5] * x5;
        loperand[5] = adjugate[7] * x28 + adjugate[8] * x5;
        loperand[6] = adjugate[2] * x29 + x24 + x6;
        loperand[7] = adjugate[5] * x29 + x17 + x26;
        loperand[8] = adjugate[8] * x29 + x19 + x27;

        for (int d = 0; d < 3; d++) {
            const scalar_t *const row = &loperand[d * 3];
            const int offset = 2 * 4 * 12 + test_idx * 12 + d * 4;
            element_matrix[offset + 0] = -row[0] - row[1] - row[2];
            element_matrix[offset + 1] = row[0];
            element_matrix[offset + 2] = row[1];
            element_matrix[offset + 3] = row[2];
        }
    }
}

// Untested
static SFEM_INLINE void tet4_linear_elasticity_crs_adj_less_registers(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t jacobian_determinant,
        accumulator_t *const SFEM_RESTRICT element_matrix) {
    {
        scalar_t gtest[3] = {-1, -1, -1};
        aux_tet4_linear_elasticity_crs_adj_less_registers(
                mu, lambda, jacobian_adjugate, jacobian_determinant, 0, gtest, element_matrix);
    }

    {
        scalar_t gtest[3] = {1, 0, 0};
        aux_tet4_linear_elasticity_crs_adj_less_registers(
                mu, lambda, jacobian_adjugate, jacobian_determinant, 1, gtest, element_matrix);
    }

    {
        scalar_t gtest[3] = {0, 1, 0};
        aux_tet4_linear_elasticity_crs_adj_less_registers(
                mu, lambda, jacobian_adjugate, jacobian_determinant, 2, gtest, element_matrix);
    }

    {
        scalar_t gtest[3] = {0, 0, 1};
        aux_tet4_linear_elasticity_crs_adj_less_registers(
                mu, lambda, jacobian_adjugate, jacobian_determinant, 3, gtest, element_matrix);
    }
}

#endif  // TET4_LINEAR_ELASTICITY_INLINE_CPU_H
