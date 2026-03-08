#ifndef TET4_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_LOCAL_H
#define TET4_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_LOCAL_H

#include "tet4_neohookean_ogden_local.hpp"

static SFEM_INLINE void tet4_neohookean_ogden_active_strain_objective(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                      const scalar_t                      jacobian_determinant,
                                                                      const scalar_t                      lmbda,
                                                                      const scalar_t                      mu,
                                                                      const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                      const scalar_t                      Ja,
                                                                      const scalar_t *const SFEM_RESTRICT dispx,
                                                                      const scalar_t *const SFEM_RESTRICT dispy,
                                                                      const scalar_t *const SFEM_RESTRICT dispz,
                                                                      scalar_t *const SFEM_RESTRICT       v) {
    scalar_t F[9];
    tet4_F(adjugate, jacobian_determinant, dispx, dispy, dispz, F);

    // mundane ops: 201 divs: 0 sqrts: 0
    // total ops: 201
    const scalar_t x0  = F[0] * Fa_inv[0];
    const scalar_t x1  = F[1] * Fa_inv[3];
    const scalar_t x2  = F[2] * Fa_inv[6];
    const scalar_t x3  = F[0] * Fa_inv[1];
    const scalar_t x4  = F[1] * Fa_inv[4];
    const scalar_t x5  = F[2] * Fa_inv[7];
    const scalar_t x6  = F[0] * Fa_inv[2];
    const scalar_t x7  = F[1] * Fa_inv[5];
    const scalar_t x8  = F[2] * Fa_inv[8];
    const scalar_t x9  = F[3] * Fa_inv[0];
    const scalar_t x10 = F[4] * Fa_inv[3];
    const scalar_t x11 = F[5] * Fa_inv[6];
    const scalar_t x12 = F[3] * Fa_inv[1];
    const scalar_t x13 = F[4] * Fa_inv[4];
    const scalar_t x14 = F[5] * Fa_inv[7];
    const scalar_t x15 = F[3] * Fa_inv[2];
    const scalar_t x16 = F[4] * Fa_inv[5];
    const scalar_t x17 = F[5] * Fa_inv[8];
    const scalar_t x18 = F[6] * Fa_inv[0];
    const scalar_t x19 = F[7] * Fa_inv[3];
    const scalar_t x20 = F[8] * Fa_inv[6];
    const scalar_t x21 = F[6] * Fa_inv[1];
    const scalar_t x22 = F[7] * Fa_inv[4];
    const scalar_t x23 = F[8] * Fa_inv[7];
    const scalar_t x24 = F[6] * Fa_inv[2];
    const scalar_t x25 = F[7] * Fa_inv[5];
    const scalar_t x26 = F[8] * Fa_inv[8];
    const scalar_t x27 = log(x0 * x13 * x26 + x0 * x14 * x25 - x0 * x16 * x23 - x0 * x17 * x22 + x1 * x12 * x26 + x1 * x14 * x24 -
                             x1 * x15 * x23 - x1 * x17 * x21 + x10 * x21 * x8 + x10 * x23 * x6 - x10 * x24 * x5 - x10 * x26 * x3 +
                             x11 * x21 * x7 + x11 * x22 * x6 - x11 * x24 * x4 - x11 * x25 * x3 - x12 * x19 * x8 + x12 * x2 * x25 -
                             x12 * x20 * x7 - x13 * x18 * x8 + x13 * x2 * x24 - x13 * x20 * x6 - x14 * x18 * x7 - x14 * x19 * x6 +
                             x15 * x19 * x5 - x15 * x2 * x22 + x15 * x20 * x4 + x16 * x18 * x5 - x16 * x2 * x21 + x16 * x20 * x3 +
                             x17 * x18 * x4 + x17 * x19 * x3 + x22 * x8 * x9 + x23 * x7 * x9 - x25 * x5 * x9 - x26 * x4 * x9);
    v[0] += (1.0 / 6.0) * Ja * jacobian_determinant *
            ((1.0 / 2.0) * lmbda * POW2(x27) - mu * x27 +
             (1.0 / 2.0) * mu *
                     (POW2(x0 + x1 + x2) + POW2(x10 + x11 + x9) + POW2(x12 + x13 + x14) + POW2(x15 + x16 + x17) +
                      POW2(x18 + x19 + x20) + POW2(x21 + x22 + x23) + POW2(x24 + x25 + x26) + POW2(x3 + x4 + x5) +
                      POW2(x6 + x7 + x8) - 3));
}

static SFEM_INLINE void tet4_neohookean_ogden_active_strain_grad(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                 const scalar_t                      jacobian_determinant,
                                                                 const scalar_t                      lmbda,
                                                                 const scalar_t                      mu,
                                                                 const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                 const scalar_t                      Ja,
                                                                 const scalar_t *const SFEM_RESTRICT dispx,
                                                                 const scalar_t *const SFEM_RESTRICT dispy,
                                                                 const scalar_t *const SFEM_RESTRICT dispz,
                                                                 scalar_t *const SFEM_RESTRICT       gx,
                                                                 scalar_t *const SFEM_RESTRICT       gy,
                                                                 scalar_t *const SFEM_RESTRICT       gz) {
    scalar_t F[9];
    {
        // mundane ops: 96 divs: 1 sqrts: 0
        // total ops: 104
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = dispx[0] - dispx[1];
        const scalar_t x2 = dispx[0] - dispx[2];
        const scalar_t x3 = dispx[0] - dispx[3];
        const scalar_t x4 = dispy[0] - dispy[1];
        const scalar_t x5 = dispy[0] - dispy[2];
        const scalar_t x6 = dispy[0] - dispy[3];
        const scalar_t x7 = dispz[0] - dispz[1];
        const scalar_t x8 = dispz[0] - dispz[2];
        const scalar_t x9 = dispz[0] - dispz[3];
        F[0]              = -adjugate[0] * x0 * x1 - adjugate[3] * x0 * x2 - adjugate[6] * x0 * x3 + 1;
        F[1]              = -x0 * (adjugate[1] * x1 + adjugate[4] * x2 + adjugate[7] * x3);
        F[2]              = -x0 * (adjugate[2] * x1 + adjugate[5] * x2 + adjugate[8] * x3);
        F[3]              = -x0 * (adjugate[0] * x4 + adjugate[3] * x5 + adjugate[6] * x6);
        F[4]              = -adjugate[1] * x0 * x4 - adjugate[4] * x0 * x5 - adjugate[7] * x0 * x6 + 1;
        F[5]              = -x0 * (adjugate[2] * x4 + adjugate[5] * x5 + adjugate[8] * x6);
        F[6]              = -x0 * (adjugate[0] * x7 + adjugate[3] * x8 + adjugate[6] * x9);
        F[7]              = -x0 * (adjugate[1] * x7 + adjugate[4] * x8 + adjugate[7] * x9);
        F[8]              = -adjugate[2] * x0 * x7 - adjugate[5] * x0 * x8 - adjugate[8] * x0 * x9 + 1;
    }

    // mundane ops: 637 divs: 1 sqrts: 0
    // total ops: 645
    const scalar_t x0  = F[0] * Fa_inv[0];
    const scalar_t x1  = F[1] * Fa_inv[3];
    const scalar_t x2  = F[2] * Fa_inv[6];
    const scalar_t x3  = x0 + x1 + x2;
    const scalar_t x4  = F[0] * Fa_inv[1];
    const scalar_t x5  = F[1] * Fa_inv[4];
    const scalar_t x6  = F[2] * Fa_inv[7];
    const scalar_t x7  = x4 + x5 + x6;
    const scalar_t x8  = F[0] * Fa_inv[2];
    const scalar_t x9  = F[1] * Fa_inv[5];
    const scalar_t x10 = F[2] * Fa_inv[8];
    const scalar_t x11 = x10 + x8 + x9;
    const scalar_t x12 = F[4] * Fa_inv[4];
    const scalar_t x13 = F[8] * Fa_inv[8];
    const scalar_t x14 = Fa_inv[0] * x13;
    const scalar_t x15 = F[4] * Fa_inv[5];
    const scalar_t x16 = F[8] * Fa_inv[6];
    const scalar_t x17 = Fa_inv[1] * x16;
    const scalar_t x18 = F[4] * Fa_inv[3];
    const scalar_t x19 = F[8] * Fa_inv[7];
    const scalar_t x20 = Fa_inv[2] * x19;
    const scalar_t x21 = F[5] * Fa_inv[7];
    const scalar_t x22 = F[7] * Fa_inv[5];
    const scalar_t x23 = Fa_inv[0] * x22;
    const scalar_t x24 = F[5] * Fa_inv[8];
    const scalar_t x25 = F[7] * Fa_inv[3];
    const scalar_t x26 = Fa_inv[1] * x25;
    const scalar_t x27 = F[5] * Fa_inv[6];
    const scalar_t x28 = F[7] * Fa_inv[4];
    const scalar_t x29 = Fa_inv[2] * x28;
    const scalar_t x30 = Fa_inv[0] * x19;
    const scalar_t x31 = Fa_inv[1] * x13;
    const scalar_t x32 = Fa_inv[2] * x16;
    const scalar_t x33 = Fa_inv[0] * x28;
    const scalar_t x34 = Fa_inv[1] * x22;
    const scalar_t x35 = Fa_inv[2] * x25;
    const scalar_t x36 = x12 * x14 - x12 * x32 + x15 * x17 - x15 * x30 + x18 * x20 - x18 * x31 + x21 * x23 - x21 * x35 +
                         x24 * x26 - x24 * x33 + x27 * x29 - x27 * x34;
    const scalar_t x37 = x0 * x12;
    const scalar_t x38 = x15 * x4;
    const scalar_t x39 = x18 * x8;
    const scalar_t x40 = x0 * x21;
    const scalar_t x41 = x24 * x4;
    const scalar_t x42 = x27 * x8;
    const scalar_t x43 = F[3] * Fa_inv[0];
    const scalar_t x44 = x43 * x9;
    const scalar_t x45 = F[3] * Fa_inv[1];
    const scalar_t x46 = x1 * x45;
    const scalar_t x47 = F[3] * Fa_inv[2];
    const scalar_t x48 = x47 * x5;
    const scalar_t x49 = F[6] * Fa_inv[0];
    const scalar_t x50 = x24 * x5;
    const scalar_t x51 = F[6] * Fa_inv[1];
    const scalar_t x52 = x27 * x9;
    const scalar_t x53 = F[6] * Fa_inv[2];
    const scalar_t x54 = x1 * x21;
    const scalar_t x55 = x10 * x43;
    const scalar_t x56 = x2 * x45;
    const scalar_t x57 = x47 * x6;
    const scalar_t x58 = x15 * x6;
    const scalar_t x59 = x10 * x18;
    const scalar_t x60 = x12 * x2;
    const scalar_t x61 = x0 * x15;
    const scalar_t x62 = x18 * x4;
    const scalar_t x63 = x12 * x8;
    const scalar_t x64 = x0 * x24;
    const scalar_t x65 = x27 * x4;
    const scalar_t x66 = x21 * x8;
    const scalar_t x67 = x43 * x5;
    const scalar_t x68 = x45 * x9;
    const scalar_t x69 = x1 * x47;
    const scalar_t x70 = x21 * x9;
    const scalar_t x71 = x1 * x24;
    const scalar_t x72 = x27 * x5;
    const scalar_t x73 = x43 * x6;
    const scalar_t x74 = x10 * x45;
    const scalar_t x75 = x2 * x47;
    const scalar_t x76 = x10 * x12;
    const scalar_t x77 = x15 * x2;
    const scalar_t x78 = x18 * x6;
    const scalar_t x79 = x13 * x37 + x13 * x46 - x13 * x62 - x13 * x67 + x16 * x38 + x16 * x48 - x16 * x63 - x16 * x68 +
                         x19 * x39 + x19 * x44 - x19 * x61 - x19 * x69 + x22 * x40 + x22 * x56 - x22 * x65 - x22 * x73 +
                         x25 * x41 + x25 * x57 - x25 * x66 - x25 * x74 + x28 * x42 + x28 * x55 - x28 * x64 - x28 * x75 +
                         x49 * x50 + x49 * x58 - x49 * x70 - x49 * x76 + x51 * x52 + x51 * x59 - x51 * x71 - x51 * x77 +
                         x53 * x54 + x53 * x60 - x53 * x72 - x53 * x78;
    const scalar_t x80 = 1.0 / x79;
    const scalar_t x81 = mu * x80;
    const scalar_t x82 = lmbda * x80 * log(x79);
    const scalar_t x83 = mu * (Fa_inv[0] * x3 + Fa_inv[1] * x7 + Fa_inv[2] * x11) - x36 * x81 + x36 * x82;
    const scalar_t x84 = Fa_inv[4] * x13;
    const scalar_t x85 = Fa_inv[5] * x16;
    const scalar_t x86 = Fa_inv[3] * x19;
    const scalar_t x87 = Fa_inv[5] * x49;
    const scalar_t x88 = Fa_inv[3] * x51;
    const scalar_t x89 = Fa_inv[4] * x53;
    const scalar_t x90 = Fa_inv[5] * x19;
    const scalar_t x91 = Fa_inv[3] * x13;
    const scalar_t x92 = Fa_inv[4] * x16;
    const scalar_t x93 = Fa_inv[4] * x49;
    const scalar_t x94 = Fa_inv[5] * x51;
    const scalar_t x95 = Fa_inv[3] * x53;
    const scalar_t x96 = x21 * x87 - x21 * x95 + x24 * x88 - x24 * x93 + x27 * x89 - x27 * x94 + x43 * x84 - x43 * x90 +
                         x45 * x85 - x45 * x91 + x47 * x86 - x47 * x92;
    const scalar_t x97  = mu * (Fa_inv[3] * x3 + Fa_inv[4] * x7 + Fa_inv[5] * x11) + x81 * x96 - x82 * x96;
    const scalar_t x98  = Fa_inv[8] * x28;
    const scalar_t x99  = Fa_inv[6] * x22;
    const scalar_t x100 = Fa_inv[7] * x25;
    const scalar_t x101 = Fa_inv[7] * x49;
    const scalar_t x102 = Fa_inv[8] * x51;
    const scalar_t x103 = Fa_inv[6] * x53;
    const scalar_t x104 = Fa_inv[7] * x22;
    const scalar_t x105 = Fa_inv[8] * x25;
    const scalar_t x106 = Fa_inv[6] * x28;
    const scalar_t x107 = Fa_inv[8] * x49;
    const scalar_t x108 = Fa_inv[6] * x51;
    const scalar_t x109 = Fa_inv[7] * x53;
    const scalar_t x110 = x100 * x47 + x101 * x15 + x102 * x18 + x103 * x12 - x104 * x43 - x105 * x45 - x106 * x47 - x107 * x12 -
                          x108 * x15 - x109 * x18 + x43 * x98 + x45 * x99;
    const scalar_t x111 = mu * (Fa_inv[6] * x3 + Fa_inv[7] * x7 + Fa_inv[8] * x11) - x110 * x81 + x110 * x82;
    const scalar_t x112 = adjugate[0] * x83 + adjugate[1] * x97 + adjugate[2] * x111;
    const scalar_t x113 = adjugate[3] * x83 + adjugate[4] * x97 + adjugate[5] * x111;
    const scalar_t x114 = adjugate[6] * x83 + adjugate[7] * x97 + adjugate[8] * x111;
    const scalar_t x115 = (1.0 / 6.0) * Ja;
    const scalar_t x116 = x18 + x27 + x43;
    const scalar_t x117 = x12 + x21 + x45;
    const scalar_t x118 = x15 + x24 + x47;
    const scalar_t x119 = x1 * x20 - x1 * x31 + x10 * x26 - x10 * x33 + x14 * x5 + x17 * x9 + x2 * x29 - x2 * x34 + x23 * x6 -
                          x30 * x9 - x32 * x5 - x35 * x6;
    const scalar_t x120 = mu * (Fa_inv[0] * x116 + Fa_inv[1] * x117 + Fa_inv[2] * x118) + x119 * x81 - x119 * x82;
    const scalar_t x121 = x0 * x84 - x0 * x90 + x10 * x88 - x10 * x93 + x2 * x89 - x2 * x94 + x4 * x85 - x4 * x91 + x6 * x87 -
                          x6 * x95 + x8 * x86 - x8 * x92;
    const scalar_t x122 = mu * (Fa_inv[3] * x116 + Fa_inv[4] * x117 + Fa_inv[5] * x118) - x121 * x81 + x121 * x82;
    const scalar_t x123 = -x0 * x104 + x0 * x98 + x1 * x102 - x1 * x109 + x100 * x8 + x101 * x9 + x103 * x5 - x105 * x4 -
                          x106 * x8 - x107 * x5 - x108 * x9 + x4 * x99;
    const scalar_t x124 = mu * (Fa_inv[6] * x116 + Fa_inv[7] * x117 + Fa_inv[8] * x118) + x123 * x81 - x123 * x82;
    const scalar_t x125 = adjugate[0] * x120 + adjugate[1] * x122 + adjugate[2] * x124;
    const scalar_t x126 = adjugate[3] * x120 + adjugate[4] * x122 + adjugate[5] * x124;
    const scalar_t x127 = adjugate[6] * x120 + adjugate[7] * x122 + adjugate[8] * x124;
    const scalar_t x128 = x16 + x25 + x49;
    const scalar_t x129 = x19 + x28 + x51;
    const scalar_t x130 = x13 + x22 + x53;
    const scalar_t x131 = Fa_inv[0] * x50 + Fa_inv[0] * x58 - Fa_inv[0] * x70 - Fa_inv[0] * x76 + Fa_inv[1] * x52 +
                          Fa_inv[1] * x59 - Fa_inv[1] * x71 - Fa_inv[1] * x77 + Fa_inv[2] * x54 + Fa_inv[2] * x60 -
                          Fa_inv[2] * x72 - Fa_inv[2] * x78;
    const scalar_t x132 = mu * (Fa_inv[0] * x128 + Fa_inv[1] * x129 + Fa_inv[2] * x130) - x131 * x81 + x131 * x82;
    const scalar_t x133 = -Fa_inv[3] * x41 - Fa_inv[3] * x57 + Fa_inv[3] * x66 + Fa_inv[3] * x74 - Fa_inv[4] * x42 -
                          Fa_inv[4] * x55 + Fa_inv[4] * x64 + Fa_inv[4] * x75 - Fa_inv[5] * x40 - Fa_inv[5] * x56 +
                          Fa_inv[5] * x65 + Fa_inv[5] * x73;
    const scalar_t x134 = mu * (Fa_inv[3] * x128 + Fa_inv[4] * x129 + Fa_inv[5] * x130) + x133 * x81 - x133 * x82;
    const scalar_t x135 = Fa_inv[6] * x38 + Fa_inv[6] * x48 - Fa_inv[6] * x63 - Fa_inv[6] * x68 + Fa_inv[7] * x39 +
                          Fa_inv[7] * x44 - Fa_inv[7] * x61 - Fa_inv[7] * x69 + Fa_inv[8] * x37 + Fa_inv[8] * x46 -
                          Fa_inv[8] * x62 - Fa_inv[8] * x67;
    const scalar_t x136 = mu * (Fa_inv[6] * x128 + Fa_inv[7] * x129 + Fa_inv[8] * x130) - x135 * x81 + x135 * x82;
    const scalar_t x137 = adjugate[0] * x132 + adjugate[1] * x134 + adjugate[2] * x136;
    const scalar_t x138 = adjugate[3] * x132 + adjugate[4] * x134 + adjugate[5] * x136;
    const scalar_t x139 = adjugate[6] * x132 + adjugate[7] * x134 + adjugate[8] * x136;
    gx[0] += -x115 * (x112 + x113 + x114);
    gx[1] += x112 * x115;
    gx[2] += x113 * x115;
    gx[3] += x114 * x115;
    gy[0] += -x115 * (x125 + x126 + x127);
    gy[1] += x115 * x125;
    gy[2] += x115 * x126;
    gy[3] += x115 * x127;
    gz[0] += -x115 * (x137 + x138 + x139);
    gz[1] += x115 * x137;
    gz[2] += x115 * x138;
    gz[3] += x115 * x139;
}
static SFEM_INLINE void tet4_neohookean_ogden_active_strain_hessian(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                    const scalar_t                      jacobian_determinant,
                                                                    const scalar_t                      lmbda,
                                                                    const scalar_t                      mu,
                                                                    const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                    const scalar_t                      Ja,
                                                                    const scalar_t *const SFEM_RESTRICT dispx,
                                                                    const scalar_t *const SFEM_RESTRICT dispy,
                                                                    const scalar_t *const SFEM_RESTRICT dispz,
                                                                    scalar_t *const SFEM_RESTRICT       H) {
    scalar_t F[9];
    {
        // mundane ops: 96 divs: 1 sqrts: 0
        // total ops: 104
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = dispx[0] - dispx[1];
        const scalar_t x2 = dispx[0] - dispx[2];
        const scalar_t x3 = dispx[0] - dispx[3];
        const scalar_t x4 = dispy[0] - dispy[1];
        const scalar_t x5 = dispy[0] - dispy[2];
        const scalar_t x6 = dispy[0] - dispy[3];
        const scalar_t x7 = dispz[0] - dispz[1];
        const scalar_t x8 = dispz[0] - dispz[2];
        const scalar_t x9 = dispz[0] - dispz[3];
        F[0]              = -adjugate[0] * x0 * x1 - adjugate[3] * x0 * x2 - adjugate[6] * x0 * x3 + 1;
        F[1]              = -x0 * (adjugate[1] * x1 + adjugate[4] * x2 + adjugate[7] * x3);
        F[2]              = -x0 * (adjugate[2] * x1 + adjugate[5] * x2 + adjugate[8] * x3);
        F[3]              = -x0 * (adjugate[0] * x4 + adjugate[3] * x5 + adjugate[6] * x6);
        F[4]              = -adjugate[1] * x0 * x4 - adjugate[4] * x0 * x5 - adjugate[7] * x0 * x6 + 1;
        F[5]              = -x0 * (adjugate[2] * x4 + adjugate[5] * x5 + adjugate[8] * x6);
        F[6]              = -x0 * (adjugate[0] * x7 + adjugate[3] * x8 + adjugate[6] * x9);
        F[7]              = -x0 * (adjugate[1] * x7 + adjugate[4] * x8 + adjugate[7] * x9);
        F[8]              = -adjugate[2] * x0 * x7 - adjugate[5] * x0 * x8 - adjugate[8] * x0 * x9 + 1;
    }

    scalar_t S_lin[81];
    {
        // mundane ops: 865 divs: 1 sqrts: 0
        // total ops: 873
        const scalar_t x0  = mu * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
        const scalar_t x1  = Fa_inv[0] * Fa_inv[4] * Fa_inv[8];
        const scalar_t x2  = F[4] * F[8];
        const scalar_t x3  = x1 * x2;
        const scalar_t x4  = Fa_inv[1] * Fa_inv[5] * Fa_inv[6];
        const scalar_t x5  = x2 * x4;
        const scalar_t x6  = Fa_inv[2] * Fa_inv[3] * Fa_inv[7];
        const scalar_t x7  = x2 * x6;
        const scalar_t x8  = Fa_inv[0] * Fa_inv[5] * Fa_inv[7];
        const scalar_t x9  = F[5] * F[7];
        const scalar_t x10 = x8 * x9;
        const scalar_t x11 = Fa_inv[1] * Fa_inv[3] * Fa_inv[8];
        const scalar_t x12 = x11 * x9;
        const scalar_t x13 = Fa_inv[2] * Fa_inv[4] * Fa_inv[6];
        const scalar_t x14 = x13 * x9;
        const scalar_t x15 = x2 * x8;
        const scalar_t x16 = x11 * x2;
        const scalar_t x17 = x13 * x2;
        const scalar_t x18 = x1 * x9;
        const scalar_t x19 = x4 * x9;
        const scalar_t x20 = x6 * x9;
        const scalar_t x21 = x10 + x12 + x14 - x15 - x16 - x17 - x18 - x19 - x20 + x3 + x5 + x7;
        const scalar_t x22 = F[3] * F[8];
        const scalar_t x23 = x22 * x8;
        const scalar_t x24 = x11 * x22;
        const scalar_t x25 = x13 * x22;
        const scalar_t x26 = F[5] * F[6];
        const scalar_t x27 = x1 * x26;
        const scalar_t x28 = x26 * x4;
        const scalar_t x29 = x26 * x6;
        const scalar_t x30 = F[3] * F[7];
        const scalar_t x31 = x1 * x30;
        const scalar_t x32 = x30 * x4;
        const scalar_t x33 = x30 * x6;
        const scalar_t x34 = F[4] * F[6];
        const scalar_t x35 = x34 * x8;
        const scalar_t x36 = x11 * x34;
        const scalar_t x37 = x13 * x34;
        const scalar_t x38 = x1 * x22;
        const scalar_t x39 = x22 * x4;
        const scalar_t x40 = x22 * x6;
        const scalar_t x41 = x26 * x8;
        const scalar_t x42 = x11 * x26;
        const scalar_t x43 = x13 * x26;
        const scalar_t x44 = x30 * x8;
        const scalar_t x45 = x11 * x30;
        const scalar_t x46 = x13 * x30;
        const scalar_t x47 = x1 * x34;
        const scalar_t x48 = x34 * x4;
        const scalar_t x49 = x34 * x6;
        const scalar_t x50 = F[0] * x10 + F[0] * x12 + F[0] * x14 - F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 -
                             F[0] * x19 - F[0] * x20 + F[0] * x3 + F[0] * x5 + F[0] * x7 + F[1] * x23 + F[1] * x24 + F[1] * x25 +
                             F[1] * x27 + F[1] * x28 + F[1] * x29 - F[1] * x38 - F[1] * x39 - F[1] * x40 - F[1] * x41 -
                             F[1] * x42 - F[1] * x43 + F[2] * x31 + F[2] * x32 + F[2] * x33 + F[2] * x35 + F[2] * x36 +
                             F[2] * x37 - F[2] * x44 - F[2] * x45 - F[2] * x46 - F[2] * x47 - F[2] * x48 - F[2] * x49;
        const scalar_t x51 = (1 / POW2(x50));
        const scalar_t x52 = POW2(x21) * x51;
        const scalar_t x53 = lmbda * log(x50);
        const scalar_t x54 = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
        const scalar_t x55 = x21 * x51;
        const scalar_t x56 = -x23 - x24 - x25 - x27 - x28 - x29 + x38 + x39 + x40 + x41 + x42 + x43;
        const scalar_t x57 = lmbda * x56;
        const scalar_t x58 = mu * x55;
        const scalar_t x59 = x53 * x55;
        const scalar_t x60 = Ja * (x54 - x55 * x57 - x56 * x58 + x56 * x59);
        const scalar_t x61 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
        const scalar_t x62 = x31 + x32 + x33 + x35 + x36 + x37 - x44 - x45 - x46 - x47 - x48 - x49;
        const scalar_t x63 = lmbda * x62;
        const scalar_t x64 = Ja * (x55 * x63 + x58 * x62 - x59 * x62 + x61);
        const scalar_t x65 = F[1] * F[8];
        const scalar_t x66 = F[2] * F[7];
        const scalar_t x67 = x1 * x65 - x1 * x66 - x11 * x65 + x11 * x66 - x13 * x65 + x13 * x66 + x4 * x65 - x4 * x66 +
                             x6 * x65 - x6 * x66 - x65 * x8 + x66 * x8;
        const scalar_t x68 = Ja * (lmbda + mu - x53);
        const scalar_t x69 = x55 * x68;
        const scalar_t x70 = -x67 * x69;
        const scalar_t x71 = x1 - x11 - x13 + x4 + x6 - x8;
        const scalar_t x72 = mu * x71;
        const scalar_t x73 = F[8] * x72;
        const scalar_t x74 = x53 * x71;
        const scalar_t x75 = F[8] * x74;
        const scalar_t x76 = 1.0 / x50;
        const scalar_t x77 = x21 * x76;
        const scalar_t x78 = F[0] * F[8];
        const scalar_t x79 = F[2] * F[6];
        const scalar_t x80 = x1 * x78 - x1 * x79 - x11 * x78 + x11 * x79 - x13 * x78 + x13 * x79 + x4 * x78 - x4 * x79 +
                             x6 * x78 - x6 * x79 - x78 * x8 + x79 * x8;
        const scalar_t x81 = lmbda * x80;
        const scalar_t x82 = mu * x77;
        const scalar_t x83 = x53 * x77;
        const scalar_t x84 = Ja * x76;
        const scalar_t x85 = x84 * (-x73 + x75 + x77 * x81 + x80 * x82 - x80 * x83);
        const scalar_t x86 = F[7] * x72;
        const scalar_t x87 = F[7] * x74;
        const scalar_t x88 = F[0] * F[7];
        const scalar_t x89 = F[1] * F[6];
        const scalar_t x90 = x1 * x88 - x1 * x89 - x11 * x88 + x11 * x89 - x13 * x88 + x13 * x89 + x4 * x88 - x4 * x89 +
                             x6 * x88 - x6 * x89 - x8 * x88 + x8 * x89;
        const scalar_t x91 = lmbda * x77;
        const scalar_t x92 = -x84 * (x82 * x90 - x83 * x90 - x86 + x87 + x90 * x91);
        const scalar_t x93 = F[1] * F[5];
        const scalar_t x94 = F[2] * F[4];
        const scalar_t x95 = x1 * x93 - x1 * x94 - x11 * x93 + x11 * x94 - x13 * x93 + x13 * x94 + x4 * x93 - x4 * x94 +
                             x6 * x93 - x6 * x94 - x8 * x93 + x8 * x94;
        const scalar_t x96  = x69 * x95;
        const scalar_t x97  = F[5] * x72;
        const scalar_t x98  = F[5] * x74;
        const scalar_t x99  = F[0] * F[5];
        const scalar_t x100 = F[2] * F[3];
        const scalar_t x101 = -x1 * x100 + x1 * x99 + x100 * x11 + x100 * x13 - x100 * x4 - x100 * x6 + x100 * x8 - x11 * x99 -
                              x13 * x99 + x4 * x99 + x6 * x99 - x8 * x99;
        const scalar_t x102 = -x84 * (x101 * x82 - x101 * x83 + x101 * x91 - x97 + x98);
        const scalar_t x103 = F[4] * x72;
        const scalar_t x104 = F[4] * x74;
        const scalar_t x105 = F[0] * F[4];
        const scalar_t x106 = F[1] * F[3];
        const scalar_t x107 = x1 * x105 - x1 * x106 - x105 * x11 - x105 * x13 + x105 * x4 + x105 * x6 - x105 * x8 + x106 * x11 +
                              x106 * x13 - x106 * x4 - x106 * x6 + x106 * x8;
        const scalar_t x108 = x84 * (-x103 + x104 + x107 * x82 - x107 * x83 + x107 * x91);
        const scalar_t x109 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
        const scalar_t x110 = x51 * POW2(x56);
        const scalar_t x111 = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
        const scalar_t x112 = x51 * x62;
        const scalar_t x113 = mu * x56;
        const scalar_t x114 = x53 * x56;
        const scalar_t x115 = Ja * (x111 - x112 * x113 + x112 * x114 - x112 * x57);
        const scalar_t x116 = x67 * x76;
        const scalar_t x117 = x84 * (x113 * x116 - x114 * x116 + x116 * x57 + x73 - x75);
        const scalar_t x118 = x56 * x68;
        const scalar_t x119 = -x118 * x51 * x80;
        const scalar_t x120 = F[6] * x72;
        const scalar_t x121 = F[6] * x74;
        const scalar_t x122 = x76 * x90;
        const scalar_t x123 = x84 * (x113 * x122 - x114 * x122 - x120 + x121 + x122 * x57);
        const scalar_t x124 = x76 * x95;
        const scalar_t x125 = -x84 * (x113 * x124 - x114 * x124 + x124 * x57 + x97 - x98);
        const scalar_t x126 = x101 * x51;
        const scalar_t x127 = x118 * x126;
        const scalar_t x128 = F[3] * x72;
        const scalar_t x129 = F[3] * x74;
        const scalar_t x130 = x107 * x76;
        const scalar_t x131 = -x84 * (x113 * x130 - x114 * x130 - x128 + x129 + x130 * x57);
        const scalar_t x132 = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
        const scalar_t x133 = x51 * POW2(x62);
        const scalar_t x134 = mu * x62;
        const scalar_t x135 = x53 * x62;
        const scalar_t x136 = -x84 * (x116 * x134 - x116 * x135 + x116 * x63 + x86 - x87);
        const scalar_t x137 = x76 * x80;
        const scalar_t x138 = x84 * (x120 - x121 + x134 * x137 - x135 * x137 + x137 * x63);
        const scalar_t x139 = x112 * x68;
        const scalar_t x140 = -x139 * x90;
        const scalar_t x141 = x84 * (x103 - x104 + x124 * x134 - x124 * x135 + x124 * x63);
        const scalar_t x142 = x101 * x76;
        const scalar_t x143 = -x84 * (x128 - x129 + x134 * x142 - x135 * x142 + x142 * x63);
        const scalar_t x144 = x107 * x139;
        const scalar_t x145 = x51 * POW2(x67);
        const scalar_t x146 = x51 * x67;
        const scalar_t x147 = mu * x146;
        const scalar_t x148 = x53 * x80;
        const scalar_t x149 = Ja * (x146 * x148 - x146 * x81 - x147 * x80 + x54);
        const scalar_t x150 = x146 * x90;
        const scalar_t x151 = Ja * (lmbda * x150 + x147 * x90 - x150 * x53 + x61);
        const scalar_t x152 = -x146 * x68 * x95;
        const scalar_t x153 = F[2] * x72;
        const scalar_t x154 = F[2] * x74;
        const scalar_t x155 = x101 * x116;
        const scalar_t x156 = x84 * (lmbda * x155 + mu * x155 - x153 + x154 - x155 * x53);
        const scalar_t x157 = F[1] * x72;
        const scalar_t x158 = F[1] * x74;
        const scalar_t x159 = x107 * x116;
        const scalar_t x160 = -x84 * (lmbda * x159 + mu * x159 - x157 + x158 - x159 * x53);
        const scalar_t x161 = x51 * POW2(x80);
        const scalar_t x162 = x51 * x90;
        const scalar_t x163 = mu * x80;
        const scalar_t x164 = Ja * (x111 + x148 * x162 - x162 * x163 - x162 * x81);
        const scalar_t x165 = x84 * (-x124 * x148 + x124 * x163 + x124 * x81 + x153 - x154);
        const scalar_t x166 = -x126 * x68 * x80;
        const scalar_t x167 = F[0] * x72;
        const scalar_t x168 = F[0] * x74;
        const scalar_t x169 = x84 * (-x130 * x148 + x130 * x163 + x130 * x81 - x167 + x168);
        const scalar_t x170 = x51 * POW2(x90);
        const scalar_t x171 = x122 * x95;
        const scalar_t x172 = -x84 * (lmbda * x171 + mu * x171 + x157 - x158 - x171 * x53);
        const scalar_t x173 = x101 * x122;
        const scalar_t x174 = x84 * (lmbda * x173 + mu * x173 + x167 - x168 - x173 * x53);
        const scalar_t x175 = -x107 * x162 * x68;
        const scalar_t x176 = x51 * POW2(x95);
        const scalar_t x177 = x51 * x95;
        const scalar_t x178 = x101 * x177;
        const scalar_t x179 = Ja * (-lmbda * x178 - mu * x178 + x178 * x53 + x54);
        const scalar_t x180 = x107 * x177;
        const scalar_t x181 = Ja * (lmbda * x180 + mu * x180 - x180 * x53 + x61);
        const scalar_t x182 = POW2(x101) * x51;
        const scalar_t x183 = x107 * x126;
        const scalar_t x184 = Ja * (-lmbda * x183 - mu * x183 + x111 + x183 * x53);
        const scalar_t x185 = POW2(x107) * x51;
        S_lin[0]            = Ja * (lmbda * x52 + mu * x52 + x0 - x52 * x53);
        S_lin[1]            = x60;
        S_lin[2]            = x64;
        S_lin[3]            = x70;
        S_lin[4]            = x85;
        S_lin[5]            = x92;
        S_lin[6]            = x96;
        S_lin[7]            = x102;
        S_lin[8]            = x108;
        S_lin[9]            = x60;
        S_lin[10]           = Ja * (lmbda * x110 + mu * x110 + x109 - x110 * x53);
        S_lin[11]           = x115;
        S_lin[12]           = x117;
        S_lin[13]           = x119;
        S_lin[14]           = x123;
        S_lin[15]           = x125;
        S_lin[16]           = x127;
        S_lin[17]           = x131;
        S_lin[18]           = x64;
        S_lin[19]           = x115;
        S_lin[20]           = Ja * (lmbda * x133 + mu * x133 + x132 - x133 * x53);
        S_lin[21]           = x136;
        S_lin[22]           = x138;
        S_lin[23]           = x140;
        S_lin[24]           = x141;
        S_lin[25]           = x143;
        S_lin[26]           = x144;
        S_lin[27]           = x70;
        S_lin[28]           = x117;
        S_lin[29]           = x136;
        S_lin[30]           = Ja * (lmbda * x145 + mu * x145 + x0 - x145 * x53);
        S_lin[31]           = x149;
        S_lin[32]           = x151;
        S_lin[33]           = x152;
        S_lin[34]           = x156;
        S_lin[35]           = x160;
        S_lin[36]           = x85;
        S_lin[37]           = x119;
        S_lin[38]           = x138;
        S_lin[39]           = x149;
        S_lin[40]           = Ja * (lmbda * x161 + mu * x161 + x109 - x161 * x53);
        S_lin[41]           = x164;
        S_lin[42]           = x165;
        S_lin[43]           = x166;
        S_lin[44]           = x169;
        S_lin[45]           = x92;
        S_lin[46]           = x123;
        S_lin[47]           = x140;
        S_lin[48]           = x151;
        S_lin[49]           = x164;
        S_lin[50]           = Ja * (lmbda * x170 + mu * x170 + x132 - x170 * x53);
        S_lin[51]           = x172;
        S_lin[52]           = x174;
        S_lin[53]           = x175;
        S_lin[54]           = x96;
        S_lin[55]           = x125;
        S_lin[56]           = x141;
        S_lin[57]           = x152;
        S_lin[58]           = x165;
        S_lin[59]           = x172;
        S_lin[60]           = Ja * (lmbda * x176 + mu * x176 + x0 - x176 * x53);
        S_lin[61]           = x179;
        S_lin[62]           = x181;
        S_lin[63]           = x102;
        S_lin[64]           = x127;
        S_lin[65]           = x143;
        S_lin[66]           = x156;
        S_lin[67]           = x166;
        S_lin[68]           = x174;
        S_lin[69]           = x179;
        S_lin[70]           = Ja * (lmbda * x182 + mu * x182 + x109 - x182 * x53);
        S_lin[71]           = x184;
        S_lin[72]           = x108;
        S_lin[73]           = x131;
        S_lin[74]           = x144;
        S_lin[75]           = x160;
        S_lin[76]           = x169;
        S_lin[77]           = x175;
        S_lin[78]           = x181;
        S_lin[79]           = x184;
        S_lin[80]           = Ja * (lmbda * x185 + mu * x185 + x132 - x185 * x53);
    }

    // mundane ops: 1136 divs: 1 sqrts: 0
    // total ops: 1144
    const scalar_t x0   = S_lin[0] * adjugate[0] + S_lin[1] * adjugate[1] + S_lin[2] * adjugate[2];
    const scalar_t x1   = S_lin[10] * adjugate[1] + S_lin[11] * adjugate[2] + S_lin[9] * adjugate[0];
    const scalar_t x2   = S_lin[18] * adjugate[0] + S_lin[19] * adjugate[1] + S_lin[20] * adjugate[2];
    const scalar_t x3   = adjugate[0] * x0 + adjugate[1] * x1 + adjugate[2] * x2;
    const scalar_t x4   = S_lin[0] * adjugate[3] + S_lin[1] * adjugate[4] + S_lin[2] * adjugate[5];
    const scalar_t x5   = S_lin[10] * adjugate[4] + S_lin[11] * adjugate[5] + S_lin[9] * adjugate[3];
    const scalar_t x6   = S_lin[18] * adjugate[3] + S_lin[19] * adjugate[4] + S_lin[20] * adjugate[5];
    const scalar_t x7   = adjugate[0] * x4 + adjugate[1] * x5 + adjugate[2] * x6;
    const scalar_t x8   = S_lin[0] * adjugate[6] + S_lin[1] * adjugate[7] + S_lin[2] * adjugate[8];
    const scalar_t x9   = S_lin[10] * adjugate[7] + S_lin[11] * adjugate[8] + S_lin[9] * adjugate[6];
    const scalar_t x10  = S_lin[18] * adjugate[6] + S_lin[19] * adjugate[7] + S_lin[20] * adjugate[8];
    const scalar_t x11  = adjugate[0] * x8 + adjugate[1] * x9 + adjugate[2] * x10;
    const scalar_t x12  = x11 + x3 + x7;
    const scalar_t x13  = adjugate[3] * x0 + adjugate[4] * x1 + adjugate[5] * x2;
    const scalar_t x14  = adjugate[3] * x4 + adjugate[4] * x5 + adjugate[5] * x6;
    const scalar_t x15  = adjugate[3] * x8 + adjugate[4] * x9 + adjugate[5] * x10;
    const scalar_t x16  = x13 + x14 + x15;
    const scalar_t x17  = adjugate[6] * x0 + adjugate[7] * x1 + adjugate[8] * x2;
    const scalar_t x18  = adjugate[6] * x4 + adjugate[7] * x5 + adjugate[8] * x6;
    const scalar_t x19  = adjugate[6] * x8 + adjugate[7] * x9 + adjugate[8] * x10;
    const scalar_t x20  = x17 + x18 + x19;
    const scalar_t x21  = (1.0 / 6.0) / jacobian_determinant;
    const scalar_t x22  = S_lin[27] * adjugate[0] + S_lin[28] * adjugate[1] + S_lin[29] * adjugate[2];
    const scalar_t x23  = S_lin[36] * adjugate[0] + S_lin[37] * adjugate[1] + S_lin[38] * adjugate[2];
    const scalar_t x24  = S_lin[45] * adjugate[0] + S_lin[46] * adjugate[1] + S_lin[47] * adjugate[2];
    const scalar_t x25  = adjugate[0] * x22 + adjugate[1] * x23 + adjugate[2] * x24;
    const scalar_t x26  = S_lin[27] * adjugate[3] + S_lin[28] * adjugate[4] + S_lin[29] * adjugate[5];
    const scalar_t x27  = S_lin[36] * adjugate[3] + S_lin[37] * adjugate[4] + S_lin[38] * adjugate[5];
    const scalar_t x28  = S_lin[45] * adjugate[3] + S_lin[46] * adjugate[4] + S_lin[47] * adjugate[5];
    const scalar_t x29  = adjugate[0] * x26 + adjugate[1] * x27 + adjugate[2] * x28;
    const scalar_t x30  = S_lin[27] * adjugate[6] + S_lin[28] * adjugate[7] + S_lin[29] * adjugate[8];
    const scalar_t x31  = S_lin[36] * adjugate[6] + S_lin[37] * adjugate[7] + S_lin[38] * adjugate[8];
    const scalar_t x32  = S_lin[45] * adjugate[6] + S_lin[46] * adjugate[7] + S_lin[47] * adjugate[8];
    const scalar_t x33  = adjugate[0] * x30 + adjugate[1] * x31 + adjugate[2] * x32;
    const scalar_t x34  = x25 + x29 + x33;
    const scalar_t x35  = adjugate[3] * x22 + adjugate[4] * x23 + adjugate[5] * x24;
    const scalar_t x36  = adjugate[3] * x26 + adjugate[4] * x27 + adjugate[5] * x28;
    const scalar_t x37  = adjugate[3] * x30 + adjugate[4] * x31 + adjugate[5] * x32;
    const scalar_t x38  = x35 + x36 + x37;
    const scalar_t x39  = adjugate[6] * x22 + adjugate[7] * x23 + adjugate[8] * x24;
    const scalar_t x40  = adjugate[6] * x26 + adjugate[7] * x27 + adjugate[8] * x28;
    const scalar_t x41  = adjugate[6] * x30 + adjugate[7] * x31 + adjugate[8] * x32;
    const scalar_t x42  = x39 + x40 + x41;
    const scalar_t x43  = S_lin[54] * adjugate[0] + S_lin[55] * adjugate[1] + S_lin[56] * adjugate[2];
    const scalar_t x44  = S_lin[63] * adjugate[0] + S_lin[64] * adjugate[1] + S_lin[65] * adjugate[2];
    const scalar_t x45  = S_lin[72] * adjugate[0] + S_lin[73] * adjugate[1] + S_lin[74] * adjugate[2];
    const scalar_t x46  = adjugate[0] * x43 + adjugate[1] * x44 + adjugate[2] * x45;
    const scalar_t x47  = S_lin[54] * adjugate[3] + S_lin[55] * adjugate[4] + S_lin[56] * adjugate[5];
    const scalar_t x48  = S_lin[63] * adjugate[3] + S_lin[64] * adjugate[4] + S_lin[65] * adjugate[5];
    const scalar_t x49  = S_lin[72] * adjugate[3] + S_lin[73] * adjugate[4] + S_lin[74] * adjugate[5];
    const scalar_t x50  = adjugate[0] * x47 + adjugate[1] * x48 + adjugate[2] * x49;
    const scalar_t x51  = S_lin[54] * adjugate[6] + S_lin[55] * adjugate[7] + S_lin[56] * adjugate[8];
    const scalar_t x52  = S_lin[63] * adjugate[6] + S_lin[64] * adjugate[7] + S_lin[65] * adjugate[8];
    const scalar_t x53  = S_lin[72] * adjugate[6] + S_lin[73] * adjugate[7] + S_lin[74] * adjugate[8];
    const scalar_t x54  = adjugate[0] * x51 + adjugate[1] * x52 + adjugate[2] * x53;
    const scalar_t x55  = x46 + x50 + x54;
    const scalar_t x56  = adjugate[3] * x43 + adjugate[4] * x44 + adjugate[5] * x45;
    const scalar_t x57  = adjugate[3] * x47 + adjugate[4] * x48 + adjugate[5] * x49;
    const scalar_t x58  = adjugate[3] * x51 + adjugate[4] * x52 + adjugate[5] * x53;
    const scalar_t x59  = x56 + x57 + x58;
    const scalar_t x60  = adjugate[6] * x43 + adjugate[7] * x44 + adjugate[8] * x45;
    const scalar_t x61  = adjugate[6] * x47 + adjugate[7] * x48 + adjugate[8] * x49;
    const scalar_t x62  = adjugate[6] * x51 + adjugate[7] * x52 + adjugate[8] * x53;
    const scalar_t x63  = x60 + x61 + x62;
    const scalar_t x64  = S_lin[3] * adjugate[0] + S_lin[4] * adjugate[1] + S_lin[5] * adjugate[2];
    const scalar_t x65  = S_lin[12] * adjugate[0] + S_lin[13] * adjugate[1] + S_lin[14] * adjugate[2];
    const scalar_t x66  = S_lin[21] * adjugate[0] + S_lin[22] * adjugate[1] + S_lin[23] * adjugate[2];
    const scalar_t x67  = adjugate[0] * x64 + adjugate[1] * x65 + adjugate[2] * x66;
    const scalar_t x68  = S_lin[3] * adjugate[3] + S_lin[4] * adjugate[4] + S_lin[5] * adjugate[5];
    const scalar_t x69  = S_lin[12] * adjugate[3] + S_lin[13] * adjugate[4] + S_lin[14] * adjugate[5];
    const scalar_t x70  = S_lin[21] * adjugate[3] + S_lin[22] * adjugate[4] + S_lin[23] * adjugate[5];
    const scalar_t x71  = adjugate[0] * x68 + adjugate[1] * x69 + adjugate[2] * x70;
    const scalar_t x72  = S_lin[3] * adjugate[6] + S_lin[4] * adjugate[7] + S_lin[5] * adjugate[8];
    const scalar_t x73  = S_lin[12] * adjugate[6] + S_lin[13] * adjugate[7] + S_lin[14] * adjugate[8];
    const scalar_t x74  = S_lin[21] * adjugate[6] + S_lin[22] * adjugate[7] + S_lin[23] * adjugate[8];
    const scalar_t x75  = adjugate[0] * x72 + adjugate[1] * x73 + adjugate[2] * x74;
    const scalar_t x76  = x67 + x71 + x75;
    const scalar_t x77  = adjugate[3] * x64 + adjugate[4] * x65 + adjugate[5] * x66;
    const scalar_t x78  = adjugate[3] * x68 + adjugate[4] * x69 + adjugate[5] * x70;
    const scalar_t x79  = adjugate[3] * x72 + adjugate[4] * x73 + adjugate[5] * x74;
    const scalar_t x80  = x77 + x78 + x79;
    const scalar_t x81  = adjugate[6] * x64 + adjugate[7] * x65 + adjugate[8] * x66;
    const scalar_t x82  = adjugate[6] * x68 + adjugate[7] * x69 + adjugate[8] * x70;
    const scalar_t x83  = adjugate[6] * x72 + adjugate[7] * x73 + adjugate[8] * x74;
    const scalar_t x84  = x81 + x82 + x83;
    const scalar_t x85  = S_lin[30] * adjugate[0] + S_lin[31] * adjugate[1] + S_lin[32] * adjugate[2];
    const scalar_t x86  = S_lin[39] * adjugate[0] + S_lin[40] * adjugate[1] + S_lin[41] * adjugate[2];
    const scalar_t x87  = S_lin[48] * adjugate[0] + S_lin[49] * adjugate[1] + S_lin[50] * adjugate[2];
    const scalar_t x88  = adjugate[0] * x85 + adjugate[1] * x86 + adjugate[2] * x87;
    const scalar_t x89  = S_lin[30] * adjugate[3] + S_lin[31] * adjugate[4] + S_lin[32] * adjugate[5];
    const scalar_t x90  = S_lin[39] * adjugate[3] + S_lin[40] * adjugate[4] + S_lin[41] * adjugate[5];
    const scalar_t x91  = S_lin[48] * adjugate[3] + S_lin[49] * adjugate[4] + S_lin[50] * adjugate[5];
    const scalar_t x92  = adjugate[0] * x89 + adjugate[1] * x90 + adjugate[2] * x91;
    const scalar_t x93  = S_lin[30] * adjugate[6] + S_lin[31] * adjugate[7] + S_lin[32] * adjugate[8];
    const scalar_t x94  = S_lin[39] * adjugate[6] + S_lin[40] * adjugate[7] + S_lin[41] * adjugate[8];
    const scalar_t x95  = S_lin[48] * adjugate[6] + S_lin[49] * adjugate[7] + S_lin[50] * adjugate[8];
    const scalar_t x96  = adjugate[0] * x93 + adjugate[1] * x94 + adjugate[2] * x95;
    const scalar_t x97  = x88 + x92 + x96;
    const scalar_t x98  = adjugate[3] * x85 + adjugate[4] * x86 + adjugate[5] * x87;
    const scalar_t x99  = adjugate[3] * x89 + adjugate[4] * x90 + adjugate[5] * x91;
    const scalar_t x100 = adjugate[3] * x93 + adjugate[4] * x94 + adjugate[5] * x95;
    const scalar_t x101 = x100 + x98 + x99;
    const scalar_t x102 = adjugate[6] * x85 + adjugate[7] * x86 + adjugate[8] * x87;
    const scalar_t x103 = adjugate[6] * x89 + adjugate[7] * x90 + adjugate[8] * x91;
    const scalar_t x104 = adjugate[6] * x93 + adjugate[7] * x94 + adjugate[8] * x95;
    const scalar_t x105 = x102 + x103 + x104;
    const scalar_t x106 = S_lin[57] * adjugate[0] + S_lin[58] * adjugate[1] + S_lin[59] * adjugate[2];
    const scalar_t x107 = S_lin[66] * adjugate[0] + S_lin[67] * adjugate[1] + S_lin[68] * adjugate[2];
    const scalar_t x108 = S_lin[75] * adjugate[0] + S_lin[76] * adjugate[1] + S_lin[77] * adjugate[2];
    const scalar_t x109 = adjugate[0] * x106 + adjugate[1] * x107 + adjugate[2] * x108;
    const scalar_t x110 = S_lin[57] * adjugate[3] + S_lin[58] * adjugate[4] + S_lin[59] * adjugate[5];
    const scalar_t x111 = S_lin[66] * adjugate[3] + S_lin[67] * adjugate[4] + S_lin[68] * adjugate[5];
    const scalar_t x112 = S_lin[75] * adjugate[3] + S_lin[76] * adjugate[4] + S_lin[77] * adjugate[5];
    const scalar_t x113 = adjugate[0] * x110 + adjugate[1] * x111 + adjugate[2] * x112;
    const scalar_t x114 = S_lin[57] * adjugate[6] + S_lin[58] * adjugate[7] + S_lin[59] * adjugate[8];
    const scalar_t x115 = S_lin[66] * adjugate[6] + S_lin[67] * adjugate[7] + S_lin[68] * adjugate[8];
    const scalar_t x116 = S_lin[75] * adjugate[6] + S_lin[76] * adjugate[7] + S_lin[77] * adjugate[8];
    const scalar_t x117 = adjugate[0] * x114 + adjugate[1] * x115 + adjugate[2] * x116;
    const scalar_t x118 = x109 + x113 + x117;
    const scalar_t x119 = adjugate[3] * x106 + adjugate[4] * x107 + adjugate[5] * x108;
    const scalar_t x120 = adjugate[3] * x110 + adjugate[4] * x111 + adjugate[5] * x112;
    const scalar_t x121 = adjugate[3] * x114 + adjugate[4] * x115 + adjugate[5] * x116;
    const scalar_t x122 = x119 + x120 + x121;
    const scalar_t x123 = adjugate[6] * x106 + adjugate[7] * x107 + adjugate[8] * x108;
    const scalar_t x124 = adjugate[6] * x110 + adjugate[7] * x111 + adjugate[8] * x112;
    const scalar_t x125 = adjugate[6] * x114 + adjugate[7] * x115 + adjugate[8] * x116;
    const scalar_t x126 = x123 + x124 + x125;
    const scalar_t x127 = S_lin[6] * adjugate[0] + S_lin[7] * adjugate[1] + S_lin[8] * adjugate[2];
    const scalar_t x128 = S_lin[15] * adjugate[0] + S_lin[16] * adjugate[1] + S_lin[17] * adjugate[2];
    const scalar_t x129 = S_lin[24] * adjugate[0] + S_lin[25] * adjugate[1] + S_lin[26] * adjugate[2];
    const scalar_t x130 = adjugate[0] * x127 + adjugate[1] * x128 + adjugate[2] * x129;
    const scalar_t x131 = S_lin[6] * adjugate[3] + S_lin[7] * adjugate[4] + S_lin[8] * adjugate[5];
    const scalar_t x132 = S_lin[15] * adjugate[3] + S_lin[16] * adjugate[4] + S_lin[17] * adjugate[5];
    const scalar_t x133 = S_lin[24] * adjugate[3] + S_lin[25] * adjugate[4] + S_lin[26] * adjugate[5];
    const scalar_t x134 = adjugate[0] * x131 + adjugate[1] * x132 + adjugate[2] * x133;
    const scalar_t x135 = S_lin[6] * adjugate[6] + S_lin[7] * adjugate[7] + S_lin[8] * adjugate[8];
    const scalar_t x136 = S_lin[15] * adjugate[6] + S_lin[16] * adjugate[7] + S_lin[17] * adjugate[8];
    const scalar_t x137 = S_lin[24] * adjugate[6] + S_lin[25] * adjugate[7] + S_lin[26] * adjugate[8];
    const scalar_t x138 = adjugate[0] * x135 + adjugate[1] * x136 + adjugate[2] * x137;
    const scalar_t x139 = x130 + x134 + x138;
    const scalar_t x140 = adjugate[3] * x127 + adjugate[4] * x128 + adjugate[5] * x129;
    const scalar_t x141 = adjugate[3] * x131 + adjugate[4] * x132 + adjugate[5] * x133;
    const scalar_t x142 = adjugate[3] * x135 + adjugate[4] * x136 + adjugate[5] * x137;
    const scalar_t x143 = x140 + x141 + x142;
    const scalar_t x144 = adjugate[6] * x127 + adjugate[7] * x128 + adjugate[8] * x129;
    const scalar_t x145 = adjugate[6] * x131 + adjugate[7] * x132 + adjugate[8] * x133;
    const scalar_t x146 = adjugate[6] * x135 + adjugate[7] * x136 + adjugate[8] * x137;
    const scalar_t x147 = x144 + x145 + x146;
    const scalar_t x148 = S_lin[33] * adjugate[0] + S_lin[34] * adjugate[1] + S_lin[35] * adjugate[2];
    const scalar_t x149 = S_lin[42] * adjugate[0] + S_lin[43] * adjugate[1] + S_lin[44] * adjugate[2];
    const scalar_t x150 = S_lin[51] * adjugate[0] + S_lin[52] * adjugate[1] + S_lin[53] * adjugate[2];
    const scalar_t x151 = adjugate[0] * x148 + adjugate[1] * x149 + adjugate[2] * x150;
    const scalar_t x152 = S_lin[33] * adjugate[3] + S_lin[34] * adjugate[4] + S_lin[35] * adjugate[5];
    const scalar_t x153 = S_lin[42] * adjugate[3] + S_lin[43] * adjugate[4] + S_lin[44] * adjugate[5];
    const scalar_t x154 = S_lin[51] * adjugate[3] + S_lin[52] * adjugate[4] + S_lin[53] * adjugate[5];
    const scalar_t x155 = adjugate[0] * x152 + adjugate[1] * x153 + adjugate[2] * x154;
    const scalar_t x156 = S_lin[33] * adjugate[6] + S_lin[34] * adjugate[7] + S_lin[35] * adjugate[8];
    const scalar_t x157 = S_lin[42] * adjugate[6] + S_lin[43] * adjugate[7] + S_lin[44] * adjugate[8];
    const scalar_t x158 = S_lin[51] * adjugate[6] + S_lin[52] * adjugate[7] + S_lin[53] * adjugate[8];
    const scalar_t x159 = adjugate[0] * x156 + adjugate[1] * x157 + adjugate[2] * x158;
    const scalar_t x160 = x151 + x155 + x159;
    const scalar_t x161 = adjugate[3] * x148 + adjugate[4] * x149 + adjugate[5] * x150;
    const scalar_t x162 = adjugate[3] * x152 + adjugate[4] * x153 + adjugate[5] * x154;
    const scalar_t x163 = adjugate[3] * x156 + adjugate[4] * x157 + adjugate[5] * x158;
    const scalar_t x164 = x161 + x162 + x163;
    const scalar_t x165 = adjugate[6] * x148 + adjugate[7] * x149 + adjugate[8] * x150;
    const scalar_t x166 = adjugate[6] * x152 + adjugate[7] * x153 + adjugate[8] * x154;
    const scalar_t x167 = adjugate[6] * x156 + adjugate[7] * x157 + adjugate[8] * x158;
    const scalar_t x168 = x165 + x166 + x167;
    const scalar_t x169 = S_lin[60] * adjugate[0] + S_lin[61] * adjugate[1] + S_lin[62] * adjugate[2];
    const scalar_t x170 = S_lin[69] * adjugate[0] + S_lin[70] * adjugate[1] + S_lin[71] * adjugate[2];
    const scalar_t x171 = S_lin[78] * adjugate[0] + S_lin[79] * adjugate[1] + S_lin[80] * adjugate[2];
    const scalar_t x172 = adjugate[0] * x169 + adjugate[1] * x170 + adjugate[2] * x171;
    const scalar_t x173 = S_lin[60] * adjugate[3] + S_lin[61] * adjugate[4] + S_lin[62] * adjugate[5];
    const scalar_t x174 = S_lin[69] * adjugate[3] + S_lin[70] * adjugate[4] + S_lin[71] * adjugate[5];
    const scalar_t x175 = S_lin[78] * adjugate[3] + S_lin[79] * adjugate[4] + S_lin[80] * adjugate[5];
    const scalar_t x176 = adjugate[0] * x173 + adjugate[1] * x174 + adjugate[2] * x175;
    const scalar_t x177 = S_lin[60] * adjugate[6] + S_lin[61] * adjugate[7] + S_lin[62] * adjugate[8];
    const scalar_t x178 = S_lin[69] * adjugate[6] + S_lin[70] * adjugate[7] + S_lin[71] * adjugate[8];
    const scalar_t x179 = S_lin[78] * adjugate[6] + S_lin[79] * adjugate[7] + S_lin[80] * adjugate[8];
    const scalar_t x180 = adjugate[0] * x177 + adjugate[1] * x178 + adjugate[2] * x179;
    const scalar_t x181 = x172 + x176 + x180;
    const scalar_t x182 = adjugate[3] * x169 + adjugate[4] * x170 + adjugate[5] * x171;
    const scalar_t x183 = adjugate[3] * x173 + adjugate[4] * x174 + adjugate[5] * x175;
    const scalar_t x184 = adjugate[3] * x177 + adjugate[4] * x178 + adjugate[5] * x179;
    const scalar_t x185 = x182 + x183 + x184;
    const scalar_t x186 = adjugate[6] * x169 + adjugate[7] * x170 + adjugate[8] * x171;
    const scalar_t x187 = adjugate[6] * x173 + adjugate[7] * x174 + adjugate[8] * x175;
    const scalar_t x188 = adjugate[6] * x177 + adjugate[7] * x178 + adjugate[8] * x179;
    const scalar_t x189 = x186 + x187 + x188;
    H[0] += x21 * (x12 + x16 + x20);
    H[1] += -x12 * x21;
    H[2] += -x16 * x21;
    H[3] += -x20 * x21;
    H[4] += x21 * (x34 + x38 + x42);
    H[5] += -x21 * x34;
    H[6] += -x21 * x38;
    H[7] += -x21 * x42;
    H[8] += x21 * (x55 + x59 + x63);
    H[9] += -x21 * x55;
    H[10] += -x21 * x59;
    H[11] += -x21 * x63;
    H[12] += -x21 * (x13 + x17 + x3);
    H[13] += x21 * x3;
    H[14] += x13 * x21;
    H[15] += x17 * x21;
    H[16] += -x21 * (x25 + x35 + x39);
    H[17] += x21 * x25;
    H[18] += x21 * x35;
    H[19] += x21 * x39;
    H[20] += -x21 * (x46 + x56 + x60);
    H[21] += x21 * x46;
    H[22] += x21 * x56;
    H[23] += x21 * x60;
    H[24] += -x21 * (x14 + x18 + x7);
    H[25] += x21 * x7;
    H[26] += x14 * x21;
    H[27] += x18 * x21;
    H[28] += -x21 * (x29 + x36 + x40);
    H[29] += x21 * x29;
    H[30] += x21 * x36;
    H[31] += x21 * x40;
    H[32] += -x21 * (x50 + x57 + x61);
    H[33] += x21 * x50;
    H[34] += x21 * x57;
    H[35] += x21 * x61;
    H[36] += -x21 * (x11 + x15 + x19);
    H[37] += x11 * x21;
    H[38] += x15 * x21;
    H[39] += x19 * x21;
    H[40] += -x21 * (x33 + x37 + x41);
    H[41] += x21 * x33;
    H[42] += x21 * x37;
    H[43] += x21 * x41;
    H[44] += -x21 * (x54 + x58 + x62);
    H[45] += x21 * x54;
    H[46] += x21 * x58;
    H[47] += x21 * x62;
    H[48] += x21 * (x76 + x80 + x84);
    H[49] += -x21 * x76;
    H[50] += -x21 * x80;
    H[51] += -x21 * x84;
    H[52] += x21 * (x101 + x105 + x97);
    H[53] += -x21 * x97;
    H[54] += -x101 * x21;
    H[55] += -x105 * x21;
    H[56] += x21 * (x118 + x122 + x126);
    H[57] += -x118 * x21;
    H[58] += -x122 * x21;
    H[59] += -x126 * x21;
    H[60] += -x21 * (x67 + x77 + x81);
    H[61] += x21 * x67;
    H[62] += x21 * x77;
    H[63] += x21 * x81;
    H[64] += -x21 * (x102 + x88 + x98);
    H[65] += x21 * x88;
    H[66] += x21 * x98;
    H[67] += x102 * x21;
    H[68] += -x21 * (x109 + x119 + x123);
    H[69] += x109 * x21;
    H[70] += x119 * x21;
    H[71] += x123 * x21;
    H[72] += -x21 * (x71 + x78 + x82);
    H[73] += x21 * x71;
    H[74] += x21 * x78;
    H[75] += x21 * x82;
    H[76] += -x21 * (x103 + x92 + x99);
    H[77] += x21 * x92;
    H[78] += x21 * x99;
    H[79] += x103 * x21;
    H[80] += -x21 * (x113 + x120 + x124);
    H[81] += x113 * x21;
    H[82] += x120 * x21;
    H[83] += x124 * x21;
    H[84] += -x21 * (x75 + x79 + x83);
    H[85] += x21 * x75;
    H[86] += x21 * x79;
    H[87] += x21 * x83;
    H[88] += -x21 * (x100 + x104 + x96);
    H[89] += x21 * x96;
    H[90] += x100 * x21;
    H[91] += x104 * x21;
    H[92] += -x21 * (x117 + x121 + x125);
    H[93] += x117 * x21;
    H[94] += x121 * x21;
    H[95] += x125 * x21;
    H[96] += x21 * (x139 + x143 + x147);
    H[97] += -x139 * x21;
    H[98] += -x143 * x21;
    H[99] += -x147 * x21;
    H[100] += x21 * (x160 + x164 + x168);
    H[101] += -x160 * x21;
    H[102] += -x164 * x21;
    H[103] += -x168 * x21;
    H[104] += x21 * (x181 + x185 + x189);
    H[105] += -x181 * x21;
    H[106] += -x185 * x21;
    H[107] += -x189 * x21;
    H[108] += -x21 * (x130 + x140 + x144);
    H[109] += x130 * x21;
    H[110] += x140 * x21;
    H[111] += x144 * x21;
    H[112] += -x21 * (x151 + x161 + x165);
    H[113] += x151 * x21;
    H[114] += x161 * x21;
    H[115] += x165 * x21;
    H[116] += -x21 * (x172 + x182 + x186);
    H[117] += x172 * x21;
    H[118] += x182 * x21;
    H[119] += x186 * x21;
    H[120] += -x21 * (x134 + x141 + x145);
    H[121] += x134 * x21;
    H[122] += x141 * x21;
    H[123] += x145 * x21;
    H[124] += -x21 * (x155 + x162 + x166);
    H[125] += x155 * x21;
    H[126] += x162 * x21;
    H[127] += x166 * x21;
    H[128] += -x21 * (x176 + x183 + x187);
    H[129] += x176 * x21;
    H[130] += x183 * x21;
    H[131] += x187 * x21;
    H[132] += -x21 * (x138 + x142 + x146);
    H[133] += x138 * x21;
    H[134] += x142 * x21;
    H[135] += x146 * x21;
    H[136] += -x21 * (x159 + x163 + x167);
    H[137] += x159 * x21;
    H[138] += x163 * x21;
    H[139] += x167 * x21;
    H[140] += -x21 * (x180 + x184 + x188);
    H[141] += x180 * x21;
    H[142] += x184 * x21;
    H[143] += x188 * x21;
}

#endif