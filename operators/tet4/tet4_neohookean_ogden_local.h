#ifndef TET4_NEOHOOKEAN_OGDEN_LOCAL_H
#define TET4_NEOHOOKEAN_OGDEN_LOCAL_H

#include <math.h>
#include "sfem_base.h"

#include "tet4_partial_assembly_neohookean_inline.h"

static SFEM_INLINE void tet4_neohookean_ogden_objective_elemental(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                  const scalar_t                      jacobian_determinant,
                                                                  const scalar_t                      lmbda,
                                                                  const scalar_t                      mu,
                                                                  const scalar_t *const SFEM_RESTRICT dispx,
                                                                  const scalar_t *const SFEM_RESTRICT dispy,
                                                                  const scalar_t *const SFEM_RESTRICT dispz,
                                                                  scalar_t *const SFEM_RESTRICT       v) {
    scalar_t F[9];
    tet4_F(adjugate, jacobian_determinant, dispx, dispy, dispz, F);

    // mundane ops: 50 divs: 0 sqrts: 0
    // total ops: 50
    const scalar_t x0 = log(F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] + F[1] * F[5] * F[6] +
                            F[2] * F[3] * F[7] - F[2] * F[4] * F[6]);
    v[0] += (1.0 / 6.0) * jacobian_determinant *
            ((1.0 / 2.0) * lmbda * POW2(x0) - mu * x0 +
             (1.0 / 2.0) * mu *
                     (POW2(F[0]) + POW2(F[1]) + POW2(F[2]) + POW2(F[3]) + POW2(F[4]) + POW2(F[5]) + POW2(F[6]) + POW2(F[7]) +
                      POW2(F[8]) - 3));
}

static SFEM_INLINE void tet4_neohookean_ogden_objective_integral(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                 const scalar_t                      jacobian_determinant,
                                                                 const scalar_t                      mu,
                                                                 const scalar_t                      lmbda,
                                                                 const scalar_t *const SFEM_RESTRICT dispx,
                                                                 const scalar_t *const SFEM_RESTRICT dispy,
                                                                 const scalar_t *const SFEM_RESTRICT dispz,
                                                                 scalar_t *const SFEM_RESTRICT       v) {
    tet4_neohookean_ogden_objective_elemental(adjugate, jacobian_determinant, mu, lmbda, dispx, dispy, dispz, v);
}

static SFEM_INLINE void tet4_neohookean_ogden_objective_steps_integral(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                       const scalar_t                      jacobian_determinant,
                                                                       const scalar_t                      mu,
                                                                       const scalar_t                      lmbda,
                                                                       const scalar_t *const SFEM_RESTRICT dispx,
                                                                       const scalar_t *const SFEM_RESTRICT dispy,
                                                                       const scalar_t *const SFEM_RESTRICT dispz,
                                                                       const scalar_t *const SFEM_RESTRICT incx,
                                                                       const scalar_t *const SFEM_RESTRICT incy,
                                                                       const scalar_t *const SFEM_RESTRICT incz,
                                                                       const int                           nsteps,
                                                                       const scalar_t *const SFEM_RESTRICT steps,
                                                                       scalar_t *const SFEM_RESTRICT       v) {
    scalar_t ux[4];
    scalar_t uy[4];
    scalar_t uz[4];

    for (int i = 0; i < nsteps; i++) {
        for (int j = 0; j < 4; j++) {
            ux[j] = dispx[j] + incx[j] * steps[i];
            uy[j] = dispy[j] + incy[j] * steps[i];
            uz[j] = dispz[j] + incz[j] * steps[i];
        }

        tet4_neohookean_ogden_objective_integral(adjugate, jacobian_determinant, mu, lmbda, ux, uy, uz, &v[i]);
    }
}

static SFEM_INLINE void tet4_neohookean_hessian(const scalar_t *const SFEM_RESTRICT adjugate,
                                                const scalar_t                      jacobian_determinant,
                                                const scalar_t                      lmbda,
                                                const scalar_t                      mu,
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
        // mundane ops: 426 divs: 1 sqrts: 0
        // total ops: 434
        const scalar_t x0   = F[4] * F[8];
        const scalar_t x1   = F[5] * F[7];
        const scalar_t x2   = x0 - x1;
        const scalar_t x3   = F[5] * F[6];
        const scalar_t x4   = F[3] * F[7];
        const scalar_t x5   = F[3] * F[8];
        const scalar_t x6   = F[4] * F[6];
        const scalar_t x7   = F[0] * x0 - F[0] * x1 + F[1] * x3 - F[1] * x5 + F[2] * x4 - F[2] * x6;
        const scalar_t x8   = (1 / POW2(x7));
        const scalar_t x9   = POW2(x2) * x8;
        const scalar_t x10  = log(x7);
        const scalar_t x11  = lmbda * x10;
        const scalar_t x12  = -x3 + x5;
        const scalar_t x13  = -lmbda * x10 + lmbda + mu;
        const scalar_t x14  = -x13;
        const scalar_t x15  = x2 * x8;
        const scalar_t x16  = x14 * x15;
        const scalar_t x17  = x12 * x16;
        const scalar_t x18  = x4 - x6;
        const scalar_t x19  = x13 * x15;
        const scalar_t x20  = x18 * x19;
        const scalar_t x21  = F[1] * F[8] - F[2] * F[7];
        const scalar_t x22  = x16 * x21;
        const scalar_t x23  = 1.0 / x7;
        const scalar_t x24  = F[8] * mu;
        const scalar_t x25  = F[8] * x11;
        const scalar_t x26  = F[0] * F[8] - F[2] * F[6];
        const scalar_t x27  = x2 * x23;
        const scalar_t x28  = x26 * x27;
        const scalar_t x29  = x23 * (lmbda * x28 + mu * x28 - x11 * x28 - x24 + x25);
        const scalar_t x30  = F[7] * mu;
        const scalar_t x31  = F[7] * x11;
        const scalar_t x32  = F[0] * F[7] - F[1] * F[6];
        const scalar_t x33  = x27 * x32;
        const scalar_t x34  = x23 * (lmbda * x10 * x2 * x23 * x32 - lmbda * x33 - mu * x33 + x30 - x31);
        const scalar_t x35  = F[1] * F[5] - F[2] * F[4];
        const scalar_t x36  = x19 * x35;
        const scalar_t x37  = F[5] * mu;
        const scalar_t x38  = F[5] * x11;
        const scalar_t x39  = F[0] * F[5] - F[2] * F[3];
        const scalar_t x40  = x27 * x39;
        const scalar_t x41  = x23 * (lmbda * x10 * x2 * x23 * x39 - lmbda * x40 - mu * x40 + x37 - x38);
        const scalar_t x42  = F[4] * mu;
        const scalar_t x43  = F[4] * x11;
        const scalar_t x44  = F[0] * F[4] - F[1] * F[3];
        const scalar_t x45  = x27 * x44;
        const scalar_t x46  = x23 * (lmbda * x45 + mu * x45 - x11 * x45 - x42 + x43);
        const scalar_t x47  = POW2(x12) * x8;
        const scalar_t x48  = x12 * x8;
        const scalar_t x49  = x14 * x48;
        const scalar_t x50  = x18 * x49;
        const scalar_t x51  = x12 * x23;
        const scalar_t x52  = x21 * x51;
        const scalar_t x53  = x23 * (lmbda * x52 + mu * x52 - x11 * x52 + x24 - x25);
        const scalar_t x54  = x26 * x49;
        const scalar_t x55  = F[6] * mu;
        const scalar_t x56  = F[6] * x11;
        const scalar_t x57  = x32 * x51;
        const scalar_t x58  = x23 * (lmbda * x57 + mu * x57 - x11 * x57 - x55 + x56);
        const scalar_t x59  = x35 * x51;
        const scalar_t x60  = x23 * (lmbda * x10 * x12 * x23 * x35 - lmbda * x59 - mu * x59 - x37 + x38);
        const scalar_t x61  = x13 * x39 * x48;
        const scalar_t x62  = F[3] * mu;
        const scalar_t x63  = F[3] * x11;
        const scalar_t x64  = x44 * x51;
        const scalar_t x65  = x23 * (lmbda * x10 * x12 * x23 * x44 - lmbda * x64 - mu * x64 + x62 - x63);
        const scalar_t x66  = POW2(x18) * x8;
        const scalar_t x67  = x18 * x23;
        const scalar_t x68  = x21 * x67;
        const scalar_t x69  = x23 * (lmbda * x10 * x18 * x21 * x23 - lmbda * x68 - mu * x68 - x30 + x31);
        const scalar_t x70  = x26 * x67;
        const scalar_t x71  = x23 * (lmbda * x70 + mu * x70 - x11 * x70 + x55 - x56);
        const scalar_t x72  = x18 * x8;
        const scalar_t x73  = x14 * x32;
        const scalar_t x74  = x72 * x73;
        const scalar_t x75  = x35 * x67;
        const scalar_t x76  = x23 * (lmbda * x75 + mu * x75 - x11 * x75 + x42 - x43);
        const scalar_t x77  = x39 * x67;
        const scalar_t x78  = x23 * (lmbda * x10 * x18 * x23 * x39 - lmbda * x77 - mu * x77 - x62 + x63);
        const scalar_t x79  = x13 * x44;
        const scalar_t x80  = x72 * x79;
        const scalar_t x81  = POW2(x21) * x8;
        const scalar_t x82  = x21 * x8;
        const scalar_t x83  = x14 * x82;
        const scalar_t x84  = x26 * x83;
        const scalar_t x85  = x13 * x32 * x82;
        const scalar_t x86  = x35 * x83;
        const scalar_t x87  = F[2] * mu;
        const scalar_t x88  = F[2] * x11;
        const scalar_t x89  = x21 * x23;
        const scalar_t x90  = x39 * x89;
        const scalar_t x91  = x23 * (lmbda * x90 + mu * x90 - x11 * x90 - x87 + x88);
        const scalar_t x92  = F[1] * mu;
        const scalar_t x93  = F[1] * x11;
        const scalar_t x94  = x44 * x89;
        const scalar_t x95  = x23 * (lmbda * x10 * x21 * x23 * x44 - lmbda * x94 - mu * x94 + x92 - x93);
        const scalar_t x96  = POW2(x26) * x8;
        const scalar_t x97  = x26 * x8;
        const scalar_t x98  = x73 * x97;
        const scalar_t x99  = x23 * x26;
        const scalar_t x100 = x35 * x99;
        const scalar_t x101 = x23 * (lmbda * x100 + mu * x100 - x100 * x11 + x87 - x88);
        const scalar_t x102 = x14 * x39;
        const scalar_t x103 = x102 * x97;
        const scalar_t x104 = F[0] * mu;
        const scalar_t x105 = F[0] * x11;
        const scalar_t x106 = x44 * x99;
        const scalar_t x107 = x23 * (lmbda * x106 + mu * x106 - x104 + x105 - x106 * x11);
        const scalar_t x108 = POW2(x32) * x8;
        const scalar_t x109 = x23 * x32;
        const scalar_t x110 = x109 * x35;
        const scalar_t x111 = x23 * (lmbda * x10 * x23 * x32 * x35 - lmbda * x110 - mu * x110 - x92 + x93);
        const scalar_t x112 = x109 * x39;
        const scalar_t x113 = x23 * (lmbda * x112 + mu * x112 + x104 - x105 - x11 * x112);
        const scalar_t x114 = x44 * x8;
        const scalar_t x115 = x114 * x73;
        const scalar_t x116 = POW2(x35) * x8;
        const scalar_t x117 = x35 * x8;
        const scalar_t x118 = x102 * x117;
        const scalar_t x119 = x117 * x79;
        const scalar_t x120 = POW2(x39) * x8;
        const scalar_t x121 = x102 * x114;
        const scalar_t x122 = POW2(x44) * x8;
        S_lin[0]            = lmbda * x9 + mu * x9 + mu - x11 * x9;
        S_lin[1]            = x17;
        S_lin[2]            = x20;
        S_lin[3]            = x22;
        S_lin[4]            = x29;
        S_lin[5]            = x34;
        S_lin[6]            = x36;
        S_lin[7]            = x41;
        S_lin[8]            = x46;
        S_lin[9]            = x17;
        S_lin[10]           = lmbda * x47 + mu * x47 + mu - x11 * x47;
        S_lin[11]           = x50;
        S_lin[12]           = x53;
        S_lin[13]           = x54;
        S_lin[14]           = x58;
        S_lin[15]           = x60;
        S_lin[16]           = x61;
        S_lin[17]           = x65;
        S_lin[18]           = x20;
        S_lin[19]           = x50;
        S_lin[20]           = lmbda * x66 + mu * x66 + mu - x11 * x66;
        S_lin[21]           = x69;
        S_lin[22]           = x71;
        S_lin[23]           = x74;
        S_lin[24]           = x76;
        S_lin[25]           = x78;
        S_lin[26]           = x80;
        S_lin[27]           = x22;
        S_lin[28]           = x53;
        S_lin[29]           = x69;
        S_lin[30]           = lmbda * x81 + mu * x81 + mu - x11 * x81;
        S_lin[31]           = x84;
        S_lin[32]           = x85;
        S_lin[33]           = x86;
        S_lin[34]           = x91;
        S_lin[35]           = x95;
        S_lin[36]           = x29;
        S_lin[37]           = x54;
        S_lin[38]           = x71;
        S_lin[39]           = x84;
        S_lin[40]           = lmbda * x96 + mu * x96 + mu - x11 * x96;
        S_lin[41]           = x98;
        S_lin[42]           = x101;
        S_lin[43]           = x103;
        S_lin[44]           = x107;
        S_lin[45]           = x34;
        S_lin[46]           = x58;
        S_lin[47]           = x74;
        S_lin[48]           = x85;
        S_lin[49]           = x98;
        S_lin[50]           = lmbda * x108 + mu * x108 + mu - x108 * x11;
        S_lin[51]           = x111;
        S_lin[52]           = x113;
        S_lin[53]           = x115;
        S_lin[54]           = x36;
        S_lin[55]           = x60;
        S_lin[56]           = x76;
        S_lin[57]           = x86;
        S_lin[58]           = x101;
        S_lin[59]           = x111;
        S_lin[60]           = lmbda * x116 + mu * x116 + mu - x11 * x116;
        S_lin[61]           = x118;
        S_lin[62]           = x119;
        S_lin[63]           = x41;
        S_lin[64]           = x61;
        S_lin[65]           = x78;
        S_lin[66]           = x91;
        S_lin[67]           = x103;
        S_lin[68]           = x113;
        S_lin[69]           = x118;
        S_lin[70]           = lmbda * x120 + mu * x120 + mu - x11 * x120;
        S_lin[71]           = x121;
        S_lin[72]           = x46;
        S_lin[73]           = x65;
        S_lin[74]           = x80;
        S_lin[75]           = x95;
        S_lin[76]           = x107;
        S_lin[77]           = x115;
        S_lin[78]           = x119;
        S_lin[79]           = x121;
        S_lin[80]           = lmbda * x122 + mu * x122 + mu - x11 * x122;
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
    H[0]                = x21 * (x12 + x16 + x20);
    H[1]                = -x12 * x21;
    H[2]                = -x16 * x21;
    H[3]                = -x20 * x21;
    H[4]                = x21 * (x34 + x38 + x42);
    H[5]                = -x21 * x34;
    H[6]                = -x21 * x38;
    H[7]                = -x21 * x42;
    H[8]                = x21 * (x55 + x59 + x63);
    H[9]                = -x21 * x55;
    H[10]               = -x21 * x59;
    H[11]               = -x21 * x63;
    H[12]               = -x21 * (x13 + x17 + x3);
    H[13]               = x21 * x3;
    H[14]               = x13 * x21;
    H[15]               = x17 * x21;
    H[16]               = -x21 * (x25 + x35 + x39);
    H[17]               = x21 * x25;
    H[18]               = x21 * x35;
    H[19]               = x21 * x39;
    H[20]               = -x21 * (x46 + x56 + x60);
    H[21]               = x21 * x46;
    H[22]               = x21 * x56;
    H[23]               = x21 * x60;
    H[24]               = -x21 * (x14 + x18 + x7);
    H[25]               = x21 * x7;
    H[26]               = x14 * x21;
    H[27]               = x18 * x21;
    H[28]               = -x21 * (x29 + x36 + x40);
    H[29]               = x21 * x29;
    H[30]               = x21 * x36;
    H[31]               = x21 * x40;
    H[32]               = -x21 * (x50 + x57 + x61);
    H[33]               = x21 * x50;
    H[34]               = x21 * x57;
    H[35]               = x21 * x61;
    H[36]               = -x21 * (x11 + x15 + x19);
    H[37]               = x11 * x21;
    H[38]               = x15 * x21;
    H[39]               = x19 * x21;
    H[40]               = -x21 * (x33 + x37 + x41);
    H[41]               = x21 * x33;
    H[42]               = x21 * x37;
    H[43]               = x21 * x41;
    H[44]               = -x21 * (x54 + x58 + x62);
    H[45]               = x21 * x54;
    H[46]               = x21 * x58;
    H[47]               = x21 * x62;
    H[48]               = x21 * (x76 + x80 + x84);
    H[49]               = -x21 * x76;
    H[50]               = -x21 * x80;
    H[51]               = -x21 * x84;
    H[52]               = x21 * (x101 + x105 + x97);
    H[53]               = -x21 * x97;
    H[54]               = -x101 * x21;
    H[55]               = -x105 * x21;
    H[56]               = x21 * (x118 + x122 + x126);
    H[57]               = -x118 * x21;
    H[58]               = -x122 * x21;
    H[59]               = -x126 * x21;
    H[60]               = -x21 * (x67 + x77 + x81);
    H[61]               = x21 * x67;
    H[62]               = x21 * x77;
    H[63]               = x21 * x81;
    H[64]               = -x21 * (x102 + x88 + x98);
    H[65]               = x21 * x88;
    H[66]               = x21 * x98;
    H[67]               = x102 * x21;
    H[68]               = -x21 * (x109 + x119 + x123);
    H[69]               = x109 * x21;
    H[70]               = x119 * x21;
    H[71]               = x123 * x21;
    H[72]               = -x21 * (x71 + x78 + x82);
    H[73]               = x21 * x71;
    H[74]               = x21 * x78;
    H[75]               = x21 * x82;
    H[76]               = -x21 * (x103 + x92 + x99);
    H[77]               = x21 * x92;
    H[78]               = x21 * x99;
    H[79]               = x103 * x21;
    H[80]               = -x21 * (x113 + x120 + x124);
    H[81]               = x113 * x21;
    H[82]               = x120 * x21;
    H[83]               = x124 * x21;
    H[84]               = -x21 * (x75 + x79 + x83);
    H[85]               = x21 * x75;
    H[86]               = x21 * x79;
    H[87]               = x21 * x83;
    H[88]               = -x21 * (x100 + x104 + x96);
    H[89]               = x21 * x96;
    H[90]               = x100 * x21;
    H[91]               = x104 * x21;
    H[92]               = -x21 * (x117 + x121 + x125);
    H[93]               = x117 * x21;
    H[94]               = x121 * x21;
    H[95]               = x125 * x21;
    H[96]               = x21 * (x139 + x143 + x147);
    H[97]               = -x139 * x21;
    H[98]               = -x143 * x21;
    H[99]               = -x147 * x21;
    H[100]              = x21 * (x160 + x164 + x168);
    H[101]              = -x160 * x21;
    H[102]              = -x164 * x21;
    H[103]              = -x168 * x21;
    H[104]              = x21 * (x181 + x185 + x189);
    H[105]              = -x181 * x21;
    H[106]              = -x185 * x21;
    H[107]              = -x189 * x21;
    H[108]              = -x21 * (x130 + x140 + x144);
    H[109]              = x130 * x21;
    H[110]              = x140 * x21;
    H[111]              = x144 * x21;
    H[112]              = -x21 * (x151 + x161 + x165);
    H[113]              = x151 * x21;
    H[114]              = x161 * x21;
    H[115]              = x165 * x21;
    H[116]              = -x21 * (x172 + x182 + x186);
    H[117]              = x172 * x21;
    H[118]              = x182 * x21;
    H[119]              = x186 * x21;
    H[120]              = -x21 * (x134 + x141 + x145);
    H[121]              = x134 * x21;
    H[122]              = x141 * x21;
    H[123]              = x145 * x21;
    H[124]              = -x21 * (x155 + x162 + x166);
    H[125]              = x155 * x21;
    H[126]              = x162 * x21;
    H[127]              = x166 * x21;
    H[128]              = -x21 * (x176 + x183 + x187);
    H[129]              = x176 * x21;
    H[130]              = x183 * x21;
    H[131]              = x187 * x21;
    H[132]              = -x21 * (x138 + x142 + x146);
    H[133]              = x138 * x21;
    H[134]              = x142 * x21;
    H[135]              = x146 * x21;
    H[136]              = -x21 * (x159 + x163 + x167);
    H[137]              = x159 * x21;
    H[138]              = x163 * x21;
    H[139]              = x167 * x21;
    H[140]              = -x21 * (x180 + x184 + x188);
    H[141]              = x180 * x21;
    H[142]              = x184 * x21;
    H[143]              = x188 * x21;
}

#endif  // TET4_NEOHOOKEAN_OGDEN_LOCAL_H