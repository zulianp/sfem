#ifndef SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H
#define SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H

#include "sfem_macros.hpp"

static SFEM_INLINE void tet4_F(const scalar_t *const SFEM_RESTRICT adjugate,
                               const scalar_t                      jacobian_determinant,
                               const scalar_t *const SFEM_RESTRICT dispx,
                               const scalar_t *const SFEM_RESTRICT dispy,
                               const scalar_t *const SFEM_RESTRICT dispz,
                               scalar_t *const SFEM_RESTRICT       F) {
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

static SFEM_INLINE void tet4_Wimpn_compressed(scalar_t *const SFEM_RESTRICT Wimpn_compressed) {
    // mundane ops: 0 divs: 0 sqrts: 0
    // total ops: 0
    Wimpn_compressed[0] = 1;
    Wimpn_compressed[1] = -1;
    Wimpn_compressed[2] = 0;
}

static SFEM_INLINE void tet4_SdotHdotG(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                       const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                       const scalar_t *const SFEM_RESTRICT incx,
                                       const scalar_t *const SFEM_RESTRICT incy,
                                       const scalar_t *const SFEM_RESTRICT incz,
                                       scalar_t *const SFEM_RESTRICT       outx,
                                       scalar_t *const SFEM_RESTRICT       outy,
                                       scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 573 divs: 0 sqrts: 0
    // total ops: 573
    const scalar_t x0  = Wimpn_compressed[2] * incx[2];
    const scalar_t x1  = Wimpn_compressed[0] * incx[0];
    const scalar_t x2  = Wimpn_compressed[2] * incx[3];
    const scalar_t x3  = x1 + x2;
    const scalar_t x4  = Wimpn_compressed[1] * incx[1] + x0 + x3;
    const scalar_t x5  = Wimpn_compressed[2] * incy[3];
    const scalar_t x6  = Wimpn_compressed[0] * incy[0];
    const scalar_t x7  = Wimpn_compressed[2] * incy[1];
    const scalar_t x8  = x6 + x7;
    const scalar_t x9  = Wimpn_compressed[1] * incy[2] + x5 + x8;
    const scalar_t x10 = Wimpn_compressed[2] * incy[2];
    const scalar_t x11 = Wimpn_compressed[1] * incy[3] + x10 + x8;
    const scalar_t x12 = Wimpn_compressed[2] * incz[2];
    const scalar_t x13 = Wimpn_compressed[0] * incz[0];
    const scalar_t x14 = Wimpn_compressed[2] * incz[3];
    const scalar_t x15 = x13 + x14;
    const scalar_t x16 = Wimpn_compressed[1] * incz[1] + x12 + x15;
    const scalar_t x17 = Wimpn_compressed[2] * incz[1];
    const scalar_t x18 = Wimpn_compressed[1] * incz[2] + x15 + x17;
    const scalar_t x19 = Wimpn_compressed[2] * incx[1];
    const scalar_t x20 = Wimpn_compressed[1] * incx[2] + x19 + x3;
    const scalar_t x21 = x12 + x17;
    const scalar_t x22 = Wimpn_compressed[1] * incz[3] + x13 + x21;
    const scalar_t x23 = x0 + x19;
    const scalar_t x24 = Wimpn_compressed[1] * incx[3] + x1 + x23;
    const scalar_t x25 = x10 + x5;
    const scalar_t x26 = Wimpn_compressed[1] * incy[1] + x25 + x6;
    const scalar_t x27 = Wimpn_compressed[1] * incx[0];
    const scalar_t x28 = x2 + x27;
    const scalar_t x29 = Wimpn_compressed[0] * incx[1] + x0 + x28;
    const scalar_t x30 = Wimpn_compressed[1] * incy[0];
    const scalar_t x31 = x30 + x7;
    const scalar_t x32 = Wimpn_compressed[0] * incy[3] + x10 + x31;
    const scalar_t x33 = Wimpn_compressed[1] * incz[0];
    const scalar_t x34 = x14 + x33;
    const scalar_t x35 = Wimpn_compressed[0] * incz[1] + x12 + x34;
    const scalar_t x36 = Wimpn_compressed[0] * incz[2] + x17 + x34;
    const scalar_t x37 = Wimpn_compressed[0] * incx[2] + x19 + x28;
    const scalar_t x38 = Wimpn_compressed[0] * incz[3] + x21 + x33;
    const scalar_t x39 = Wimpn_compressed[0] * incx[3] + x23 + x27;
    const scalar_t x40 = Wimpn_compressed[0] * incy[1] + x25 + x30;
    const scalar_t x41 = Wimpn_compressed[0] * incy[2] + x31 + x5;
    const scalar_t x42 = Wimpn_compressed[2] * (incx[0] + incx[1] + incx[2] + incx[3]);
    const scalar_t x43 = S_ikmn_canonical[4] * x42;
    const scalar_t x44 = Wimpn_compressed[2] * (incy[0] + incy[1] + incy[2] + incy[3]);
    const scalar_t x45 = Wimpn_compressed[2] * (incz[0] + incz[1] + incz[2] + incz[3]);
    const scalar_t x46 = S_ikmn_canonical[1] * x42;
    const scalar_t x47 = S_ikmn_canonical[11] * x44 + S_ikmn_canonical[14] * x44 + S_ikmn_canonical[17] * x45 +
                         S_ikmn_canonical[20] * x45 + S_ikmn_canonical[23] * x45 + S_ikmn_canonical[5] * x42 +
                         S_ikmn_canonical[8] * x44 + x46;
    const scalar_t x48 = S_ikmn_canonical[2] * x42;
    const scalar_t x49 = S_ikmn_canonical[10] * x44 + S_ikmn_canonical[13] * x44 + S_ikmn_canonical[16] * x45 +
                         S_ikmn_canonical[19] * x45 + S_ikmn_canonical[22] * x45 + S_ikmn_canonical[3] * x42 +
                         S_ikmn_canonical[7] * x44 + x48;
    const scalar_t x50 = S_ikmn_canonical[0] * x42 + S_ikmn_canonical[12] * x44 + S_ikmn_canonical[15] * x45 +
                         S_ikmn_canonical[18] * x45 + S_ikmn_canonical[21] * x45 + S_ikmn_canonical[6] * x44 +
                         S_ikmn_canonical[9] * x44 + x43;
    const scalar_t x51 = S_ikmn_canonical[28] * x44;
    const scalar_t x52 = S_ikmn_canonical[25] * x44;
    const scalar_t x53 = S_ikmn_canonical[12] * x42 + S_ikmn_canonical[13] * x42 + S_ikmn_canonical[14] * x42 +
                         S_ikmn_canonical[29] * x44 + S_ikmn_canonical[32] * x45 + S_ikmn_canonical[35] * x45 +
                         S_ikmn_canonical[38] * x45 + x52;
    const scalar_t x54 = S_ikmn_canonical[26] * x44;
    const scalar_t x55 = S_ikmn_canonical[10] * x42 + S_ikmn_canonical[11] * x42 + S_ikmn_canonical[27] * x44 +
                         S_ikmn_canonical[31] * x45 + S_ikmn_canonical[34] * x45 + S_ikmn_canonical[37] * x45 +
                         S_ikmn_canonical[9] * x42 + x54;
    const scalar_t x56 = S_ikmn_canonical[24] * x44 + S_ikmn_canonical[30] * x45 + S_ikmn_canonical[33] * x45 +
                         S_ikmn_canonical[36] * x45 + S_ikmn_canonical[6] * x42 + S_ikmn_canonical[7] * x42 +
                         S_ikmn_canonical[8] * x42 + x51;
    const scalar_t x57 = S_ikmn_canonical[43] * x45;
    const scalar_t x58 = S_ikmn_canonical[40] * x45;
    const scalar_t x59 = S_ikmn_canonical[21] * x42 + S_ikmn_canonical[22] * x42 + S_ikmn_canonical[23] * x42 +
                         S_ikmn_canonical[36] * x44 + S_ikmn_canonical[37] * x44 + S_ikmn_canonical[38] * x44 +
                         S_ikmn_canonical[44] * x45 + x58;
    const scalar_t x60 = S_ikmn_canonical[41] * x45;
    const scalar_t x61 = S_ikmn_canonical[18] * x42 + S_ikmn_canonical[19] * x42 + S_ikmn_canonical[20] * x42 +
                         S_ikmn_canonical[33] * x44 + S_ikmn_canonical[34] * x44 + S_ikmn_canonical[35] * x44 +
                         S_ikmn_canonical[42] * x45 + x60;
    const scalar_t x62 = S_ikmn_canonical[15] * x42 + S_ikmn_canonical[16] * x42 + S_ikmn_canonical[17] * x42 +
                         S_ikmn_canonical[30] * x44 + S_ikmn_canonical[31] * x44 + S_ikmn_canonical[32] * x44 +
                         S_ikmn_canonical[39] * x45 + x57;
    outx[0] = S_ikmn_canonical[0] * x4 + S_ikmn_canonical[10] * x9 + S_ikmn_canonical[11] * x9 + S_ikmn_canonical[12] * x11 +
              S_ikmn_canonical[13] * x11 + S_ikmn_canonical[14] * x11 + S_ikmn_canonical[15] * x16 + S_ikmn_canonical[16] * x16 +
              S_ikmn_canonical[17] * x16 + S_ikmn_canonical[18] * x18 + S_ikmn_canonical[19] * x18 + S_ikmn_canonical[1] * x20 +
              S_ikmn_canonical[1] * x4 + S_ikmn_canonical[20] * x18 + S_ikmn_canonical[21] * x22 + S_ikmn_canonical[22] * x22 +
              S_ikmn_canonical[23] * x22 + S_ikmn_canonical[2] * x24 + S_ikmn_canonical[2] * x4 + S_ikmn_canonical[3] * x20 +
              S_ikmn_canonical[4] * x20 + S_ikmn_canonical[4] * x24 + S_ikmn_canonical[5] * x24 + S_ikmn_canonical[6] * x26 +
              S_ikmn_canonical[7] * x26 + S_ikmn_canonical[8] * x26 + S_ikmn_canonical[9] * x9;
    outx[1] = S_ikmn_canonical[0] * x29 + S_ikmn_canonical[12] * x32 + S_ikmn_canonical[15] * x35 + S_ikmn_canonical[18] * x36 +
              S_ikmn_canonical[1] * x37 + S_ikmn_canonical[21] * x38 + S_ikmn_canonical[2] * x39 + S_ikmn_canonical[6] * x40 +
              S_ikmn_canonical[9] * x41 + 2 * x43 + x47 + x49;
    outx[2] = S_ikmn_canonical[10] * x41 + S_ikmn_canonical[13] * x32 + S_ikmn_canonical[16] * x35 + S_ikmn_canonical[19] * x36 +
              S_ikmn_canonical[1] * x29 + S_ikmn_canonical[22] * x38 + S_ikmn_canonical[3] * x37 + S_ikmn_canonical[4] * x39 +
              S_ikmn_canonical[7] * x40 + x47 + 2 * x48 + x50;
    outx[3] = S_ikmn_canonical[11] * x41 + S_ikmn_canonical[14] * x32 + S_ikmn_canonical[17] * x35 + S_ikmn_canonical[20] * x36 +
              S_ikmn_canonical[23] * x38 + S_ikmn_canonical[2] * x29 + S_ikmn_canonical[4] * x37 + S_ikmn_canonical[5] * x39 +
              S_ikmn_canonical[8] * x40 + 2 * x46 + x49 + x50;
    outy[0] = S_ikmn_canonical[10] * x20 + S_ikmn_canonical[11] * x24 + S_ikmn_canonical[12] * x4 + S_ikmn_canonical[13] * x20 +
              S_ikmn_canonical[14] * x24 + S_ikmn_canonical[24] * x26 + S_ikmn_canonical[25] * x26 + S_ikmn_canonical[25] * x9 +
              S_ikmn_canonical[26] * x11 + S_ikmn_canonical[26] * x26 + S_ikmn_canonical[27] * x9 + S_ikmn_canonical[28] * x11 +
              S_ikmn_canonical[28] * x9 + S_ikmn_canonical[29] * x11 + S_ikmn_canonical[30] * x16 + S_ikmn_canonical[31] * x16 +
              S_ikmn_canonical[32] * x16 + S_ikmn_canonical[33] * x18 + S_ikmn_canonical[34] * x18 + S_ikmn_canonical[35] * x18 +
              S_ikmn_canonical[36] * x22 + S_ikmn_canonical[37] * x22 + S_ikmn_canonical[38] * x22 + S_ikmn_canonical[6] * x4 +
              S_ikmn_canonical[7] * x20 + S_ikmn_canonical[8] * x24 + S_ikmn_canonical[9] * x4;
    outy[1] = S_ikmn_canonical[24] * x40 + S_ikmn_canonical[25] * x41 + S_ikmn_canonical[26] * x32 + S_ikmn_canonical[30] * x35 +
              S_ikmn_canonical[33] * x36 + S_ikmn_canonical[36] * x38 + S_ikmn_canonical[6] * x29 + S_ikmn_canonical[7] * x37 +
              S_ikmn_canonical[8] * x39 + 2 * x51 + x53 + x55;
    outy[2] = S_ikmn_canonical[10] * x37 + S_ikmn_canonical[11] * x39 + S_ikmn_canonical[25] * x40 + S_ikmn_canonical[27] * x41 +
              S_ikmn_canonical[28] * x32 + S_ikmn_canonical[31] * x35 + S_ikmn_canonical[34] * x36 + S_ikmn_canonical[37] * x38 +
              S_ikmn_canonical[9] * x29 + x53 + 2 * x54 + x56;
    outy[3] = S_ikmn_canonical[12] * x29 + S_ikmn_canonical[13] * x37 + S_ikmn_canonical[14] * x39 + S_ikmn_canonical[26] * x40 +
              S_ikmn_canonical[28] * x41 + S_ikmn_canonical[29] * x32 + S_ikmn_canonical[32] * x35 + S_ikmn_canonical[35] * x36 +
              S_ikmn_canonical[38] * x38 + 2 * x52 + x55 + x56;
    outz[0] = S_ikmn_canonical[15] * x4 + S_ikmn_canonical[16] * x20 + S_ikmn_canonical[17] * x24 + S_ikmn_canonical[18] * x4 +
              S_ikmn_canonical[19] * x20 + S_ikmn_canonical[20] * x24 + S_ikmn_canonical[21] * x4 + S_ikmn_canonical[22] * x20 +
              S_ikmn_canonical[23] * x24 + S_ikmn_canonical[30] * x26 + S_ikmn_canonical[31] * x9 + S_ikmn_canonical[32] * x11 +
              S_ikmn_canonical[33] * x26 + S_ikmn_canonical[34] * x9 + S_ikmn_canonical[35] * x11 + S_ikmn_canonical[36] * x26 +
              S_ikmn_canonical[37] * x9 + S_ikmn_canonical[38] * x11 + S_ikmn_canonical[39] * x16 + S_ikmn_canonical[40] * x16 +
              S_ikmn_canonical[40] * x18 + S_ikmn_canonical[41] * x16 + S_ikmn_canonical[41] * x22 + S_ikmn_canonical[42] * x18 +
              S_ikmn_canonical[43] * x18 + S_ikmn_canonical[43] * x22 + S_ikmn_canonical[44] * x22;
    outz[1] = S_ikmn_canonical[15] * x29 + S_ikmn_canonical[16] * x37 + S_ikmn_canonical[17] * x39 + S_ikmn_canonical[30] * x40 +
              S_ikmn_canonical[31] * x41 + S_ikmn_canonical[32] * x32 + S_ikmn_canonical[39] * x35 + S_ikmn_canonical[40] * x36 +
              S_ikmn_canonical[41] * x38 + 2 * x57 + x59 + x61;
    outz[2] = S_ikmn_canonical[18] * x29 + S_ikmn_canonical[19] * x37 + S_ikmn_canonical[20] * x39 + S_ikmn_canonical[33] * x40 +
              S_ikmn_canonical[34] * x41 + S_ikmn_canonical[35] * x32 + S_ikmn_canonical[40] * x35 + S_ikmn_canonical[42] * x36 +
              S_ikmn_canonical[43] * x38 + x59 + 2 * x60 + x62;
    outz[3] = S_ikmn_canonical[21] * x29 + S_ikmn_canonical[22] * x37 + S_ikmn_canonical[23] * x39 + S_ikmn_canonical[36] * x40 +
              S_ikmn_canonical[37] * x41 + S_ikmn_canonical[38] * x32 + S_ikmn_canonical[41] * x35 + S_ikmn_canonical[43] * x36 +
              S_ikmn_canonical[44] * x38 + 2 * x58 + x61 + x62;
}

static SFEM_INLINE void tet4_ref_inc_grad(const scalar_t *const SFEM_RESTRICT incx,
                                          const scalar_t *const SFEM_RESTRICT incy,
                                          const scalar_t *const SFEM_RESTRICT incz,
                                          scalar_t *const SFEM_RESTRICT       inc_grad) {
    // mundane ops: 18 divs: 0 sqrts: 0
    // total ops: 18
    inc_grad[0] = -incx[0] + incx[1];
    inc_grad[1] = -incx[0] + incx[2];
    inc_grad[2] = -incx[0] + incx[3];
    inc_grad[3] = -incy[0] + incy[1];
    inc_grad[4] = -incy[0] + incy[2];
    inc_grad[5] = -incy[0] + incy[3];
    inc_grad[6] = -incz[0] + incz[1];
    inc_grad[7] = -incz[0] + incz[2];
    inc_grad[8] = -incz[0] + incz[3];
}

static SFEM_INLINE void tet4_Zpkmn(const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                   const scalar_t *const SFEM_RESTRICT incx,
                                   const scalar_t *const SFEM_RESTRICT incy,
                                   const scalar_t *const SFEM_RESTRICT incz,
                                   scalar_t *const SFEM_RESTRICT       Zpkmn) {
    // mundane ops: 90 divs: 0 sqrts: 0
    // total ops: 90
    const scalar_t x0  = Wimpn_compressed[2] * incx[2];
    const scalar_t x1  = Wimpn_compressed[0] * incx[0];
    const scalar_t x2  = Wimpn_compressed[2] * incx[3];
    const scalar_t x3  = x1 + x2;
    const scalar_t x4  = Wimpn_compressed[1] * incx[1] + x0 + x3;
    const scalar_t x5  = Wimpn_compressed[2] * incx[1];
    const scalar_t x6  = Wimpn_compressed[1] * incx[2] + x3 + x5;
    const scalar_t x7  = x0 + x5;
    const scalar_t x8  = Wimpn_compressed[1] * incx[3] + x1 + x7;
    const scalar_t x9  = Wimpn_compressed[2] * incy[2];
    const scalar_t x10 = Wimpn_compressed[0] * incy[0];
    const scalar_t x11 = Wimpn_compressed[2] * incy[3];
    const scalar_t x12 = x10 + x11;
    const scalar_t x13 = Wimpn_compressed[1] * incy[1] + x12 + x9;
    const scalar_t x14 = Wimpn_compressed[2] * incy[1];
    const scalar_t x15 = Wimpn_compressed[1] * incy[2] + x12 + x14;
    const scalar_t x16 = x14 + x9;
    const scalar_t x17 = Wimpn_compressed[1] * incy[3] + x10 + x16;
    const scalar_t x18 = Wimpn_compressed[2] * incz[2];
    const scalar_t x19 = Wimpn_compressed[0] * incz[0];
    const scalar_t x20 = Wimpn_compressed[2] * incz[3];
    const scalar_t x21 = x19 + x20;
    const scalar_t x22 = Wimpn_compressed[1] * incz[1] + x18 + x21;
    const scalar_t x23 = Wimpn_compressed[2] * incz[1];
    const scalar_t x24 = Wimpn_compressed[1] * incz[2] + x21 + x23;
    const scalar_t x25 = x18 + x23;
    const scalar_t x26 = Wimpn_compressed[1] * incz[3] + x19 + x25;
    const scalar_t x27 = Wimpn_compressed[1] * incx[0];
    const scalar_t x28 = x2 + x27;
    const scalar_t x29 = Wimpn_compressed[0] * incx[1] + x0 + x28;
    const scalar_t x30 = Wimpn_compressed[2] * (incx[0] + incx[1] + incx[2] + incx[3]);
    const scalar_t x31 = Wimpn_compressed[0] * incx[2] + x28 + x5;
    const scalar_t x32 = Wimpn_compressed[0] * incx[3] + x27 + x7;
    const scalar_t x33 = Wimpn_compressed[1] * incy[0];
    const scalar_t x34 = x11 + x33;
    const scalar_t x35 = Wimpn_compressed[0] * incy[1] + x34 + x9;
    const scalar_t x36 = Wimpn_compressed[2] * (incy[0] + incy[1] + incy[2] + incy[3]);
    const scalar_t x37 = Wimpn_compressed[0] * incy[2] + x14 + x34;
    const scalar_t x38 = Wimpn_compressed[0] * incy[3] + x16 + x33;
    const scalar_t x39 = Wimpn_compressed[1] * incz[0];
    const scalar_t x40 = x20 + x39;
    const scalar_t x41 = Wimpn_compressed[0] * incz[1] + x18 + x40;
    const scalar_t x42 = Wimpn_compressed[2] * (incz[0] + incz[1] + incz[2] + incz[3]);
    const scalar_t x43 = Wimpn_compressed[0] * incz[2] + x23 + x40;
    const scalar_t x44 = Wimpn_compressed[0] * incz[3] + x25 + x39;
    Zpkmn[0]           = x4;
    Zpkmn[1]           = x4;
    Zpkmn[2]           = x4;
    Zpkmn[3]           = x6;
    Zpkmn[4]           = x6;
    Zpkmn[5]           = x6;
    Zpkmn[6]           = x8;
    Zpkmn[7]           = x8;
    Zpkmn[8]           = x8;
    Zpkmn[9]           = x13;
    Zpkmn[10]          = x13;
    Zpkmn[11]          = x13;
    Zpkmn[12]          = x15;
    Zpkmn[13]          = x15;
    Zpkmn[14]          = x15;
    Zpkmn[15]          = x17;
    Zpkmn[16]          = x17;
    Zpkmn[17]          = x17;
    Zpkmn[18]          = x22;
    Zpkmn[19]          = x22;
    Zpkmn[20]          = x22;
    Zpkmn[21]          = x24;
    Zpkmn[22]          = x24;
    Zpkmn[23]          = x24;
    Zpkmn[24]          = x26;
    Zpkmn[25]          = x26;
    Zpkmn[26]          = x26;
    Zpkmn[27]          = x29;
    Zpkmn[28]          = x30;
    Zpkmn[29]          = x30;
    Zpkmn[30]          = x31;
    Zpkmn[31]          = x30;
    Zpkmn[32]          = x30;
    Zpkmn[33]          = x32;
    Zpkmn[34]          = x30;
    Zpkmn[35]          = x30;
    Zpkmn[36]          = x35;
    Zpkmn[37]          = x36;
    Zpkmn[38]          = x36;
    Zpkmn[39]          = x37;
    Zpkmn[40]          = x36;
    Zpkmn[41]          = x36;
    Zpkmn[42]          = x38;
    Zpkmn[43]          = x36;
    Zpkmn[44]          = x36;
    Zpkmn[45]          = x41;
    Zpkmn[46]          = x42;
    Zpkmn[47]          = x42;
    Zpkmn[48]          = x43;
    Zpkmn[49]          = x42;
    Zpkmn[50]          = x42;
    Zpkmn[51]          = x44;
    Zpkmn[52]          = x42;
    Zpkmn[53]          = x42;
    Zpkmn[54]          = x30;
    Zpkmn[55]          = x29;
    Zpkmn[56]          = x30;
    Zpkmn[57]          = x30;
    Zpkmn[58]          = x31;
    Zpkmn[59]          = x30;
    Zpkmn[60]          = x30;
    Zpkmn[61]          = x32;
    Zpkmn[62]          = x30;
    Zpkmn[63]          = x36;
    Zpkmn[64]          = x35;
    Zpkmn[65]          = x36;
    Zpkmn[66]          = x36;
    Zpkmn[67]          = x37;
    Zpkmn[68]          = x36;
    Zpkmn[69]          = x36;
    Zpkmn[70]          = x38;
    Zpkmn[71]          = x36;
    Zpkmn[72]          = x42;
    Zpkmn[73]          = x41;
    Zpkmn[74]          = x42;
    Zpkmn[75]          = x42;
    Zpkmn[76]          = x43;
    Zpkmn[77]          = x42;
    Zpkmn[78]          = x42;
    Zpkmn[79]          = x44;
    Zpkmn[80]          = x42;
    Zpkmn[81]          = x30;
    Zpkmn[82]          = x30;
    Zpkmn[83]          = x29;
    Zpkmn[84]          = x30;
    Zpkmn[85]          = x30;
    Zpkmn[86]          = x31;
    Zpkmn[87]          = x30;
    Zpkmn[88]          = x30;
    Zpkmn[89]          = x32;
    Zpkmn[90]          = x36;
    Zpkmn[91]          = x36;
    Zpkmn[92]          = x35;
    Zpkmn[93]          = x36;
    Zpkmn[94]          = x36;
    Zpkmn[95]          = x37;
    Zpkmn[96]          = x36;
    Zpkmn[97]          = x36;
    Zpkmn[98]          = x38;
    Zpkmn[99]          = x42;
    Zpkmn[100]         = x42;
    Zpkmn[101]         = x41;
    Zpkmn[102]         = x42;
    Zpkmn[103]         = x42;
    Zpkmn[104]         = x43;
    Zpkmn[105]         = x42;
    Zpkmn[106]         = x42;
    Zpkmn[107]         = x44;
}

static SFEM_INLINE void tet4_SdotZ(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                   const scalar_t *const SFEM_RESTRICT Zpkmn,
                                   scalar_t *const SFEM_RESTRICT       outx,
                                   scalar_t *const SFEM_RESTRICT       outy,
                                   scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 636 divs: 0 sqrts: 0
    // total ops: 636
    outx[0] = S_ikmn_canonical[0] * Zpkmn[0] + S_ikmn_canonical[10] * Zpkmn[13] + S_ikmn_canonical[11] * Zpkmn[14] +
              S_ikmn_canonical[12] * Zpkmn[15] + S_ikmn_canonical[13] * Zpkmn[16] + S_ikmn_canonical[14] * Zpkmn[17] +
              S_ikmn_canonical[15] * Zpkmn[18] + S_ikmn_canonical[16] * Zpkmn[19] + S_ikmn_canonical[17] * Zpkmn[20] +
              S_ikmn_canonical[18] * Zpkmn[21] + S_ikmn_canonical[19] * Zpkmn[22] + S_ikmn_canonical[1] * Zpkmn[1] +
              S_ikmn_canonical[1] * Zpkmn[3] + S_ikmn_canonical[20] * Zpkmn[23] + S_ikmn_canonical[21] * Zpkmn[24] +
              S_ikmn_canonical[22] * Zpkmn[25] + S_ikmn_canonical[23] * Zpkmn[26] + S_ikmn_canonical[2] * Zpkmn[2] +
              S_ikmn_canonical[2] * Zpkmn[6] + S_ikmn_canonical[3] * Zpkmn[4] + S_ikmn_canonical[4] * Zpkmn[5] +
              S_ikmn_canonical[4] * Zpkmn[7] + S_ikmn_canonical[5] * Zpkmn[8] + S_ikmn_canonical[6] * Zpkmn[9] +
              S_ikmn_canonical[7] * Zpkmn[10] + S_ikmn_canonical[8] * Zpkmn[11] + S_ikmn_canonical[9] * Zpkmn[12];
    outx[1] = S_ikmn_canonical[0] * Zpkmn[27] + S_ikmn_canonical[10] * Zpkmn[40] + S_ikmn_canonical[11] * Zpkmn[41] +
              S_ikmn_canonical[12] * Zpkmn[42] + S_ikmn_canonical[13] * Zpkmn[43] + S_ikmn_canonical[14] * Zpkmn[44] +
              S_ikmn_canonical[15] * Zpkmn[45] + S_ikmn_canonical[16] * Zpkmn[46] + S_ikmn_canonical[17] * Zpkmn[47] +
              S_ikmn_canonical[18] * Zpkmn[48] + S_ikmn_canonical[19] * Zpkmn[49] + S_ikmn_canonical[1] * Zpkmn[28] +
              S_ikmn_canonical[1] * Zpkmn[30] + S_ikmn_canonical[20] * Zpkmn[50] + S_ikmn_canonical[21] * Zpkmn[51] +
              S_ikmn_canonical[22] * Zpkmn[52] + S_ikmn_canonical[23] * Zpkmn[53] + S_ikmn_canonical[2] * Zpkmn[29] +
              S_ikmn_canonical[2] * Zpkmn[33] + S_ikmn_canonical[3] * Zpkmn[31] + S_ikmn_canonical[4] * Zpkmn[32] +
              S_ikmn_canonical[4] * Zpkmn[34] + S_ikmn_canonical[5] * Zpkmn[35] + S_ikmn_canonical[6] * Zpkmn[36] +
              S_ikmn_canonical[7] * Zpkmn[37] + S_ikmn_canonical[8] * Zpkmn[38] + S_ikmn_canonical[9] * Zpkmn[39];
    outx[2] = S_ikmn_canonical[0] * Zpkmn[54] + S_ikmn_canonical[10] * Zpkmn[67] + S_ikmn_canonical[11] * Zpkmn[68] +
              S_ikmn_canonical[12] * Zpkmn[69] + S_ikmn_canonical[13] * Zpkmn[70] + S_ikmn_canonical[14] * Zpkmn[71] +
              S_ikmn_canonical[15] * Zpkmn[72] + S_ikmn_canonical[16] * Zpkmn[73] + S_ikmn_canonical[17] * Zpkmn[74] +
              S_ikmn_canonical[18] * Zpkmn[75] + S_ikmn_canonical[19] * Zpkmn[76] + S_ikmn_canonical[1] * Zpkmn[55] +
              S_ikmn_canonical[1] * Zpkmn[57] + S_ikmn_canonical[20] * Zpkmn[77] + S_ikmn_canonical[21] * Zpkmn[78] +
              S_ikmn_canonical[22] * Zpkmn[79] + S_ikmn_canonical[23] * Zpkmn[80] + S_ikmn_canonical[2] * Zpkmn[56] +
              S_ikmn_canonical[2] * Zpkmn[60] + S_ikmn_canonical[3] * Zpkmn[58] + S_ikmn_canonical[4] * Zpkmn[59] +
              S_ikmn_canonical[4] * Zpkmn[61] + S_ikmn_canonical[5] * Zpkmn[62] + S_ikmn_canonical[6] * Zpkmn[63] +
              S_ikmn_canonical[7] * Zpkmn[64] + S_ikmn_canonical[8] * Zpkmn[65] + S_ikmn_canonical[9] * Zpkmn[66];
    outx[3] = S_ikmn_canonical[0] * Zpkmn[81] + S_ikmn_canonical[10] * Zpkmn[94] + S_ikmn_canonical[11] * Zpkmn[95] +
              S_ikmn_canonical[12] * Zpkmn[96] + S_ikmn_canonical[13] * Zpkmn[97] + S_ikmn_canonical[14] * Zpkmn[98] +
              S_ikmn_canonical[15] * Zpkmn[99] + S_ikmn_canonical[16] * Zpkmn[100] + S_ikmn_canonical[17] * Zpkmn[101] +
              S_ikmn_canonical[18] * Zpkmn[102] + S_ikmn_canonical[19] * Zpkmn[103] + S_ikmn_canonical[1] * Zpkmn[82] +
              S_ikmn_canonical[1] * Zpkmn[84] + S_ikmn_canonical[20] * Zpkmn[104] + S_ikmn_canonical[21] * Zpkmn[105] +
              S_ikmn_canonical[22] * Zpkmn[106] + S_ikmn_canonical[23] * Zpkmn[107] + S_ikmn_canonical[2] * Zpkmn[83] +
              S_ikmn_canonical[2] * Zpkmn[87] + S_ikmn_canonical[3] * Zpkmn[85] + S_ikmn_canonical[4] * Zpkmn[86] +
              S_ikmn_canonical[4] * Zpkmn[88] + S_ikmn_canonical[5] * Zpkmn[89] + S_ikmn_canonical[6] * Zpkmn[90] +
              S_ikmn_canonical[7] * Zpkmn[91] + S_ikmn_canonical[8] * Zpkmn[92] + S_ikmn_canonical[9] * Zpkmn[93];
    outy[0] = S_ikmn_canonical[10] * Zpkmn[4] + S_ikmn_canonical[11] * Zpkmn[7] + S_ikmn_canonical[12] * Zpkmn[2] +
              S_ikmn_canonical[13] * Zpkmn[5] + S_ikmn_canonical[14] * Zpkmn[8] + S_ikmn_canonical[24] * Zpkmn[9] +
              S_ikmn_canonical[25] * Zpkmn[10] + S_ikmn_canonical[25] * Zpkmn[12] + S_ikmn_canonical[26] * Zpkmn[11] +
              S_ikmn_canonical[26] * Zpkmn[15] + S_ikmn_canonical[27] * Zpkmn[13] + S_ikmn_canonical[28] * Zpkmn[14] +
              S_ikmn_canonical[28] * Zpkmn[16] + S_ikmn_canonical[29] * Zpkmn[17] + S_ikmn_canonical[30] * Zpkmn[18] +
              S_ikmn_canonical[31] * Zpkmn[19] + S_ikmn_canonical[32] * Zpkmn[20] + S_ikmn_canonical[33] * Zpkmn[21] +
              S_ikmn_canonical[34] * Zpkmn[22] + S_ikmn_canonical[35] * Zpkmn[23] + S_ikmn_canonical[36] * Zpkmn[24] +
              S_ikmn_canonical[37] * Zpkmn[25] + S_ikmn_canonical[38] * Zpkmn[26] + S_ikmn_canonical[6] * Zpkmn[0] +
              S_ikmn_canonical[7] * Zpkmn[3] + S_ikmn_canonical[8] * Zpkmn[6] + S_ikmn_canonical[9] * Zpkmn[1];
    outy[1] = S_ikmn_canonical[10] * Zpkmn[31] + S_ikmn_canonical[11] * Zpkmn[34] + S_ikmn_canonical[12] * Zpkmn[29] +
              S_ikmn_canonical[13] * Zpkmn[32] + S_ikmn_canonical[14] * Zpkmn[35] + S_ikmn_canonical[24] * Zpkmn[36] +
              S_ikmn_canonical[25] * Zpkmn[37] + S_ikmn_canonical[25] * Zpkmn[39] + S_ikmn_canonical[26] * Zpkmn[38] +
              S_ikmn_canonical[26] * Zpkmn[42] + S_ikmn_canonical[27] * Zpkmn[40] + S_ikmn_canonical[28] * Zpkmn[41] +
              S_ikmn_canonical[28] * Zpkmn[43] + S_ikmn_canonical[29] * Zpkmn[44] + S_ikmn_canonical[30] * Zpkmn[45] +
              S_ikmn_canonical[31] * Zpkmn[46] + S_ikmn_canonical[32] * Zpkmn[47] + S_ikmn_canonical[33] * Zpkmn[48] +
              S_ikmn_canonical[34] * Zpkmn[49] + S_ikmn_canonical[35] * Zpkmn[50] + S_ikmn_canonical[36] * Zpkmn[51] +
              S_ikmn_canonical[37] * Zpkmn[52] + S_ikmn_canonical[38] * Zpkmn[53] + S_ikmn_canonical[6] * Zpkmn[27] +
              S_ikmn_canonical[7] * Zpkmn[30] + S_ikmn_canonical[8] * Zpkmn[33] + S_ikmn_canonical[9] * Zpkmn[28];
    outy[2] = S_ikmn_canonical[10] * Zpkmn[58] + S_ikmn_canonical[11] * Zpkmn[61] + S_ikmn_canonical[12] * Zpkmn[56] +
              S_ikmn_canonical[13] * Zpkmn[59] + S_ikmn_canonical[14] * Zpkmn[62] + S_ikmn_canonical[24] * Zpkmn[63] +
              S_ikmn_canonical[25] * Zpkmn[64] + S_ikmn_canonical[25] * Zpkmn[66] + S_ikmn_canonical[26] * Zpkmn[65] +
              S_ikmn_canonical[26] * Zpkmn[69] + S_ikmn_canonical[27] * Zpkmn[67] + S_ikmn_canonical[28] * Zpkmn[68] +
              S_ikmn_canonical[28] * Zpkmn[70] + S_ikmn_canonical[29] * Zpkmn[71] + S_ikmn_canonical[30] * Zpkmn[72] +
              S_ikmn_canonical[31] * Zpkmn[73] + S_ikmn_canonical[32] * Zpkmn[74] + S_ikmn_canonical[33] * Zpkmn[75] +
              S_ikmn_canonical[34] * Zpkmn[76] + S_ikmn_canonical[35] * Zpkmn[77] + S_ikmn_canonical[36] * Zpkmn[78] +
              S_ikmn_canonical[37] * Zpkmn[79] + S_ikmn_canonical[38] * Zpkmn[80] + S_ikmn_canonical[6] * Zpkmn[54] +
              S_ikmn_canonical[7] * Zpkmn[57] + S_ikmn_canonical[8] * Zpkmn[60] + S_ikmn_canonical[9] * Zpkmn[55];
    outy[3] = S_ikmn_canonical[10] * Zpkmn[85] + S_ikmn_canonical[11] * Zpkmn[88] + S_ikmn_canonical[12] * Zpkmn[83] +
              S_ikmn_canonical[13] * Zpkmn[86] + S_ikmn_canonical[14] * Zpkmn[89] + S_ikmn_canonical[24] * Zpkmn[90] +
              S_ikmn_canonical[25] * Zpkmn[91] + S_ikmn_canonical[25] * Zpkmn[93] + S_ikmn_canonical[26] * Zpkmn[92] +
              S_ikmn_canonical[26] * Zpkmn[96] + S_ikmn_canonical[27] * Zpkmn[94] + S_ikmn_canonical[28] * Zpkmn[95] +
              S_ikmn_canonical[28] * Zpkmn[97] + S_ikmn_canonical[29] * Zpkmn[98] + S_ikmn_canonical[30] * Zpkmn[99] +
              S_ikmn_canonical[31] * Zpkmn[100] + S_ikmn_canonical[32] * Zpkmn[101] + S_ikmn_canonical[33] * Zpkmn[102] +
              S_ikmn_canonical[34] * Zpkmn[103] + S_ikmn_canonical[35] * Zpkmn[104] + S_ikmn_canonical[36] * Zpkmn[105] +
              S_ikmn_canonical[37] * Zpkmn[106] + S_ikmn_canonical[38] * Zpkmn[107] + S_ikmn_canonical[6] * Zpkmn[81] +
              S_ikmn_canonical[7] * Zpkmn[84] + S_ikmn_canonical[8] * Zpkmn[87] + S_ikmn_canonical[9] * Zpkmn[82];
    outz[0] = S_ikmn_canonical[15] * Zpkmn[0] + S_ikmn_canonical[16] * Zpkmn[3] + S_ikmn_canonical[17] * Zpkmn[6] +
              S_ikmn_canonical[18] * Zpkmn[1] + S_ikmn_canonical[19] * Zpkmn[4] + S_ikmn_canonical[20] * Zpkmn[7] +
              S_ikmn_canonical[21] * Zpkmn[2] + S_ikmn_canonical[22] * Zpkmn[5] + S_ikmn_canonical[23] * Zpkmn[8] +
              S_ikmn_canonical[30] * Zpkmn[9] + S_ikmn_canonical[31] * Zpkmn[12] + S_ikmn_canonical[32] * Zpkmn[15] +
              S_ikmn_canonical[33] * Zpkmn[10] + S_ikmn_canonical[34] * Zpkmn[13] + S_ikmn_canonical[35] * Zpkmn[16] +
              S_ikmn_canonical[36] * Zpkmn[11] + S_ikmn_canonical[37] * Zpkmn[14] + S_ikmn_canonical[38] * Zpkmn[17] +
              S_ikmn_canonical[39] * Zpkmn[18] + S_ikmn_canonical[40] * Zpkmn[19] + S_ikmn_canonical[40] * Zpkmn[21] +
              S_ikmn_canonical[41] * Zpkmn[20] + S_ikmn_canonical[41] * Zpkmn[24] + S_ikmn_canonical[42] * Zpkmn[22] +
              S_ikmn_canonical[43] * Zpkmn[23] + S_ikmn_canonical[43] * Zpkmn[25] + S_ikmn_canonical[44] * Zpkmn[26];
    outz[1] = S_ikmn_canonical[15] * Zpkmn[27] + S_ikmn_canonical[16] * Zpkmn[30] + S_ikmn_canonical[17] * Zpkmn[33] +
              S_ikmn_canonical[18] * Zpkmn[28] + S_ikmn_canonical[19] * Zpkmn[31] + S_ikmn_canonical[20] * Zpkmn[34] +
              S_ikmn_canonical[21] * Zpkmn[29] + S_ikmn_canonical[22] * Zpkmn[32] + S_ikmn_canonical[23] * Zpkmn[35] +
              S_ikmn_canonical[30] * Zpkmn[36] + S_ikmn_canonical[31] * Zpkmn[39] + S_ikmn_canonical[32] * Zpkmn[42] +
              S_ikmn_canonical[33] * Zpkmn[37] + S_ikmn_canonical[34] * Zpkmn[40] + S_ikmn_canonical[35] * Zpkmn[43] +
              S_ikmn_canonical[36] * Zpkmn[38] + S_ikmn_canonical[37] * Zpkmn[41] + S_ikmn_canonical[38] * Zpkmn[44] +
              S_ikmn_canonical[39] * Zpkmn[45] + S_ikmn_canonical[40] * Zpkmn[46] + S_ikmn_canonical[40] * Zpkmn[48] +
              S_ikmn_canonical[41] * Zpkmn[47] + S_ikmn_canonical[41] * Zpkmn[51] + S_ikmn_canonical[42] * Zpkmn[49] +
              S_ikmn_canonical[43] * Zpkmn[50] + S_ikmn_canonical[43] * Zpkmn[52] + S_ikmn_canonical[44] * Zpkmn[53];
    outz[2] = S_ikmn_canonical[15] * Zpkmn[54] + S_ikmn_canonical[16] * Zpkmn[57] + S_ikmn_canonical[17] * Zpkmn[60] +
              S_ikmn_canonical[18] * Zpkmn[55] + S_ikmn_canonical[19] * Zpkmn[58] + S_ikmn_canonical[20] * Zpkmn[61] +
              S_ikmn_canonical[21] * Zpkmn[56] + S_ikmn_canonical[22] * Zpkmn[59] + S_ikmn_canonical[23] * Zpkmn[62] +
              S_ikmn_canonical[30] * Zpkmn[63] + S_ikmn_canonical[31] * Zpkmn[66] + S_ikmn_canonical[32] * Zpkmn[69] +
              S_ikmn_canonical[33] * Zpkmn[64] + S_ikmn_canonical[34] * Zpkmn[67] + S_ikmn_canonical[35] * Zpkmn[70] +
              S_ikmn_canonical[36] * Zpkmn[65] + S_ikmn_canonical[37] * Zpkmn[68] + S_ikmn_canonical[38] * Zpkmn[71] +
              S_ikmn_canonical[39] * Zpkmn[72] + S_ikmn_canonical[40] * Zpkmn[73] + S_ikmn_canonical[40] * Zpkmn[75] +
              S_ikmn_canonical[41] * Zpkmn[74] + S_ikmn_canonical[41] * Zpkmn[78] + S_ikmn_canonical[42] * Zpkmn[76] +
              S_ikmn_canonical[43] * Zpkmn[77] + S_ikmn_canonical[43] * Zpkmn[79] + S_ikmn_canonical[44] * Zpkmn[80];
    outz[3] = S_ikmn_canonical[15] * Zpkmn[81] + S_ikmn_canonical[16] * Zpkmn[84] + S_ikmn_canonical[17] * Zpkmn[87] +
              S_ikmn_canonical[18] * Zpkmn[82] + S_ikmn_canonical[19] * Zpkmn[85] + S_ikmn_canonical[20] * Zpkmn[88] +
              S_ikmn_canonical[21] * Zpkmn[83] + S_ikmn_canonical[22] * Zpkmn[86] + S_ikmn_canonical[23] * Zpkmn[89] +
              S_ikmn_canonical[30] * Zpkmn[90] + S_ikmn_canonical[31] * Zpkmn[93] + S_ikmn_canonical[32] * Zpkmn[96] +
              S_ikmn_canonical[33] * Zpkmn[91] + S_ikmn_canonical[34] * Zpkmn[94] + S_ikmn_canonical[35] * Zpkmn[97] +
              S_ikmn_canonical[36] * Zpkmn[92] + S_ikmn_canonical[37] * Zpkmn[95] + S_ikmn_canonical[38] * Zpkmn[98] +
              S_ikmn_canonical[39] * Zpkmn[99] + S_ikmn_canonical[40] * Zpkmn[100] + S_ikmn_canonical[40] * Zpkmn[102] +
              S_ikmn_canonical[41] * Zpkmn[101] + S_ikmn_canonical[41] * Zpkmn[105] + S_ikmn_canonical[42] * Zpkmn[103] +
              S_ikmn_canonical[43] * Zpkmn[104] + S_ikmn_canonical[43] * Zpkmn[106] + S_ikmn_canonical[44] * Zpkmn[107];
}

static SFEM_INLINE void tet4_expand_S(const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
                                      scalar_t *const SFEM_RESTRICT              S_ikmn) {
    // mundane ops: 0 divs: 0 sqrts: 0
    // total ops: 0
    S_ikmn[0]  = S_ikmn_canonical[0];
    S_ikmn[1]  = S_ikmn_canonical[1];
    S_ikmn[2]  = S_ikmn_canonical[2];
    S_ikmn[3]  = S_ikmn_canonical[1];
    S_ikmn[4]  = S_ikmn_canonical[3];
    S_ikmn[5]  = S_ikmn_canonical[4];
    S_ikmn[6]  = S_ikmn_canonical[2];
    S_ikmn[7]  = S_ikmn_canonical[4];
    S_ikmn[8]  = S_ikmn_canonical[5];
    S_ikmn[9]  = S_ikmn_canonical[6];
    S_ikmn[10] = S_ikmn_canonical[7];
    S_ikmn[11] = S_ikmn_canonical[8];
    S_ikmn[12] = S_ikmn_canonical[9];
    S_ikmn[13] = S_ikmn_canonical[10];
    S_ikmn[14] = S_ikmn_canonical[11];
    S_ikmn[15] = S_ikmn_canonical[12];
    S_ikmn[16] = S_ikmn_canonical[13];
    S_ikmn[17] = S_ikmn_canonical[14];
    S_ikmn[18] = S_ikmn_canonical[15];
    S_ikmn[19] = S_ikmn_canonical[16];
    S_ikmn[20] = S_ikmn_canonical[17];
    S_ikmn[21] = S_ikmn_canonical[18];
    S_ikmn[22] = S_ikmn_canonical[19];
    S_ikmn[23] = S_ikmn_canonical[20];
    S_ikmn[24] = S_ikmn_canonical[21];
    S_ikmn[25] = S_ikmn_canonical[22];
    S_ikmn[26] = S_ikmn_canonical[23];
    S_ikmn[27] = S_ikmn_canonical[6];
    S_ikmn[28] = S_ikmn_canonical[9];
    S_ikmn[29] = S_ikmn_canonical[12];
    S_ikmn[30] = S_ikmn_canonical[7];
    S_ikmn[31] = S_ikmn_canonical[10];
    S_ikmn[32] = S_ikmn_canonical[13];
    S_ikmn[33] = S_ikmn_canonical[8];
    S_ikmn[34] = S_ikmn_canonical[11];
    S_ikmn[35] = S_ikmn_canonical[14];
    S_ikmn[36] = S_ikmn_canonical[24];
    S_ikmn[37] = S_ikmn_canonical[25];
    S_ikmn[38] = S_ikmn_canonical[26];
    S_ikmn[39] = S_ikmn_canonical[25];
    S_ikmn[40] = S_ikmn_canonical[27];
    S_ikmn[41] = S_ikmn_canonical[28];
    S_ikmn[42] = S_ikmn_canonical[26];
    S_ikmn[43] = S_ikmn_canonical[28];
    S_ikmn[44] = S_ikmn_canonical[29];
    S_ikmn[45] = S_ikmn_canonical[30];
    S_ikmn[46] = S_ikmn_canonical[31];
    S_ikmn[47] = S_ikmn_canonical[32];
    S_ikmn[48] = S_ikmn_canonical[33];
    S_ikmn[49] = S_ikmn_canonical[34];
    S_ikmn[50] = S_ikmn_canonical[35];
    S_ikmn[51] = S_ikmn_canonical[36];
    S_ikmn[52] = S_ikmn_canonical[37];
    S_ikmn[53] = S_ikmn_canonical[38];
    S_ikmn[54] = S_ikmn_canonical[15];
    S_ikmn[55] = S_ikmn_canonical[18];
    S_ikmn[56] = S_ikmn_canonical[21];
    S_ikmn[57] = S_ikmn_canonical[16];
    S_ikmn[58] = S_ikmn_canonical[19];
    S_ikmn[59] = S_ikmn_canonical[22];
    S_ikmn[60] = S_ikmn_canonical[17];
    S_ikmn[61] = S_ikmn_canonical[20];
    S_ikmn[62] = S_ikmn_canonical[23];
    S_ikmn[63] = S_ikmn_canonical[30];
    S_ikmn[64] = S_ikmn_canonical[33];
    S_ikmn[65] = S_ikmn_canonical[36];
    S_ikmn[66] = S_ikmn_canonical[31];
    S_ikmn[67] = S_ikmn_canonical[34];
    S_ikmn[68] = S_ikmn_canonical[37];
    S_ikmn[69] = S_ikmn_canonical[32];
    S_ikmn[70] = S_ikmn_canonical[35];
    S_ikmn[71] = S_ikmn_canonical[38];
    S_ikmn[72] = S_ikmn_canonical[39];
    S_ikmn[73] = S_ikmn_canonical[40];
    S_ikmn[74] = S_ikmn_canonical[41];
    S_ikmn[75] = S_ikmn_canonical[40];
    S_ikmn[76] = S_ikmn_canonical[42];
    S_ikmn[77] = S_ikmn_canonical[43];
    S_ikmn[78] = S_ikmn_canonical[41];
    S_ikmn[79] = S_ikmn_canonical[43];
    S_ikmn[80] = S_ikmn_canonical[44];
}

static SFEM_INLINE void tet4_SdotZ_expanded(const scalar_t *const SFEM_RESTRICT S_ikmn,
                                            const scalar_t *const SFEM_RESTRICT Zpkmn,
                                            scalar_t *const SFEM_RESTRICT       outx,
                                            scalar_t *const SFEM_RESTRICT       outy,
                                            scalar_t *const SFEM_RESTRICT       outz)

{
    static const int pstride = 3 * 3 * 3;
    static const int ksize   = 3 * 3 * 3;

    for (int p = 0; p < 4; p++) {
        scalar_t                            acc[3] = {0};
        const scalar_t *const SFEM_RESTRICT Zkmn   = &Zpkmn[p * pstride];
        for (int i = 0; i < 3; i++) {
            const scalar_t *const SFEM_RESTRICT Skmn = &S_ikmn[i * ksize];
            for (int k = 0; k < ksize; k++) {
                acc[i] += Skmn[k] * Zkmn[k];
            }
        }

        outx[p] = acc[0];
        outy[p] = acc[1];
        outz[p] = acc[2];
    }
}

static SFEM_INLINE void tet4_neohookean_ogden_active_strain_hessian_from_S_ikmn(
        const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
        const scalar_t *const SFEM_RESTRICT        Wimpn_compressed,
        scalar_t *const SFEM_RESTRICT              H) {
    // mundane ops: 483 divs: 0 sqrts: 0
    // total ops: 483
    const scalar_t x0   = 2 * S_ikmn_canonical[1];
    const scalar_t x1   = 2 * S_ikmn_canonical[2];
    const scalar_t x2   = 2 * S_ikmn_canonical[4];
    const scalar_t x3   = Wimpn_compressed[2] * x2;
    const scalar_t x4   = S_ikmn_canonical[1] * Wimpn_compressed[1];
    const scalar_t x5   = S_ikmn_canonical[2] * Wimpn_compressed[1];
    const scalar_t x6   = S_ikmn_canonical[3] * Wimpn_compressed[2];
    const scalar_t x7   = S_ikmn_canonical[1] * Wimpn_compressed[2];
    const scalar_t x8   = x6 + x7;
    const scalar_t x9   = S_ikmn_canonical[5] * Wimpn_compressed[2];
    const scalar_t x10  = S_ikmn_canonical[2] * Wimpn_compressed[2];
    const scalar_t x11  = x10 + x9;
    const scalar_t x12  = S_ikmn_canonical[0] * Wimpn_compressed[1] + x11 + x3 + x4 + x5 + x8;
    const scalar_t x13  = S_ikmn_canonical[0] * Wimpn_compressed[2];
    const scalar_t x14  = S_ikmn_canonical[4] * Wimpn_compressed[1];
    const scalar_t x15  = S_ikmn_canonical[4] * Wimpn_compressed[2];
    const scalar_t x16  = Wimpn_compressed[2] * x1;
    const scalar_t x17  = x16 + x9;
    const scalar_t x18  = x15 + x17;
    const scalar_t x19  = S_ikmn_canonical[3] * Wimpn_compressed[1] + x13 + x14 + x18 + x4 + x7;
    const scalar_t x20  = Wimpn_compressed[2] * x0;
    const scalar_t x21  = x20 + x6;
    const scalar_t x22  = x13 + x21;
    const scalar_t x23  = S_ikmn_canonical[5] * Wimpn_compressed[1] + x10 + x14 + x15 + x22 + x5;
    const scalar_t x24  = Wimpn_compressed[0] * (S_ikmn_canonical[10] + S_ikmn_canonical[11] + S_ikmn_canonical[12] +
                                                S_ikmn_canonical[13] + S_ikmn_canonical[14] + S_ikmn_canonical[6] +
                                                S_ikmn_canonical[7] + S_ikmn_canonical[8] + S_ikmn_canonical[9]);
    const scalar_t x25  = S_ikmn_canonical[6] * Wimpn_compressed[1];
    const scalar_t x26  = S_ikmn_canonical[7] * Wimpn_compressed[1];
    const scalar_t x27  = S_ikmn_canonical[8] * Wimpn_compressed[1];
    const scalar_t x28  = S_ikmn_canonical[12] * Wimpn_compressed[2];
    const scalar_t x29  = S_ikmn_canonical[13] * Wimpn_compressed[2];
    const scalar_t x30  = S_ikmn_canonical[14] * Wimpn_compressed[2];
    const scalar_t x31  = x28 + x29 + x30;
    const scalar_t x32  = S_ikmn_canonical[10] * Wimpn_compressed[2];
    const scalar_t x33  = S_ikmn_canonical[11] * Wimpn_compressed[2];
    const scalar_t x34  = S_ikmn_canonical[9] * Wimpn_compressed[2];
    const scalar_t x35  = x32 + x33 + x34;
    const scalar_t x36  = x31 + x35;
    const scalar_t x37  = x25 + x26 + x27 + x36;
    const scalar_t x38  = S_ikmn_canonical[10] * Wimpn_compressed[1];
    const scalar_t x39  = S_ikmn_canonical[11] * Wimpn_compressed[1];
    const scalar_t x40  = S_ikmn_canonical[9] * Wimpn_compressed[1];
    const scalar_t x41  = S_ikmn_canonical[6] * Wimpn_compressed[2];
    const scalar_t x42  = S_ikmn_canonical[7] * Wimpn_compressed[2];
    const scalar_t x43  = S_ikmn_canonical[8] * Wimpn_compressed[2];
    const scalar_t x44  = x42 + x43;
    const scalar_t x45  = x41 + x44;
    const scalar_t x46  = x31 + x45;
    const scalar_t x47  = x38 + x39 + x40 + x46;
    const scalar_t x48  = S_ikmn_canonical[12] * Wimpn_compressed[1];
    const scalar_t x49  = S_ikmn_canonical[13] * Wimpn_compressed[1];
    const scalar_t x50  = S_ikmn_canonical[14] * Wimpn_compressed[1];
    const scalar_t x51  = x35 + x45;
    const scalar_t x52  = x48 + x49 + x50 + x51;
    const scalar_t x53  = Wimpn_compressed[0] * (S_ikmn_canonical[15] + S_ikmn_canonical[16] + S_ikmn_canonical[17] +
                                                S_ikmn_canonical[18] + S_ikmn_canonical[19] + S_ikmn_canonical[20] +
                                                S_ikmn_canonical[21] + S_ikmn_canonical[22] + S_ikmn_canonical[23]);
    const scalar_t x54  = S_ikmn_canonical[15] * Wimpn_compressed[1];
    const scalar_t x55  = S_ikmn_canonical[16] * Wimpn_compressed[1];
    const scalar_t x56  = S_ikmn_canonical[17] * Wimpn_compressed[1];
    const scalar_t x57  = S_ikmn_canonical[21] * Wimpn_compressed[2];
    const scalar_t x58  = S_ikmn_canonical[22] * Wimpn_compressed[2];
    const scalar_t x59  = S_ikmn_canonical[23] * Wimpn_compressed[2];
    const scalar_t x60  = x57 + x58 + x59;
    const scalar_t x61  = S_ikmn_canonical[18] * Wimpn_compressed[2];
    const scalar_t x62  = S_ikmn_canonical[19] * Wimpn_compressed[2];
    const scalar_t x63  = S_ikmn_canonical[20] * Wimpn_compressed[2];
    const scalar_t x64  = x61 + x62 + x63;
    const scalar_t x65  = x60 + x64;
    const scalar_t x66  = x54 + x55 + x56 + x65;
    const scalar_t x67  = S_ikmn_canonical[18] * Wimpn_compressed[1];
    const scalar_t x68  = S_ikmn_canonical[19] * Wimpn_compressed[1];
    const scalar_t x69  = S_ikmn_canonical[20] * Wimpn_compressed[1];
    const scalar_t x70  = S_ikmn_canonical[15] * Wimpn_compressed[2];
    const scalar_t x71  = S_ikmn_canonical[16] * Wimpn_compressed[2];
    const scalar_t x72  = S_ikmn_canonical[17] * Wimpn_compressed[2];
    const scalar_t x73  = x71 + x72;
    const scalar_t x74  = x70 + x73;
    const scalar_t x75  = x60 + x74;
    const scalar_t x76  = x67 + x68 + x69 + x75;
    const scalar_t x77  = S_ikmn_canonical[21] * Wimpn_compressed[1];
    const scalar_t x78  = S_ikmn_canonical[22] * Wimpn_compressed[1];
    const scalar_t x79  = S_ikmn_canonical[23] * Wimpn_compressed[1];
    const scalar_t x80  = x64 + x74;
    const scalar_t x81  = x77 + x78 + x79 + x80;
    const scalar_t x82  = x17 + x3;
    const scalar_t x83  = x13 + x82;
    const scalar_t x84  = S_ikmn_canonical[1] * Wimpn_compressed[0] + x8 + x83;
    const scalar_t x85  = x22 + x3;
    const scalar_t x86  = S_ikmn_canonical[2] * Wimpn_compressed[0] + x11 + x85;
    const scalar_t x87  = x32 + x33;
    const scalar_t x88  = x29 + x30;
    const scalar_t x89  = x25 + x40 + x44 + x48 + x87 + x88;
    const scalar_t x90  = S_ikmn_canonical[6] * Wimpn_compressed[0] + x36 + x44;
    const scalar_t x91  = S_ikmn_canonical[9] * Wimpn_compressed[0] + x46 + x87;
    const scalar_t x92  = S_ikmn_canonical[12] * Wimpn_compressed[0] + x51 + x88;
    const scalar_t x93  = x62 + x63;
    const scalar_t x94  = x58 + x59;
    const scalar_t x95  = x54 + x67 + x73 + x77 + x93 + x94;
    const scalar_t x96  = S_ikmn_canonical[15] * Wimpn_compressed[0] + x65 + x73;
    const scalar_t x97  = S_ikmn_canonical[18] * Wimpn_compressed[0] + x75 + x93;
    const scalar_t x98  = S_ikmn_canonical[21] * Wimpn_compressed[0] + x80 + x94;
    const scalar_t x99  = S_ikmn_canonical[4] * Wimpn_compressed[0] + x18 + x22;
    const scalar_t x100 = x41 + x43;
    const scalar_t x101 = x33 + x34;
    const scalar_t x102 = x28 + x30;
    const scalar_t x103 = x100 + x101 + x102 + x26 + x38 + x49;
    const scalar_t x104 = S_ikmn_canonical[7] * Wimpn_compressed[0] + x100 + x36;
    const scalar_t x105 = S_ikmn_canonical[10] * Wimpn_compressed[0] + x101 + x46;
    const scalar_t x106 = S_ikmn_canonical[13] * Wimpn_compressed[0] + x102 + x51;
    const scalar_t x107 = x70 + x72;
    const scalar_t x108 = x61 + x63;
    const scalar_t x109 = x57 + x59;
    const scalar_t x110 = x107 + x108 + x109 + x55 + x68 + x78;
    const scalar_t x111 = S_ikmn_canonical[16] * Wimpn_compressed[0] + x107 + x65;
    const scalar_t x112 = S_ikmn_canonical[19] * Wimpn_compressed[0] + x108 + x75;
    const scalar_t x113 = S_ikmn_canonical[22] * Wimpn_compressed[0] + x109 + x80;
    const scalar_t x114 = x41 + x42;
    const scalar_t x115 = x32 + x34;
    const scalar_t x116 = x28 + x29;
    const scalar_t x117 = x114 + x115 + x116 + x27 + x39 + x50;
    const scalar_t x118 = S_ikmn_canonical[8] * Wimpn_compressed[0] + x114 + x36;
    const scalar_t x119 = S_ikmn_canonical[11] * Wimpn_compressed[0] + x115 + x46;
    const scalar_t x120 = S_ikmn_canonical[14] * Wimpn_compressed[0] + x116 + x51;
    const scalar_t x121 = x70 + x71;
    const scalar_t x122 = x61 + x62;
    const scalar_t x123 = x57 + x58;
    const scalar_t x124 = x121 + x122 + x123 + x56 + x69 + x79;
    const scalar_t x125 = S_ikmn_canonical[17] * Wimpn_compressed[0] + x121 + x65;
    const scalar_t x126 = S_ikmn_canonical[20] * Wimpn_compressed[0] + x122 + x75;
    const scalar_t x127 = S_ikmn_canonical[23] * Wimpn_compressed[0] + x123 + x80;
    const scalar_t x128 = 2 * S_ikmn_canonical[25];
    const scalar_t x129 = 2 * S_ikmn_canonical[26];
    const scalar_t x130 = 2 * S_ikmn_canonical[28];
    const scalar_t x131 = Wimpn_compressed[2] * x130;
    const scalar_t x132 = S_ikmn_canonical[25] * Wimpn_compressed[1];
    const scalar_t x133 = S_ikmn_canonical[26] * Wimpn_compressed[1];
    const scalar_t x134 = S_ikmn_canonical[27] * Wimpn_compressed[2];
    const scalar_t x135 = S_ikmn_canonical[25] * Wimpn_compressed[2];
    const scalar_t x136 = x134 + x135;
    const scalar_t x137 = S_ikmn_canonical[29] * Wimpn_compressed[2];
    const scalar_t x138 = S_ikmn_canonical[26] * Wimpn_compressed[2];
    const scalar_t x139 = x137 + x138;
    const scalar_t x140 = S_ikmn_canonical[24] * Wimpn_compressed[1] + x131 + x132 + x133 + x136 + x139;
    const scalar_t x141 = S_ikmn_canonical[24] * Wimpn_compressed[2];
    const scalar_t x142 = S_ikmn_canonical[28] * Wimpn_compressed[1];
    const scalar_t x143 = S_ikmn_canonical[28] * Wimpn_compressed[2];
    const scalar_t x144 = Wimpn_compressed[2] * x129;
    const scalar_t x145 = x137 + x144;
    const scalar_t x146 = x143 + x145;
    const scalar_t x147 = S_ikmn_canonical[27] * Wimpn_compressed[1] + x132 + x135 + x141 + x142 + x146;
    const scalar_t x148 = Wimpn_compressed[2] * x128;
    const scalar_t x149 = x134 + x148;
    const scalar_t x150 = x141 + x149;
    const scalar_t x151 = S_ikmn_canonical[29] * Wimpn_compressed[1] + x133 + x138 + x142 + x143 + x150;
    const scalar_t x152 = Wimpn_compressed[0] * (S_ikmn_canonical[30] + S_ikmn_canonical[31] + S_ikmn_canonical[32] +
                                                 S_ikmn_canonical[33] + S_ikmn_canonical[34] + S_ikmn_canonical[35] +
                                                 S_ikmn_canonical[36] + S_ikmn_canonical[37] + S_ikmn_canonical[38]);
    const scalar_t x153 = S_ikmn_canonical[30] * Wimpn_compressed[1];
    const scalar_t x154 = S_ikmn_canonical[31] * Wimpn_compressed[1];
    const scalar_t x155 = S_ikmn_canonical[32] * Wimpn_compressed[1];
    const scalar_t x156 = S_ikmn_canonical[36] * Wimpn_compressed[2];
    const scalar_t x157 = S_ikmn_canonical[37] * Wimpn_compressed[2];
    const scalar_t x158 = S_ikmn_canonical[38] * Wimpn_compressed[2];
    const scalar_t x159 = x156 + x157 + x158;
    const scalar_t x160 = S_ikmn_canonical[33] * Wimpn_compressed[2];
    const scalar_t x161 = S_ikmn_canonical[34] * Wimpn_compressed[2];
    const scalar_t x162 = S_ikmn_canonical[35] * Wimpn_compressed[2];
    const scalar_t x163 = x160 + x161 + x162;
    const scalar_t x164 = x159 + x163;
    const scalar_t x165 = x153 + x154 + x155 + x164;
    const scalar_t x166 = S_ikmn_canonical[33] * Wimpn_compressed[1];
    const scalar_t x167 = S_ikmn_canonical[34] * Wimpn_compressed[1];
    const scalar_t x168 = S_ikmn_canonical[35] * Wimpn_compressed[1];
    const scalar_t x169 = S_ikmn_canonical[30] * Wimpn_compressed[2];
    const scalar_t x170 = S_ikmn_canonical[31] * Wimpn_compressed[2];
    const scalar_t x171 = S_ikmn_canonical[32] * Wimpn_compressed[2];
    const scalar_t x172 = x170 + x171;
    const scalar_t x173 = x169 + x172;
    const scalar_t x174 = x159 + x173;
    const scalar_t x175 = x166 + x167 + x168 + x174;
    const scalar_t x176 = S_ikmn_canonical[36] * Wimpn_compressed[1];
    const scalar_t x177 = S_ikmn_canonical[37] * Wimpn_compressed[1];
    const scalar_t x178 = S_ikmn_canonical[38] * Wimpn_compressed[1];
    const scalar_t x179 = x163 + x173;
    const scalar_t x180 = x176 + x177 + x178 + x179;
    const scalar_t x181 = x131 + x145;
    const scalar_t x182 = x141 + x181;
    const scalar_t x183 = S_ikmn_canonical[25] * Wimpn_compressed[0] + x136 + x182;
    const scalar_t x184 = x131 + x150;
    const scalar_t x185 = S_ikmn_canonical[26] * Wimpn_compressed[0] + x139 + x184;
    const scalar_t x186 = x161 + x162;
    const scalar_t x187 = x157 + x158;
    const scalar_t x188 = x153 + x166 + x172 + x176 + x186 + x187;
    const scalar_t x189 = S_ikmn_canonical[30] * Wimpn_compressed[0] + x164 + x172;
    const scalar_t x190 = S_ikmn_canonical[33] * Wimpn_compressed[0] + x174 + x186;
    const scalar_t x191 = S_ikmn_canonical[36] * Wimpn_compressed[0] + x179 + x187;
    const scalar_t x192 = S_ikmn_canonical[28] * Wimpn_compressed[0] + x146 + x150;
    const scalar_t x193 = x169 + x171;
    const scalar_t x194 = x160 + x162;
    const scalar_t x195 = x156 + x158;
    const scalar_t x196 = x154 + x167 + x177 + x193 + x194 + x195;
    const scalar_t x197 = S_ikmn_canonical[31] * Wimpn_compressed[0] + x164 + x193;
    const scalar_t x198 = S_ikmn_canonical[34] * Wimpn_compressed[0] + x174 + x194;
    const scalar_t x199 = S_ikmn_canonical[37] * Wimpn_compressed[0] + x179 + x195;
    const scalar_t x200 = x169 + x170;
    const scalar_t x201 = x160 + x161;
    const scalar_t x202 = x156 + x157;
    const scalar_t x203 = x155 + x168 + x178 + x200 + x201 + x202;
    const scalar_t x204 = S_ikmn_canonical[32] * Wimpn_compressed[0] + x164 + x200;
    const scalar_t x205 = S_ikmn_canonical[35] * Wimpn_compressed[0] + x174 + x201;
    const scalar_t x206 = S_ikmn_canonical[38] * Wimpn_compressed[0] + x179 + x202;
    const scalar_t x207 = 2 * S_ikmn_canonical[40];
    const scalar_t x208 = 2 * S_ikmn_canonical[41];
    const scalar_t x209 = 2 * S_ikmn_canonical[43];
    const scalar_t x210 = Wimpn_compressed[2] * x209;
    const scalar_t x211 = S_ikmn_canonical[40] * Wimpn_compressed[1];
    const scalar_t x212 = S_ikmn_canonical[41] * Wimpn_compressed[1];
    const scalar_t x213 = S_ikmn_canonical[42] * Wimpn_compressed[2];
    const scalar_t x214 = S_ikmn_canonical[40] * Wimpn_compressed[2];
    const scalar_t x215 = x213 + x214;
    const scalar_t x216 = S_ikmn_canonical[44] * Wimpn_compressed[2];
    const scalar_t x217 = S_ikmn_canonical[41] * Wimpn_compressed[2];
    const scalar_t x218 = x216 + x217;
    const scalar_t x219 = S_ikmn_canonical[39] * Wimpn_compressed[1] + x210 + x211 + x212 + x215 + x218;
    const scalar_t x220 = S_ikmn_canonical[39] * Wimpn_compressed[2];
    const scalar_t x221 = S_ikmn_canonical[43] * Wimpn_compressed[1];
    const scalar_t x222 = S_ikmn_canonical[43] * Wimpn_compressed[2];
    const scalar_t x223 = Wimpn_compressed[2] * x208;
    const scalar_t x224 = x216 + x223;
    const scalar_t x225 = x222 + x224;
    const scalar_t x226 = S_ikmn_canonical[42] * Wimpn_compressed[1] + x211 + x214 + x220 + x221 + x225;
    const scalar_t x227 = Wimpn_compressed[2] * x207;
    const scalar_t x228 = x213 + x227;
    const scalar_t x229 = x220 + x228;
    const scalar_t x230 = S_ikmn_canonical[44] * Wimpn_compressed[1] + x212 + x217 + x221 + x222 + x229;
    const scalar_t x231 = x210 + x224;
    const scalar_t x232 = x220 + x231;
    const scalar_t x233 = S_ikmn_canonical[40] * Wimpn_compressed[0] + x215 + x232;
    const scalar_t x234 = x210 + x229;
    const scalar_t x235 = S_ikmn_canonical[41] * Wimpn_compressed[0] + x218 + x234;
    const scalar_t x236 = S_ikmn_canonical[43] * Wimpn_compressed[0] + x225 + x229;
    H[0]                = Wimpn_compressed[0] * (S_ikmn_canonical[0] + S_ikmn_canonical[3] + S_ikmn_canonical[5] + x0 + x1 + x2);
    H[1]                = x12;
    H[2]                = x19;
    H[3]                = x23;
    H[4]                = x24;
    H[5]                = x37;
    H[6]                = x47;
    H[7]                = x52;
    H[8]                = x53;
    H[9]                = x66;
    H[10]               = x76;
    H[11]               = x81;
    H[12]               = x12;
    H[13]               = S_ikmn_canonical[0] * Wimpn_compressed[0] + x21 + x82;
    H[14]               = x84;
    H[15]               = x86;
    H[16]               = x89;
    H[17]               = x90;
    H[18]               = x91;
    H[19]               = x92;
    H[20]               = x95;
    H[21]               = x96;
    H[22]               = x97;
    H[23]               = x98;
    H[24]               = x19;
    H[25]               = x84;
    H[26]               = S_ikmn_canonical[3] * Wimpn_compressed[0] + x20 + x83;
    H[27]               = x99;
    H[28]               = x103;
    H[29]               = x104;
    H[30]               = x105;
    H[31]               = x106;
    H[32]               = x110;
    H[33]               = x111;
    H[34]               = x112;
    H[35]               = x113;
    H[36]               = x23;
    H[37]               = x86;
    H[38]               = x99;
    H[39]               = S_ikmn_canonical[5] * Wimpn_compressed[0] + x16 + x85;
    H[40]               = x117;
    H[41]               = x118;
    H[42]               = x119;
    H[43]               = x120;
    H[44]               = x124;
    H[45]               = x125;
    H[46]               = x126;
    H[47]               = x127;
    H[48]               = x24;
    H[49]               = x89;
    H[50]               = x103;
    H[51]               = x117;
    H[52]  = Wimpn_compressed[0] * (S_ikmn_canonical[24] + S_ikmn_canonical[27] + S_ikmn_canonical[29] + x128 + x129 + x130);
    H[53]  = x140;
    H[54]  = x147;
    H[55]  = x151;
    H[56]  = x152;
    H[57]  = x165;
    H[58]  = x175;
    H[59]  = x180;
    H[60]  = x37;
    H[61]  = x90;
    H[62]  = x104;
    H[63]  = x118;
    H[64]  = x140;
    H[65]  = S_ikmn_canonical[24] * Wimpn_compressed[0] + x149 + x181;
    H[66]  = x183;
    H[67]  = x185;
    H[68]  = x188;
    H[69]  = x189;
    H[70]  = x190;
    H[71]  = x191;
    H[72]  = x47;
    H[73]  = x91;
    H[74]  = x105;
    H[75]  = x119;
    H[76]  = x147;
    H[77]  = x183;
    H[78]  = S_ikmn_canonical[27] * Wimpn_compressed[0] + x148 + x182;
    H[79]  = x192;
    H[80]  = x196;
    H[81]  = x197;
    H[82]  = x198;
    H[83]  = x199;
    H[84]  = x52;
    H[85]  = x92;
    H[86]  = x106;
    H[87]  = x120;
    H[88]  = x151;
    H[89]  = x185;
    H[90]  = x192;
    H[91]  = S_ikmn_canonical[29] * Wimpn_compressed[0] + x144 + x184;
    H[92]  = x203;
    H[93]  = x204;
    H[94]  = x205;
    H[95]  = x206;
    H[96]  = x53;
    H[97]  = x95;
    H[98]  = x110;
    H[99]  = x124;
    H[100] = x152;
    H[101] = x188;
    H[102] = x196;
    H[103] = x203;
    H[104] = Wimpn_compressed[0] * (S_ikmn_canonical[39] + S_ikmn_canonical[42] + S_ikmn_canonical[44] + x207 + x208 + x209);
    H[105] = x219;
    H[106] = x226;
    H[107] = x230;
    H[108] = x66;
    H[109] = x96;
    H[110] = x111;
    H[111] = x125;
    H[112] = x165;
    H[113] = x189;
    H[114] = x197;
    H[115] = x204;
    H[116] = x219;
    H[117] = S_ikmn_canonical[39] * Wimpn_compressed[0] + x228 + x231;
    H[118] = x233;
    H[119] = x235;
    H[120] = x76;
    H[121] = x97;
    H[122] = x112;
    H[123] = x126;
    H[124] = x175;
    H[125] = x190;
    H[126] = x198;
    H[127] = x205;
    H[128] = x226;
    H[129] = x233;
    H[130] = S_ikmn_canonical[42] * Wimpn_compressed[0] + x227 + x232;
    H[131] = x236;
    H[132] = x81;
    H[133] = x98;
    H[134] = x113;
    H[135] = x127;
    H[136] = x180;
    H[137] = x191;
    H[138] = x199;
    H[139] = x206;
    H[140] = x230;
    H[141] = x235;
    H[142] = x236;
    H[143] = S_ikmn_canonical[44] * Wimpn_compressed[0] + x223 + x234;
}

#define TET4_S_IKMN_SIZE 45
static SFEM_INLINE void tet4_S_ikmn_neohookean_ogden_active_strain(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                   const scalar_t                      jacobian_determinant,
                                                                   const scalar_t                      qw,
                                                                   const scalar_t *const SFEM_RESTRICT F,
                                                                   const scalar_t                      lmbda,
                                                                   const scalar_t                      mu,
                                                                   const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                   const scalar_t                      Ja,
                                                                   scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1408 divs: 2 sqrts: 0
    // total ops: 1424
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
    const scalar_t x50 = F[0] * x10 + F[0] * x12 + F[0] * x14 - F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 - F[0] * x19 -
                         F[0] * x20 + F[0] * x3 + F[0] * x5 + F[0] * x7 + F[1] * x23 + F[1] * x24 + F[1] * x25 + F[1] * x27 +
                         F[1] * x28 + F[1] * x29 - F[1] * x38 - F[1] * x39 - F[1] * x40 - F[1] * x41 - F[1] * x42 - F[1] * x43 +
                         F[2] * x31 + F[2] * x32 + F[2] * x33 + F[2] * x35 + F[2] * x36 + F[2] * x37 - F[2] * x44 - F[2] * x45 -
                         F[2] * x46 - F[2] * x47 - F[2] * x48 - F[2] * x49;
    const scalar_t x51 = (1 / POW2(x50));
    const scalar_t x52 = POW2(x21) * x51;
    const scalar_t x53 = lmbda * log(x50);
    const scalar_t x54 = lmbda * x52 + mu * x52 + x0 - x52 * x53;
    const scalar_t x55 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x56 = x21 * x51;
    const scalar_t x57 = x31 + x32 + x33 + x35 + x36 + x37 - x44 - x45 - x46 - x47 - x48 - x49;
    const scalar_t x58 = lmbda * x57;
    const scalar_t x59 = mu * x56;
    const scalar_t x60 = x53 * x56;
    const scalar_t x61 = x55 + x56 * x58 + x57 * x59 - x57 * x60;
    const scalar_t x62 = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x63 = -x23 - x24 - x25 - x27 - x28 - x29 + x38 + x39 + x40 + x41 + x42 + x43;
    const scalar_t x64 = lmbda * x63;
    const scalar_t x65 = -x56 * x64 - x59 * x63 + x60 * x63 + x62;
    const scalar_t x66 = adjugate[0] * x54 + adjugate[1] * x65 + adjugate[2] * x61;
    const scalar_t x67 = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x68 = x51 * POW2(x57);
    const scalar_t x69 = lmbda * x68 + mu * x68 - x53 * x68 + x67;
    const scalar_t x70 = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x71 = x51 * x57;
    const scalar_t x72 = mu * x63;
    const scalar_t x73 = x53 * x63;
    const scalar_t x74 = -x64 * x71 + x70 - x71 * x72 + x71 * x73;
    const scalar_t x75 = adjugate[0] * x61 + adjugate[1] * x74 + adjugate[2] * x69;
    const scalar_t x76 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x77 = x51 * POW2(x63);
    const scalar_t x78 = lmbda * x77 + mu * x77 - x53 * x77 + x76;
    const scalar_t x79 = adjugate[0] * x65 + adjugate[1] * x78 + adjugate[2] * x74;
    const scalar_t x80 = (1.0 / 6.0) * Ja * qw / jacobian_determinant;
    const scalar_t x81 = adjugate[3] * x54 + adjugate[4] * x65 + adjugate[5] * x61;
    const scalar_t x82 = adjugate[3] * x61 + adjugate[4] * x74 + adjugate[5] * x69;
    const scalar_t x83 = adjugate[3] * x65 + adjugate[4] * x78 + adjugate[5] * x74;
    const scalar_t x84 = 1.0 / x50;
    const scalar_t x85 = F[0] * F[8];
    const scalar_t x86 = F[2] * F[6];
    const scalar_t x87 = x1 * x85 - x1 * x86 - x11 * x85 + x11 * x86 - x13 * x85 + x13 * x86 + x4 * x85 - x4 * x86 + x6 * x85 -
                         x6 * x86 - x8 * x85 + x8 * x86;
    const scalar_t x88 = x84 * x87;
    const scalar_t x89 = lmbda + mu - x53;
    const scalar_t x90 = x63 * x88 * x89;
    const scalar_t x91 = x1 - x11 - x13 + x4 + x6 - x8;
    const scalar_t x92 = mu * x91;
    const scalar_t x93 = F[8] * x92;
    const scalar_t x94 = x53 * x91;
    const scalar_t x95 = F[8] * x94;
    const scalar_t x96 = F[1] * F[8];
    const scalar_t x97 = F[2] * F[7];
    const scalar_t x98 = x1 * x96 - x1 * x97 - x11 * x96 + x11 * x97 - x13 * x96 + x13 * x97 + x4 * x96 - x4 * x97 + x6 * x96 -
                         x6 * x97 - x8 * x96 + x8 * x97;
    const scalar_t x99  = x84 * x98;
    const scalar_t x100 = x64 * x99 + x72 * x99 - x73 * x99 + x93 - x95;
    const scalar_t x101 = F[6] * x92;
    const scalar_t x102 = F[6] * x94;
    const scalar_t x103 = F[0] * F[7];
    const scalar_t x104 = F[1] * F[6];
    const scalar_t x105 = x1 * x103 - x1 * x104 - x103 * x11 - x103 * x13 + x103 * x4 + x103 * x6 - x103 * x8 + x104 * x11 +
                          x104 * x13 - x104 * x4 - x104 * x6 + x104 * x8;
    const scalar_t x106 = x105 * x84;
    const scalar_t x107 = -x101 + x102 + x106 * x64 + x106 * x72 - x106 * x73;
    const scalar_t x108 = adjugate[0] * x100 - adjugate[1] * x90 + adjugate[2] * x107;
    const scalar_t x109 = x21 * x84;
    const scalar_t x110 = x109 * x89;
    const scalar_t x111 = x110 * x98;
    const scalar_t x112 = F[7] * x92;
    const scalar_t x113 = F[7] * x94;
    const scalar_t x114 = lmbda * x109;
    const scalar_t x115 = mu * x109;
    const scalar_t x116 = x109 * x53;
    const scalar_t x117 = x105 * x114 + x105 * x115 - x105 * x116 - x112 + x113;
    const scalar_t x118 = lmbda * x87;
    const scalar_t x119 = x109 * x118 + x115 * x87 - x116 * x87 - x93 + x95;
    const scalar_t x120 = -adjugate[0] * x111 + adjugate[1] * x119 - adjugate[2] * x117;
    const scalar_t x121 = x57 * x89;
    const scalar_t x122 = x106 * x121;
    const scalar_t x123 = mu * x57;
    const scalar_t x124 = x53 * x57;
    const scalar_t x125 = x112 - x113 + x123 * x99 - x124 * x99 + x58 * x99;
    const scalar_t x126 = x101 - x102 + x123 * x88 - x124 * x88 + x58 * x88;
    const scalar_t x127 = -adjugate[0] * x125 + adjugate[1] * x126 - adjugate[2] * x122;
    const scalar_t x128 = x80 * x84;
    const scalar_t x129 = adjugate[3] * x100 - adjugate[4] * x90 + adjugate[5] * x107;
    const scalar_t x130 = -adjugate[3] * x111 + adjugate[4] * x119 - adjugate[5] * x117;
    const scalar_t x131 = -adjugate[3] * x125 + adjugate[4] * x126 - adjugate[5] * x122;
    const scalar_t x132 = adjugate[6] * x100 - adjugate[7] * x90 + adjugate[8] * x107;
    const scalar_t x133 = -adjugate[6] * x111 + adjugate[7] * x119 - adjugate[8] * x117;
    const scalar_t x134 = -adjugate[6] * x125 + adjugate[7] * x126 - adjugate[8] * x122;
    const scalar_t x135 = F[1] * F[5];
    const scalar_t x136 = F[2] * F[4];
    const scalar_t x137 = x1 * x135 - x1 * x136 - x11 * x135 + x11 * x136 - x13 * x135 + x13 * x136 + x135 * x4 + x135 * x6 -
                          x135 * x8 - x136 * x4 - x136 * x6 + x136 * x8;
    const scalar_t x138 = x110 * x137;
    const scalar_t x139 = F[4] * x92;
    const scalar_t x140 = F[4] * x94;
    const scalar_t x141 = F[0] * F[4];
    const scalar_t x142 = F[1] * F[3];
    const scalar_t x143 = x1 * x141 - x1 * x142 - x11 * x141 + x11 * x142 - x13 * x141 + x13 * x142 + x141 * x4 + x141 * x6 -
                          x141 * x8 - x142 * x4 - x142 * x6 + x142 * x8;
    const scalar_t x144 = x114 * x143 + x115 * x143 - x116 * x143 - x139 + x140;
    const scalar_t x145 = F[5] * x92;
    const scalar_t x146 = F[5] * x94;
    const scalar_t x147 = F[0] * F[5];
    const scalar_t x148 = F[2] * F[3];
    const scalar_t x149 = x1 * x147 - x1 * x148 - x11 * x147 + x11 * x148 - x13 * x147 + x13 * x148 + x147 * x4 + x147 * x6 -
                          x147 * x8 - x148 * x4 - x148 * x6 + x148 * x8;
    const scalar_t x150  = x114 * x149 + x115 * x149 - x116 * x149 - x145 + x146;
    const scalar_t x151  = adjugate[0] * x138 - adjugate[1] * x150 + adjugate[2] * x144;
    const scalar_t x152  = x143 * x84;
    const scalar_t x153  = x121 * x152;
    const scalar_t x154  = x137 * x84;
    const scalar_t x155  = x123 * x154 - x124 * x154 + x139 - x140 + x154 * x58;
    const scalar_t x156  = F[3] * x92;
    const scalar_t x157  = F[3] * x94;
    const scalar_t x158  = x149 * x84;
    const scalar_t x159  = x123 * x158 - x124 * x158 + x156 - x157 + x158 * x58;
    const scalar_t x160  = adjugate[0] * x155 - adjugate[1] * x159 + adjugate[2] * x153;
    const scalar_t x161  = x145 - x146 + x154 * x64 + x154 * x72 - x154 * x73;
    const scalar_t x162  = x152 * x64 + x152 * x72 - x152 * x73 - x156 + x157;
    const scalar_t x163  = -adjugate[0] * x161 + adjugate[1] * x149 * x63 * x84 * x89 - adjugate[2] * x162;
    const scalar_t x164  = adjugate[3] * x138 - adjugate[4] * x150 + adjugate[5] * x144;
    const scalar_t x165  = adjugate[3] * x155 - adjugate[4] * x159 + adjugate[5] * x153;
    const scalar_t x166  = -adjugate[3] * x161 + adjugate[4] * x149 * x63 * x84 * x89 - adjugate[5] * x162;
    const scalar_t x167  = adjugate[6] * x138 - adjugate[7] * x150 + adjugate[8] * x144;
    const scalar_t x168  = adjugate[6] * x155 - adjugate[7] * x159 + adjugate[8] * x153;
    const scalar_t x169  = -adjugate[6] * x161 + adjugate[7] * x149 * x63 * x84 * x89 - adjugate[8] * x162;
    const scalar_t x170  = x51 * POW2(x98);
    const scalar_t x171  = lmbda * x170 + mu * x170 + x0 - x170 * x53;
    const scalar_t x172  = x51 * x98;
    const scalar_t x173  = x105 * x172;
    const scalar_t x174  = mu * x172;
    const scalar_t x175  = lmbda * x173 + x105 * x174 - x173 * x53 + x55;
    const scalar_t x176  = -x118 * x172 + x172 * x53 * x87 - x174 * x87 + x62;
    const scalar_t x177  = adjugate[0] * x171 + adjugate[1] * x176 + adjugate[2] * x175;
    const scalar_t x178  = POW2(x105) * x51;
    const scalar_t x179  = lmbda * x178 + mu * x178 - x178 * x53 + x67;
    const scalar_t x180  = x105 * x51;
    const scalar_t x181  = x180 * x87;
    const scalar_t x182  = -mu * x181 - x118 * x180 + x181 * x53 + x70;
    const scalar_t x183  = adjugate[0] * x175 + adjugate[1] * x182 + adjugate[2] * x179;
    const scalar_t x184  = x51 * POW2(x87);
    const scalar_t x185  = lmbda * x184 + mu * x184 - x184 * x53 + x76;
    const scalar_t x186  = adjugate[0] * x176 + adjugate[1] * x185 + adjugate[2] * x182;
    const scalar_t x187  = adjugate[3] * x171 + adjugate[4] * x176 + adjugate[5] * x175;
    const scalar_t x188  = adjugate[3] * x175 + adjugate[4] * x182 + adjugate[5] * x179;
    const scalar_t x189  = adjugate[3] * x176 + adjugate[4] * x185 + adjugate[5] * x182;
    const scalar_t x190  = x149 * x88 * x89;
    const scalar_t x191  = F[2] * x92;
    const scalar_t x192  = F[2] * x94;
    const scalar_t x193  = mu * x88;
    const scalar_t x194  = x53 * x88;
    const scalar_t x195  = x118 * x154 + x137 * x193 - x137 * x194 + x191 - x192;
    const scalar_t x196  = F[0] * x92;
    const scalar_t x197  = F[0] * x94;
    const scalar_t x198  = x118 * x152 + x143 * x193 - x143 * x194 - x196 + x197;
    const scalar_t x199  = adjugate[0] * x195 - adjugate[1] * x190 + adjugate[2] * x198;
    const scalar_t x200  = x137 * x89 * x99;
    const scalar_t x201  = F[1] * x92;
    const scalar_t x202  = F[1] * x94;
    const scalar_t x203  = x143 * x99;
    const scalar_t x204  = lmbda * x203 + mu * x203 - x201 + x202 - x203 * x53;
    const scalar_t x205  = x149 * x99;
    const scalar_t x206  = lmbda * x205 + mu * x205 - x191 + x192 - x205 * x53;
    const scalar_t x207  = -adjugate[0] * x200 + adjugate[1] * x206 - adjugate[2] * x204;
    const scalar_t x208  = x106 * x143 * x89;
    const scalar_t x209  = x106 * x137;
    const scalar_t x210  = lmbda * x209 + mu * x209 + x201 - x202 - x209 * x53;
    const scalar_t x211  = x106 * x149;
    const scalar_t x212  = lmbda * x211 + mu * x211 + x196 - x197 - x211 * x53;
    const scalar_t x213  = -adjugate[0] * x210 + adjugate[1] * x212 - adjugate[2] * x208;
    const scalar_t x214  = adjugate[3] * x195 - adjugate[4] * x190 + adjugate[5] * x198;
    const scalar_t x215  = -adjugate[3] * x200 + adjugate[4] * x206 - adjugate[5] * x204;
    const scalar_t x216  = -adjugate[3] * x210 + adjugate[4] * x212 - adjugate[5] * x208;
    const scalar_t x217  = adjugate[6] * x195 - adjugate[7] * x190 + adjugate[8] * x198;
    const scalar_t x218  = -adjugate[6] * x200 + adjugate[7] * x206 - adjugate[8] * x204;
    const scalar_t x219  = -adjugate[6] * x210 + adjugate[7] * x212 - adjugate[8] * x208;
    const scalar_t x220  = POW2(x137) * x51;
    const scalar_t x221  = lmbda * x220 + mu * x220 + x0 - x220 * x53;
    const scalar_t x222  = x137 * x51;
    const scalar_t x223  = x143 * x222;
    const scalar_t x224  = lmbda * x223 + mu * x223 - x223 * x53 + x55;
    const scalar_t x225  = x149 * x222;
    const scalar_t x226  = -lmbda * x225 - mu * x225 + x225 * x53 + x62;
    const scalar_t x227  = adjugate[0] * x221 + adjugate[1] * x226 + adjugate[2] * x224;
    const scalar_t x228  = POW2(x143) * x51;
    const scalar_t x229  = lmbda * x228 + mu * x228 - x228 * x53 + x67;
    const scalar_t x230  = x143 * x149 * x51;
    const scalar_t x231  = -lmbda * x230 - mu * x230 + x230 * x53 + x70;
    const scalar_t x232  = adjugate[0] * x224 + adjugate[1] * x231 + adjugate[2] * x229;
    const scalar_t x233  = POW2(x149) * x51;
    const scalar_t x234  = lmbda * x233 + mu * x233 - x233 * x53 + x76;
    const scalar_t x235  = adjugate[0] * x226 + adjugate[1] * x234 + adjugate[2] * x231;
    const scalar_t x236  = adjugate[3] * x221 + adjugate[4] * x226 + adjugate[5] * x224;
    const scalar_t x237  = adjugate[3] * x224 + adjugate[4] * x231 + adjugate[5] * x229;
    const scalar_t x238  = adjugate[3] * x226 + adjugate[4] * x234 + adjugate[5] * x231;
    S_ikmn_canonical[0]  = x80 * (adjugate[0] * x66 + adjugate[1] * x79 + adjugate[2] * x75);
    S_ikmn_canonical[1]  = x80 * (adjugate[3] * x66 + adjugate[4] * x79 + adjugate[5] * x75);
    S_ikmn_canonical[2]  = x80 * (adjugate[6] * x66 + adjugate[7] * x79 + adjugate[8] * x75);
    S_ikmn_canonical[3]  = x80 * (adjugate[3] * x81 + adjugate[4] * x83 + adjugate[5] * x82);
    S_ikmn_canonical[4]  = x80 * (adjugate[6] * x81 + adjugate[7] * x83 + adjugate[8] * x82);
    S_ikmn_canonical[5]  = x80 * (adjugate[6] * (adjugate[6] * x54 + adjugate[7] * x65 + adjugate[8] * x61) +
                                 adjugate[7] * (adjugate[6] * x65 + adjugate[7] * x78 + adjugate[8] * x74) +
                                 adjugate[8] * (adjugate[6] * x61 + adjugate[7] * x74 + adjugate[8] * x69));
    S_ikmn_canonical[6]  = x128 * (adjugate[0] * x120 + adjugate[1] * x108 + adjugate[2] * x127);
    S_ikmn_canonical[7]  = x128 * (adjugate[3] * x120 + adjugate[4] * x108 + adjugate[5] * x127);
    S_ikmn_canonical[8]  = x128 * (adjugate[6] * x120 + adjugate[7] * x108 + adjugate[8] * x127);
    S_ikmn_canonical[9]  = x128 * (adjugate[0] * x130 + adjugate[1] * x129 + adjugate[2] * x131);
    S_ikmn_canonical[10] = x128 * (adjugate[3] * x130 + adjugate[4] * x129 + adjugate[5] * x131);
    S_ikmn_canonical[11] = x128 * (adjugate[6] * x130 + adjugate[7] * x129 + adjugate[8] * x131);
    S_ikmn_canonical[12] = x128 * (adjugate[0] * x133 + adjugate[1] * x132 + adjugate[2] * x134);
    S_ikmn_canonical[13] = x128 * (adjugate[3] * x133 + adjugate[4] * x132 + adjugate[5] * x134);
    S_ikmn_canonical[14] = x128 * (adjugate[6] * x133 + adjugate[7] * x132 + adjugate[8] * x134);
    S_ikmn_canonical[15] = x128 * (adjugate[0] * x151 + adjugate[1] * x163 + adjugate[2] * x160);
    S_ikmn_canonical[16] = x128 * (adjugate[3] * x151 + adjugate[4] * x163 + adjugate[5] * x160);
    S_ikmn_canonical[17] = x128 * (adjugate[6] * x151 + adjugate[7] * x163 + adjugate[8] * x160);
    S_ikmn_canonical[18] = x128 * (adjugate[0] * x164 + adjugate[1] * x166 + adjugate[2] * x165);
    S_ikmn_canonical[19] = x128 * (adjugate[3] * x164 + adjugate[4] * x166 + adjugate[5] * x165);
    S_ikmn_canonical[20] = x128 * (adjugate[6] * x164 + adjugate[7] * x166 + adjugate[8] * x165);
    S_ikmn_canonical[21] = x128 * (adjugate[0] * x167 + adjugate[1] * x169 + adjugate[2] * x168);
    S_ikmn_canonical[22] = x128 * (adjugate[3] * x167 + adjugate[4] * x169 + adjugate[5] * x168);
    S_ikmn_canonical[23] = x128 * (adjugate[6] * x167 + adjugate[7] * x169 + adjugate[8] * x168);
    S_ikmn_canonical[24] = x80 * (adjugate[0] * x177 + adjugate[1] * x186 + adjugate[2] * x183);
    S_ikmn_canonical[25] = x80 * (adjugate[3] * x177 + adjugate[4] * x186 + adjugate[5] * x183);
    S_ikmn_canonical[26] = x80 * (adjugate[6] * x177 + adjugate[7] * x186 + adjugate[8] * x183);
    S_ikmn_canonical[27] = x80 * (adjugate[3] * x187 + adjugate[4] * x189 + adjugate[5] * x188);
    S_ikmn_canonical[28] = x80 * (adjugate[6] * x187 + adjugate[7] * x189 + adjugate[8] * x188);
    S_ikmn_canonical[29] = x80 * (adjugate[6] * (adjugate[6] * x171 + adjugate[7] * x176 + adjugate[8] * x175) +
                                  adjugate[7] * (adjugate[6] * x176 + adjugate[7] * x185 + adjugate[8] * x182) +
                                  adjugate[8] * (adjugate[6] * x175 + adjugate[7] * x182 + adjugate[8] * x179));
    S_ikmn_canonical[30] = x128 * (adjugate[0] * x207 + adjugate[1] * x199 + adjugate[2] * x213);
    S_ikmn_canonical[31] = x128 * (adjugate[3] * x207 + adjugate[4] * x199 + adjugate[5] * x213);
    S_ikmn_canonical[32] = x128 * (adjugate[6] * x207 + adjugate[7] * x199 + adjugate[8] * x213);
    S_ikmn_canonical[33] = x128 * (adjugate[0] * x215 + adjugate[1] * x214 + adjugate[2] * x216);
    S_ikmn_canonical[34] = x128 * (adjugate[3] * x215 + adjugate[4] * x214 + adjugate[5] * x216);
    S_ikmn_canonical[35] = x128 * (adjugate[6] * x215 + adjugate[7] * x214 + adjugate[8] * x216);
    S_ikmn_canonical[36] = x128 * (adjugate[0] * x218 + adjugate[1] * x217 + adjugate[2] * x219);
    S_ikmn_canonical[37] = x128 * (adjugate[3] * x218 + adjugate[4] * x217 + adjugate[5] * x219);
    S_ikmn_canonical[38] = x128 * (adjugate[6] * x218 + adjugate[7] * x217 + adjugate[8] * x219);
    S_ikmn_canonical[39] = x80 * (adjugate[0] * x227 + adjugate[1] * x235 + adjugate[2] * x232);
    S_ikmn_canonical[40] = x80 * (adjugate[3] * x227 + adjugate[4] * x235 + adjugate[5] * x232);
    S_ikmn_canonical[41] = x80 * (adjugate[6] * x227 + adjugate[7] * x235 + adjugate[8] * x232);
    S_ikmn_canonical[42] = x80 * (adjugate[3] * x236 + adjugate[4] * x238 + adjugate[5] * x237);
    S_ikmn_canonical[43] = x80 * (adjugate[6] * x236 + adjugate[7] * x238 + adjugate[8] * x237);
    S_ikmn_canonical[44] = x80 * (adjugate[6] * (adjugate[6] * x221 + adjugate[7] * x226 + adjugate[8] * x224) +
                                  adjugate[7] * (adjugate[6] * x226 + adjugate[7] * x234 + adjugate[8] * x231) +
                                  adjugate[8] * (adjugate[6] * x224 + adjugate[7] * x231 + adjugate[8] * x229));
}

static SFEM_INLINE void tet4_apply_S_ikmn(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,  // 3x3x3x3, includes dV
                                          const scalar_t *const SFEM_RESTRICT inc_grad,          // 3x3 reference trial gradient R
                                          scalar_t *const SFEM_RESTRICT       eoutx,
                                          scalar_t *const SFEM_RESTRICT       eouty,
                                          scalar_t *const SFEM_RESTRICT       eoutz) {
    // mundane ops: 168 divs: 0 sqrts: 0
    // total ops: 168
    const scalar_t x0 = S_ikmn_canonical[0] * inc_grad[0] + S_ikmn_canonical[12] * inc_grad[5] +
                        S_ikmn_canonical[15] * inc_grad[6] + S_ikmn_canonical[18] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[1] + S_ikmn_canonical[21] * inc_grad[8] +
                        S_ikmn_canonical[2] * inc_grad[2] + S_ikmn_canonical[6] * inc_grad[3] + S_ikmn_canonical[9] * inc_grad[4];
    const scalar_t x1 = S_ikmn_canonical[10] * inc_grad[4] + S_ikmn_canonical[13] * inc_grad[5] +
                        S_ikmn_canonical[16] * inc_grad[6] + S_ikmn_canonical[19] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[8] +
                        S_ikmn_canonical[3] * inc_grad[1] + S_ikmn_canonical[4] * inc_grad[2] + S_ikmn_canonical[7] * inc_grad[3];
    const scalar_t x2 = S_ikmn_canonical[11] * inc_grad[4] + S_ikmn_canonical[14] * inc_grad[5] +
                        S_ikmn_canonical[17] * inc_grad[6] + S_ikmn_canonical[20] * inc_grad[7] +
                        S_ikmn_canonical[23] * inc_grad[8] + S_ikmn_canonical[2] * inc_grad[0] +
                        S_ikmn_canonical[4] * inc_grad[1] + S_ikmn_canonical[5] * inc_grad[2] + S_ikmn_canonical[8] * inc_grad[3];
    const scalar_t x3 = S_ikmn_canonical[24] * inc_grad[3] + S_ikmn_canonical[25] * inc_grad[4] +
                        S_ikmn_canonical[26] * inc_grad[5] + S_ikmn_canonical[30] * inc_grad[6] +
                        S_ikmn_canonical[33] * inc_grad[7] + S_ikmn_canonical[36] * inc_grad[8] +
                        S_ikmn_canonical[6] * inc_grad[0] + S_ikmn_canonical[7] * inc_grad[1] + S_ikmn_canonical[8] * inc_grad[2];
    const scalar_t x4 =
            S_ikmn_canonical[10] * inc_grad[1] + S_ikmn_canonical[11] * inc_grad[2] + S_ikmn_canonical[25] * inc_grad[3] +
            S_ikmn_canonical[27] * inc_grad[4] + S_ikmn_canonical[28] * inc_grad[5] + S_ikmn_canonical[31] * inc_grad[6] +
            S_ikmn_canonical[34] * inc_grad[7] + S_ikmn_canonical[37] * inc_grad[8] + S_ikmn_canonical[9] * inc_grad[0];
    const scalar_t x5 =
            S_ikmn_canonical[12] * inc_grad[0] + S_ikmn_canonical[13] * inc_grad[1] + S_ikmn_canonical[14] * inc_grad[2] +
            S_ikmn_canonical[26] * inc_grad[3] + S_ikmn_canonical[28] * inc_grad[4] + S_ikmn_canonical[29] * inc_grad[5] +
            S_ikmn_canonical[32] * inc_grad[6] + S_ikmn_canonical[35] * inc_grad[7] + S_ikmn_canonical[38] * inc_grad[8];
    const scalar_t x6 =
            S_ikmn_canonical[15] * inc_grad[0] + S_ikmn_canonical[16] * inc_grad[1] + S_ikmn_canonical[17] * inc_grad[2] +
            S_ikmn_canonical[30] * inc_grad[3] + S_ikmn_canonical[31] * inc_grad[4] + S_ikmn_canonical[32] * inc_grad[5] +
            S_ikmn_canonical[39] * inc_grad[6] + S_ikmn_canonical[40] * inc_grad[7] + S_ikmn_canonical[41] * inc_grad[8];
    const scalar_t x7 =
            S_ikmn_canonical[18] * inc_grad[0] + S_ikmn_canonical[19] * inc_grad[1] + S_ikmn_canonical[20] * inc_grad[2] +
            S_ikmn_canonical[33] * inc_grad[3] + S_ikmn_canonical[34] * inc_grad[4] + S_ikmn_canonical[35] * inc_grad[5] +
            S_ikmn_canonical[40] * inc_grad[6] + S_ikmn_canonical[42] * inc_grad[7] + S_ikmn_canonical[43] * inc_grad[8];
    const scalar_t x8 =
            S_ikmn_canonical[21] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[1] + S_ikmn_canonical[23] * inc_grad[2] +
            S_ikmn_canonical[36] * inc_grad[3] + S_ikmn_canonical[37] * inc_grad[4] + S_ikmn_canonical[38] * inc_grad[5] +
            S_ikmn_canonical[41] * inc_grad[6] + S_ikmn_canonical[43] * inc_grad[7] + S_ikmn_canonical[44] * inc_grad[8];
    eoutx[0] = -x0 - x1 - x2;
    eoutx[1] = x0;
    eoutx[2] = x1;
    eoutx[3] = x2;
    eouty[0] = -x3 - x4 - x5;
    eouty[1] = x3;
    eouty[2] = x4;
    eouty[3] = x5;
    eoutz[0] = -x6 - x7 - x8;
    eoutz[1] = x6;
    eoutz[2] = x7;
    eoutz[3] = x8;
}

#endif /* SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H */
