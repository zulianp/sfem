#ifndef SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H
#define SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H

static SFEM_INLINE void tet4_F(const scalar_t *const SFEM_RESTRICT adjugate,
                               const scalar_t                      jacobian_determinant,
                               const scalar_t *const SFEM_RESTRICT dispx,
                               const scalar_t *const SFEM_RESTRICT dispy,
                               const scalar_t *const SFEM_RESTRICT dispz,
                               scalar_t *const SFEM_RESTRICT       F) {
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

static SFEM_INLINE void tet4_ref_inc_grad(const scalar_t *const SFEM_RESTRICT incx,
                                          const scalar_t *const SFEM_RESTRICT incy,
                                          const scalar_t *const SFEM_RESTRICT incz,
                                          scalar_t *const SFEM_RESTRICT       inc_grad) {
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

static SFEM_INLINE void tet4_S_ikmn(const scalar_t *const SFEM_RESTRICT adjugate,
                                    const scalar_t                      jacobian_determinant,
                                    const scalar_t *const SFEM_RESTRICT F,
                                    const scalar_t                      mu,
                                    const scalar_t                      lmbda,
                                    const scalar_t                      qw,
                                    scalar_t *const SFEM_RESTRICT       S_ikmn) {
    scalar_t S_lin[81];  // Check if used in SSA mode
    {
        const scalar_t x0   = F[4] * F[8];
        const scalar_t x1   = F[5] * F[6];
        const scalar_t x2   = F[3] * F[7];
        const scalar_t x3   = F[5] * F[7];
        const scalar_t x4   = F[3] * F[8];
        const scalar_t x5   = F[4] * F[6];
        const scalar_t x6   = F[0] * x0 - F[0] * x3 + F[1] * x1 - F[1] * x4 + F[2] * x2 - F[2] * x5;
        const scalar_t x7   = 1.0 / x6;
        const scalar_t x8   = x0 - x3;
        const scalar_t x9   = mu * x8;
        const scalar_t x10  = mu * x6;
        const scalar_t x11  = lmbda * x7;
        const scalar_t x12  = lmbda * log(x6);
        const scalar_t x13  = F[0] * x10 + x12 * x8 - x9;
        const scalar_t x14  = x13 * x7;
        const scalar_t x15  = -x1 + x4;
        const scalar_t x16  = F[0] * mu;
        const scalar_t x17  = x11 * x8;
        const scalar_t x18  = -x13 * x7 + x16 + x17;
        const scalar_t x19  = -x18 * x7;
        const scalar_t x20  = x2 - x5;
        const scalar_t x21  = x18 * x7;
        const scalar_t x22  = F[1] * F[8] - F[2] * F[7];
        const scalar_t x23  = F[0] * F[8] - F[2] * F[6];
        const scalar_t x24  = F[8] * mu;
        const scalar_t x25  = F[8] * x12;
        const scalar_t x26  = x17 * x23 - x24 + x25;
        const scalar_t x27  = F[0] * F[7] - F[1] * F[6];
        const scalar_t x28  = F[7] * mu;
        const scalar_t x29  = -x28;
        const scalar_t x30  = F[7] * x12;
        const scalar_t x31  = x17 * x27 + x29 + x30;
        const scalar_t x32  = F[1] * F[5] - F[2] * F[4];
        const scalar_t x33  = F[0] * F[5] - F[2] * F[3];
        const scalar_t x34  = F[5] * mu;
        const scalar_t x35  = -x34;
        const scalar_t x36  = F[5] * x12;
        const scalar_t x37  = x17 * x33 + x35 + x36;
        const scalar_t x38  = F[0] * F[4] - F[1] * F[3];
        const scalar_t x39  = F[4] * mu;
        const scalar_t x40  = F[4] * x12;
        const scalar_t x41  = x17 * x38 - x39 + x40;
        const scalar_t x42  = F[1] * mu;
        const scalar_t x43  = -x42;
        const scalar_t x44  = x11 * x15;
        const scalar_t x45  = mu * x15;
        const scalar_t x46  = x7 * (F[1] * x10 - x12 * x15 + x45);
        const scalar_t x47  = x43 + x44 + x46;
        const scalar_t x48  = -x47 * x7;
        const scalar_t x49  = x22 * x44 + x24 - x25;
        const scalar_t x50  = F[6] * mu;
        const scalar_t x51  = F[6] * x12;
        const scalar_t x52  = x27 * x44 - x50 + x51;
        const scalar_t x53  = x32 * x44 + x34 - x36;
        const scalar_t x54  = x33 * x7;
        const scalar_t x55  = F[3] * mu;
        const scalar_t x56  = -x55;
        const scalar_t x57  = F[3] * x12;
        const scalar_t x58  = x38 * x44 + x56 + x57;
        const scalar_t x59  = F[2] * mu;
        const scalar_t x60  = x11 * x20;
        const scalar_t x61  = F[2] * x10 - mu * x20 + x12 * x20;
        const scalar_t x62  = x61 * x7;
        const scalar_t x63  = x59 + x60 - x62;
        const scalar_t x64  = x63 * x7;
        const scalar_t x65  = -x63 * x7;
        const scalar_t x66  = x22 * x60 + x28 - x30;
        const scalar_t x67  = x23 * x60 + x50 - x51;
        const scalar_t x68  = x32 * x60 + x39 - x40;
        const scalar_t x69  = x33 * x60 + x55 - x57;
        const scalar_t x70  = x11 * x22;
        const scalar_t x71  = x7 * (F[3] * x10 + mu * x22 - x12 * x22);
        const scalar_t x72  = x56 + x70 + x71;
        const scalar_t x73  = -x7 * x72;
        const scalar_t x74  = x27 * x7;
        const scalar_t x75  = F[2] * x12;
        const scalar_t x76  = x33 * x70 - x59 + x75;
        const scalar_t x77  = F[1] * x12;
        const scalar_t x78  = x38 * x70 + x43 + x77;
        const scalar_t x79  = F[4] * x10 - mu * x23 + x12 * x23;
        const scalar_t x80  = x7 * x79;
        const scalar_t x81  = x11 * x23;
        const scalar_t x82  = -x39 + x7 * x79 - x81;
        const scalar_t x83  = x7 * x82;
        const scalar_t x84  = x32 * x81 + x59 - x75;
        const scalar_t x85  = F[0] * x12;
        const scalar_t x86  = -x16 + x38 * x81 + x85;
        const scalar_t x87  = x7 * (F[5] * x10 + mu * x27 - x12 * x27);
        const scalar_t x88  = x11 * x27;
        const scalar_t x89  = x35 + x87 + x88;
        const scalar_t x90  = -x7 * x89;
        const scalar_t x91  = x22 * x7;
        const scalar_t x92  = x32 * x88 + x42 - x77;
        const scalar_t x93  = x16 + x33 * x88 - x85;
        const scalar_t x94  = F[6] * x10 - mu * x32 + x12 * x32;
        const scalar_t x95  = x7 * x94;
        const scalar_t x96  = x11 * x32 + x50 - x95;
        const scalar_t x97  = x7 * x96;
        const scalar_t x98  = -x96;
        const scalar_t x99  = x7 * (F[7] * x10 + mu * x33 - x12 * x33);
        const scalar_t x100 = x11 * x33 + x29 + x99;
        const scalar_t x101 = -x100 * x7;
        const scalar_t x102 = F[8] * x10 - mu * x38 + x12 * x38;
        const scalar_t x103 = x102 * x7;
        const scalar_t x104 = -x103 + x11 * x38 + x24;
        const scalar_t x105 = x104 * x7;
        const scalar_t x106 = -x104;
        S_lin[0]            = x7 * (F[0] * x9 + x10 + x11 * POW2(x8) - x14 * x8);
        S_lin[1]            = x15 * x19;
        S_lin[2]            = x20 * x21;
        S_lin[3]            = x19 * x22;
        S_lin[4]            = x7 * (-x14 * x23 + x16 * x23 + x26);
        S_lin[5]            = x7 * (x13 * x27 * x7 - x16 * x27 - x31);
        S_lin[6]            = x21 * x32;
        S_lin[7]            = x7 * (x13 * x33 * x7 - x16 * x33 - x37);
        S_lin[8]            = x7 * (-x14 * x38 + x16 * x38 + x41);
        S_lin[9]            = x48 * x8;
        S_lin[10]           = x7 * (x10 + x11 * POW2(x15) - x15 * x42 + x15 * x46);
        S_lin[11]           = x20 * x48;
        S_lin[12]           = x7 * (-x22 * x42 + x22 * x46 + x49);
        S_lin[13]           = x23 * x48;
        S_lin[14]           = x7 * (-x27 * x42 + x27 * x46 + x52);
        S_lin[15]           = x7 * (F[1] * mu * x32 - x32 * x46 - x53);
        S_lin[16]           = x47 * x54;
        S_lin[17]           = x7 * (F[1] * mu * x38 - x38 * x46 - x58);
        S_lin[18]           = x64 * x8;
        S_lin[19]           = x15 * x65;
        S_lin[20]           = x7 * (x10 + x11 * POW2(x20) + x20 * x59 - x20 * x62);
        S_lin[21]           = x7 * (-x22 * x59 + x22 * x61 * x7 - x66);
        S_lin[22]           = x7 * (x23 * x59 - x23 * x62 + x67);
        S_lin[23]           = x27 * x65;
        S_lin[24]           = x7 * (x32 * x59 - x32 * x62 + x68);
        S_lin[25]           = x7 * (-x33 * x59 + x33 * x61 * x7 - x69);
        S_lin[26]           = x38 * x64;
        S_lin[27]           = x73 * x8;
        S_lin[28]           = x7 * (-F[3] * x45 + x15 * x71 + x49);
        S_lin[29]           = x7 * (F[3] * mu * x20 - x20 * x71 - x66);
        S_lin[30]           = x7 * (x10 + x11 * POW2(x22) - x22 * x55 + x22 * x71);
        S_lin[31]           = x23 * x73;
        S_lin[32]           = x72 * x74;
        S_lin[33]           = x32 * x73;
        S_lin[34]           = x7 * (-x33 * x55 + x33 * x71 + x76);
        S_lin[35]           = -x7 * (-x38 * x55 + x38 * x71 + x78);
        S_lin[36]           = x7 * (F[4] * x9 + x26 - x8 * x80);
        S_lin[37]           = x15 * x83;
        S_lin[38]           = x7 * (x20 * x39 - x20 * x80 + x67);
        S_lin[39]           = x22 * x83;
        S_lin[40]           = x7 * (x10 + x11 * POW2(x23) + x23 * x39 - x23 * x80);
        S_lin[41]           = x74 * x82;
        S_lin[42]           = x7 * (x32 * x39 - x32 * x80 + x84);
        S_lin[43]           = x54 * x82;
        S_lin[44]           = x7 * (x38 * x39 - x38 * x80 + x86);
        S_lin[45]           = x7 * (F[5] * mu * x8 - x31 - x8 * x87);
        S_lin[46]           = x7 * (-x15 * x34 + x15 * x87 + x52);
        S_lin[47]           = x20 * x90;
        S_lin[48]           = x89 * x91;
        S_lin[49]           = x23 * x90;
        S_lin[50]           = x7 * (x10 + x11 * POW2(x27) - x27 * x34 + x27 * x87);
        S_lin[51]           = x7 * (F[5] * mu * x32 - x32 * x87 - x92);
        S_lin[52]           = x7 * (-x33 * x34 + x33 * x87 + x93);
        S_lin[53]           = x38 * x90;
        S_lin[54]           = x8 * x97;
        S_lin[55]           = x7 * (-F[6] * x45 + x15 * x7 * x94 - x53);
        S_lin[56]           = x7 * (x20 * x50 - x20 * x95 + x68);
        S_lin[57]           = x91 * x98;
        S_lin[58]           = x7 * (x23 * x50 - x23 * x95 + x84);
        S_lin[59]           = x7 * (-x27 * x50 + x27 * x7 * x94 - x92);
        S_lin[60]           = x7 * (x10 + x11 * POW2(x32) + x32 * x50 - x32 * x95);
        S_lin[61]           = x54 * x98;
        S_lin[62]           = x38 * x97;
        S_lin[63]           = -x7 * (-F[7] * x9 + x37 + x8 * x99);
        S_lin[64]           = x100 * x15 * x7;
        S_lin[65]           = x7 * (F[7] * mu * x20 - x20 * x99 - x69);
        S_lin[66]           = x7 * (-x22 * x28 + x22 * x99 + x76);
        S_lin[67]           = x101 * x23;
        S_lin[68]           = x7 * (-x27 * x28 + x27 * x99 + x93);
        S_lin[69]           = x101 * x32;
        S_lin[70]           = x7 * (x10 + x11 * POW2(x33) - x28 * x33 + x33 * x99);
        S_lin[71]           = x101 * x38;
        S_lin[72]           = x7 * (F[8] * x9 - x103 * x8 + x41);
        S_lin[73]           = x7 * (x102 * x15 * x7 - x15 * x24 - x58);
        S_lin[74]           = x105 * x20;
        S_lin[75]           = x7 * (x102 * x22 * x7 - x22 * x24 - x78);
        S_lin[76]           = x7 * (-x103 * x23 + x23 * x24 + x86);
        S_lin[77]           = x106 * x74;
        S_lin[78]           = x105 * x32;
        S_lin[79]           = x106 * x54;
        S_lin[80]           = x7 * (x10 - x103 * x38 + x11 * POW2(x38) + x24 * x38);
    }
    const scalar_t x0  = S_lin[0] * adjugate[0] + S_lin[1] * adjugate[1] + S_lin[2] * adjugate[2];
    const scalar_t x1  = S_lin[10] * adjugate[1] + S_lin[11] * adjugate[2] + S_lin[9] * adjugate[0];
    const scalar_t x2  = S_lin[18] * adjugate[0] + S_lin[19] * adjugate[1] + S_lin[20] * adjugate[2];
    const scalar_t x3  = (1.0 / 6.0) * qw / jacobian_determinant;
    const scalar_t x4  = S_lin[0] * adjugate[3] + S_lin[1] * adjugate[4] + S_lin[2] * adjugate[5];
    const scalar_t x5  = S_lin[10] * adjugate[4] + S_lin[11] * adjugate[5] + S_lin[9] * adjugate[3];
    const scalar_t x6  = S_lin[18] * adjugate[3] + S_lin[19] * adjugate[4] + S_lin[20] * adjugate[5];
    const scalar_t x7  = S_lin[0] * adjugate[6] + S_lin[1] * adjugate[7] + S_lin[2] * adjugate[8];
    const scalar_t x8  = S_lin[10] * adjugate[7] + S_lin[11] * adjugate[8] + S_lin[9] * adjugate[6];
    const scalar_t x9  = S_lin[18] * adjugate[6] + S_lin[19] * adjugate[7] + S_lin[20] * adjugate[8];
    const scalar_t x10 = S_lin[3] * adjugate[0] + S_lin[4] * adjugate[1] + S_lin[5] * adjugate[2];
    const scalar_t x11 = S_lin[12] * adjugate[0] + S_lin[13] * adjugate[1] + S_lin[14] * adjugate[2];
    const scalar_t x12 = S_lin[21] * adjugate[0] + S_lin[22] * adjugate[1] + S_lin[23] * adjugate[2];
    const scalar_t x13 = S_lin[3] * adjugate[3] + S_lin[4] * adjugate[4] + S_lin[5] * adjugate[5];
    const scalar_t x14 = S_lin[12] * adjugate[3] + S_lin[13] * adjugate[4] + S_lin[14] * adjugate[5];
    const scalar_t x15 = S_lin[21] * adjugate[3] + S_lin[22] * adjugate[4] + S_lin[23] * adjugate[5];
    const scalar_t x16 = S_lin[3] * adjugate[6] + S_lin[4] * adjugate[7] + S_lin[5] * adjugate[8];
    const scalar_t x17 = S_lin[12] * adjugate[6] + S_lin[13] * adjugate[7] + S_lin[14] * adjugate[8];
    const scalar_t x18 = S_lin[21] * adjugate[6] + S_lin[22] * adjugate[7] + S_lin[23] * adjugate[8];
    const scalar_t x19 = S_lin[6] * adjugate[0] + S_lin[7] * adjugate[1] + S_lin[8] * adjugate[2];
    const scalar_t x20 = S_lin[15] * adjugate[0] + S_lin[16] * adjugate[1] + S_lin[17] * adjugate[2];
    const scalar_t x21 = S_lin[24] * adjugate[0] + S_lin[25] * adjugate[1] + S_lin[26] * adjugate[2];
    const scalar_t x22 = S_lin[6] * adjugate[3] + S_lin[7] * adjugate[4] + S_lin[8] * adjugate[5];
    const scalar_t x23 = S_lin[15] * adjugate[3] + S_lin[16] * adjugate[4] + S_lin[17] * adjugate[5];
    const scalar_t x24 = S_lin[24] * adjugate[3] + S_lin[25] * adjugate[4] + S_lin[26] * adjugate[5];
    const scalar_t x25 = S_lin[6] * adjugate[6] + S_lin[7] * adjugate[7] + S_lin[8] * adjugate[8];
    const scalar_t x26 = S_lin[15] * adjugate[6] + S_lin[16] * adjugate[7] + S_lin[17] * adjugate[8];
    const scalar_t x27 = S_lin[24] * adjugate[6] + S_lin[25] * adjugate[7] + S_lin[26] * adjugate[8];
    const scalar_t x28 = S_lin[27] * adjugate[0] + S_lin[28] * adjugate[1] + S_lin[29] * adjugate[2];
    const scalar_t x29 = S_lin[36] * adjugate[0] + S_lin[37] * adjugate[1] + S_lin[38] * adjugate[2];
    const scalar_t x30 = S_lin[45] * adjugate[0] + S_lin[46] * adjugate[1] + S_lin[47] * adjugate[2];
    const scalar_t x31 = S_lin[27] * adjugate[3] + S_lin[28] * adjugate[4] + S_lin[29] * adjugate[5];
    const scalar_t x32 = S_lin[36] * adjugate[3] + S_lin[37] * adjugate[4] + S_lin[38] * adjugate[5];
    const scalar_t x33 = S_lin[45] * adjugate[3] + S_lin[46] * adjugate[4] + S_lin[47] * adjugate[5];
    const scalar_t x34 = S_lin[27] * adjugate[6] + S_lin[28] * adjugate[7] + S_lin[29] * adjugate[8];
    const scalar_t x35 = S_lin[36] * adjugate[6] + S_lin[37] * adjugate[7] + S_lin[38] * adjugate[8];
    const scalar_t x36 = S_lin[45] * adjugate[6] + S_lin[46] * adjugate[7] + S_lin[47] * adjugate[8];
    const scalar_t x37 = S_lin[30] * adjugate[0] + S_lin[31] * adjugate[1] + S_lin[32] * adjugate[2];
    const scalar_t x38 = S_lin[39] * adjugate[0] + S_lin[40] * adjugate[1] + S_lin[41] * adjugate[2];
    const scalar_t x39 = S_lin[48] * adjugate[0] + S_lin[49] * adjugate[1] + S_lin[50] * adjugate[2];
    const scalar_t x40 = S_lin[30] * adjugate[3] + S_lin[31] * adjugate[4] + S_lin[32] * adjugate[5];
    const scalar_t x41 = S_lin[39] * adjugate[3] + S_lin[40] * adjugate[4] + S_lin[41] * adjugate[5];
    const scalar_t x42 = S_lin[48] * adjugate[3] + S_lin[49] * adjugate[4] + S_lin[50] * adjugate[5];
    const scalar_t x43 = S_lin[30] * adjugate[6] + S_lin[31] * adjugate[7] + S_lin[32] * adjugate[8];
    const scalar_t x44 = S_lin[39] * adjugate[6] + S_lin[40] * adjugate[7] + S_lin[41] * adjugate[8];
    const scalar_t x45 = S_lin[48] * adjugate[6] + S_lin[49] * adjugate[7] + S_lin[50] * adjugate[8];
    const scalar_t x46 = S_lin[33] * adjugate[0] + S_lin[34] * adjugate[1] + S_lin[35] * adjugate[2];
    const scalar_t x47 = S_lin[42] * adjugate[0] + S_lin[43] * adjugate[1] + S_lin[44] * adjugate[2];
    const scalar_t x48 = S_lin[51] * adjugate[0] + S_lin[52] * adjugate[1] + S_lin[53] * adjugate[2];
    const scalar_t x49 = S_lin[33] * adjugate[3] + S_lin[34] * adjugate[4] + S_lin[35] * adjugate[5];
    const scalar_t x50 = S_lin[42] * adjugate[3] + S_lin[43] * adjugate[4] + S_lin[44] * adjugate[5];
    const scalar_t x51 = S_lin[51] * adjugate[3] + S_lin[52] * adjugate[4] + S_lin[53] * adjugate[5];
    const scalar_t x52 = S_lin[33] * adjugate[6] + S_lin[34] * adjugate[7] + S_lin[35] * adjugate[8];
    const scalar_t x53 = S_lin[42] * adjugate[6] + S_lin[43] * adjugate[7] + S_lin[44] * adjugate[8];
    const scalar_t x54 = S_lin[51] * adjugate[6] + S_lin[52] * adjugate[7] + S_lin[53] * adjugate[8];
    const scalar_t x55 = S_lin[54] * adjugate[0] + S_lin[55] * adjugate[1] + S_lin[56] * adjugate[2];
    const scalar_t x56 = S_lin[63] * adjugate[0] + S_lin[64] * adjugate[1] + S_lin[65] * adjugate[2];
    const scalar_t x57 = S_lin[72] * adjugate[0] + S_lin[73] * adjugate[1] + S_lin[74] * adjugate[2];
    const scalar_t x58 = S_lin[54] * adjugate[3] + S_lin[55] * adjugate[4] + S_lin[56] * adjugate[5];
    const scalar_t x59 = S_lin[63] * adjugate[3] + S_lin[64] * adjugate[4] + S_lin[65] * adjugate[5];
    const scalar_t x60 = S_lin[72] * adjugate[3] + S_lin[73] * adjugate[4] + S_lin[74] * adjugate[5];
    const scalar_t x61 = S_lin[54] * adjugate[6] + S_lin[55] * adjugate[7] + S_lin[56] * adjugate[8];
    const scalar_t x62 = S_lin[63] * adjugate[6] + S_lin[64] * adjugate[7] + S_lin[65] * adjugate[8];
    const scalar_t x63 = S_lin[72] * adjugate[6] + S_lin[73] * adjugate[7] + S_lin[74] * adjugate[8];
    const scalar_t x64 = S_lin[57] * adjugate[0] + S_lin[58] * adjugate[1] + S_lin[59] * adjugate[2];
    const scalar_t x65 = S_lin[66] * adjugate[0] + S_lin[67] * adjugate[1] + S_lin[68] * adjugate[2];
    const scalar_t x66 = S_lin[75] * adjugate[0] + S_lin[76] * adjugate[1] + S_lin[77] * adjugate[2];
    const scalar_t x67 = S_lin[57] * adjugate[3] + S_lin[58] * adjugate[4] + S_lin[59] * adjugate[5];
    const scalar_t x68 = S_lin[66] * adjugate[3] + S_lin[67] * adjugate[4] + S_lin[68] * adjugate[5];
    const scalar_t x69 = S_lin[75] * adjugate[3] + S_lin[76] * adjugate[4] + S_lin[77] * adjugate[5];
    const scalar_t x70 = S_lin[57] * adjugate[6] + S_lin[58] * adjugate[7] + S_lin[59] * adjugate[8];
    const scalar_t x71 = S_lin[66] * adjugate[6] + S_lin[67] * adjugate[7] + S_lin[68] * adjugate[8];
    const scalar_t x72 = S_lin[75] * adjugate[6] + S_lin[76] * adjugate[7] + S_lin[77] * adjugate[8];
    const scalar_t x73 = S_lin[60] * adjugate[0] + S_lin[61] * adjugate[1] + S_lin[62] * adjugate[2];
    const scalar_t x74 = S_lin[69] * adjugate[0] + S_lin[70] * adjugate[1] + S_lin[71] * adjugate[2];
    const scalar_t x75 = S_lin[78] * adjugate[0] + S_lin[79] * adjugate[1] + S_lin[80] * adjugate[2];
    const scalar_t x76 = S_lin[60] * adjugate[3] + S_lin[61] * adjugate[4] + S_lin[62] * adjugate[5];
    const scalar_t x77 = S_lin[69] * adjugate[3] + S_lin[70] * adjugate[4] + S_lin[71] * adjugate[5];
    const scalar_t x78 = S_lin[78] * adjugate[3] + S_lin[79] * adjugate[4] + S_lin[80] * adjugate[5];
    const scalar_t x79 = S_lin[60] * adjugate[6] + S_lin[61] * adjugate[7] + S_lin[62] * adjugate[8];
    const scalar_t x80 = S_lin[69] * adjugate[6] + S_lin[70] * adjugate[7] + S_lin[71] * adjugate[8];
    const scalar_t x81 = S_lin[78] * adjugate[6] + S_lin[79] * adjugate[7] + S_lin[80] * adjugate[8];
    S_ikmn[0]          = x3 * (adjugate[0] * x0 + adjugate[1] * x1 + adjugate[2] * x2);
    S_ikmn[1]          = x3 * (adjugate[0] * x4 + adjugate[1] * x5 + adjugate[2] * x6);
    S_ikmn[2]          = x3 * (adjugate[0] * x7 + adjugate[1] * x8 + adjugate[2] * x9);
    S_ikmn[3]          = x3 * (adjugate[0] * x10 + adjugate[1] * x11 + adjugate[2] * x12);
    S_ikmn[4]          = x3 * (adjugate[0] * x13 + adjugate[1] * x14 + adjugate[2] * x15);
    S_ikmn[5]          = x3 * (adjugate[0] * x16 + adjugate[1] * x17 + adjugate[2] * x18);
    S_ikmn[6]          = x3 * (adjugate[0] * x19 + adjugate[1] * x20 + adjugate[2] * x21);
    S_ikmn[7]          = x3 * (adjugate[0] * x22 + adjugate[1] * x23 + adjugate[2] * x24);
    S_ikmn[8]          = x3 * (adjugate[0] * x25 + adjugate[1] * x26 + adjugate[2] * x27);
    S_ikmn[9]          = x3 * (adjugate[3] * x0 + adjugate[4] * x1 + adjugate[5] * x2);
    S_ikmn[10]         = x3 * (adjugate[3] * x4 + adjugate[4] * x5 + adjugate[5] * x6);
    S_ikmn[11]         = x3 * (adjugate[3] * x7 + adjugate[4] * x8 + adjugate[5] * x9);
    S_ikmn[12]         = x3 * (adjugate[3] * x10 + adjugate[4] * x11 + adjugate[5] * x12);
    S_ikmn[13]         = x3 * (adjugate[3] * x13 + adjugate[4] * x14 + adjugate[5] * x15);
    S_ikmn[14]         = x3 * (adjugate[3] * x16 + adjugate[4] * x17 + adjugate[5] * x18);
    S_ikmn[15]         = x3 * (adjugate[3] * x19 + adjugate[4] * x20 + adjugate[5] * x21);
    S_ikmn[16]         = x3 * (adjugate[3] * x22 + adjugate[4] * x23 + adjugate[5] * x24);
    S_ikmn[17]         = x3 * (adjugate[3] * x25 + adjugate[4] * x26 + adjugate[5] * x27);
    S_ikmn[18]         = x3 * (adjugate[6] * x0 + adjugate[7] * x1 + adjugate[8] * x2);
    S_ikmn[19]         = x3 * (adjugate[6] * x4 + adjugate[7] * x5 + adjugate[8] * x6);
    S_ikmn[20]         = x3 * (adjugate[6] * x7 + adjugate[7] * x8 + adjugate[8] * x9);
    S_ikmn[21]         = x3 * (adjugate[6] * x10 + adjugate[7] * x11 + adjugate[8] * x12);
    S_ikmn[22]         = x3 * (adjugate[6] * x13 + adjugate[7] * x14 + adjugate[8] * x15);
    S_ikmn[23]         = x3 * (adjugate[6] * x16 + adjugate[7] * x17 + adjugate[8] * x18);
    S_ikmn[24]         = x3 * (adjugate[6] * x19 + adjugate[7] * x20 + adjugate[8] * x21);
    S_ikmn[25]         = x3 * (adjugate[6] * x22 + adjugate[7] * x23 + adjugate[8] * x24);
    S_ikmn[26]         = x3 * (adjugate[6] * x25 + adjugate[7] * x26 + adjugate[8] * x27);
    S_ikmn[27]         = x3 * (adjugate[0] * x28 + adjugate[1] * x29 + adjugate[2] * x30);
    S_ikmn[28]         = x3 * (adjugate[0] * x31 + adjugate[1] * x32 + adjugate[2] * x33);
    S_ikmn[29]         = x3 * (adjugate[0] * x34 + adjugate[1] * x35 + adjugate[2] * x36);
    S_ikmn[30]         = x3 * (adjugate[0] * x37 + adjugate[1] * x38 + adjugate[2] * x39);
    S_ikmn[31]         = x3 * (adjugate[0] * x40 + adjugate[1] * x41 + adjugate[2] * x42);
    S_ikmn[32]         = x3 * (adjugate[0] * x43 + adjugate[1] * x44 + adjugate[2] * x45);
    S_ikmn[33]         = x3 * (adjugate[0] * x46 + adjugate[1] * x47 + adjugate[2] * x48);
    S_ikmn[34]         = x3 * (adjugate[0] * x49 + adjugate[1] * x50 + adjugate[2] * x51);
    S_ikmn[35]         = x3 * (adjugate[0] * x52 + adjugate[1] * x53 + adjugate[2] * x54);
    S_ikmn[36]         = x3 * (adjugate[3] * x28 + adjugate[4] * x29 + adjugate[5] * x30);
    S_ikmn[37]         = x3 * (adjugate[3] * x31 + adjugate[4] * x32 + adjugate[5] * x33);
    S_ikmn[38]         = x3 * (adjugate[3] * x34 + adjugate[4] * x35 + adjugate[5] * x36);
    S_ikmn[39]         = x3 * (adjugate[3] * x37 + adjugate[4] * x38 + adjugate[5] * x39);
    S_ikmn[40]         = x3 * (adjugate[3] * x40 + adjugate[4] * x41 + adjugate[5] * x42);
    S_ikmn[41]         = x3 * (adjugate[3] * x43 + adjugate[4] * x44 + adjugate[5] * x45);
    S_ikmn[42]         = x3 * (adjugate[3] * x46 + adjugate[4] * x47 + adjugate[5] * x48);
    S_ikmn[43]         = x3 * (adjugate[3] * x49 + adjugate[4] * x50 + adjugate[5] * x51);
    S_ikmn[44]         = x3 * (adjugate[3] * x52 + adjugate[4] * x53 + adjugate[5] * x54);
    S_ikmn[45]         = x3 * (adjugate[6] * x28 + adjugate[7] * x29 + adjugate[8] * x30);
    S_ikmn[46]         = x3 * (adjugate[6] * x31 + adjugate[7] * x32 + adjugate[8] * x33);
    S_ikmn[47]         = x3 * (adjugate[6] * x34 + adjugate[7] * x35 + adjugate[8] * x36);
    S_ikmn[48]         = x3 * (adjugate[6] * x37 + adjugate[7] * x38 + adjugate[8] * x39);
    S_ikmn[49]         = x3 * (adjugate[6] * x40 + adjugate[7] * x41 + adjugate[8] * x42);
    S_ikmn[50]         = x3 * (adjugate[6] * x43 + adjugate[7] * x44 + adjugate[8] * x45);
    S_ikmn[51]         = x3 * (adjugate[6] * x46 + adjugate[7] * x47 + adjugate[8] * x48);
    S_ikmn[52]         = x3 * (adjugate[6] * x49 + adjugate[7] * x50 + adjugate[8] * x51);
    S_ikmn[53]         = x3 * (adjugate[6] * x52 + adjugate[7] * x53 + adjugate[8] * x54);
    S_ikmn[54]         = x3 * (adjugate[0] * x55 + adjugate[1] * x56 + adjugate[2] * x57);
    S_ikmn[55]         = x3 * (adjugate[0] * x58 + adjugate[1] * x59 + adjugate[2] * x60);
    S_ikmn[56]         = x3 * (adjugate[0] * x61 + adjugate[1] * x62 + adjugate[2] * x63);
    S_ikmn[57]         = x3 * (adjugate[0] * x64 + adjugate[1] * x65 + adjugate[2] * x66);
    S_ikmn[58]         = x3 * (adjugate[0] * x67 + adjugate[1] * x68 + adjugate[2] * x69);
    S_ikmn[59]         = x3 * (adjugate[0] * x70 + adjugate[1] * x71 + adjugate[2] * x72);
    S_ikmn[60]         = x3 * (adjugate[0] * x73 + adjugate[1] * x74 + adjugate[2] * x75);
    S_ikmn[61]         = x3 * (adjugate[0] * x76 + adjugate[1] * x77 + adjugate[2] * x78);
    S_ikmn[62]         = x3 * (adjugate[0] * x79 + adjugate[1] * x80 + adjugate[2] * x81);
    S_ikmn[63]         = x3 * (adjugate[3] * x55 + adjugate[4] * x56 + adjugate[5] * x57);
    S_ikmn[64]         = x3 * (adjugate[3] * x58 + adjugate[4] * x59 + adjugate[5] * x60);
    S_ikmn[65]         = x3 * (adjugate[3] * x61 + adjugate[4] * x62 + adjugate[5] * x63);
    S_ikmn[66]         = x3 * (adjugate[3] * x64 + adjugate[4] * x65 + adjugate[5] * x66);
    S_ikmn[67]         = x3 * (adjugate[3] * x67 + adjugate[4] * x68 + adjugate[5] * x69);
    S_ikmn[68]         = x3 * (adjugate[3] * x70 + adjugate[4] * x71 + adjugate[5] * x72);
    S_ikmn[69]         = x3 * (adjugate[3] * x73 + adjugate[4] * x74 + adjugate[5] * x75);
    S_ikmn[70]         = x3 * (adjugate[3] * x76 + adjugate[4] * x77 + adjugate[5] * x78);
    S_ikmn[71]         = x3 * (adjugate[3] * x79 + adjugate[4] * x80 + adjugate[5] * x81);
    S_ikmn[72]         = x3 * (adjugate[6] * x55 + adjugate[7] * x56 + adjugate[8] * x57);
    S_ikmn[73]         = x3 * (adjugate[6] * x58 + adjugate[7] * x59 + adjugate[8] * x60);
    S_ikmn[74]         = x3 * (adjugate[6] * x61 + adjugate[7] * x62 + adjugate[8] * x63);
    S_ikmn[75]         = x3 * (adjugate[6] * x64 + adjugate[7] * x65 + adjugate[8] * x66);
    S_ikmn[76]         = x3 * (adjugate[6] * x67 + adjugate[7] * x68 + adjugate[8] * x69);
    S_ikmn[77]         = x3 * (adjugate[6] * x70 + adjugate[7] * x71 + adjugate[8] * x72);
    S_ikmn[78]         = x3 * (adjugate[6] * x73 + adjugate[7] * x74 + adjugate[8] * x75);
    S_ikmn[79]         = x3 * (adjugate[6] * x76 + adjugate[7] * x77 + adjugate[8] * x78);
    S_ikmn[80]         = x3 * (adjugate[6] * x79 + adjugate[7] * x80 + adjugate[8] * x81);
}

static SFEM_INLINE void tet4_apply_S_ikmn(const scalar_t *const SFEM_RESTRICT S_ikmn,    // 3x3x3x3, includes dV
                                          const scalar_t *const SFEM_RESTRICT inc_grad,  // 3x3 reference trial gradient R
                                          scalar_t *const SFEM_RESTRICT       eoutx,
                                          scalar_t *const SFEM_RESTRICT       eouty,
                                          scalar_t *const SFEM_RESTRICT       eoutz) {
    scalar_t SdotH_km[9];
    SdotH_km[0] = S_ikmn[0] * inc_grad[0] + S_ikmn[1] * inc_grad[1] + S_ikmn[27] * inc_grad[3] + S_ikmn[28] * inc_grad[4] +
                  S_ikmn[29] * inc_grad[5] + S_ikmn[2] * inc_grad[2] + S_ikmn[54] * inc_grad[6] + S_ikmn[55] * inc_grad[7] +
                  S_ikmn[56] * inc_grad[8];
    SdotH_km[1] = S_ikmn[30] * inc_grad[3] + S_ikmn[31] * inc_grad[4] + S_ikmn[32] * inc_grad[5] + S_ikmn[3] * inc_grad[0] +
                  S_ikmn[4] * inc_grad[1] + S_ikmn[57] * inc_grad[6] + S_ikmn[58] * inc_grad[7] + S_ikmn[59] * inc_grad[8] +
                  S_ikmn[5] * inc_grad[2];
    SdotH_km[2] = S_ikmn[33] * inc_grad[3] + S_ikmn[34] * inc_grad[4] + S_ikmn[35] * inc_grad[5] + S_ikmn[60] * inc_grad[6] +
                  S_ikmn[61] * inc_grad[7] + S_ikmn[62] * inc_grad[8] + S_ikmn[6] * inc_grad[0] + S_ikmn[7] * inc_grad[1] +
                  S_ikmn[8] * inc_grad[2];
    SdotH_km[3] = S_ikmn[10] * inc_grad[1] + S_ikmn[11] * inc_grad[2] + S_ikmn[36] * inc_grad[3] + S_ikmn[37] * inc_grad[4] +
                  S_ikmn[38] * inc_grad[5] + S_ikmn[63] * inc_grad[6] + S_ikmn[64] * inc_grad[7] + S_ikmn[65] * inc_grad[8] +
                  S_ikmn[9] * inc_grad[0];
    SdotH_km[4] = S_ikmn[12] * inc_grad[0] + S_ikmn[13] * inc_grad[1] + S_ikmn[14] * inc_grad[2] + S_ikmn[39] * inc_grad[3] +
                  S_ikmn[40] * inc_grad[4] + S_ikmn[41] * inc_grad[5] + S_ikmn[66] * inc_grad[6] + S_ikmn[67] * inc_grad[7] +
                  S_ikmn[68] * inc_grad[8];
    SdotH_km[5] = S_ikmn[15] * inc_grad[0] + S_ikmn[16] * inc_grad[1] + S_ikmn[17] * inc_grad[2] + S_ikmn[42] * inc_grad[3] +
                  S_ikmn[43] * inc_grad[4] + S_ikmn[44] * inc_grad[5] + S_ikmn[69] * inc_grad[6] + S_ikmn[70] * inc_grad[7] +
                  S_ikmn[71] * inc_grad[8];
    SdotH_km[6] = S_ikmn[18] * inc_grad[0] + S_ikmn[19] * inc_grad[1] + S_ikmn[20] * inc_grad[2] + S_ikmn[45] * inc_grad[3] +
                  S_ikmn[46] * inc_grad[4] + S_ikmn[47] * inc_grad[5] + S_ikmn[72] * inc_grad[6] + S_ikmn[73] * inc_grad[7] +
                  S_ikmn[74] * inc_grad[8];
    SdotH_km[7] = S_ikmn[21] * inc_grad[0] + S_ikmn[22] * inc_grad[1] + S_ikmn[23] * inc_grad[2] + S_ikmn[48] * inc_grad[3] +
                  S_ikmn[49] * inc_grad[4] + S_ikmn[50] * inc_grad[5] + S_ikmn[75] * inc_grad[6] + S_ikmn[76] * inc_grad[7] +
                  S_ikmn[77] * inc_grad[8];
    SdotH_km[8] = S_ikmn[24] * inc_grad[0] + S_ikmn[25] * inc_grad[1] + S_ikmn[26] * inc_grad[2] + S_ikmn[51] * inc_grad[3] +
                  S_ikmn[52] * inc_grad[4] + S_ikmn[53] * inc_grad[5] + S_ikmn[78] * inc_grad[6] + S_ikmn[79] * inc_grad[7] +
                  S_ikmn[80] * inc_grad[8];
    eoutx[0] = -SdotH_km[0] - SdotH_km[1] - SdotH_km[2];
    eoutx[1] = SdotH_km[0];
    eoutx[2] = SdotH_km[1];
    eoutx[3] = SdotH_km[2];
    eouty[0] = -SdotH_km[3] - SdotH_km[4] - SdotH_km[5];
    eouty[1] = SdotH_km[3];
    eouty[2] = SdotH_km[4];
    eouty[3] = SdotH_km[5];
    eoutz[0] = -SdotH_km[6] - SdotH_km[7] - SdotH_km[8];
    eoutz[1] = SdotH_km[6];
    eoutz[2] = SdotH_km[7];
    eoutz[3] = SdotH_km[8];
}

#endif /* SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H */
