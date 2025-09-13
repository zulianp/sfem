#ifndef SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H
#define SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H

static SFEM_INLINE void hex8_F(const scalar_t *const SFEM_RESTRICT adjugate,
                               const scalar_t                      jacobian_determinant,
                               const scalar_t                      qx,
                               const scalar_t                      qy,
                               const scalar_t                      qz,
                               const scalar_t *const SFEM_RESTRICT dispx,
                               const scalar_t *const SFEM_RESTRICT dispy,
                               const scalar_t *const SFEM_RESTRICT dispz,
                               scalar_t *const SFEM_RESTRICT       F) {
    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = qy * qz;
    const scalar_t x2 = 1 - qz;
    const scalar_t x3 = qy * x2;
    const scalar_t x4 = 1 - qy;
    const scalar_t x5 = qz * x4;
    const scalar_t x6 = x2 * x4;
    const scalar_t x7 = dispx[0] * x6 - dispx[1] * x6 - dispx[2] * x3 + dispx[3] * x3 + dispx[4] * x5 - dispx[5] * x5 -
                        dispx[6] * x1 + dispx[7] * x1;
    const scalar_t x8  = qx * qz;
    const scalar_t x9  = qx * x2;
    const scalar_t x10 = 1 - qx;
    const scalar_t x11 = qz * x10;
    const scalar_t x12 = x10 * x2;
    const scalar_t x13 = dispx[0] * x12 + dispx[1] * x9 - dispx[2] * x9 - dispx[3] * x12 + dispx[4] * x11 + dispx[5] * x8 -
                         dispx[6] * x8 - dispx[7] * x11;
    const scalar_t x14 = qx * qy;
    const scalar_t x15 = qx * x4;
    const scalar_t x16 = qy * x10;
    const scalar_t x17 = x10 * x4;
    const scalar_t x18 = dispx[0] * x17 + dispx[1] * x15 + dispx[2] * x14 + dispx[3] * x16 - dispx[4] * x17 - dispx[5] * x15 -
                         dispx[6] * x14 - dispx[7] * x16;
    const scalar_t x19 = dispy[0] * x6 - dispy[1] * x6 - dispy[2] * x3 + dispy[3] * x3 + dispy[4] * x5 - dispy[5] * x5 -
                         dispy[6] * x1 + dispy[7] * x1;
    const scalar_t x20 = dispy[0] * x12 + dispy[1] * x9 - dispy[2] * x9 - dispy[3] * x12 + dispy[4] * x11 + dispy[5] * x8 -
                         dispy[6] * x8 - dispy[7] * x11;
    const scalar_t x21 = dispy[0] * x17 + dispy[1] * x15 + dispy[2] * x14 + dispy[3] * x16 - dispy[4] * x17 - dispy[5] * x15 -
                         dispy[6] * x14 - dispy[7] * x16;
    const scalar_t x22 = dispz[0] * x6 - dispz[1] * x6 - dispz[2] * x3 + dispz[3] * x3 + dispz[4] * x5 - dispz[5] * x5 -
                         dispz[6] * x1 + dispz[7] * x1;
    const scalar_t x23 = dispz[0] * x12 + dispz[1] * x9 - dispz[2] * x9 - dispz[3] * x12 + dispz[4] * x11 + dispz[5] * x8 -
                         dispz[6] * x8 - dispz[7] * x11;
    const scalar_t x24 = dispz[0] * x17 + dispz[1] * x15 + dispz[2] * x14 + dispz[3] * x16 - dispz[4] * x17 - dispz[5] * x15 -
                         dispz[6] * x14 - dispz[7] * x16;
    F[0] = -adjugate[0] * x0 * x7 - adjugate[3] * x0 * x13 - adjugate[6] * x0 * x18 + 1;
    F[1] = -x0 * (adjugate[1] * x7 + adjugate[4] * x13 + adjugate[7] * x18);
    F[2] = -x0 * (adjugate[2] * x7 + adjugate[5] * x13 + adjugate[8] * x18);
    F[3] = -x0 * (adjugate[0] * x19 + adjugate[3] * x20 + adjugate[6] * x21);
    F[4] = -adjugate[1] * x0 * x19 - adjugate[4] * x0 * x20 - adjugate[7] * x0 * x21 + 1;
    F[5] = -x0 * (adjugate[2] * x19 + adjugate[5] * x20 + adjugate[8] * x21);
    F[6] = -x0 * (adjugate[0] * x22 + adjugate[3] * x23 + adjugate[6] * x24);
    F[7] = -x0 * (adjugate[1] * x22 + adjugate[4] * x23 + adjugate[7] * x24);
    F[8] = -adjugate[2] * x0 * x22 - adjugate[5] * x0 * x23 - adjugate[8] * x0 * x24 + 1;
}

static SFEM_INLINE void hex8_ref_inc_grad(const scalar_t                      qx,
                                          const scalar_t                      qy,
                                          const scalar_t                      qz,
                                          const scalar_t *const SFEM_RESTRICT incx,
                                          const scalar_t *const SFEM_RESTRICT incy,
                                          const scalar_t *const SFEM_RESTRICT incz,
                                          scalar_t *const SFEM_RESTRICT       inc_grad) {
    const scalar_t x0  = qy * qz;
    const scalar_t x1  = qz - 1;
    const scalar_t x2  = qy * x1;
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qz * x3;
    const scalar_t x5  = x1 * x3;
    const scalar_t x6  = qx * qz;
    const scalar_t x7  = qx * x1;
    const scalar_t x8  = qx - 1;
    const scalar_t x9  = qz * x8;
    const scalar_t x10 = x1 * x8;
    const scalar_t x11 = qx * qy;
    const scalar_t x12 = qx * x3;
    const scalar_t x13 = qy * x8;
    const scalar_t x14 = x3 * x8;
    inc_grad[0]        = -incx[0] * x5 + incx[1] * x1 * x3 - incx[2] * x2 + incx[3] * qy * x1 + incx[4] * qz * x3 - incx[5] * x4 +
                  incx[6] * qy * qz - incx[7] * x0;
    inc_grad[1] = -incx[0] * x10 + incx[1] * qx * x1 - incx[2] * x7 + incx[3] * x1 * x8 + incx[4] * qz * x8 - incx[5] * x6 +
                  incx[6] * qx * qz - incx[7] * x9;
    inc_grad[2] = -incx[0] * x14 + incx[1] * qx * x3 - incx[2] * x11 + incx[3] * qy * x8 + incx[4] * x3 * x8 - incx[5] * x12 +
                  incx[6] * qx * qy - incx[7] * x13;
    inc_grad[3] = -incy[0] * x5 + incy[1] * x1 * x3 - incy[2] * x2 + incy[3] * qy * x1 + incy[4] * qz * x3 - incy[5] * x4 +
                  incy[6] * qy * qz - incy[7] * x0;
    inc_grad[4] = -incy[0] * x10 + incy[1] * qx * x1 - incy[2] * x7 + incy[3] * x1 * x8 + incy[4] * qz * x8 - incy[5] * x6 +
                  incy[6] * qx * qz - incy[7] * x9;
    inc_grad[5] = -incy[0] * x14 + incy[1] * qx * x3 - incy[2] * x11 + incy[3] * qy * x8 + incy[4] * x3 * x8 - incy[5] * x12 +
                  incy[6] * qx * qy - incy[7] * x13;
    inc_grad[6] = -incz[0] * x5 + incz[1] * x1 * x3 - incz[2] * x2 + incz[3] * qy * x1 + incz[4] * qz * x3 - incz[5] * x4 +
                  incz[6] * qy * qz - incz[7] * x0;
    inc_grad[7] = -incz[0] * x10 + incz[1] * qx * x1 - incz[2] * x7 + incz[3] * x1 * x8 + incz[4] * qz * x8 - incz[5] * x6 +
                  incz[6] * qx * qz - incz[7] * x9;
    inc_grad[8] = -incz[0] * x14 + incz[1] * qx * x3 - incz[2] * x11 + incz[3] * qy * x8 + incz[4] * x3 * x8 - incz[5] * x12 +
                  incz[6] * qx * qy - incz[7] * x13;
}

#define HEX8_S_IKMN_SIZE 45
static SFEM_INLINE void hex8_S_ikmn_neohookean(const scalar_t *const SFEM_RESTRICT adjugate,
                                               const scalar_t                      jacobian_determinant,
                                               const scalar_t                      qx,
                                               const scalar_t                      qy,
                                               const scalar_t                      qz,
                                               const scalar_t *const SFEM_RESTRICT F,
                                               const scalar_t                      mu,
                                               const scalar_t                      lmbda,
                                               const scalar_t                      qw,
                                               scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    const scalar_t x0    = F[4] * F[8];
    const scalar_t x1    = F[5] * F[6];
    const scalar_t x2    = F[3] * F[7];
    const scalar_t x3    = F[5] * F[7];
    const scalar_t x4    = F[3] * F[8];
    const scalar_t x5    = F[4] * F[6];
    const scalar_t x6    = F[0] * x0 - F[0] * x3 + F[1] * x1 - F[1] * x4 + F[2] * x2 - F[2] * x5;
    const scalar_t x7    = (1 / POW2(x6));
    const scalar_t x8    = x0 - x3;
    const scalar_t x9    = x7 * x8;
    const scalar_t x10   = x2 - x5;
    const scalar_t x11   = lmbda * log(x6);
    const scalar_t x12   = lmbda + mu - x11;
    const scalar_t x13   = adjugate[2] * x12;
    const scalar_t x14   = x10 * x13;
    const scalar_t x15   = -x1 + x4;
    const scalar_t x16   = x12 * x9;
    const scalar_t x17   = x15 * x16;
    const scalar_t x18   = x7 * POW2(x8);
    const scalar_t x19   = lmbda * x18 + mu * x18 + mu - x11 * x18;
    const scalar_t x20   = adjugate[0] * x19 - adjugate[1] * x17 + x14 * x9;
    const scalar_t x21   = x10 * x16;
    const scalar_t x22   = adjugate[1] * x12;
    const scalar_t x23   = x15 * x7;
    const scalar_t x24   = x10 * x23;
    const scalar_t x25   = POW2(x10) * x7;
    const scalar_t x26   = lmbda * x25 + mu * x25 + mu - x11 * x25;
    const scalar_t x27   = adjugate[0] * x21 + adjugate[2] * x26 - x22 * x24;
    const scalar_t x28   = POW2(x15) * x7;
    const scalar_t x29   = lmbda * x28 + mu * x28 + mu - x11 * x28;
    const scalar_t x30   = -adjugate[0] * x17 + adjugate[1] * x29 - x14 * x23;
    const scalar_t x31   = qw / jacobian_determinant;
    const scalar_t x32   = adjugate[3] * x19 - adjugate[4] * x17 + adjugate[5] * x21;
    const scalar_t x33   = x12 * x24;
    const scalar_t x34   = adjugate[3] * x21 - adjugate[4] * x33 + adjugate[5] * x26;
    const scalar_t x35   = -adjugate[3] * x17 + adjugate[4] * x29 - adjugate[5] * x33;
    const scalar_t x36   = F[0] * F[8] - F[2] * F[6];
    const scalar_t x37   = 1.0 / x6;
    const scalar_t x38   = x15 * x37;
    const scalar_t x39   = x36 * x38;
    const scalar_t x40   = F[8] * mu;
    const scalar_t x41   = F[8] * x11;
    const scalar_t x42   = F[1] * F[8] - F[2] * F[7];
    const scalar_t x43   = x38 * x42;
    const scalar_t x44   = lmbda * x43 + mu * x43 - x11 * x43 + x40 - x41;
    const scalar_t x45   = F[6] * mu;
    const scalar_t x46   = F[6] * x11;
    const scalar_t x47   = F[0] * F[7] - F[1] * F[6];
    const scalar_t x48   = x38 * x47;
    const scalar_t x49   = lmbda * x48 + mu * x48 - x11 * x48 - x45 + x46;
    const scalar_t x50   = adjugate[0] * x44 + adjugate[2] * x49 - x22 * x39;
    const scalar_t x51   = x37 * x8;
    const scalar_t x52   = x12 * x51;
    const scalar_t x53   = x42 * x52;
    const scalar_t x54   = F[7] * mu;
    const scalar_t x55   = F[7] * x11;
    const scalar_t x56   = x47 * x51;
    const scalar_t x57   = lmbda * x56 + mu * x56 - x11 * x56 - x54 + x55;
    const scalar_t x58   = x36 * x51;
    const scalar_t x59   = lmbda * x58 + mu * x58 - x11 * x58 - x40 + x41;
    const scalar_t x60   = -adjugate[0] * x53 + adjugate[1] * x59 - adjugate[2] * x57;
    const scalar_t x61   = x10 * x37;
    const scalar_t x62   = x47 * x61;
    const scalar_t x63   = x42 * x61;
    const scalar_t x64   = lmbda * x63 + mu * x63 - x11 * x63 + x54 - x55;
    const scalar_t x65   = x36 * x61;
    const scalar_t x66   = lmbda * x65 + mu * x65 - x11 * x65 + x45 - x46;
    const scalar_t x67   = -adjugate[0] * x64 + adjugate[1] * x66 - x13 * x62;
    const scalar_t x68   = x31 * x37;
    const scalar_t x69   = x12 * x39;
    const scalar_t x70   = adjugate[3] * x44 - adjugate[4] * x69 + adjugate[5] * x49;
    const scalar_t x71   = -adjugate[3] * x53 + adjugate[4] * x59 - adjugate[5] * x57;
    const scalar_t x72   = x12 * x62;
    const scalar_t x73   = -adjugate[3] * x64 + adjugate[4] * x66 - adjugate[5] * x72;
    const scalar_t x74   = adjugate[6] * x44 - adjugate[7] * x69 + adjugate[8] * x49;
    const scalar_t x75   = -adjugate[6] * x53 + adjugate[7] * x59 - adjugate[8] * x57;
    const scalar_t x76   = -adjugate[6] * x64 + adjugate[7] * x66 - adjugate[8] * x72;
    const scalar_t x77   = F[1] * F[5] - F[2] * F[4];
    const scalar_t x78   = x52 * x77;
    const scalar_t x79   = F[4] * mu;
    const scalar_t x80   = F[4] * x11;
    const scalar_t x81   = F[0] * F[4] - F[1] * F[3];
    const scalar_t x82   = x51 * x81;
    const scalar_t x83   = lmbda * x82 + mu * x82 - x11 * x82 - x79 + x80;
    const scalar_t x84   = F[5] * mu;
    const scalar_t x85   = F[5] * x11;
    const scalar_t x86   = F[0] * F[5] - F[2] * F[3];
    const scalar_t x87   = x51 * x86;
    const scalar_t x88   = lmbda * x87 + mu * x87 - x11 * x87 - x84 + x85;
    const scalar_t x89   = adjugate[0] * x78 - adjugate[1] * x88 + adjugate[2] * x83;
    const scalar_t x90   = x61 * x81;
    const scalar_t x91   = x61 * x77;
    const scalar_t x92   = lmbda * x91 + mu * x91 - x11 * x91 + x79 - x80;
    const scalar_t x93   = F[3] * mu;
    const scalar_t x94   = F[3] * x11;
    const scalar_t x95   = x61 * x86;
    const scalar_t x96   = lmbda * x95 + mu * x95 - x11 * x95 + x93 - x94;
    const scalar_t x97   = adjugate[0] * x92 - adjugate[1] * x96 + x13 * x90;
    const scalar_t x98   = x38 * x77;
    const scalar_t x99   = lmbda * x98 + mu * x98 - x11 * x98 + x84 - x85;
    const scalar_t x100  = x38 * x81;
    const scalar_t x101  = lmbda * x100 + mu * x100 - x100 * x11 - x93 + x94;
    const scalar_t x102  = -adjugate[0] * x99 + adjugate[1] * x12 * x15 * x37 * x86 - adjugate[2] * x101;
    const scalar_t x103  = adjugate[3] * x78 - adjugate[4] * x88 + adjugate[5] * x83;
    const scalar_t x104  = x12 * x90;
    const scalar_t x105  = adjugate[3] * x92 - adjugate[4] * x96 + adjugate[5] * x104;
    const scalar_t x106  = -adjugate[3] * x99 + adjugate[4] * x12 * x15 * x37 * x86 - adjugate[5] * x101;
    const scalar_t x107  = adjugate[6] * x78 - adjugate[7] * x88 + adjugate[8] * x83;
    const scalar_t x108  = adjugate[6] * x92 - adjugate[7] * x96 + adjugate[8] * x104;
    const scalar_t x109  = -adjugate[6] * x99 + adjugate[7] * x12 * x15 * x37 * x86 - adjugate[8] * x101;
    const scalar_t x110  = x42 * x7;
    const scalar_t x111  = x13 * x47;
    const scalar_t x112  = x110 * x36;
    const scalar_t x113  = POW2(x42) * x7;
    const scalar_t x114  = lmbda * x113 + mu * x113 + mu - x11 * x113;
    const scalar_t x115  = adjugate[0] * x114 + x110 * x111 - x112 * x22;
    const scalar_t x116  = adjugate[0] * x12;
    const scalar_t x117  = x110 * x47;
    const scalar_t x118  = x36 * x7;
    const scalar_t x119  = x118 * x47;
    const scalar_t x120  = POW2(x47) * x7;
    const scalar_t x121  = lmbda * x120 + mu * x120 + mu - x11 * x120;
    const scalar_t x122  = adjugate[2] * x121 + x116 * x117 - x119 * x22;
    const scalar_t x123  = POW2(x36) * x7;
    const scalar_t x124  = lmbda * x123 + mu * x123 + mu - x11 * x123;
    const scalar_t x125  = adjugate[1] * x124 - x111 * x118 - x112 * x116;
    const scalar_t x126  = adjugate[5] * x12;
    const scalar_t x127  = x112 * x12;
    const scalar_t x128  = adjugate[3] * x114 - adjugate[4] * x127 + x117 * x126;
    const scalar_t x129  = x117 * x12;
    const scalar_t x130  = x119 * x12;
    const scalar_t x131  = adjugate[3] * x129 - adjugate[4] * x130 + adjugate[5] * x121;
    const scalar_t x132  = -adjugate[3] * x127 + adjugate[4] * x124 - x119 * x126;
    const scalar_t x133  = x36 * x37;
    const scalar_t x134  = x133 * x86;
    const scalar_t x135  = F[2] * mu;
    const scalar_t x136  = F[2] * x11;
    const scalar_t x137  = x133 * x77;
    const scalar_t x138  = lmbda * x137 + mu * x137 - x11 * x137 + x135 - x136;
    const scalar_t x139  = F[0] * mu;
    const scalar_t x140  = F[0] * x11;
    const scalar_t x141  = x133 * x81;
    const scalar_t x142  = lmbda * x141 + mu * x141 - x11 * x141 - x139 + x140;
    const scalar_t x143  = adjugate[0] * x138 + adjugate[2] * x142 - x134 * x22;
    const scalar_t x144  = x37 * x42;
    const scalar_t x145  = x144 * x77;
    const scalar_t x146  = F[1] * mu;
    const scalar_t x147  = F[1] * x11;
    const scalar_t x148  = x144 * x81;
    const scalar_t x149  = lmbda * x148 + mu * x148 - x11 * x148 - x146 + x147;
    const scalar_t x150  = x144 * x86;
    const scalar_t x151  = lmbda * x150 + mu * x150 - x11 * x150 - x135 + x136;
    const scalar_t x152  = adjugate[1] * x151 - adjugate[2] * x149 - x116 * x145;
    const scalar_t x153  = x37 * x47;
    const scalar_t x154  = x153 * x81;
    const scalar_t x155  = x153 * x77;
    const scalar_t x156  = lmbda * x155 + mu * x155 - x11 * x155 + x146 - x147;
    const scalar_t x157  = x153 * x86;
    const scalar_t x158  = lmbda * x157 + mu * x157 - x11 * x157 + x139 - x140;
    const scalar_t x159  = -adjugate[0] * x156 + adjugate[1] * x158 - x13 * x154;
    const scalar_t x160  = x12 * x134;
    const scalar_t x161  = adjugate[3] * x138 - adjugate[4] * x160 + adjugate[5] * x142;
    const scalar_t x162  = x12 * x145;
    const scalar_t x163  = -adjugate[3] * x162 + adjugate[4] * x151 - adjugate[5] * x149;
    const scalar_t x164  = -adjugate[3] * x156 + adjugate[4] * x158 - x126 * x154;
    const scalar_t x165  = adjugate[6] * x138 - adjugate[7] * x160 + adjugate[8] * x142;
    const scalar_t x166  = -adjugate[6] * x162 + adjugate[7] * x151 - adjugate[8] * x149;
    const scalar_t x167  = adjugate[8] * x12;
    const scalar_t x168  = -adjugate[6] * x156 + adjugate[7] * x158 - x154 * x167;
    const scalar_t x169  = x7 * x77;
    const scalar_t x170  = x13 * x81;
    const scalar_t x171  = x169 * x86;
    const scalar_t x172  = x7 * POW2(x77);
    const scalar_t x173  = lmbda * x172 + mu * x172 + mu - x11 * x172;
    const scalar_t x174  = adjugate[0] * x173 + x169 * x170 - x171 * x22;
    const scalar_t x175  = x169 * x81;
    const scalar_t x176  = x7 * x86;
    const scalar_t x177  = x176 * x81;
    const scalar_t x178  = x7 * POW2(x81);
    const scalar_t x179  = lmbda * x178 + mu * x178 + mu - x11 * x178;
    const scalar_t x180  = adjugate[2] * x179 + x116 * x175 - x177 * x22;
    const scalar_t x181  = x7 * POW2(x86);
    const scalar_t x182  = lmbda * x181 + mu * x181 + mu - x11 * x181;
    const scalar_t x183  = adjugate[1] * x182 - x116 * x171 - x170 * x176;
    const scalar_t x184  = x12 * x171;
    const scalar_t x185  = adjugate[3] * x173 - adjugate[4] * x184 + x126 * x175;
    const scalar_t x186  = x12 * x175;
    const scalar_t x187  = x12 * x177;
    const scalar_t x188  = adjugate[3] * x186 - adjugate[4] * x187 + adjugate[5] * x179;
    const scalar_t x189  = -adjugate[3] * x184 + adjugate[4] * x182 - x126 * x177;
    S_ikmn_canonical[0]  = x31 * (adjugate[0] * x20 + adjugate[1] * x30 + adjugate[2] * x27);
    S_ikmn_canonical[1]  = x31 * (adjugate[3] * x20 + adjugate[4] * x30 + adjugate[5] * x27);
    S_ikmn_canonical[2]  = x31 * (adjugate[6] * x20 + adjugate[7] * x30 + adjugate[8] * x27);
    S_ikmn_canonical[3]  = x31 * (adjugate[3] * x32 + adjugate[4] * x35 + adjugate[5] * x34);
    S_ikmn_canonical[4]  = x31 * (adjugate[6] * x32 + adjugate[7] * x35 + adjugate[8] * x34);
    S_ikmn_canonical[5]  = x31 * (adjugate[6] * (adjugate[6] * x19 - adjugate[7] * x17 + adjugate[8] * x21) +
                                 adjugate[7] * (-adjugate[6] * x17 + adjugate[7] * x29 - adjugate[8] * x33) +
                                 adjugate[8] * (adjugate[6] * x21 - adjugate[7] * x33 + adjugate[8] * x26));
    S_ikmn_canonical[6]  = x68 * (adjugate[0] * x60 + adjugate[1] * x50 + adjugate[2] * x67);
    S_ikmn_canonical[7]  = x68 * (adjugate[3] * x60 + adjugate[4] * x50 + adjugate[5] * x67);
    S_ikmn_canonical[8]  = x68 * (adjugate[6] * x60 + adjugate[7] * x50 + adjugate[8] * x67);
    S_ikmn_canonical[9]  = x68 * (adjugate[0] * x71 + adjugate[1] * x70 + adjugate[2] * x73);
    S_ikmn_canonical[10] = x68 * (adjugate[3] * x71 + adjugate[4] * x70 + adjugate[5] * x73);
    S_ikmn_canonical[11] = x68 * (adjugate[6] * x71 + adjugate[7] * x70 + adjugate[8] * x73);
    S_ikmn_canonical[12] = x68 * (adjugate[0] * x75 + adjugate[1] * x74 + adjugate[2] * x76);
    S_ikmn_canonical[13] = x68 * (adjugate[3] * x75 + adjugate[4] * x74 + adjugate[5] * x76);
    S_ikmn_canonical[14] = x68 * (adjugate[6] * x75 + adjugate[7] * x74 + adjugate[8] * x76);
    S_ikmn_canonical[15] = x68 * (adjugate[0] * x89 + adjugate[1] * x102 + adjugate[2] * x97);
    S_ikmn_canonical[16] = x68 * (adjugate[3] * x89 + adjugate[4] * x102 + adjugate[5] * x97);
    S_ikmn_canonical[17] = x68 * (adjugate[6] * x89 + adjugate[7] * x102 + adjugate[8] * x97);
    S_ikmn_canonical[18] = x68 * (adjugate[0] * x103 + adjugate[1] * x106 + adjugate[2] * x105);
    S_ikmn_canonical[19] = x68 * (adjugate[3] * x103 + adjugate[4] * x106 + adjugate[5] * x105);
    S_ikmn_canonical[20] = x68 * (adjugate[6] * x103 + adjugate[7] * x106 + adjugate[8] * x105);
    S_ikmn_canonical[21] = x68 * (adjugate[0] * x107 + adjugate[1] * x109 + adjugate[2] * x108);
    S_ikmn_canonical[22] = x68 * (adjugate[3] * x107 + adjugate[4] * x109 + adjugate[5] * x108);
    S_ikmn_canonical[23] = x68 * (adjugate[6] * x107 + adjugate[7] * x109 + adjugate[8] * x108);
    S_ikmn_canonical[24] = x31 * (adjugate[0] * x115 + adjugate[1] * x125 + adjugate[2] * x122);
    S_ikmn_canonical[25] = x31 * (adjugate[3] * x115 + adjugate[4] * x125 + adjugate[5] * x122);
    S_ikmn_canonical[26] = x31 * (adjugate[6] * x115 + adjugate[7] * x125 + adjugate[8] * x122);
    S_ikmn_canonical[27] = x31 * (adjugate[3] * x128 + adjugate[4] * x132 + adjugate[5] * x131);
    S_ikmn_canonical[28] = x31 * (adjugate[6] * x128 + adjugate[7] * x132 + adjugate[8] * x131);
    S_ikmn_canonical[29] = x31 * (adjugate[6] * (adjugate[6] * x114 - adjugate[7] * x127 + adjugate[8] * x129) +
                                  adjugate[7] * (-adjugate[6] * x127 + adjugate[7] * x124 - adjugate[8] * x130) +
                                  adjugate[8] * (adjugate[6] * x129 - adjugate[7] * x130 + adjugate[8] * x121));
    S_ikmn_canonical[30] = x68 * (adjugate[0] * x152 + adjugate[1] * x143 + adjugate[2] * x159);
    S_ikmn_canonical[31] = x68 * (adjugate[3] * x152 + adjugate[4] * x143 + adjugate[5] * x159);
    S_ikmn_canonical[32] = x68 * (adjugate[6] * x152 + adjugate[7] * x143 + adjugate[8] * x159);
    S_ikmn_canonical[33] = x68 * (adjugate[0] * x163 + adjugate[1] * x161 + adjugate[2] * x164);
    S_ikmn_canonical[34] = x68 * (adjugate[3] * x163 + adjugate[4] * x161 + adjugate[5] * x164);
    S_ikmn_canonical[35] = x68 * (adjugate[6] * x163 + adjugate[7] * x161 + adjugate[8] * x164);
    S_ikmn_canonical[36] = x68 * (adjugate[0] * x166 + adjugate[1] * x165 + adjugate[2] * x168);
    S_ikmn_canonical[37] = x68 * (adjugate[3] * x166 + adjugate[4] * x165 + adjugate[5] * x168);
    S_ikmn_canonical[38] = x68 * (adjugate[6] * x166 + adjugate[7] * x165 + adjugate[8] * x168);
    S_ikmn_canonical[39] = x31 * (adjugate[0] * x174 + adjugate[1] * x183 + adjugate[2] * x180);
    S_ikmn_canonical[40] = x31 * (adjugate[3] * x174 + adjugate[4] * x183 + adjugate[5] * x180);
    S_ikmn_canonical[41] = x31 * (adjugate[6] * x174 + adjugate[7] * x183 + adjugate[8] * x180);
    S_ikmn_canonical[42] = x31 * (adjugate[3] * x185 + adjugate[4] * x189 + adjugate[5] * x188);
    S_ikmn_canonical[43] = x31 * (adjugate[6] * x185 + adjugate[7] * x189 + adjugate[8] * x188);
    S_ikmn_canonical[44] = x31 * (adjugate[6] * (adjugate[6] * x173 - adjugate[7] * x184 + x167 * x175) +
                                  adjugate[7] * (-adjugate[6] * x184 + adjugate[7] * x182 - x167 * x177) +
                                  adjugate[8] * (adjugate[6] * x186 - adjugate[7] * x187 + adjugate[8] * x179));
}

static SFEM_INLINE void hex8_apply_S_ikmn(const scalar_t                      qx,
                                          const scalar_t                      qy,
                                          const scalar_t                      qz,
                                          const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,  // 3x3x3x3, includes dV
                                          const scalar_t *const SFEM_RESTRICT inc_grad,          // 3x3 reference trial gradient R
                                          scalar_t *const SFEM_RESTRICT       eoutx,
                                          scalar_t *const SFEM_RESTRICT       eouty,
                                          scalar_t *const SFEM_RESTRICT       eoutz) {
    const scalar_t x0 = qx - 1;
    const scalar_t x1 = qy - 1;
    const scalar_t x2 = S_ikmn_canonical[11] * inc_grad[4] + S_ikmn_canonical[14] * inc_grad[5] +
                        S_ikmn_canonical[17] * inc_grad[6] + S_ikmn_canonical[20] * inc_grad[7] +
                        S_ikmn_canonical[23] * inc_grad[8] + S_ikmn_canonical[2] * inc_grad[0] +
                        S_ikmn_canonical[4] * inc_grad[1] + S_ikmn_canonical[5] * inc_grad[2] + S_ikmn_canonical[8] * inc_grad[3];
    const scalar_t x3 = x1 * x2;
    const scalar_t x4 = x0 * x3;
    const scalar_t x5 = qz - 1;
    const scalar_t x6 = S_ikmn_canonical[10] * inc_grad[4] + S_ikmn_canonical[13] * inc_grad[5] +
                        S_ikmn_canonical[16] * inc_grad[6] + S_ikmn_canonical[19] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[8] +
                        S_ikmn_canonical[3] * inc_grad[1] + S_ikmn_canonical[4] * inc_grad[2] + S_ikmn_canonical[7] * inc_grad[3];
    const scalar_t x7 = x5 * x6;
    const scalar_t x8 = x0 * x7;
    const scalar_t x9 = S_ikmn_canonical[0] * inc_grad[0] + S_ikmn_canonical[12] * inc_grad[5] +
                        S_ikmn_canonical[15] * inc_grad[6] + S_ikmn_canonical[18] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[1] + S_ikmn_canonical[21] * inc_grad[8] +
                        S_ikmn_canonical[2] * inc_grad[2] + S_ikmn_canonical[6] * inc_grad[3] + S_ikmn_canonical[9] * inc_grad[4];
    const scalar_t x10 = x5 * x9;
    const scalar_t x11 = x1 * x10;
    const scalar_t x12 = qx * x3;
    const scalar_t x13 = qx * x7;
    const scalar_t x14 = qy * x2;
    const scalar_t x15 = qx * x14;
    const scalar_t x16 = qy * x10;
    const scalar_t x17 = x0 * x14;
    const scalar_t x18 = qz * x6;
    const scalar_t x19 = x0 * x18;
    const scalar_t x20 = qz * x9;
    const scalar_t x21 = x1 * x20;
    const scalar_t x22 = qx * x18;
    const scalar_t x23 = qy * x20;
    const scalar_t x24 =
            S_ikmn_canonical[12] * inc_grad[0] + S_ikmn_canonical[13] * inc_grad[1] + S_ikmn_canonical[14] * inc_grad[2] +
            S_ikmn_canonical[26] * inc_grad[3] + S_ikmn_canonical[28] * inc_grad[4] + S_ikmn_canonical[29] * inc_grad[5] +
            S_ikmn_canonical[32] * inc_grad[6] + S_ikmn_canonical[35] * inc_grad[7] + S_ikmn_canonical[38] * inc_grad[8];
    const scalar_t x25 = x1 * x24;
    const scalar_t x26 = x0 * x25;
    const scalar_t x27 =
            S_ikmn_canonical[10] * inc_grad[1] + S_ikmn_canonical[11] * inc_grad[2] + S_ikmn_canonical[25] * inc_grad[3] +
            S_ikmn_canonical[27] * inc_grad[4] + S_ikmn_canonical[28] * inc_grad[5] + S_ikmn_canonical[31] * inc_grad[6] +
            S_ikmn_canonical[34] * inc_grad[7] + S_ikmn_canonical[37] * inc_grad[8] + S_ikmn_canonical[9] * inc_grad[0];
    const scalar_t x28 = x27 * x5;
    const scalar_t x29 = x0 * x28;
    const scalar_t x30 =
            S_ikmn_canonical[24] * inc_grad[3] + S_ikmn_canonical[25] * inc_grad[4] + S_ikmn_canonical[26] * inc_grad[5] +
            S_ikmn_canonical[30] * inc_grad[6] + S_ikmn_canonical[33] * inc_grad[7] + S_ikmn_canonical[36] * inc_grad[8] +
            S_ikmn_canonical[6] * inc_grad[0] + S_ikmn_canonical[7] * inc_grad[1] + S_ikmn_canonical[8] * inc_grad[2];
    const scalar_t x31 = x30 * x5;
    const scalar_t x32 = x1 * x31;
    const scalar_t x33 = qx * x25;
    const scalar_t x34 = qx * x28;
    const scalar_t x35 = qy * x24;
    const scalar_t x36 = qx * x35;
    const scalar_t x37 = qy * x31;
    const scalar_t x38 = x0 * x35;
    const scalar_t x39 = qz * x27;
    const scalar_t x40 = x0 * x39;
    const scalar_t x41 = qz * x30;
    const scalar_t x42 = x1 * x41;
    const scalar_t x43 = qx * x39;
    const scalar_t x44 = qy * x41;
    const scalar_t x45 =
            S_ikmn_canonical[21] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[1] + S_ikmn_canonical[23] * inc_grad[2] +
            S_ikmn_canonical[36] * inc_grad[3] + S_ikmn_canonical[37] * inc_grad[4] + S_ikmn_canonical[38] * inc_grad[5] +
            S_ikmn_canonical[41] * inc_grad[6] + S_ikmn_canonical[43] * inc_grad[7] + S_ikmn_canonical[44] * inc_grad[8];
    const scalar_t x46 = x1 * x45;
    const scalar_t x47 = x0 * x46;
    const scalar_t x48 =
            S_ikmn_canonical[18] * inc_grad[0] + S_ikmn_canonical[19] * inc_grad[1] + S_ikmn_canonical[20] * inc_grad[2] +
            S_ikmn_canonical[33] * inc_grad[3] + S_ikmn_canonical[34] * inc_grad[4] + S_ikmn_canonical[35] * inc_grad[5] +
            S_ikmn_canonical[40] * inc_grad[6] + S_ikmn_canonical[42] * inc_grad[7] + S_ikmn_canonical[43] * inc_grad[8];
    const scalar_t x49 = x48 * x5;
    const scalar_t x50 = x0 * x49;
    const scalar_t x51 =
            S_ikmn_canonical[15] * inc_grad[0] + S_ikmn_canonical[16] * inc_grad[1] + S_ikmn_canonical[17] * inc_grad[2] +
            S_ikmn_canonical[30] * inc_grad[3] + S_ikmn_canonical[31] * inc_grad[4] + S_ikmn_canonical[32] * inc_grad[5] +
            S_ikmn_canonical[39] * inc_grad[6] + S_ikmn_canonical[40] * inc_grad[7] + S_ikmn_canonical[41] * inc_grad[8];
    const scalar_t x52 = x5 * x51;
    const scalar_t x53 = x1 * x52;
    const scalar_t x54 = qx * x46;
    const scalar_t x55 = qx * x49;
    const scalar_t x56 = qy * x45;
    const scalar_t x57 = qx * x56;
    const scalar_t x58 = qy * x52;
    const scalar_t x59 = x0 * x56;
    const scalar_t x60 = qz * x48;
    const scalar_t x61 = x0 * x60;
    const scalar_t x62 = qz * x51;
    const scalar_t x63 = x1 * x62;
    const scalar_t x64 = qx * x60;
    const scalar_t x65 = qy * x62;
    eoutx[0]           = -x11 - x4 - x8;
    eoutx[1]           = x11 + x12 + x13;
    eoutx[2]           = -x13 - x15 - x16;
    eoutx[3]           = x16 + x17 + x8;
    eoutx[4]           = x19 + x21 + x4;
    eoutx[5]           = -x12 - x21 - x22;
    eoutx[6]           = x15 + x22 + x23;
    eoutx[7]           = -x17 - x19 - x23;
    eouty[0]           = -x26 - x29 - x32;
    eouty[1]           = x32 + x33 + x34;
    eouty[2]           = -x34 - x36 - x37;
    eouty[3]           = x29 + x37 + x38;
    eouty[4]           = x26 + x40 + x42;
    eouty[5]           = -x33 - x42 - x43;
    eouty[6]           = x36 + x43 + x44;
    eouty[7]           = -x38 - x40 - x44;
    eoutz[0]           = -x47 - x50 - x53;
    eoutz[1]           = x53 + x54 + x55;
    eoutz[2]           = -x55 - x57 - x58;
    eoutz[3]           = x50 + x58 + x59;
    eoutz[4]           = x47 + x61 + x63;
    eoutz[5]           = -x54 - x63 - x64;
    eoutz[6]           = x57 + x64 + x65;
    eoutz[7]           = -x59 - x61 - x65;
}

#endif /* SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H */
