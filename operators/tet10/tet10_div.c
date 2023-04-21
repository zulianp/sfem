#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#include <mpi.h>

#include "sfem_base.h"

static SFEM_INLINE void tet10_div_apply_kernel(const real_t px0,
                                               const real_t px1,
                                               const real_t px2,
                                               const real_t px3,
                                               const real_t py0,
                                               const real_t py1,
                                               const real_t py2,
                                               const real_t py3,
                                               const real_t pz0,
                                               const real_t pz1,
                                               const real_t pz2,
                                               const real_t pz3,
                                               const real_t *ux,
                                               const real_t *uy,
                                               const real_t *uz,
                                               real_t *element_vector) {
//     // FLOATING POINT OPS!
//     //      - Result: 10*ADD + 10*ASSIGNMENT + 46*MUL
//     //      - Subexpressions: 72*ADD + 3*DIV + 181*MUL + 14*NEG + 100*SUB
//     const real_t x0 = px0 - px1;
//     const real_t x1 = -x0;
//     const real_t x2 = py0 - py2;
//     const real_t x3 = -x2;
//     const real_t x4 = pz0 - pz3;
//     const real_t x5 = -x4;
//     const real_t x6 = x3 * x5;
//     const real_t x7 = pz0 - pz1;
//     const real_t x8 = -x7;
//     const real_t x9 = px0 - px2;
//     const real_t x10 = -x9;
//     const real_t x11 = py0 - py3;
//     const real_t x12 = -x11;
//     const real_t x13 = x10 * x12;
//     const real_t x14 = py0 - py1;
//     const real_t x15 = -x14;
//     const real_t x16 = px0 - px3;
//     const real_t x17 = -x16;
//     const real_t x18 = pz0 - pz2;
//     const real_t x19 = -x18;
//     const real_t x20 = x12 * x19;
//     const real_t x21 = x10 * x5;
//     const real_t x22 = x17 * x3;
//     const real_t x23 = 1.0 / (-x1 * x20 + x1 * x6 + x13 * x8 + x15 * x17 * x19 - x15 * x21 - x22 * x8);
//     const real_t x24 = x23 * (x17 * x19 - x21);
//     const real_t x25 = ux[2] * x24;
//     const real_t x26 = x23 * (x13 - x22);
//     const real_t x27 = ux[3] * x26;
//     const real_t x28 = x23 * (x1 * x5 - x17 * x8);
//     const real_t x29 = uy[2] * x28;
//     const real_t x30 = x23 * (-x1 * x12 + x15 * x17);
//     const real_t x31 = uy[3] * x30;
//     const real_t x32 = x23 * (-x1 * x19 + x10 * x8);
//     const real_t x33 = uz[2] * x32;
//     const real_t x34 = x23 * (x1 * x3 - x10 * x15);
//     const real_t x35 = uz[3] * x34;
//     const real_t x36 = x23 * (-x20 + x6);
//     const real_t x37 = ux[1] * x36;
//     const real_t x38 = x23 * (x12 * x8 - x15 * x5);
//     const real_t x39 = uy[1] * x38;
//     const real_t x40 = x23 * (x15 * x19 - x3 * x8);
//     const real_t x41 = uz[1] * x40;
//     const real_t x42 = ux[0] * x26;
//     const real_t x43 = ux[0] * x24;
//     const real_t x44 = ux[0] * x36;
//     const real_t x45 = uy[0] * x28;
//     const real_t x46 = uy[0] * x30;
//     const real_t x47 = uy[0] * x38;
//     const real_t x48 = uz[0] * x34;
//     const real_t x49 = uz[0] * x32;
//     const real_t x50 = uz[0] * x40;
//     const real_t x51 = 4 * ux[4];
//     const real_t x52 = 4 * uy[4];
//     const real_t x53 = 4 * uz[4];
//     const real_t x54 = -x36 * x51 - x38 * x52 - x40 * x53;
//     const real_t x55 = 4 * ux[6];
//     const real_t x56 = 4 * uy[6];
//     const real_t x57 = 4 * uz[6];
//     const real_t x58 = -x24 * x55 - x28 * x56 - x32 * x57;
//     const real_t x59 = 4 * ux[7];
//     const real_t x60 = 4 * uy[7];
//     const real_t x61 = 4 * uz[7];
//     const real_t x62 = -x26 * x59 - x30 * x60 - x34 * x61;
//     const real_t x63 = x26 * x55;
//     const real_t x64 = x36 * x55;
//     const real_t x65 = x30 * x56;
//     const real_t x66 = x38 * x56;
//     const real_t x67 = x34 * x57;
//     const real_t x68 = x40 * x57;
//     const real_t x69 = 4 * ux[5];
//     const real_t x70 = x36 * x69;
//     const real_t x71 = 4 * ux[9];
//     const real_t x72 = x26 * x71;
//     const real_t x73 = 4 * uy[5];
//     const real_t x74 = x38 * x73;
//     const real_t x75 = 4 * uy[9];
//     const real_t x76 = x30 * x75;
//     const real_t x77 = 4 * uz[5];
//     const real_t x78 = x40 * x77;
//     const real_t x79 = 4 * uz[9];
//     const real_t x80 = x34 * x79;
//     const real_t x81 = -x63 - x64 - x65 - x66 - x67 - x68 + x70 + x72 + x74 + x76 + x78 + x80;
//     const real_t x82 = x26 * x51;
//     const real_t x83 = x24 * x51;
//     const real_t x84 = x28 * x52;
//     const real_t x85 = x30 * x52;
//     const real_t x86 = x34 * x53;
//     const real_t x87 = x32 * x53;
//     const real_t x88 = x24 * x69;
//     const real_t x89 = 4 * ux[8];
//     const real_t x90 = x26 * x89;
//     const real_t x91 = x28 * x73;
//     const real_t x92 = 4 * uy[8];
//     const real_t x93 = x30 * x92;
//     const real_t x94 = x32 * x77;
//     const real_t x95 = 4 * uz[8];
//     const real_t x96 = x34 * x95;
//     const real_t x97 = -x82 - x83 - x84 - x85 - x86 - x87 + x88 + x90 + x91 + x93 + x94 + x96;
//     const real_t x98 = x81 + x97;
//     const real_t x99 = x24 * x59;
//     const real_t x100 = x36 * x59;
//     const real_t x101 = x28 * x60;
//     const real_t x102 = x38 * x60;
//     const real_t x103 = x32 * x61;
//     const real_t x104 = x40 * x61;
//     const real_t x105 = x36 * x89;
//     const real_t x106 = x24 * x71;
//     const real_t x107 = x38 * x92;
//     const real_t x108 = x28 * x75;
//     const real_t x109 = x40 * x95;
//     const real_t x110 = x32 * x79;
//     const real_t x111 = -x100 - x101 - x102 - x103 - x104 + x105 + x106 + x107 + x108 + x109 + x110 - x99;
//     const real_t x112 =
//         -x0 * x11 * x18 + x0 * x2 * x4 + x11 * x7 * x9 + x14 * x16 * x18 - x14 * x4 * x9 - x16 * x2 * x7;
//     const real_t x113 = (1.0 / 360.0) * x112;
//     const real_t x114 = x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50;
//     const real_t x115 = x114 - x27 - x31 - x35;
//     const real_t x116 = x100 + x101 + x102 + x103 + x104 - x105 - x106 - x107 - x108 - x109 - x110 + x115 + x99;
//     const real_t x117 = -x25 - x29 - x33;
//     const real_t x118 = x117 + x63 + x64 + x65 + x66 + x67 + x68 - x70 - x72 - x74 - x76 - x78 - x80;
//     const real_t x119 = -x37 - x39 - x41;
//     const real_t x120 = x119 + x82 + x83 + x84 + x85 + x86 + x87 - x88 - x90 - x91 - x93 - x94 - x96;
//     const real_t x121 = x114 + x120;
//     const real_t x122 = 2 * x24;
//     const real_t x123 = 2 * x28;
//     const real_t x124 = 2 * x32;
//     const real_t x125 = -ux[6] * x122 - uy[6] * x123 - uz[6] * x124 + x25 + x29 + x33;
//     const real_t x126 = 2 * x26;
//     const real_t x127 = 2 * x30;
//     const real_t x128 = 2 * x34;
//     const real_t x129 = -ux[7] * x126 - uy[7] * x127 - uz[7] * x128 + x27 + x31 + x35;
//     const real_t x130 = x125 + x129;
//     const real_t x131 = 2 * x36;
//     const real_t x132 = ux[8] * x131;
//     const real_t x133 = ux[9] * x122;
//     const real_t x134 = 2 * x38;
//     const real_t x135 = uy[8] * x134;
//     const real_t x136 = uy[9] * x123;
//     const real_t x137 = 2 * x40;
//     const real_t x138 = uz[8] * x137;
//     const real_t x139 = uz[9] * x124;
//     const real_t x140 = ux[7] * x122;
//     const real_t x141 = ux[7] * x131;
//     const real_t x142 = uy[7] * x123;
//     const real_t x143 = uy[7] * x134;
//     const real_t x144 = uz[7] * x124;
//     const real_t x145 = uz[7] * x137;
//     const real_t x146 = -x132 - x133 - x135 - x136 - x138 - x139 + x140 + x141 + x142 + x143 + x144 + x145;
//     const real_t x147 = ux[5] * x131;
//     const real_t x148 = ux[9] * x126;
//     const real_t x149 = uy[5] * x134;
//     const real_t x150 = uy[9] * x127;
//     const real_t x151 = uz[5] * x137;
//     const real_t x152 = uz[9] * x128;
//     const real_t x153 = ux[6] * x126;
//     const real_t x154 = ux[6] * x131;
//     const real_t x155 = uy[6] * x127;
//     const real_t x156 = uy[6] * x134;
//     const real_t x157 = uz[6] * x128;
//     const real_t x158 = uz[6] * x137;
//     const real_t x159 = -x147 - x148 - x149 - x150 - x151 - x152 + x153 + x154 + x155 + x156 + x157 + x158;
//     const real_t x160 = (1.0 / 90.0) * x112;
//     const real_t x161 = -ux[4] * x131 - uy[4] * x134 - uz[4] * x137 + x37 + x39 + x41;
//     const real_t x162 = x125 + x161;
//     const real_t x163 = x114 + x129 + x161;
//     const real_t x164 = ux[5] * x122;
//     const real_t x165 = ux[8] * x126;
//     const real_t x166 = uy[5] * x123;
//     const real_t x167 = uy[8] * x127;
//     const real_t x168 = uz[5] * x124;
//     const real_t x169 = uz[8] * x128;
//     const real_t x170 = ux[4] * x126;
//     const real_t x171 = ux[4] * x122;
//     const real_t x172 = uy[4] * x123;
//     const real_t x173 = uy[4] * x127;
//     const real_t x174 = uz[4] * x128;
//     const real_t x175 = uz[4] * x124;
//     const real_t x176 = -x164 - x165 - x166 - x167 - x168 - x169 + x170 + x171 + x172 + x173 + x174 + x175;
//     element_vector[0] =
//         x113 * (x111 + x25 + x27 + x29 + x31 + x33 + x35 + x37 + x39 + x41 + 3 * x42 + 3 * x43 + 3 * x44 + 3 * x45 +
//                 3 * x46 + 3 * x47 + 3 * x48 + 3 * x49 + 3 * x50 + x54 + x58 + x62 + x98);
//     element_vector[1] = -x113 * (x116 + x118 + 3 * x37 + 3 * x39 + 3 * x41 + x54);
//     element_vector[2] = -x113 * (x116 + x120 + 3 * x25 + 3 * x29 + 3 * x33 + x58);
//     element_vector[3] = -x113 * (x118 + x121 + 3 * x27 + 3 * x31 + 3 * x35 + x62);
//     element_vector[4] = x160 * (x121 + x130 + x146 + x159);
//     element_vector[5] =
//         -x160 * (x115 + x132 + x133 + x135 + x136 + x138 + x139 - x140 - x141 - x142 - x143 - x144 - x145 + x162 + x98);
//     element_vector[6] = x160 * (x118 + x146 + x163 + x176);
//     element_vector[7] = x160 * (x116 + x159 + x162 + x176);
//     element_vector[8] = -x160 * (x111 + x117 + x147 + x148 + x149 + x150 + x151 + x152 - x153 - x154 - x155 - x156 -
//                                  x157 - x158 + x163 + x97);
//     element_vector[9] = -x160 * (x111 + x114 + x119 + x130 + x164 + x165 + x166 + x167 + x168 + x169 - x170 - x171 -
//                                  x172 - x173 - x174 - x175 + x81);


    // generated code
    //FLOATING POINT OPS!
    //      - Result: 10*ADD + 10*ASSIGNMENT + 46*MUL
    //      - Subexpressions: 72*ADD + 3*DIV + 181*MUL + 14*NEG + 100*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = -x0;
    const real_t x2 = py0 - py2;
    const real_t x3 = -x2;
    const real_t x4 = pz0 - pz3;
    const real_t x5 = -x4;
    const real_t x6 = x3*x5;
    const real_t x7 = px0 - px2;
    const real_t x8 = -x7;
    const real_t x9 = py0 - py3;
    const real_t x10 = -x9;
    const real_t x11 = pz0 - pz1;
    const real_t x12 = -x11;
    const real_t x13 = px0 - px3;
    const real_t x14 = -x13;
    const real_t x15 = py0 - py1;
    const real_t x16 = -x15;
    const real_t x17 = pz0 - pz2;
    const real_t x18 = -x17;
    const real_t x19 = x16*x18;
    const real_t x20 = x10*x18;
    const real_t x21 = x16*x5;
    const real_t x22 = x12*x3;
    const real_t x23 = 1.0/(-x1*x20 + x1*x6 + x10*x12*x8 + x14*x19 - x14*x22 - x21*x8);
    const real_t x24 = x23*(x10*x12 - x21);
    const real_t x25 = ux[2]*x24;
    const real_t x26 = x23*(x19 - x22);
    const real_t x27 = ux[3]*x26;
    const real_t x28 = x23*(x1*x5 - x12*x14);
    const real_t x29 = uy[2]*x28;
    const real_t x30 = x23*(-x1*x18 + x12*x8);
    const real_t x31 = uy[3]*x30;
    const real_t x32 = x23*(-x1*x10 + x14*x16);
    const real_t x33 = uz[2]*x32;
    const real_t x34 = x23*(x1*x3 - x16*x8);
    const real_t x35 = uz[3]*x34;
    const real_t x36 = x23*(-x20 + x6);
    const real_t x37 = ux[1]*x36;
    const real_t x38 = x23*(x14*x18 - x5*x8);
    const real_t x39 = uy[1]*x38;
    const real_t x40 = x23*(x10*x8 - x14*x3);
    const real_t x41 = uz[1]*x40;
    const real_t x42 = ux[0]*x26;
    const real_t x43 = ux[0]*x36;
    const real_t x44 = ux[0]*x24;
    const real_t x45 = uy[0]*x28;
    const real_t x46 = uy[0]*x30;
    const real_t x47 = uy[0]*x38;
    const real_t x48 = uz[0]*x34;
    const real_t x49 = uz[0]*x40;
    const real_t x50 = uz[0]*x32;
    const real_t x51 = 4*ux[4];
    const real_t x52 = 4*uy[4];
    const real_t x53 = 4*uz[4];
    const real_t x54 = -x36*x51 - x38*x52 - x40*x53;
    const real_t x55 = 4*ux[6];
    const real_t x56 = 4*uy[6];
    const real_t x57 = 4*uz[6];
    const real_t x58 = -x24*x55 - x28*x56 - x32*x57;
    const real_t x59 = 4*ux[7];
    const real_t x60 = 4*uy[7];
    const real_t x61 = 4*uz[7];
    const real_t x62 = -x26*x59 - x30*x60 - x34*x61;
    const real_t x63 = x26*x55;
    const real_t x64 = x36*x55;
    const real_t x65 = x30*x56;
    const real_t x66 = x38*x56;
    const real_t x67 = x34*x57;
    const real_t x68 = x40*x57;
    const real_t x69 = 4*ux[5];
    const real_t x70 = x36*x69;
    const real_t x71 = 4*ux[9];
    const real_t x72 = x26*x71;
    const real_t x73 = 4*uy[5];
    const real_t x74 = x38*x73;
    const real_t x75 = 4*uy[9];
    const real_t x76 = x30*x75;
    const real_t x77 = 4*uz[5];
    const real_t x78 = x40*x77;
    const real_t x79 = 4*uz[9];
    const real_t x80 = x34*x79;
    const real_t x81 = -x63 - x64 - x65 - x66 - x67 - x68 + x70 + x72 + x74 + x76 + x78 + x80;
    const real_t x82 = x26*x51;
    const real_t x83 = x24*x51;
    const real_t x84 = x28*x52;
    const real_t x85 = x30*x52;
    const real_t x86 = x34*x53;
    const real_t x87 = x32*x53;
    const real_t x88 = x24*x69;
    const real_t x89 = 4*ux[8];
    const real_t x90 = x26*x89;
    const real_t x91 = x28*x73;
    const real_t x92 = 4*uy[8];
    const real_t x93 = x30*x92;
    const real_t x94 = x32*x77;
    const real_t x95 = 4*uz[8];
    const real_t x96 = x34*x95;
    const real_t x97 = -x82 - x83 - x84 - x85 - x86 - x87 + x88 + x90 + x91 + x93 + x94 + x96;
    const real_t x98 = x81 + x97;
    const real_t x99 = x36*x59;
    const real_t x100 = x24*x59;
    const real_t x101 = x28*x60;
    const real_t x102 = x38*x60;
    const real_t x103 = x40*x61;
    const real_t x104 = x32*x61;
    const real_t x105 = x36*x89;
    const real_t x106 = x24*x71;
    const real_t x107 = x38*x92;
    const real_t x108 = x28*x75;
    const real_t x109 = x40*x95;
    const real_t x110 = x32*x79;
    const real_t x111 = -x100 - x101 - x102 - x103 - x104 + x105 + x106 + x107 + x108 + x109 + x110 - x99;
    const real_t x112 = -x0*x17*x9 + x0*x2*x4 - x11*x13*x2 + x11*x7*x9 + x13*x15*x17 - x15*x4*x7;
    const real_t x113 = (1.0/360.0)*x112;
    const real_t x114 = x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50;
    const real_t x115 = x114 - x27 - x31 - x35;
    const real_t x116 = x100 + x101 + x102 + x103 + x104 - x105 - x106 - x107 - x108 - x109 - x110 + x115 + x99;
    const real_t x117 = -x25 - x29 - x33;
    const real_t x118 = x117 + x63 + x64 + x65 + x66 + x67 + x68 - x70 - x72 - x74 - x76 - x78 - x80;
    const real_t x119 = -x37 - x39 - x41;
    const real_t x120 = x119 + x82 + x83 + x84 + x85 + x86 + x87 - x88 - x90 - x91 - x93 - x94 - x96;
    const real_t x121 = x114 + x120;
    const real_t x122 = 2*x24;
    const real_t x123 = 2*x28;
    const real_t x124 = 2*x32;
    const real_t x125 = -ux[6]*x122 - uy[6]*x123 - uz[6]*x124 + x25 + x29 + x33;
    const real_t x126 = 2*x26;
    const real_t x127 = 2*x30;
    const real_t x128 = 2*x34;
    const real_t x129 = -ux[7]*x126 - uy[7]*x127 - uz[7]*x128 + x27 + x31 + x35;
    const real_t x130 = x125 + x129;
    const real_t x131 = 2*x36;
    const real_t x132 = ux[8]*x131;
    const real_t x133 = ux[9]*x122;
    const real_t x134 = 2*x38;
    const real_t x135 = uy[8]*x134;
    const real_t x136 = uy[9]*x123;
    const real_t x137 = 2*x40;
    const real_t x138 = uz[8]*x137;
    const real_t x139 = uz[9]*x124;
    const real_t x140 = ux[7]*x131;
    const real_t x141 = ux[7]*x122;
    const real_t x142 = uy[7]*x123;
    const real_t x143 = uy[7]*x134;
    const real_t x144 = uz[7]*x137;
    const real_t x145 = uz[7]*x124;
    const real_t x146 = -x132 - x133 - x135 - x136 - x138 - x139 + x140 + x141 + x142 + x143 + x144 + x145;
    const real_t x147 = ux[5]*x131;
    const real_t x148 = ux[9]*x126;
    const real_t x149 = uy[5]*x134;
    const real_t x150 = uy[9]*x127;
    const real_t x151 = uz[5]*x137;
    const real_t x152 = uz[9]*x128;
    const real_t x153 = ux[6]*x126;
    const real_t x154 = ux[6]*x131;
    const real_t x155 = uy[6]*x127;
    const real_t x156 = uy[6]*x134;
    const real_t x157 = uz[6]*x128;
    const real_t x158 = uz[6]*x137;
    const real_t x159 = -x147 - x148 - x149 - x150 - x151 - x152 + x153 + x154 + x155 + x156 + x157 + x158;
    const real_t x160 = (1.0/90.0)*x112;
    const real_t x161 = -ux[4]*x131 - uy[4]*x134 - uz[4]*x137 + x37 + x39 + x41;
    const real_t x162 = x125 + x161;
    const real_t x163 = x114 + x129 + x161;
    const real_t x164 = ux[5]*x122;
    const real_t x165 = ux[8]*x126;
    const real_t x166 = uy[5]*x123;
    const real_t x167 = uy[8]*x127;
    const real_t x168 = uz[5]*x124;
    const real_t x169 = uz[8]*x128;
    const real_t x170 = ux[4]*x126;
    const real_t x171 = ux[4]*x122;
    const real_t x172 = uy[4]*x123;
    const real_t x173 = uy[4]*x127;
    const real_t x174 = uz[4]*x128;
    const real_t x175 = uz[4]*x124;
    const real_t x176 = -x164 - x165 - x166 - x167 - x168 - x169 + x170 + x171 + x172 + x173 + x174 + x175;
    element_vector[0] = x113*(x111 + x25 + x27 + x29 + x31 + x33 + x35 + x37 + x39 + x41 + 3*x42 + 3*x43 + 3*x44 + 3*x45 + 3*x46 + 3*x47 + 3*x48 + 3*x49 + 
    3*x50 + x54 + x58 + x62 + x98);
    element_vector[1] = -x113*(x116 + x118 + 3*x37 + 3*x39 + 3*x41 + x54);
    element_vector[2] = -x113*(x116 + x120 + 3*x25 + 3*x29 + 3*x33 + x58);
    element_vector[3] = -x113*(x118 + x121 + 3*x27 + 3*x31 + 3*x35 + x62);
    element_vector[4] = x160*(x121 + x130 + x146 + x159);
    element_vector[5] = -x160*(x115 + x132 + x133 + x135 + x136 + x138 + x139 - x140 - x141 - x142 - x143 - x144 - x145 + x162 + x98);
    element_vector[6] = x160*(x118 + x146 + x163 + x176);
    element_vector[7] = x160*(x116 + x159 + x162 + x176);
    element_vector[8] = -x160*(x111 + x117 + x147 + x148 + x149 + x150 + x151 + x152 - x153 - x154 - x155 - x156 - x157 - x158 + x163 + x97);
    element_vector[9] = -x160*(x111 + x114 + x119 + x130 + x164 + x165 + x166 + x167 + x168 + x169 - x170 - x171 - x172 - x173 - x174 - x175 + x81);
}

void tet10_div_apply(const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const elems,
                     geom_t **const xyz,
                     const real_t *const ux,
                     const real_t *const uy,
                     const real_t *const uz,
                     real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_vector[10];
    real_t element_ux[10];
    real_t element_uy[10];
    real_t element_uz[10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 10; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        for (int v = 0; v < 10; ++v) {
            element_uz[v] = uz[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_div_apply_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            element_ux,
            element_uy,
            element_uz,
            // Output
            element_vector);

#pragma unroll(10)
        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_div.c: tet10_div_apply\t%g seconds\n", tock - tick);
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

static SFEM_INLINE void tet10_integrate_div_kernel(const real_t px0,
                                                   const real_t px1,
                                                   const real_t px2,
                                                   const real_t px3,
                                                   const real_t py0,
                                                   const real_t py1,
                                                   const real_t py2,
                                                   const real_t py3,
                                                   const real_t pz0,
                                                   const real_t pz1,
                                                   const real_t pz2,
                                                   const real_t pz3,
                                                   const real_t *SFEM_RESTRICT ux,
                                                   const real_t *SFEM_RESTRICT uy,
                                                   const real_t *SFEM_RESTRICT uz,
                                                   real_t *SFEM_RESTRICT element_value) {
    // FLOATING POINT OPS!
    //       - Result: 2*ADD + ASSIGNMENT + 43*MUL
    //       - Subexpressions: 2*ADD + DIV + 34*MUL + 9*NEG + 21*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py3;
    const real_t x5 = pz0 - pz1;
    const real_t x6 = px0 - px3;
    const real_t x7 = py0 - py1;
    const real_t x8 = pz0 - pz2;
    const real_t x9 = -x3;
    const real_t x10 = -x4;
    const real_t x11 = x10 * x9;
    const real_t x12 = -x6;
    const real_t x13 = -x1;
    const real_t x14 = x12 * x13;
    const real_t x15 = x11 - x14;
    const real_t x16 = -x0;
    const real_t x17 = -x2;
    const real_t x18 = x13 * x17;
    const real_t x19 = -x5;
    const real_t x20 = -x7;
    const real_t x21 = -x8;
    const real_t x22 = x10 * x21;
    const real_t x23 = x17 * x9;
    const real_t x24 = 1.0 / (x11 * x19 + x12 * x20 * x21 - x14 * x19 + x16 * x18 - x16 * x22 - x20 * x23);
    const real_t x25 = ux[4] * x24;
    const real_t x26 = x12 * x21 - x23;
    const real_t x27 = ux[6] * x24;
    const real_t x28 = x18 - x22;
    const real_t x29 = ux[7] * x24;
    const real_t x30 = -x10 * x16 + x12 * x20;
    const real_t x31 = uy[4] * x24;
    const real_t x32 = -x12 * x19 + x16 * x17;
    const real_t x33 = uy[6] * x24;
    const real_t x34 = x10 * x19 - x17 * x20;
    const real_t x35 = uy[7] * x24;
    const real_t x36 = x13 * x16 - x20 * x9;
    const real_t x37 = uz[4] * x24;
    const real_t x38 = -x16 * x21 + x19 * x9;
    const real_t x39 = uz[6] * x24;
    const real_t x40 = -x13 * x19 + x20 * x21;
    const real_t x41 = uz[7] * x24;
    element_value[0] =
        -1.0 / 6.0 * (x0 * x1 * x2 - x0 * x4 * x8 - x1 * x5 * x6 - x2 * x3 * x7 + x3 * x4 * x5 + x6 * x7 * x8) *
        (ux[5] * x24 * x26 + ux[5] * x24 * x28 + ux[8] * x15 * x24 + ux[8] * x24 * x28 + ux[9] * x15 * x24 +
         ux[9] * x24 * x26 + uy[5] * x24 * x32 + uy[5] * x24 * x34 + uy[8] * x24 * x30 + uy[8] * x24 * x34 +
         uy[9] * x24 * x30 + uy[9] * x24 * x32 + uz[5] * x24 * x38 + uz[5] * x24 * x40 + uz[8] * x24 * x36 +
         uz[8] * x24 * x40 + uz[9] * x24 * x36 + uz[9] * x24 * x38 - x15 * x25 - x15 * x27 - x25 * x26 - x26 * x29 -
         x27 * x28 - x28 * x29 - x30 * x31 - x30 * x33 - x31 * x32 - x32 * x35 - x33 * x34 - x34 * x35 - x36 * x37 -
         x36 * x39 - x37 * x38 - x38 * x41 - x39 * x40 - x40 * x41);
}

void tet10_integrate_div(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const elems,
                         geom_t **const xyz,
                         const real_t *const ux,
                         const real_t *const uy,
                         const real_t *const uz,
                         real_t *const value) {
        SFEM_UNUSED(nnodes);

        double tick = MPI_Wtime();

        idx_t ev[10];
        real_t element_ux[10];
        real_t element_uy[10];
        real_t element_uz[10];

        *value = 0.;

        for (ptrdiff_t i = 0; i < nelements; ++i) {
    #pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];

                assert(ev[v] >= 0);
                assert(ev[v] < nnodes);
            }

            for (int v = 0; v < 10; ++v) {
                element_ux[v] = ux[ev[v]];
            }

            for (int v = 0; v < 10; ++v) {
                element_uy[v] = uy[ev[v]];
            }

            for (int v = 0; v < 10; ++v) {
                element_uz[v] = uz[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            real_t element_scalar = 0;

            tet10_integrate_div_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],
                // Data
                element_ux,
                element_uy,
                element_uz,
                // Output
                &element_scalar);

            *value += element_scalar;
        }

        double tock = MPI_Wtime();
        printf("tet10_div.c: tet10_integrate_div\t%g seconds\n", tock - tick);
}

void tet10_cdiv(const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **const SFEM_RESTRICT xyz,
                const real_t *const SFEM_RESTRICT ux,
                const real_t *const SFEM_RESTRICT uy,
                const real_t *const SFEM_RESTRICT uz,
                real_t *const SFEM_RESTRICT div) {
    // TODO
    fprintf(stderr, "tet10_cdiv not implemented!\n");
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, -1);
    
}
