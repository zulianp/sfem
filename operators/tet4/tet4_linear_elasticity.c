#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"
#include "tet4_linear_elasticity_inline_cpu.h"

#define RPOW2(l) (1. / ((l) * (l)))
// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MAX(a, b) ((a) > (b) ? (a) : (b))

static SFEM_INLINE void value_kernel(const scalar_t mu,
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
                                     accumulator_t *const SFEM_RESTRICT element_scalar) {
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

static SFEM_INLINE void hessian_kernel(const real_t mu,
                                       const real_t lambda,
                                       const real_t px0,
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
                                       real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = -px0 + px2;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz1;
    const real_t x7 = -px0 + px3;
    const real_t x8 = -py0 + py1;
    const real_t x9 = -pz0 + pz2;
    const real_t x10 = x8 * x9;
    const real_t x11 = x5 * x9;
    const real_t x12 = x2 * x8;
    const real_t x13 = x1 * x6;
    const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const real_t x15 = -x11 + x3;
    const real_t x16 = RPOW2(x14);
    const real_t x17 = x10 - x13;
    const real_t x18 = x16 * x17;
    const real_t x19 = x15 * x18;
    const real_t x20 = (1.0 / 3.0) * lambda;
    const real_t x21 = -x12 + x5 * x6;
    const real_t x22 = x20 * x21;
    const real_t x23 = x15 * x16;
    const real_t x24 = (2.0 / 3.0) * mu;
    const real_t x25 = x21 * x24;
    const real_t x26 = POW2(x15);
    const real_t x27 = (1.0 / 6.0) * lambda;
    const real_t x28 = x16 * x27;
    const real_t x29 = -x1 * x7 + x4 * x5;
    const real_t x30 = POW2(x29);
    const real_t x31 = (1.0 / 6.0) * mu;
    const real_t x32 = x16 * x31;
    const real_t x33 = x30 * x32;
    const real_t x34 = -x2 * x4 + x7 * x9;
    const real_t x35 = POW2(x34);
    const real_t x36 = x32 * x35;
    const real_t x37 = (1.0 / 3.0) * mu;
    const real_t x38 = x16 * x37;
    const real_t x39 = x26 * x28 + x26 * x38 + x33 + x36;
    const real_t x40 = POW2(x21);
    const real_t x41 = -x0 * x5 + x7 * x8;
    const real_t x42 = POW2(x41);
    const real_t x43 = x32 * x42;
    const real_t x44 = x0 * x2 - x6 * x7;
    const real_t x45 = POW2(x44);
    const real_t x46 = x32 * x45;
    const real_t x47 = x28 * x40 + x38 * x40 + x43 + x46;
    const real_t x48 = x16 * POW2(x17);
    const real_t x49 = x0 * x1 - x4 * x8;
    const real_t x50 = POW2(x49);
    const real_t x51 = x32 * x50;
    const real_t x52 = -x0 * x9 + x4 * x6;
    const real_t x53 = POW2(x52);
    const real_t x54 = x32 * x53;
    const real_t x55 = x27 * x48 + x37 * x48 + x51 + x54;
    const real_t x56 = x29 * x41;
    const real_t x57 = x38 * x56;
    const real_t x58 = x38 * x49;
    const real_t x59 = x29 * x58;
    const real_t x60 = x41 * x58;
    const real_t x61 = x57 + x59 + x60;
    const real_t x62 = x38 * x44;
    const real_t x63 = x34 * x62;
    const real_t x64 = x34 * x52;
    const real_t x65 = x38 * x64;
    const real_t x66 = x52 * x62;
    const real_t x67 = x63 + x65 + x66;
    const real_t x68 = x15 * x28;
    const real_t x69 = x32 * x56;
    const real_t x70 = x32 * x44;
    const real_t x71 = x34 * x70;
    const real_t x72 = x15 * x38;
    const real_t x73 = x21 * x72;
    const real_t x74 = x21 * x68 + x69 + x71 + x73;
    const real_t x75 = x32 * x49;
    const real_t x76 = x29 * x75;
    const real_t x77 = x32 * x64;
    const real_t x78 = x17 * x72;
    const real_t x79 = x17 * x68 + x76 + x77 + x78;
    const real_t x80 = x14 * (-x39 - x74 - x79);
    const real_t x81 = x17 * x21;
    const real_t x82 = x41 * x75;
    const real_t x83 = x52 * x70;
    const real_t x84 = x38 * x81;
    const real_t x85 = x28 * x81 + x82 + x83 + x84;
    const real_t x86 = x14 * (-x47 - x74 - x85);
    const real_t x87 = x14 * (-x55 - x79 - x85);
    const real_t x88 = x32 * x34;
    const real_t x89 = x15 * x88 + x34 * x68;
    const real_t x90 = x28 * x34;
    const real_t x91 = x15 * x70 + x21 * x90;
    const real_t x92 = x32 * x52;
    const real_t x93 = x15 * x92 + x17 * x90;
    const real_t x94 = x89 + x91 + x93;
    const real_t x95 = x21 * x88 + x44 * x68;
    const real_t x96 = x28 * x44;
    const real_t x97 = x21 * x70 + x21 * x96;
    const real_t x98 = x17 * x96 + x21 * x92;
    const real_t x99 = x95 + x97 + x98;
    const real_t x100 = x17 * x88 + x52 * x68;
    const real_t x101 = x28 * x52;
    const real_t x102 = x101 * x21 + x17 * x70;
    const real_t x103 = x101 * x17 + x17 * x92;
    const real_t x104 = x100 + x102 + x103;
    const real_t x105 = x14 * (x104 + x94 + x99);
    const real_t x106 = -x14 * x94;
    const real_t x107 = -x14 * x99;
    const real_t x108 = -x104 * x14;
    const real_t x109 = x29 * x32;
    const real_t x110 = x109 * x15 + x29 * x68;
    const real_t x111 = x28 * x29;
    const real_t x112 = x32 * x41;
    const real_t x113 = x111 * x21 + x112 * x15;
    const real_t x114 = x111 * x17 + x15 * x75;
    const real_t x115 = x110 + x113 + x114;
    const real_t x116 = x109 * x21 + x41 * x68;
    const real_t x117 = x28 * x41;
    const real_t x118 = x112 * x21 + x117 * x21;
    const real_t x119 = x117 * x17 + x21 * x75;
    const real_t x120 = x116 + x118 + x119;
    const real_t x121 = x109 * x17 + x49 * x68;
    const real_t x122 = x28 * x49;
    const real_t x123 = x112 * x17 + x122 * x21;
    const real_t x124 = x122 * x17 + x17 * x75;
    const real_t x125 = x121 + x123 + x124;
    const real_t x126 = x14 * (x115 + x120 + x125);
    const real_t x127 = -x115 * x14;
    const real_t x128 = -x120 * x14;
    const real_t x129 = -x125 * x14;
    const real_t x130 = x14 * x74;
    const real_t x131 = x14 * x79;
    const real_t x132 = x14 * (-x100 - x89 - x95);
    const real_t x133 = x14 * x89;
    const real_t x134 = x14 * x95;
    const real_t x135 = x100 * x14;
    const real_t x136 = x14 * (-x110 - x116 - x121);
    const real_t x137 = x110 * x14;
    const real_t x138 = x116 * x14;
    const real_t x139 = x121 * x14;
    const real_t x140 = x14 * x85;
    const real_t x141 = x14 * (-x102 - x91 - x97);
    const real_t x142 = x14 * x91;
    const real_t x143 = x14 * x97;
    const real_t x144 = x102 * x14;
    const real_t x145 = x14 * (-x113 - x118 - x123);
    const real_t x146 = x113 * x14;
    const real_t x147 = x118 * x14;
    const real_t x148 = x123 * x14;
    const real_t x149 = x14 * (-x103 - x93 - x98);
    const real_t x150 = x14 * x93;
    const real_t x151 = x14 * x98;
    const real_t x152 = x103 * x14;
    const real_t x153 = x14 * (-x114 - x119 - x124);
    const real_t x154 = x114 * x14;
    const real_t x155 = x119 * x14;
    const real_t x156 = x124 * x14;
    const real_t x157 = x16 * x20;
    const real_t x158 = x157 * x44;
    const real_t x159 = x16 * x24;
    const real_t x160 = x159 * x44;
    const real_t x161 = x26 * x32;
    const real_t x162 = x161 + x28 * x35 + x33 + x35 * x38;
    const real_t x163 = x32 * x40;
    const real_t x164 = x163 + x28 * x45 + x38 * x45 + x43;
    const real_t x165 = x31 * x48;
    const real_t x166 = x165 + x28 * x53 + x38 * x53 + x51;
    const real_t x167 = x73 + x78 + x84;
    const real_t x168 = x15 * x32;
    const real_t x169 = x168 * x21;
    const real_t x170 = x169 + x34 * x96 + x63 + x69;
    const real_t x171 = x168 * x17;
    const real_t x172 = x171 + x28 * x64 + x65 + x76;
    const real_t x173 = x14 * (-x162 - x170 - x172);
    const real_t x174 = x32 * x81;
    const real_t x175 = x174 + x52 * x96 + x66 + x82;
    const real_t x176 = x14 * (-x164 - x170 - x175);
    const real_t x177 = x14 * (-x166 - x172 - x175);
    const real_t x178 = x29 * x88 + x29 * x90;
    const real_t x179 = x29 * x96 + x41 * x88;
    const real_t x180 = x101 * x29 + x34 * x75;
    const real_t x181 = x178 + x179 + x180;
    const real_t x182 = x29 * x70 + x41 * x90;
    const real_t x183 = x41 * x70 + x41 * x96;
    const real_t x184 = x101 * x41 + x44 * x75;
    const real_t x185 = x182 + x183 + x184;
    const real_t x186 = x29 * x92 + x49 * x90;
    const real_t x187 = x41 * x92 + x49 * x96;
    const real_t x188 = x101 * x49 + x52 * x75;
    const real_t x189 = x186 + x187 + x188;
    const real_t x190 = x14 * (x181 + x185 + x189);
    const real_t x191 = -x14 * x181;
    const real_t x192 = -x14 * x185;
    const real_t x193 = -x14 * x189;
    const real_t x194 = x14 * x170;
    const real_t x195 = x14 * x172;
    const real_t x196 = x14 * (-x178 - x182 - x186);
    const real_t x197 = x14 * x178;
    const real_t x198 = x14 * x182;
    const real_t x199 = x14 * x186;
    const real_t x200 = x14 * x175;
    const real_t x201 = x14 * (-x179 - x183 - x187);
    const real_t x202 = x14 * x179;
    const real_t x203 = x14 * x183;
    const real_t x204 = x14 * x187;
    const real_t x205 = x14 * (-x180 - x184 - x188);
    const real_t x206 = x14 * x180;
    const real_t x207 = x14 * x184;
    const real_t x208 = x14 * x188;
    const real_t x209 = x157 * x49;
    const real_t x210 = x159 * x49;
    const real_t x211 = x161 + x28 * x30 + x30 * x38 + x36;
    const real_t x212 = x163 + x28 * x42 + x38 * x42 + x46;
    const real_t x213 = x165 + x28 * x50 + x38 * x50 + x54;
    const real_t x214 = x169 + x28 * x56 + x57 + x71;
    const real_t x215 = x122 * x29 + x171 + x59 + x77;
    const real_t x216 = x14 * (-x211 - x214 - x215);
    const real_t x217 = x122 * x41 + x174 + x60 + x83;
    const real_t x218 = x14 * (-x212 - x214 - x217);
    const real_t x219 = x14 * (-x213 - x215 - x217);
    const real_t x220 = x14 * x214;
    const real_t x221 = x14 * x215;
    const real_t x222 = x14 * x217;
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

int tet4_linear_elasticity_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const real_t *const uz,
                                 real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    real_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        scalar_t element_ux[4];
        scalar_t element_uy[4];
        scalar_t element_uz[4];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int enode = 0; enode < 4; ++enode) {
            idx_t dof = ev[enode] * u_stride;
            element_ux[enode] = ux[dof];
            element_uy[enode] = uy[dof];
            element_uz[enode] = uz[dof];
        }

        real_t element_scalar = 0;
        value_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_ux,
                element_uy,
                element_uz,
                // output vector
                &element_scalar);

        acc += element_scalar;
    }

    *value += acc;

    return 0;
}

int tet4_linear_elasticity_hessian(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t mu,
                                   const real_t lambda,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        idx_t ks[4];

        real_t element_matrix[(4 * 3) * (4 * 3)];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        hessian_kernel(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                points[0][i0],
                points[0][i1],
                points[0][i2],
                points[0][i3],
                // Y-coordinates
                points[1][i0],
                points[1][i1],
                points[1][i2],
                points[1][i3],
                // Z-coordinates
                points[2][i0],
                points[2][i1],
                points[2][i2],
                points[2][i3],

                // output matrix
                element_matrix);

        // assert(!check_symmetric(4 * block_size, element_matrix));

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = elements[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            {
                const idx_t *row = &colidx[rowptr[dof_i]];
                tet4_find_cols(ev, row, lenrow, ks);
            }

            // Blocks for row
            real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                const idx_t offset_j = ks[edof_j] * block_size;

                for (int bi = 0; bi < block_size; ++bi) {
                    const int ii = bi * 4 + edof_i;

                    // Jump rows (including the block-size for the columns)
                    real_t *row = &block_start[bi * lenrow * block_size];

                    for (int bj = 0; bj < block_size; ++bj) {
                        const int jj = bj * 4 + edof_j;
                        const real_t val = element_matrix[ii * 12 + jj];

#pragma omp atomic update
                        row[offset_j + bj] += val;
                    }
                }
            }
        }
    }

    return 0;
}

int tet4_linear_elasticity_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const real_t *const uz,
                                 const ptrdiff_t out_stride,
                                 real_t *const outx,
                                 real_t *const outy,
                                 real_t *const outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        scalar_t element_ux[4];
        scalar_t element_uy[4];
        scalar_t element_uz[4];

        accumulator_t element_outx[4];
        accumulator_t element_outy[4];
        accumulator_t element_outz[4];

        jacobian_t jacobian_adjugate[9];
        jacobian_t jacobian_determinant = 0;

        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v] * u_stride];
            element_uy[v] = uy[ev[v] * u_stride];
            element_uz[v] = uz[ev[v] * u_stride];
        }

        tet4_adjugate_and_det(x[ev[0]],
                              x[ev[1]],
                              x[ev[2]],
                              x[ev[3]],
                              // Y-coordinates
                              y[ev[0]],
                              y[ev[1]],
                              y[ev[2]],
                              y[ev[3]],
                              // Z-coordinates
                              z[ev[0]],
                              z[ev[1]],
                              z[ev[2]],
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        tet4_linear_elasticity_apply_adj(
                                         jacobian_adjugate,
                                         jacobian_determinant,
                                         mu,
                                                                                  lambda,
                                         element_ux,
                                         element_uy,
                                         element_uz,
                                         element_outx,
                                         element_outy,
                                         element_outz);

        for (int edof_i = 0; edof_i < 4; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return 0;
}

int tet4_linear_elasticity_diag(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const ptrdiff_t out_stride,
                                real_t *const outx,
                                real_t *const outy,
                                real_t *const outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];

        accumulator_t element_outx[4];
        accumulator_t element_outy[4];
        accumulator_t element_outz[4];

        jacobian_t jacobian_adjugate[9];
        jacobian_t jacobian_determinant = 0;

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det(x[ev[0]],
                              x[ev[1]],
                              x[ev[2]],
                              x[ev[3]],
                              // Y-coordinates
                              y[ev[0]],
                              y[ev[1]],
                              y[ev[2]],
                              y[ev[3]],
                              // Z-coordinates
                              z[ev[0]],
                              z[ev[1]],
                              z[ev[2]],
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        tet4_linear_elasticity_diag_adj(mu,
                                        lambda,
                                        jacobian_adjugate,
                                        jacobian_determinant,
                                        // Output
                                        element_outx,
                                        element_outy,
                                        element_outz);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return 0;
}
