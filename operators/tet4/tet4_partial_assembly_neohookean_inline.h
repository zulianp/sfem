#ifndef TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H
#define TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H

static SFEM_INLINE void tet4_F(const scalar_t *const SFEM_RESTRICT adjugate,
                               const scalar_t                      jacobian_determinant,
                               const scalar_t *const SFEM_RESTRICT ux,
                               const scalar_t *const SFEM_RESTRICT uy,
                               const scalar_t *const SFEM_RESTRICT uz,
                               scalar_t *const SFEM_RESTRICT       F) {
    // mundane ops: 96 divs: 1 sqrts: 0
    // total ops: 104
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

static SFEM_INLINE void tet4_inc_grad(const scalar_t *const SFEM_RESTRICT adjugate,
                                      const scalar_t                      jacobian_determinant,
                                      const scalar_t *const SFEM_RESTRICT ux,
                                      const scalar_t *const SFEM_RESTRICT uy,
                                      const scalar_t *const SFEM_RESTRICT uz,
                                      scalar_t *const SFEM_RESTRICT       inc_grad) {
    // mundane ops: 81 divs: 1 sqrts: 0
    // total ops: 89

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
    inc_grad[0]       = -x0 * (adjugate[0] * x1 + adjugate[3] * x2 + adjugate[6] * x3);
    inc_grad[1]       = -x0 * (adjugate[1] * x1 + adjugate[4] * x2 + adjugate[7] * x3);
    inc_grad[2]       = -x0 * (adjugate[2] * x1 + adjugate[5] * x2 + adjugate[8] * x3);
    inc_grad[3]       = -x0 * (adjugate[0] * x4 + adjugate[3] * x5 + adjugate[6] * x6);
    inc_grad[4]       = -x0 * (adjugate[1] * x4 + adjugate[4] * x5 + adjugate[7] * x6);
    inc_grad[5]       = -x0 * (adjugate[2] * x4 + adjugate[5] * x5 + adjugate[8] * x6);
    inc_grad[6]       = -x0 * (adjugate[0] * x7 + adjugate[3] * x8 + adjugate[6] * x9);
    inc_grad[7]       = -x0 * (adjugate[1] * x7 + adjugate[4] * x8 + adjugate[7] * x9);
    inc_grad[8]       = -x0 * (adjugate[2] * x7 + adjugate[5] * x8 + adjugate[8] * x9);
}

static SFEM_INLINE void tet4_S_iklm(const scalar_t *const SFEM_RESTRICT adjugate,
                                    const scalar_t                      jacobian_determinant,
                                    const scalar_t *const SFEM_RESTRICT F,
                                    const scalar_t                      mu,
                                    const scalar_t                      lmbda,
                                    scalar_t *const SFEM_RESTRICT       S_iklm) {
    // mundane ops: 1759 divs: 2 sqrts: 0
    // total ops: 1775

    const scalar_t x0   = F[3] * F[8];
    const scalar_t x1   = -F[5] * F[6] + x0;
    const scalar_t x2   = F[4] * F[8];
    const scalar_t x3   = F[5] * F[7];
    const scalar_t x4   = x2 - x3;
    const scalar_t x5   = F[3] * F[7];
    const scalar_t x6   = F[4] * F[6];
    const scalar_t x7   = F[0] * x2 - F[0] * x3 + F[1] * F[5] * F[6] - F[1] * x0 + F[2] * x5 - F[2] * x6;
    const scalar_t x8   = 9 * lmbda;
    const scalar_t x9   = x8 / POW2(x7);
    const scalar_t x10  = x4 * x9;
    const scalar_t x11  = x1 * x10;
    const scalar_t x12  = log(x7);
    const scalar_t x13  = x11 * x12;
    const scalar_t x14  = 3 * F[1];
    const scalar_t x15  = pow(x7, -5.0 / 3.0);
    const scalar_t x16  = POW2(F[0]) + POW2(F[1]) + POW2(F[2]);
    const scalar_t x17  = POW2(F[3]) + POW2(F[4]) + POW2(F[5]);
    const scalar_t x18  = POW2(F[6]) + POW2(F[7]) + POW2(F[8]);
    const scalar_t x19  = x16 + x17 + x18;
    const scalar_t x20  = x15 * x19;
    const scalar_t x21  = x1 * x20 + x14;
    const scalar_t x22  = x15 * x4;
    const scalar_t x23  = 2 * mu;
    const scalar_t x24  = x22 * x23;
    const scalar_t x25  = -x1;
    const scalar_t x26  = 1.0 / x7;
    const scalar_t x27  = -x4;
    const scalar_t x28  = x26 * x27;
    const scalar_t x29  = x16 * x28;
    const scalar_t x30  = x17 * x28;
    const scalar_t x31  = x18 * x28;
    const scalar_t x32  = 3 * x19;
    const scalar_t x33  = x28 * x32;
    const scalar_t x34  = mu / pow(x7, 7.0 / 3.0);
    const scalar_t x35  = x34 * (6 * F[0] + 2 * x29 + 2 * x30 + 2 * x31 + x33);
    const scalar_t x36  = x11 - x13 + x21 * x24 + x25 * x35;
    const scalar_t x37  = x5 - x6;
    const scalar_t x38  = 3 * F[2];
    const scalar_t x39  = -x20 * x37 + x38;
    const scalar_t x40  = x10 * x37;
    const scalar_t x41  = x12 * x40 - x40;
    const scalar_t x42  = x24 * x39 + x35 * x37 + x41;
    const scalar_t x43  = POW2(x4) * x9;
    const scalar_t x44  = 3 * F[0];
    const scalar_t x45  = -x20 * x4 + x44;
    const scalar_t x46  = x32 / pow(x7, 8.0 / 3.0);
    const scalar_t x47  = 2 * x29 + 2 * x30 + 2 * x31 + 2 * x44;
    const scalar_t x48  = mu / pow(x7, 2.0 / 3.0);
    const scalar_t x49  = x12 * x43 + x24 * x45 - x43 + x48 * (x22 * x47 + x27 * x4 * x46 - 9);
    const scalar_t x50  = (1.0 / 9.0) / jacobian_determinant;
    const scalar_t x51  = x23 * x45;
    const scalar_t x52  = x1 * x15;
    const scalar_t x53  = x1 * x26;
    const scalar_t x54  = x16 * x53;
    const scalar_t x55  = x17 * x53;
    const scalar_t x56  = x18 * x53;
    const scalar_t x57  = x32 * x53;
    const scalar_t x58  = x34 * (6 * F[1] + 2 * x54 + 2 * x55 + 2 * x56 + x57);
    const scalar_t x59  = -x11 + x13 - x4 * x58 + x51 * x52;
    const scalar_t x60  = x37 * x9;
    const scalar_t x61  = x1 * x60;
    const scalar_t x62  = x12 * x61;
    const scalar_t x63  = x23 * x52;
    const scalar_t x64  = -x37 * x58 + x39 * x63 - x61 + x62;
    const scalar_t x65  = POW2(x1) * x9;
    const scalar_t x66  = 2 * x14 + 2 * x54 + 2 * x55 + 2 * x56;
    const scalar_t x67  = x12 * x65 - x21 * x63 + x48 * (x1 * x25 * x46 + x15 * x25 * x66 - 9) - x65;
    const scalar_t x68  = x15 * x37;
    const scalar_t x69  = x23 * x68;
    const scalar_t x70  = -x37;
    const scalar_t x71  = x26 * x70;
    const scalar_t x72  = x16 * x71;
    const scalar_t x73  = x17 * x71;
    const scalar_t x74  = x18 * x71;
    const scalar_t x75  = x32 * x71;
    const scalar_t x76  = x34 * (6 * F[2] + 2 * x72 + 2 * x73 + 2 * x74 + x75);
    const scalar_t x77  = x21 * x69 + x25 * x76 + x61 - x62;
    const scalar_t x78  = x4 * x76 + x41 + x51 * x68;
    const scalar_t x79  = POW2(x37) * x9;
    const scalar_t x80  = 2 * x38 + 2 * x72 + 2 * x73 + 2 * x74;
    const scalar_t x81  = x12 * x79 + x39 * x69 + x48 * (x37 * x46 * x70 + x68 * x80 - 9) - x79;
    const scalar_t x82  = F[1] * F[8] - F[2] * F[7];
    const scalar_t x83  = x10 * x82;
    const scalar_t x84  = x12 * x83;
    const scalar_t x85  = x15 * x82;
    const scalar_t x86  = x26 * x82;
    const scalar_t x87  = x16 * x86;
    const scalar_t x88  = x17 * x86;
    const scalar_t x89  = x18 * x86;
    const scalar_t x90  = x32 * x86;
    const scalar_t x91  = 6 * F[3] + 2 * x87 + 2 * x88 + 2 * x89 + x90;
    const scalar_t x92  = x34 * x4;
    const scalar_t x93  = x51 * x85 - x83 + x84 - x91 * x92;
    const scalar_t x94  = 3 * F[8];
    const scalar_t x95  = x19 * x94;
    const scalar_t x96  = -x95;
    const scalar_t x97  = 3 * F[3];
    const scalar_t x98  = 2 * x87 + 2 * x88 + 2 * x89 + 2 * x97;
    const scalar_t x99  = x23 * x85;
    const scalar_t x100 = x1 * x9;
    const scalar_t x101 = x100 * x82;
    const scalar_t x102 = x12 * x26 * x8;
    const scalar_t x103 = F[8] * x102;
    const scalar_t x104 = x101 * x12 - x101 + x103;
    const scalar_t x105 = x104 - x21 * x99 + x34 * (x25 * x90 + x25 * x98 + x96);
    const scalar_t x106 = x60 * x82;
    const scalar_t x107 = F[7] * x102;
    const scalar_t x108 = x106 * x12;
    const scalar_t x109 = 3 * F[7];
    const scalar_t x110 = x109 * x19;
    const scalar_t x111 = -x106 + x107 + x108 - x34 * (x110 + x37 * x90 + x37 * x98) + x39 * x99;
    const scalar_t x112 = F[0] * F[8] - F[2] * F[6];
    const scalar_t x113 = x100 * x112;
    const scalar_t x114 = x113 * x12;
    const scalar_t x115 = x112 * x15;
    const scalar_t x116 = x115 * x23;
    const scalar_t x117 = -x112;
    const scalar_t x118 = x117 * x26;
    const scalar_t x119 = x118 * x16;
    const scalar_t x120 = x118 * x17;
    const scalar_t x121 = x118 * x18;
    const scalar_t x122 = x118 * x32;
    const scalar_t x123 = 6 * F[4] + 2 * x119 + 2 * x120 + 2 * x121 + x122;
    const scalar_t x124 = x25 * x34;
    const scalar_t x125 = x113 - x114 + x116 * x21 + x123 * x124;
    const scalar_t x126 = 3 * F[4];
    const scalar_t x127 = 2 * x119 + 2 * x120 + 2 * x121 + 2 * x126;
    const scalar_t x128 = x10 * x112;
    const scalar_t x129 = -x103 + x12 * x128 - x128;
    const scalar_t x130 = x115 * x51 + x129 + x34 * (x122 * x4 + x127 * x4 + x95);
    const scalar_t x131 = 3 * F[6];
    const scalar_t x132 = -x131 * x19;
    const scalar_t x133 = x112 * x60;
    const scalar_t x134 = F[6] * x102;
    const scalar_t x135 = x12 * x133 - x133 + x134;
    const scalar_t x136 = x116 * x39 + x135 + x34 * (x122 * x37 + x127 * x37 + x132);
    const scalar_t x137 = F[0] * F[7] - F[1] * F[6];
    const scalar_t x138 = x137 * x60;
    const scalar_t x139 = x12 * x138;
    const scalar_t x140 = x137 * x15;
    const scalar_t x141 = x23 * x39;
    const scalar_t x142 = x137 * x26;
    const scalar_t x143 = x142 * x16;
    const scalar_t x144 = x142 * x17;
    const scalar_t x145 = x142 * x18;
    const scalar_t x146 = x142 * x32;
    const scalar_t x147 = 6 * F[5] + 2 * x143 + 2 * x144 + 2 * x145 + x146;
    const scalar_t x148 = x34 * x37;
    const scalar_t x149 = -x138 + x139 + x140 * x141 - x147 * x148;
    const scalar_t x150 = 3 * F[5];
    const scalar_t x151 = 2 * x143 + 2 * x144 + 2 * x145 + 2 * x150;
    const scalar_t x152 = x132 + x137 * x57;
    const scalar_t x153 = x21 * x23;
    const scalar_t x154 = x100 * x137;
    const scalar_t x155 = -x12 * x154 + x134 + x154;
    const scalar_t x156 = x140 * x153 + x155 - x34 * (-x1 * x151 - x152);
    const scalar_t x157 = -x110;
    const scalar_t x158 = x10 * x137;
    const scalar_t x159 = x107 - x12 * x158 + x158;
    const scalar_t x160 = -x140 * x51 + x159 + x34 * (x146 * x4 + x151 * x4 + x157);
    const scalar_t x161 = F[1] * F[5] - F[2] * F[4];
    const scalar_t x162 = -x161;
    const scalar_t x163 = x162 * x26;
    const scalar_t x164 = x16 * x163;
    const scalar_t x165 = x163 * x17;
    const scalar_t x166 = x163 * x18;
    const scalar_t x167 = x163 * x32;
    const scalar_t x168 = 6 * F[6] + 2 * x164 + 2 * x165 + 2 * x166 + x167;
    const scalar_t x169 = x15 * x161;
    const scalar_t x170 = x10 * x161;
    const scalar_t x171 = x12 * x170 - x170;
    const scalar_t x172 = x168 * x92 + x169 * x51 + x171;
    const scalar_t x173 = x100 * x161;
    const scalar_t x174 = F[5] * x102;
    const scalar_t x175 = x12 * x173;
    const scalar_t x176 = x150 * x19;
    const scalar_t x177 = 2 * x131 + 2 * x164 + 2 * x165 + 2 * x166;
    const scalar_t x178 = x153 * x169 + x173 - x174 - x175 + x34 * (x167 * x25 + x176 + x177 * x25);
    const scalar_t x179 = x126 * x19;
    const scalar_t x180 = -x179;
    const scalar_t x181 = x161 * x60;
    const scalar_t x182 = F[4] * x102;
    const scalar_t x183 = x12 * x181 - x181 + x182;
    const scalar_t x184 = x141 * x169 + x183 + x34 * (x167 * x37 + x177 * x37 + x180);
    const scalar_t x185 = F[0] * F[5] - F[2] * F[3];
    const scalar_t x186 = x185 * x26;
    const scalar_t x187 = x16 * x186;
    const scalar_t x188 = x17 * x186;
    const scalar_t x189 = x18 * x186;
    const scalar_t x190 = x186 * x32;
    const scalar_t x191 = 6 * F[7] + 2 * x187 + 2 * x188 + 2 * x189 + x190;
    const scalar_t x192 = x15 * x185;
    const scalar_t x193 = x100 * x185;
    const scalar_t x194 = x12 * x193 - x193;
    const scalar_t x195 = x124 * x191 - x153 * x192 + x194;
    const scalar_t x196 = -x176;
    const scalar_t x197 = 2 * x109 + 2 * x187 + 2 * x188 + 2 * x189;
    const scalar_t x198 = x10 * x185;
    const scalar_t x199 = -x12 * x198 + x174 + x198;
    const scalar_t x200 = -x192 * x51 + x199 + x34 * (x190 * x4 + x196 + x197 * x4);
    const scalar_t x201 = x185 * x60;
    const scalar_t x202 = F[3] * x102;
    const scalar_t x203 = x12 * x201;
    const scalar_t x204 = x19 * x97;
    const scalar_t x205 = x141 * x192 - x201 + x202 + x203 - x34 * (x190 * x37 + x197 * x37 + x204);
    const scalar_t x206 = F[0] * F[4] - F[1] * F[3];
    const scalar_t x207 = -x206;
    const scalar_t x208 = x207 * x26;
    const scalar_t x209 = x16 * x208;
    const scalar_t x210 = x17 * x208;
    const scalar_t x211 = x18 * x208;
    const scalar_t x212 = x208 * x32;
    const scalar_t x213 = 6 * F[8] + 2 * x209 + 2 * x210 + 2 * x211 + x212;
    const scalar_t x214 = x15 * x206;
    const scalar_t x215 = x206 * x60;
    const scalar_t x216 = x12 * x215 - x215;
    const scalar_t x217 = x141 * x214 + x148 * x213 + x216;
    const scalar_t x218 = -x204;
    const scalar_t x219 = 2 * x209 + 2 * x210 + 2 * x211 + 2 * x94;
    const scalar_t x220 = x100 * x206;
    const scalar_t x221 = -x12 * x220 + x202 + x220;
    const scalar_t x222 = x153 * x214 + x221 + x34 * (x212 * x25 + x218 + x219 * x25);
    const scalar_t x223 = x10 * x206;
    const scalar_t x224 = x12 * x223 - x182 - x223;
    const scalar_t x225 = x214 * x51 + x224 + x34 * (x179 + x212 * x4 + x219 * x4);
    const scalar_t x226 = x20 * x82 + x97;
    const scalar_t x227 = -x82;
    const scalar_t x228 = x226 * x24 + x227 * x35 + x83 - x84;
    const scalar_t x229 = -x137;
    const scalar_t x230 = x137 * x20 + x150;
    const scalar_t x231 = x159 + x230 * x24 + x34 * (x157 + x229 * x33 + x229 * x47);
    const scalar_t x232 = -x112 * x20 + x126;
    const scalar_t x233 = x129 + x232 * x24 + x34 * (x112 * x33 + x112 * x47 + x95);
    const scalar_t x234 = -x112 * x58 - x113 + x114 + x232 * x63;
    const scalar_t x235 = x104 - x226 * x63 + x34 * (x227 * x57 + x227 * x66 + x96);
    const scalar_t x236 = x155 + x230 * x63 - x34 * (-x137 * x66 - x152);
    const scalar_t x237 = x138 - x139 + x229 * x76 + x230 * x69;
    const scalar_t x238 = x106 - x107 - x108 + x226 * x69 + x34 * (x110 + x227 * x75 + x227 * x80);
    const scalar_t x239 = x135 + x232 * x69 + x34 * (x112 * x75 + x112 * x80 + x132);
    const scalar_t x240 = x34 * x91;
    const scalar_t x241 = x82 * x9;
    const scalar_t x242 = x137 * x241;
    const scalar_t x243 = x12 * x242 - x242;
    const scalar_t x244 = x229 * x240 - x230 * x99 + x243;
    const scalar_t x245 = x112 * x241;
    const scalar_t x246 = x12 * x245;
    const scalar_t x247 = -x112 * x240 + x232 * x99 - x245 + x246;
    const scalar_t x248 = POW2(x82) * x9;
    const scalar_t x249 = x12 * x248 - x226 * x99 - x248 + x48 * (x15 * x227 * x98 + x227 * x46 * x82 - 9);
    const scalar_t x250 = x123 * x34;
    const scalar_t x251 = x116 * x226 + x227 * x250 + x245 - x246;
    const scalar_t x252 = x112 * x9;
    const scalar_t x253 = x137 * x252;
    const scalar_t x254 = x12 * x253;
    const scalar_t x255 = x116 * x230 + x229 * x250 + x253 - x254;
    const scalar_t x256 = POW2(x112) * x9;
    const scalar_t x257 = x116 * x232 + x12 * x256 - x256 + x48 * (x112 * x117 * x46 + x115 * x127 - 9);
    const scalar_t x258 = x147 * x34;
    const scalar_t x259 = x140 * x23;
    const scalar_t x260 = -x226 * x259 + x227 * x258 + x243;
    const scalar_t x261 = -x112 * x258 + x232 * x259 - x253 + x254;
    const scalar_t x262 = POW2(x137) * x9;
    const scalar_t x263 = x12 * x262 - x230 * x259 - x262 + x48 * (x137 * x229 * x46 + x15 * x151 * x229 - 9);
    const scalar_t x264 = x161 * x241;
    const scalar_t x265 = x12 * x264;
    const scalar_t x266 = x169 * x23;
    const scalar_t x267 = x168 * x34;
    const scalar_t x268 = x226 * x266 + x227 * x267 + x264 - x265;
    const scalar_t x269 = x137 * x9;
    const scalar_t x270 = x161 * x269;
    const scalar_t x271 = F[1] * x102;
    const scalar_t x272 = x12 * x270;
    const scalar_t x273 = x14 * x19;
    const scalar_t x274 = x230 * x266 + x270 - x271 - x272 + x34 * (x167 * x229 + x177 * x229 + x273);
    const scalar_t x275 = -x19 * x38;
    const scalar_t x276 = x161 * x252;
    const scalar_t x277 = F[2] * x102;
    const scalar_t x278 = x12 * x276 - x276 + x277;
    const scalar_t x279 = x232 * x266 + x278 + x34 * (x112 * x167 + x112 * x177 + x275);
    const scalar_t x280 = x185 * x252;
    const scalar_t x281 = x12 * x280;
    const scalar_t x282 = x192 * x23;
    const scalar_t x283 = x191 * x34;
    const scalar_t x284 = -x112 * x283 + x232 * x282 - x280 + x281;
    const scalar_t x285 = x185 * x90 + x275;
    const scalar_t x286 = x185 * x241;
    const scalar_t x287 = -x12 * x286 + x277 + x286;
    const scalar_t x288 = x226 * x282 + x287 - x34 * (-x197 * x82 - x285);
    const scalar_t x289 = x19 * x44;
    const scalar_t x290 = -x289;
    const scalar_t x291 = x185 * x269;
    const scalar_t x292 = F[0] * x102;
    const scalar_t x293 = x12 * x291 - x291 + x292;
    const scalar_t x294 = -x230 * x282 + x293 + x34 * (x190 * x229 + x197 * x229 + x290);
    const scalar_t x295 = x206 * x269;
    const scalar_t x296 = x12 * x295;
    const scalar_t x297 = x214 * x23;
    const scalar_t x298 = x213 * x34;
    const scalar_t x299 = x229 * x298 + x230 * x297 + x295 - x296;
    const scalar_t x300 = -x273;
    const scalar_t x301 = x206 * x241;
    const scalar_t x302 = -x12 * x301 + x271 + x301;
    const scalar_t x303 = x226 * x297 + x302 + x34 * (x212 * x227 + x219 * x227 + x300);
    const scalar_t x304 = x206 * x252;
    const scalar_t x305 = x12 * x304 - x292 - x304;
    const scalar_t x306 = x232 * x297 + x305 + x34 * (x112 * x212 + x112 * x219 + x289);
    const scalar_t x307 = x131 - x161 * x20;
    const scalar_t x308 = x161 * x35 + x171 + x24 * x307;
    const scalar_t x309 = -x185;
    const scalar_t x310 = x109 + x185 * x20;
    const scalar_t x311 = x199 + x24 * x310 + x34 * (x196 + x309 * x33 + x309 * x47);
    const scalar_t x312 = -x20 * x206 + x94;
    const scalar_t x313 = x224 + x24 * x312 + x34 * (x179 + x206 * x33 + x206 * x47);
    const scalar_t x314 = x194 + x309 * x58 - x310 * x63;
    const scalar_t x315 = x221 - x312 * x63 + x34 * (x206 * x57 + x206 * x66 + x218);
    const scalar_t x316 = -x173 + x174 + x175 + x307 * x63 - x34 * (x161 * x57 + x161 * x66 + x176);
    const scalar_t x317 = x206 * x76 + x216 + x312 * x69;
    const scalar_t x318 = x201 - x202 - x203 + x310 * x69 + x34 * (x204 + x309 * x75 + x309 * x80);
    const scalar_t x319 = x183 + x307 * x69 + x34 * (x161 * x75 + x161 * x80 + x180);
    const scalar_t x320 = -x161 * x240 - x264 + x265 + x307 * x99;
    const scalar_t x321 = x287 + x310 * x99 - x34 * (-x185 * x98 - x285);
    const scalar_t x322 = x302 - x312 * x99 + x34 * (x206 * x90 + x206 * x98 + x300);
    const scalar_t x323 = x116 * x310 + x250 * x309 + x280 - x281;
    const scalar_t x324 = x116 * x307 + x278 + x34 * (x122 * x161 + x127 * x161 + x275);
    const scalar_t x325 = x116 * x312 + x305 + x34 * (x122 * x206 + x127 * x206 + x289);
    const scalar_t x326 = -x206 * x258 + x259 * x312 - x295 + x296;
    const scalar_t x327 = -x259 * x310 + x293 + x34 * (x146 * x309 + x151 * x309 + x290);
    const scalar_t x328 = x259 * x307 - x270 + x271 + x272 - x34 * (x146 * x161 + x151 * x161 + x273);
    const scalar_t x329 = x161 * x9;
    const scalar_t x330 = x185 * x329;
    const scalar_t x331 = x12 * x330;
    const scalar_t x332 = x266 * x310 + x267 * x309 + x330 - x331;
    const scalar_t x333 = x206 * x329;
    const scalar_t x334 = x12 * x333 - x333;
    const scalar_t x335 = x206 * x267 + x266 * x312 + x334;
    const scalar_t x336 = POW2(x161) * x9;
    const scalar_t x337 = x12 * x336 + x266 * x307 - x336 + x48 * (x161 * x162 * x46 + x169 * x177 - 9);
    const scalar_t x338 = -x161 * x283 + x282 * x307 - x330 + x331;
    const scalar_t x339 = x185 * x206 * x9;
    const scalar_t x340 = x12 * x339;
    const scalar_t x341 = -x206 * x283 + x282 * x312 - x339 + x340;
    const scalar_t x342 = POW2(x185) * x9;
    const scalar_t x343 = x12 * x342 - x282 * x310 - x342 + x48 * (x15 * x197 * x309 + x185 * x309 * x46 - 9);
    const scalar_t x344 = x297 * x310 + x298 * x309 + x339 - x340;
    const scalar_t x345 = x161 * x298 + x297 * x307 + x334;
    const scalar_t x346 = POW2(x206) * x9;
    const scalar_t x347 = x12 * x346 + x297 * x312 - x346 + x48 * (x206 * x207 * x46 + x214 * x219 - 9);
    S_iklm[0]           = -x50 * (adjugate[0] * x49 + adjugate[1] * x36 + adjugate[2] * x42);
    S_iklm[1]           = -x50 * (adjugate[3] * x49 + adjugate[4] * x36 + adjugate[5] * x42);
    S_iklm[2]           = -x50 * (adjugate[6] * x49 + adjugate[7] * x36 + adjugate[8] * x42);
    S_iklm[3]           = x50 * (adjugate[0] * x59 - adjugate[1] * x67 + adjugate[2] * x64);
    S_iklm[4]           = x50 * (adjugate[3] * x59 - adjugate[4] * x67 + adjugate[5] * x64);
    S_iklm[5]           = x50 * (adjugate[6] * x59 - adjugate[7] * x67 + adjugate[8] * x64);
    S_iklm[6]           = -x50 * (adjugate[0] * x78 + adjugate[1] * x77 + adjugate[2] * x81);
    S_iklm[7]           = -x50 * (adjugate[3] * x78 + adjugate[4] * x77 + adjugate[5] * x81);
    S_iklm[8]           = -x50 * (adjugate[6] * x78 + adjugate[7] * x77 + adjugate[8] * x81);
    S_iklm[9]           = x50 * (adjugate[0] * x93 - adjugate[1] * x105 + adjugate[2] * x111);
    S_iklm[10]          = x50 * (adjugate[3] * x93 - adjugate[4] * x105 + adjugate[5] * x111);
    S_iklm[11]          = x50 * (adjugate[6] * x93 - adjugate[7] * x105 + adjugate[8] * x111);
    S_iklm[12]          = -x50 * (adjugate[0] * x130 + adjugate[1] * x125 + adjugate[2] * x136);
    S_iklm[13]          = -x50 * (adjugate[3] * x130 + adjugate[4] * x125 + adjugate[5] * x136);
    S_iklm[14]          = -x50 * (adjugate[6] * x130 + adjugate[7] * x125 + adjugate[8] * x136);
    S_iklm[15]          = x50 * (-adjugate[0] * x160 + adjugate[1] * x156 + adjugate[2] * x149);
    S_iklm[16]          = x50 * (-adjugate[3] * x160 + adjugate[4] * x156 + adjugate[5] * x149);
    S_iklm[17]          = x50 * (-adjugate[6] * x160 + adjugate[7] * x156 + adjugate[8] * x149);
    S_iklm[18]          = -x50 * (adjugate[0] * x172 + adjugate[1] * x178 + adjugate[2] * x184);
    S_iklm[19]          = -x50 * (adjugate[3] * x172 + adjugate[4] * x178 + adjugate[5] * x184);
    S_iklm[20]          = -x50 * (adjugate[6] * x172 + adjugate[7] * x178 + adjugate[8] * x184);
    S_iklm[21]          = x50 * (-adjugate[0] * x200 - adjugate[1] * x195 + adjugate[2] * x205);
    S_iklm[22]          = x50 * (-adjugate[3] * x200 - adjugate[4] * x195 + adjugate[5] * x205);
    S_iklm[23]          = x50 * (-adjugate[6] * x200 - adjugate[7] * x195 + adjugate[8] * x205);
    S_iklm[24]          = -x50 * (adjugate[0] * x225 + adjugate[1] * x222 + adjugate[2] * x217);
    S_iklm[25]          = -x50 * (adjugate[3] * x225 + adjugate[4] * x222 + adjugate[5] * x217);
    S_iklm[26]          = -x50 * (adjugate[6] * x225 + adjugate[7] * x222 + adjugate[8] * x217);
    S_iklm[27]          = -x50 * (adjugate[0] * x228 + adjugate[1] * x233 + adjugate[2] * x231);
    S_iklm[28]          = -x50 * (adjugate[3] * x228 + adjugate[4] * x233 + adjugate[5] * x231);
    S_iklm[29]          = -x50 * (adjugate[6] * x228 + adjugate[7] * x233 + adjugate[8] * x231);
    S_iklm[30]          = x50 * (-adjugate[0] * x235 + adjugate[1] * x234 + adjugate[2] * x236);
    S_iklm[31]          = x50 * (-adjugate[3] * x235 + adjugate[4] * x234 + adjugate[5] * x236);
    S_iklm[32]          = x50 * (-adjugate[6] * x235 + adjugate[7] * x234 + adjugate[8] * x236);
    S_iklm[33]          = -x50 * (adjugate[0] * x238 + adjugate[1] * x239 + adjugate[2] * x237);
    S_iklm[34]          = -x50 * (adjugate[3] * x238 + adjugate[4] * x239 + adjugate[5] * x237);
    S_iklm[35]          = -x50 * (adjugate[6] * x238 + adjugate[7] * x239 + adjugate[8] * x237);
    S_iklm[36]          = x50 * (-adjugate[0] * x249 + adjugate[1] * x247 - adjugate[2] * x244);
    S_iklm[37]          = x50 * (-adjugate[3] * x249 + adjugate[4] * x247 - adjugate[5] * x244);
    S_iklm[38]          = x50 * (-adjugate[6] * x249 + adjugate[7] * x247 - adjugate[8] * x244);
    S_iklm[39]          = -x50 * (adjugate[0] * x251 + adjugate[1] * x257 + adjugate[2] * x255);
    S_iklm[40]          = -x50 * (adjugate[3] * x251 + adjugate[4] * x257 + adjugate[5] * x255);
    S_iklm[41]          = -x50 * (adjugate[6] * x251 + adjugate[7] * x257 + adjugate[8] * x255);
    S_iklm[42]          = x50 * (-adjugate[0] * x260 + adjugate[1] * x261 - adjugate[2] * x263);
    S_iklm[43]          = x50 * (-adjugate[3] * x260 + adjugate[4] * x261 - adjugate[5] * x263);
    S_iklm[44]          = x50 * (-adjugate[6] * x260 + adjugate[7] * x261 - adjugate[8] * x263);
    S_iklm[45]          = -x50 * (adjugate[0] * x268 + adjugate[1] * x279 + adjugate[2] * x274);
    S_iklm[46]          = -x50 * (adjugate[3] * x268 + adjugate[4] * x279 + adjugate[5] * x274);
    S_iklm[47]          = -x50 * (adjugate[6] * x268 + adjugate[7] * x279 + adjugate[8] * x274);
    S_iklm[48]          = x50 * (adjugate[0] * x288 + adjugate[1] * x284 - adjugate[2] * x294);
    S_iklm[49]          = x50 * (adjugate[3] * x288 + adjugate[4] * x284 - adjugate[5] * x294);
    S_iklm[50]          = x50 * (adjugate[6] * x288 + adjugate[7] * x284 - adjugate[8] * x294);
    S_iklm[51]          = -x50 * (adjugate[0] * x303 + adjugate[1] * x306 + adjugate[2] * x299);
    S_iklm[52]          = -x50 * (adjugate[3] * x303 + adjugate[4] * x306 + adjugate[5] * x299);
    S_iklm[53]          = -x50 * (adjugate[6] * x303 + adjugate[7] * x306 + adjugate[8] * x299);
    S_iklm[54]          = -x50 * (adjugate[0] * x308 + adjugate[1] * x311 + adjugate[2] * x313);
    S_iklm[55]          = -x50 * (adjugate[3] * x308 + adjugate[4] * x311 + adjugate[5] * x313);
    S_iklm[56]          = -x50 * (adjugate[6] * x308 + adjugate[7] * x311 + adjugate[8] * x313);
    S_iklm[57]          = x50 * (adjugate[0] * x316 - adjugate[1] * x314 - adjugate[2] * x315);
    S_iklm[58]          = x50 * (adjugate[3] * x316 - adjugate[4] * x314 - adjugate[5] * x315);
    S_iklm[59]          = x50 * (adjugate[6] * x316 - adjugate[7] * x314 - adjugate[8] * x315);
    S_iklm[60]          = -x50 * (adjugate[0] * x319 + adjugate[1] * x318 + adjugate[2] * x317);
    S_iklm[61]          = -x50 * (adjugate[3] * x319 + adjugate[4] * x318 + adjugate[5] * x317);
    S_iklm[62]          = -x50 * (adjugate[6] * x319 + adjugate[7] * x318 + adjugate[8] * x317);
    S_iklm[63]          = x50 * (adjugate[0] * x320 + adjugate[1] * x321 - adjugate[2] * x322);
    S_iklm[64]          = x50 * (adjugate[3] * x320 + adjugate[4] * x321 - adjugate[5] * x322);
    S_iklm[65]          = x50 * (adjugate[6] * x320 + adjugate[7] * x321 - adjugate[8] * x322);
    S_iklm[66]          = -x50 * (adjugate[0] * x324 + adjugate[1] * x323 + adjugate[2] * x325);
    S_iklm[67]          = -x50 * (adjugate[3] * x324 + adjugate[4] * x323 + adjugate[5] * x325);
    S_iklm[68]          = -x50 * (adjugate[6] * x324 + adjugate[7] * x323 + adjugate[8] * x325);
    S_iklm[69]          = x50 * (adjugate[0] * x328 - adjugate[1] * x327 + adjugate[2] * x326);
    S_iklm[70]          = x50 * (adjugate[3] * x328 - adjugate[4] * x327 + adjugate[5] * x326);
    S_iklm[71]          = x50 * (adjugate[6] * x328 - adjugate[7] * x327 + adjugate[8] * x326);
    S_iklm[72]          = -x50 * (adjugate[0] * x337 + adjugate[1] * x332 + adjugate[2] * x335);
    S_iklm[73]          = -x50 * (adjugate[3] * x337 + adjugate[4] * x332 + adjugate[5] * x335);
    S_iklm[74]          = -x50 * (adjugate[6] * x337 + adjugate[7] * x332 + adjugate[8] * x335);
    S_iklm[75]          = x50 * (adjugate[0] * x338 - adjugate[1] * x343 + adjugate[2] * x341);
    S_iklm[76]          = x50 * (adjugate[3] * x338 - adjugate[4] * x343 + adjugate[5] * x341);
    S_iklm[77]          = x50 * (adjugate[6] * x338 - adjugate[7] * x343 + adjugate[8] * x341);
    S_iklm[78]          = -x50 * (adjugate[0] * x345 + adjugate[1] * x344 + adjugate[2] * x347);
    S_iklm[79]          = -x50 * (adjugate[3] * x345 + adjugate[4] * x344 + adjugate[5] * x347);
    S_iklm[80]          = -x50 * (adjugate[6] * x345 + adjugate[7] * x344 + adjugate[8] * x347);
}

static SFEM_INLINE void tet4_partial_assembly_neohookean(const scalar_t *const SFEM_RESTRICT S_iklm,
                                                         const scalar_t *const SFEM_RESTRICT inc_grad,
                                                         scalar_t *const SFEM_RESTRICT       element_outx,
                                                         scalar_t *const SFEM_RESTRICT       element_outy,
                                                         scalar_t *const SFEM_RESTRICT       element_outz) {
    // mundane ops: 168 divs: 0 sqrts: 0
    // total ops: 168

    const scalar_t x0 = S_iklm[0] * inc_grad[0] + S_iklm[12] * inc_grad[4] + S_iklm[15] * inc_grad[5] + S_iklm[18] * inc_grad[6] +
                        S_iklm[21] * inc_grad[7] + S_iklm[24] * inc_grad[8] + S_iklm[3] * inc_grad[1] + S_iklm[6] * inc_grad[2] +
                        S_iklm[9] * inc_grad[3];
    const scalar_t x1 = S_iklm[10] * inc_grad[3] + S_iklm[13] * inc_grad[4] + S_iklm[16] * inc_grad[5] +
                        S_iklm[19] * inc_grad[6] + S_iklm[1] * inc_grad[0] + S_iklm[22] * inc_grad[7] + S_iklm[25] * inc_grad[8] +
                        S_iklm[4] * inc_grad[1] + S_iklm[7] * inc_grad[2];
    const scalar_t x2 = S_iklm[11] * inc_grad[3] + S_iklm[14] * inc_grad[4] + S_iklm[17] * inc_grad[5] +
                        S_iklm[20] * inc_grad[6] + S_iklm[23] * inc_grad[7] + S_iklm[26] * inc_grad[8] + S_iklm[2] * inc_grad[0] +
                        S_iklm[5] * inc_grad[1] + S_iklm[8] * inc_grad[2];
    const scalar_t x3 = S_iklm[27] * inc_grad[0] + S_iklm[30] * inc_grad[1] + S_iklm[33] * inc_grad[2] +
                        S_iklm[36] * inc_grad[3] + S_iklm[39] * inc_grad[4] + S_iklm[42] * inc_grad[5] +
                        S_iklm[45] * inc_grad[6] + S_iklm[48] * inc_grad[7] + S_iklm[51] * inc_grad[8];
    const scalar_t x4 = S_iklm[28] * inc_grad[0] + S_iklm[31] * inc_grad[1] + S_iklm[34] * inc_grad[2] +
                        S_iklm[37] * inc_grad[3] + S_iklm[40] * inc_grad[4] + S_iklm[43] * inc_grad[5] +
                        S_iklm[46] * inc_grad[6] + S_iklm[49] * inc_grad[7] + S_iklm[52] * inc_grad[8];
    const scalar_t x5 = S_iklm[29] * inc_grad[0] + S_iklm[32] * inc_grad[1] + S_iklm[35] * inc_grad[2] +
                        S_iklm[38] * inc_grad[3] + S_iklm[41] * inc_grad[4] + S_iklm[44] * inc_grad[5] +
                        S_iklm[47] * inc_grad[6] + S_iklm[50] * inc_grad[7] + S_iklm[53] * inc_grad[8];
    const scalar_t x6 = S_iklm[54] * inc_grad[0] + S_iklm[57] * inc_grad[1] + S_iklm[60] * inc_grad[2] +
                        S_iklm[63] * inc_grad[3] + S_iklm[66] * inc_grad[4] + S_iklm[69] * inc_grad[5] +
                        S_iklm[72] * inc_grad[6] + S_iklm[75] * inc_grad[7] + S_iklm[78] * inc_grad[8];
    const scalar_t x7 = S_iklm[55] * inc_grad[0] + S_iklm[58] * inc_grad[1] + S_iklm[61] * inc_grad[2] +
                        S_iklm[64] * inc_grad[3] + S_iklm[67] * inc_grad[4] + S_iklm[70] * inc_grad[5] +
                        S_iklm[73] * inc_grad[6] + S_iklm[76] * inc_grad[7] + S_iklm[79] * inc_grad[8];
    const scalar_t x8 = S_iklm[56] * inc_grad[0] + S_iklm[59] * inc_grad[1] + S_iklm[62] * inc_grad[2] +
                        S_iklm[65] * inc_grad[3] + S_iklm[68] * inc_grad[4] + S_iklm[71] * inc_grad[5] +
                        S_iklm[74] * inc_grad[6] + S_iklm[77] * inc_grad[7] + S_iklm[80] * inc_grad[8];
    element_outx[0] = -x0 - x1 - x2;
    element_outx[1] = x0;
    element_outx[2] = x1;
    element_outx[3] = x2;
    element_outy[0] = -x3 - x4 - x5;
    element_outy[1] = x3;
    element_outy[2] = x4;
    element_outy[3] = x5;
    element_outz[0] = -x6 - x7 - x8;
    element_outz[1] = x6;
    element_outz[2] = x7;
    element_outz[3] = x8;
}

#endif
