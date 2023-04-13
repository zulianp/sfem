#include <stddef.h>
#include <assert.h>

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
    // FLOATING POINT OPS!
    //       - Result: 10*ADD + 10*ASSIGNMENT + 157*MUL
    //       - Subexpressions: 83*ADD + 24*DIV + 277*MUL + 3*NEG + 108*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -px0 + px2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x5 * x6;
    const real_t x8 = -py0 + py1;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -pz0 + pz2;
    const real_t x11 = x10 * x6;
    const real_t x12 = x2 * x5;
    const real_t x13 = x1 * x9;
    const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
    const real_t x15 = -x13 + x7;
    const real_t x16 = 1.0 / x14;
    const real_t x17 = x10 * x9 - x12;
    const real_t x18 = -x11 + x3;
    const real_t x19 = x0 * x2 - x4 * x9;
    const real_t x20 = -x0 * x6 + x8 * x9;
    const real_t x21 = -x2 * x8 + x4 * x6;
    const real_t x22 = x0 * x1 - x5 * x8;
    const real_t x23 = -x0 * x10 + x4 * x5;
    const real_t x24 = -x1 * x4 + x10 * x8;
    const real_t x25 = (1.0 / 90.0) * x16;
    const real_t x26 = ux[5] * x25;
    const real_t x27 = x17 * x26;
    const real_t x28 = x18 * x26;
    const real_t x29 = ux[8] * x25;
    const real_t x30 = x15 * x29;
    const real_t x31 = x18 * x29;
    const real_t x32 = ux[9] * x25;
    const real_t x33 = x15 * x32;
    const real_t x34 = x17 * x32;
    const real_t x35 = uy[5] * x25;
    const real_t x36 = x19 * x35;
    const real_t x37 = x21 * x35;
    const real_t x38 = uy[8] * x25;
    const real_t x39 = x20 * x38;
    const real_t x40 = x21 * x38;
    const real_t x41 = uy[9] * x25;
    const real_t x42 = x19 * x41;
    const real_t x43 = x20 * x41;
    const real_t x44 = uz[5] * x25;
    const real_t x45 = x23 * x44;
    const real_t x46 = x24 * x44;
    const real_t x47 = uz[8] * x25;
    const real_t x48 = x22 * x47;
    const real_t x49 = x24 * x47;
    const real_t x50 = uz[9] * x25;
    const real_t x51 = x22 * x50;
    const real_t x52 = x23 * x50;
    const real_t x53 = (1.0 / 120.0) * x16;
    const real_t x54 = ux[0] * x15;
    const real_t x55 = ux[0] * x53;
    const real_t x56 = uy[0] * x53;
    const real_t x57 = uz[0] * x53;
    const real_t x58 = (1.0 / 360.0) * x16;
    const real_t x59 = ux[1] * x18;
    const real_t x60 = x58 * x59;
    const real_t x61 = x17 * x58;
    const real_t x62 = ux[2] * x61;
    const real_t x63 = ux[3] * x15;
    const real_t x64 = x58 * x63;
    const real_t x65 = uy[1] * x21;
    const real_t x66 = x58 * x65;
    const real_t x67 = x19 * x58;
    const real_t x68 = uy[2] * x67;
    const real_t x69 = x20 * x58;
    const real_t x70 = uy[3] * x69;
    const real_t x71 = uz[1] * x24;
    const real_t x72 = x58 * x71;
    const real_t x73 = x23 * x58;
    const real_t x74 = uz[2] * x73;
    const real_t x75 = x22 * x58;
    const real_t x76 = uz[3] * x75;
    const real_t x77 = ux[4] * x25;
    const real_t x78 = uy[4] * x25;
    const real_t x79 = uz[4] * x25;
    const real_t x80 = -x18 * x77 - x21 * x78 - x24 * x79;
    const real_t x81 = ux[6] * x25;
    const real_t x82 = uy[6] * x25;
    const real_t x83 = uz[6] * x25;
    const real_t x84 = -x17 * x81 - x19 * x82 - x23 * x83;
    const real_t x85 = ux[7] * x25;
    const real_t x86 = uy[7] * x25;
    const real_t x87 = uz[7] * x25;
    const real_t x88 = -x15 * x85 - x20 * x86 - x22 * x87;
    const real_t x89 = x54 * x58;
    const real_t x90 = ux[0] * x61;
    const real_t x91 = ux[0] * x18;
    const real_t x92 = x58 * x91;
    const real_t x93 = uy[0] * x67;
    const real_t x94 = uy[0] * x69;
    const real_t x95 = uy[0] * x21;
    const real_t x96 = x58 * x95;
    const real_t x97 = uz[0] * x75;
    const real_t x98 = uz[0] * x73;
    const real_t x99 = uz[0] * x24;
    const real_t x100 = x58 * x99;
    const real_t x101 = x100 + x17 * x85 + x18 * x85 + x19 * x86 + x21 * x86 + x23 * x87 + x24 * x87 - x31 - x34 - x40 -
                        x42 - x49 - x52 - x64 - x70 - x76 + x89 + x90 + x92 + x93 + x94 + x96 + x97 + x98;
    const real_t x102 = x15 * x81 + x18 * x81 + x20 * x82 + x21 * x82 + x22 * x83 + x24 * x83 - x28 - x33 - x37 - x43 -
                        x46 - x51 - x62 - x68 - x74;
    const real_t x103 = ux[2] * x17;
    const real_t x104 = uy[2] * x19;
    const real_t x105 = uz[2] * x23;
    const real_t x106 = x15 * x77 + x17 * x77 + x19 * x78 + x20 * x78 + x22 * x79 + x23 * x79 - x27 - x30 - x36 - x39 -
                        x45 - x48 - x60 - x66 - x72;
    const real_t x107 = uy[3] * x20;
    const real_t x108 = uz[3] * x22;
    const real_t x109 = (2.0 / 45.0) * x16;
    const real_t x110 = ux[4] * x109;
    const real_t x111 = x110 * x15;
    const real_t x112 = x110 * x17;
    const real_t x113 = uy[4] * x109;
    const real_t x114 = x113 * x19;
    const real_t x115 = x113 * x20;
    const real_t x116 = uz[4] * x109;
    const real_t x117 = x116 * x22;
    const real_t x118 = x116 * x23;
    const real_t x119 = x25 * x59;
    const real_t x120 = x25 * x65;
    const real_t x121 = x25 * x71;
    const real_t x122 = (1.0 / 45.0) * x16;
    const real_t x123 = x122 * x17;
    const real_t x124 = uy[6] * x122;
    const real_t x125 = uz[6] * x122;
    const real_t x126 = x25 * x54;
    const real_t x127 = ux[0] * x17 * x25;
    const real_t x128 = x25 * x91;
    const real_t x129 = x103 * x25;
    const real_t x130 = uy[0] * x25;
    const real_t x131 = x130 * x19;
    const real_t x132 = x130 * x20;
    const real_t x133 = x25 * x95;
    const real_t x134 = x104 * x25;
    const real_t x135 = uz[0] * x25;
    const real_t x136 = x135 * x22;
    const real_t x137 = x135 * x23;
    const real_t x138 = x25 * x99;
    const real_t x139 = x105 * x25;
    const real_t x140 = -ux[6] * x123 - x124 * x19 - x125 * x23 + x126 + x127 + x128 + x129 + x131 + x132 + x133 +
                        x134 + x136 + x137 + x138 + x139;
    const real_t x141 = x122 * x15;
    const real_t x142 = uy[7] * x122;
    const real_t x143 = uz[7] * x122;
    const real_t x144 = x25 * x63;
    const real_t x145 = x107 * x25;
    const real_t x146 = x108 * x25;
    const real_t x147 = -ux[7] * x141 - x142 * x20 - x143 * x22 + x144 + x145 + x146;
    const real_t x148 = -x119 - x120 - x121 + x140 + x147;
    const real_t x149 = ux[7] * x123;
    const real_t x150 = x122 * x18;
    const real_t x151 = ux[7] * x150;
    const real_t x152 = x142 * x19;
    const real_t x153 = x142 * x21;
    const real_t x154 = x143 * x23;
    const real_t x155 = x143 * x24;
    const real_t x156 = -1.0 / 45.0 * ux[8] * x16 * x18 - 1.0 / 45.0 * ux[9] * x16 * x17 -
                        1.0 / 45.0 * uy[8] * x16 * x21 - 1.0 / 45.0 * uy[9] * x16 * x19 -
                        1.0 / 45.0 * uz[8] * x16 * x24 - 1.0 / 45.0 * uz[9] * x16 * x23 + x149 + x151 + x152 + x153 +
                        x154 + x155;
    const real_t x157 = ux[6] * x15;
    const real_t x158 = x122 * x157;
    const real_t x159 = ux[6] * x150;
    const real_t x160 = x124 * x20;
    const real_t x161 = x124 * x21;
    const real_t x162 = x125 * x22;
    const real_t x163 = x125 * x24;
    const real_t x164 = -1.0 / 45.0 * ux[5] * x16 * x18 - 1.0 / 45.0 * ux[9] * x15 * x16 -
                        1.0 / 45.0 * uy[5] * x16 * x21 - 1.0 / 45.0 * uy[9] * x16 * x20 -
                        1.0 / 45.0 * uz[5] * x16 * x24 - 1.0 / 45.0 * uz[9] * x16 * x22 + x158 + x159 + x160 + x161 +
                        x162 + x163;
    const real_t x165 = x122 * x21;
    const real_t x166 = uy[9] * x122;
    const real_t x167 = x122 * x24;
    const real_t x168 = uz[9] * x122;
    const real_t x169 = -ux[4] * x150 - uy[4] * x165 - uz[4] * x167 + x119 + x120 + x121;
    const real_t x170 = x140 - x144 - x145 - x146 + x169;
    const real_t x171 = x109 * x157;
    const real_t x172 = x109 * x18;
    const real_t x173 = ux[6] * x172;
    const real_t x174 = x109 * x20;
    const real_t x175 = uy[6] * x174;
    const real_t x176 = x109 * x21;
    const real_t x177 = uy[6] * x176;
    const real_t x178 = x109 * x22;
    const real_t x179 = uz[6] * x178;
    const real_t x180 = x109 * x24;
    const real_t x181 = uz[6] * x180;
    const real_t x182 = ux[5] * x109;
    const real_t x183 = x109 * x15;
    const real_t x184 = uy[5] * x109;
    const real_t x185 = uz[5] * x109;
    const real_t x186 = ux[9] * x183 + uy[9] * x174 + uz[9] * x178 - x171 - x173 - x175 - x177 - x179 + x18 * x182 -
                        x181 + x184 * x21 + x185 * x24;
    const real_t x187 = ux[8] * x183 + uy[8] * x174 + uz[8] * x178 - x111 - x112 - x114 - x115 - x117 - x118 +
                        x17 * x182 + x184 * x19 + x185 * x23;
    const real_t x188 = ux[4] * x141;
    const real_t x189 = ux[4] * x123;
    const real_t x190 = uy[4] * x122;
    const real_t x191 = x19 * x190;
    const real_t x192 = x190 * x20;
    const real_t x193 = uz[4] * x122;
    const real_t x194 = x193 * x22;
    const real_t x195 = x193 * x23;
    const real_t x196 = -1.0 / 45.0 * ux[5] * x16 * x17 - 1.0 / 45.0 * ux[8] * x15 * x16 -
                        1.0 / 45.0 * uy[5] * x16 * x19 - 1.0 / 45.0 * uy[8] * x16 * x20 -
                        1.0 / 45.0 * uz[5] * x16 * x23 - 1.0 / 45.0 * uz[8] * x16 * x22 + x188 + x189 + x191 + x192 +
                        x194 + x195;
    const real_t x197 = x126 + x127 + x128 - x129 + x131 + x132 + x133 - x134 + x136 + x137 + x138 - x139 + x147 + x169;
    const real_t x198 = x109 * x17;
    const real_t x199 = ux[7] * x198;
    const real_t x200 = ux[7] * x172;
    const real_t x201 = x109 * x19;
    const real_t x202 = uy[7] * x201;
    const real_t x203 = uy[7] * x176;
    const real_t x204 = x109 * x23;
    const real_t x205 = uz[7] * x204;
    const real_t x206 = uz[7] * x180;
    const real_t x207 = ux[8] * x172 + ux[9] * x198 + uy[8] * x176 + uy[9] * x201 + uz[8] * x180 + uz[9] * x204 - x199 -
                        x200 - x202 - x203 - x205 - x206;
    element_vector[0] =
        x14 * ((1.0 / 90.0) * ux[4] * x15 * x16 + (1.0 / 90.0) * ux[4] * x16 * x17 + (1.0 / 90.0) * ux[6] * x15 * x16 +
               (1.0 / 90.0) * ux[6] * x16 * x18 + (1.0 / 90.0) * ux[7] * x16 * x17 + (1.0 / 90.0) * ux[7] * x16 * x18 +
               (1.0 / 90.0) * uy[4] * x16 * x19 + (1.0 / 90.0) * uy[4] * x16 * x20 + (1.0 / 90.0) * uy[6] * x16 * x20 +
               (1.0 / 90.0) * uy[6] * x16 * x21 + (1.0 / 90.0) * uy[7] * x16 * x19 + (1.0 / 90.0) * uy[7] * x16 * x21 +
               (1.0 / 90.0) * uz[4] * x16 * x22 + (1.0 / 90.0) * uz[4] * x16 * x23 + (1.0 / 90.0) * uz[6] * x16 * x22 +
               (1.0 / 90.0) * uz[6] * x16 * x24 + (1.0 / 90.0) * uz[7] * x16 * x23 + (1.0 / 90.0) * uz[7] * x16 * x24 -
               x17 * x55 - x18 * x55 - x19 * x56 - x20 * x56 - x21 * x56 - x22 * x57 - x23 * x57 - x24 * x57 - x27 -
               x28 - x30 - x31 - x33 - x34 - x36 - x37 - x39 - x40 - x42 - x43 - x45 - x46 - x48 - x49 - x51 - x52 -
               x53 * x54 - x60 - x62 - x64 - x66 - x68 - x70 - x72 - x74 - x76 - x80 - x84 - x88);
    element_vector[1] = x14 * (x101 + x102 + x53 * x59 + x53 * x65 + x53 * x71 + x80);
    element_vector[2] = x14 * (x101 + x103 * x53 + x104 * x53 + x105 * x53 + x106 + x84);
    element_vector[3] = x14 * (x100 + x102 + x106 + x107 * x53 + x108 * x53 + x53 * x63 + x88 + x89 + x90 + x92 + x93 +
                               x94 + x96 + x97 + x98);
    element_vector[4] =
        x14 * ((2.0 / 45.0) * ux[5] * x16 * x17 + (2.0 / 45.0) * ux[8] * x15 * x16 + (2.0 / 45.0) * uy[5] * x16 * x19 +
               (2.0 / 45.0) * uy[8] * x16 * x20 + (2.0 / 45.0) * uz[5] * x16 * x23 + (2.0 / 45.0) * uz[8] * x16 * x22 -
               x111 - x112 - x114 - x115 - x117 - x118 - x148 - x156 - x164);
    element_vector[5] = x14 * (ux[8] * x150 + ux[9] * x123 + uy[8] * x165 + uz[8] * x167 - x149 - x151 - x152 - x153 -
                               x154 - x155 + x166 * x19 + x168 * x23 + x170 + x186 + x187);
    element_vector[6] =
        x14 * ((2.0 / 45.0) * ux[5] * x16 * x18 + (2.0 / 45.0) * ux[9] * x15 * x16 + (2.0 / 45.0) * uy[5] * x16 * x21 +
               (2.0 / 45.0) * uy[9] * x16 * x20 + (2.0 / 45.0) * uz[5] * x16 * x24 + (2.0 / 45.0) * uz[9] * x16 * x22 -
               x156 - x171 - x173 - x175 - x177 - x179 - x181 - x196 - x197);
    element_vector[7] =
        x14 * ((2.0 / 45.0) * ux[8] * x16 * x18 + (2.0 / 45.0) * ux[9] * x16 * x17 + (2.0 / 45.0) * uy[8] * x16 * x21 +
               (2.0 / 45.0) * uy[9] * x16 * x19 + (2.0 / 45.0) * uz[8] * x16 * x24 + (2.0 / 45.0) * uz[9] * x16 * x23 -
               x164 - x170 - x196 - x199 - x200 - x202 - x203 - x205 - x206);
    element_vector[8] = x14 * (ux[5] * x150 + ux[9] * x141 + uy[5] * x165 + uz[5] * x167 - x158 - x159 - x160 - x161 -
                               x162 - x163 + x166 * x20 + x168 * x22 + x187 + x197 + x207);
    element_vector[9] =
        x14 * (ux[5] * x123 + ux[8] * x141 + uy[5] * x122 * x19 + uy[8] * x122 * x20 + uz[5] * x122 * x23 +
               uz[8] * x122 * x22 + x148 + x186 - x188 - x189 - x191 - x192 - x194 - x195 + x207);
}

// Implement this first!!
void tet10_div_apply(const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const elems,
                     geom_t **const xyz,
                     const real_t *const ux,
                     const real_t *const uy,
                     const real_t *const uz,
                     real_t *const values) {
    // TODO
    assert(0);
}

void tet10_integrate_div(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const elems,
                         geom_t **const xyz,
                         const real_t *const ux,
                         const real_t *const uy,
                         const real_t *const uz,
                         real_t *const value) {
    // TODO
    assert(0);
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
    assert(0);
}
