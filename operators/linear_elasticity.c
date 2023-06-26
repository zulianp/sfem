#include "linear_elasticity.h"

#include "tri3_linear_elasticity.h"

#include <mpi.h>

void linear_elasticity_assemble_value_soa(const enum ElemType element_type,
                                          const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t mu,
                                          const real_t lambda,
                                          const real_t **const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_value_soa(
                nelements, nnodes, elems, xyz, mu, lambda, u, value);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_gradient_soa(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t lambda,
                                             const real_t **const SFEM_RESTRICT u,
                                             real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_gradient_soa(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_hessian_soa(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t mu,
                                            const real_t lambda,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, lambda, rowptr, colidx, values);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_apply_soa(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t **const SFEM_RESTRICT u,
                                 real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_apply_soa(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

// static SFEM_INLINE void linear_elasticity(const real_t mu,
//                                           const real_t lambda,
//                                           const real_t x0,
//                                           const real_t x1,
//                                           const real_t x2,
//                                           const real_t x3,
//                                           const real_t y0,
//                                           const real_t y1,
//                                           const real_t y2,
//                                           const real_t y3,
//                                           const real_t z0,
//                                           const real_t z1,
//                                           const real_t z2,
//                                           const real_t z3,
//                                           real_t *element_matrix) {
//     real_t x4 = x0 - x1;
//     real_t x5 = y0 - y2;
//     real_t x6 = z0 - z3;
//     real_t x7 = x5 * x6;
//     real_t x8 = x0 - x2;
//     real_t x9 = y0 - y3;
//     real_t x10 = z0 - z1;
//     real_t x11 = x10 * x9;
//     real_t x12 = x0 - x3;
//     real_t x13 = y0 - y1;
//     real_t x14 = z0 - z2;
//     real_t x15 = x13 * x14;
//     real_t x16 = x14 * x9;
//     real_t x17 = x13 * x6;
//     real_t x18 = x10 * x5;
//     real_t x19 = 1.0 / (x11 * x8 + x12 * x15 - x12 * x18 - x16 * x4 - x17 * x8 + x4 * x7);
//     real_t x20 = -x16 + x7;
//     real_t x21 = x15 - x18;
//     real_t x22 = x11 - x17 + x20 + x21;
//     real_t x23 = pow(x22, 2);
//     real_t x24 = (1.0 / 6.0) * lambda;
//     real_t x25 = x12 * x13;
//     real_t x26 = x4 * x9;
//     real_t x27 = -x12 * x5 + x8 * x9;
//     real_t x28 = -x13 * x8 + x4 * x5;
//     real_t x29 = x25 - x26 + x27 + x28;
//     real_t x30 = pow(x29, 2);
//     real_t x31 = x10 * x12;
//     real_t x32 = x4 * x6;
//     real_t x33 = -x12 * x14 + x6 * x8;
//     real_t x34 = -x10 * x8 + x14 * x4;
//     real_t x35 = x31 - x32 + x33 + x34;
//     real_t x36 = pow(x35, 2);
//     real_t x37 = (1.0 / 6.0) * mu;
//     real_t x38 = (1.0 / 6.0) * x19;
//     real_t x39 = x22 * x38;
//     real_t x40 = lambda + mu;
//     real_t x41 = x35 * x40;
//     real_t x42 = x39 * x41;
//     real_t x43 = -x40;
//     real_t x44 = x29 * x39 * x43;
//     real_t x45 = x20 * x22;
//     real_t x46 = x27 * x29;
//     real_t x47 = x33 * x35;
//     real_t x48 = x19 * (x24 * x45 - x37 * (-2 * x45 - x46 - x47));
//     real_t x49 = x22 * x24;
//     real_t x50 = x20 * x37;
//     real_t x51 = x19 * (-x33 * x49 - x35 * x50);
//     real_t x52 = x19 * (x27 * x49 + x29 * x50);
//     real_t x53 = -x11 + x17;
//     real_t x54 = x22 * x53;
//     real_t x55 = -x25 + x26;
//     real_t x56 = x29 * x55;
//     real_t x57 = -x31 + x32;
//     real_t x58 = x35 * x57;
//     real_t x59 = x19 * (-x24 * x54 - x37 * (2 * x54 + x56 + x58));
//     real_t x60 = x37 * x53;
//     real_t x61 = x19 * (x35 * x60 + x49 * x57);
//     real_t x62 = x19 * (-x29 * x60 - x49 * x55);
//     real_t x63 = x21 * x22;
//     real_t x64 = x28 * x29;
//     real_t x65 = x34 * x35;
//     real_t x66 = x19 * (x24 * x63 - x37 * (-2 * x63 - x64 - x65));
//     real_t x67 = x21 * x37;
//     real_t x68 = x19 * (-x34 * x49 - x35 * x67);
//     real_t x69 = x19 * (x28 * x49 + x29 * x67);
//     real_t x70 = x29 * x38 * x41;
//     real_t x71 = x24 * x35;
//     real_t x72 = x33 * x37;
//     real_t x73 = x19 * (-x20 * x71 - x22 * x72);
//     real_t x74 = x19 * (x24 * x47 - x37 * (-x45 - x46 - 2 * x47));
//     real_t x75 = x19 * (-x27 * x71 - x29 * x72);
//     real_t x76 = x37 * x57;
//     real_t x77 = x19 * (x22 * x76 + x53 * x71);
//     real_t x78 = x19 * (-x24 * x58 - x37 * (x54 + x56 + 2 * x58));
//     real_t x79 = x19 * (x29 * x76 + x55 * x71);
//     real_t x80 = x34 * x37;
//     real_t x81 = x19 * (-x21 * x71 - x22 * x80);
//     real_t x82 = x19 * (x24 * x65 - x37 * (-x63 - x64 - 2 * x65));
//     real_t x83 = x19 * (-x28 * x71 - x29 * x80);
//     real_t x84 = x24 * x29;
//     real_t x85 = x27 * x37;
//     real_t x86 = x19 * (x20 * x84 + x22 * x85);
//     real_t x87 = x19 * (-x33 * x84 - x35 * x85);
//     real_t x88 = x19 * (x24 * x46 - x37 * (-x45 - 2 * x46 - x47));
//     real_t x89 = x37 * x55;
//     real_t x90 = x19 * (-x22 * x89 - x53 * x84);
//     real_t x91 = x19 * (x35 * x89 + x57 * x84);
//     real_t x92 = x19 * (-x24 * x56 - x37 * (x54 + 2 * x56 + x58));
//     real_t x93 = x28 * x37;
//     real_t x94 = x19 * (x21 * x84 + x22 * x93);
//     real_t x95 = x19 * (-x34 * x84 - x35 * x93);
//     real_t x96 = x19 * (x24 * x64 - x37 * (-x63 - 2 * x64 - x65));
//     real_t x97 = pow(x20, 2);
//     real_t x98 = pow(x27, 2);
//     real_t x99 = pow(x33, 2);
//     real_t x100 = x20 * x38;
//     real_t x101 = x33 * x40;
//     real_t x102 = x100 * x101;
//     real_t x103 = x100 * x27 * x43;
//     real_t x104 = x20 * x53;
//     real_t x105 = x27 * x55;
//     real_t x106 = x33 * x57;
//     real_t x107 = x19 * (x104 * x24 - x37 * (-2 * x104 - x105 - x106));
//     real_t x108 = x20 * x24;
//     real_t x109 = x19 * (-x108 * x57 - x33 * x60);
//     real_t x110 = x19 * (x108 * x55 + x27 * x60);
//     real_t x111 = x20 * x21;
//     real_t x112 = x27 * x28;
//     real_t x113 = x33 * x34;
//     real_t x114 = x19 * (-x111 * x24 - x37 * (2 * x111 + x112 + x113));
//     real_t x115 = x19 * (x108 * x34 + x33 * x67);
//     real_t x116 = x19 * (-x108 * x28 - x27 * x67);
//     real_t x117 = x101 * x27 * x38;
//     real_t x118 = x24 * x33;
//     real_t x119 = x19 * (-x118 * x53 - x50 * x57);
//     real_t x120 = x19 * (x106 * x24 - x37 * (-x104 - x105 - 2 * x106));
//     real_t x121 = x19 * (-x118 * x55 - x27 * x76);
//     real_t x122 = x19 * (x118 * x21 + x34 * x50);
//     real_t x123 = x19 * (-x113 * x24 - x37 * (x111 + x112 + 2 * x113));
//     real_t x124 = x19 * (x118 * x28 + x27 * x80);
//     real_t x125 = x24 * x27;
//     real_t x126 = x19 * (x125 * x53 + x50 * x55);
//     real_t x127 = x19 * (-x125 * x57 - x55 * x72);
//     real_t x128 = x19 * (x105 * x24 - x37 * (-x104 - 2 * x105 - x106));
//     real_t x129 = x19 * (-x125 * x21 - x28 * x50);
//     real_t x130 = x19 * (x125 * x34 + x28 * x72);
//     real_t x131 = x19 * (-x112 * x24 - x37 * (x111 + 2 * x112 + x113));
//     real_t x132 = pow(x53, 2);
//     real_t x133 = pow(x55, 2);
//     real_t x134 = pow(x57, 2);
//     real_t x135 = x38 * x53;
//     real_t x136 = x40 * x57;
//     real_t x137 = x135 * x136;
//     real_t x138 = x135 * x43 * x55;
//     real_t x139 = x21 * x53;
//     real_t x140 = x28 * x55;
//     real_t x141 = x34 * x57;
//     real_t x142 = x19 * (x139 * x24 - x37 * (-2 * x139 - x140 - x141));
//     real_t x143 = x24 * x53;
//     real_t x144 = x19 * (-x143 * x34 - x57 * x67);
//     real_t x145 = x19 * (x143 * x28 + x55 * x67);
//     real_t x146 = x136 * x38 * x55;
//     real_t x147 = x24 * x57;
//     real_t x148 = x19 * (-x147 * x21 - x34 * x60);
//     real_t x149 = x19 * (x141 * x24 - x37 * (-x139 - x140 - 2 * x141));
//     real_t x150 = x19 * (-x147 * x28 - x55 * x80);
//     real_t x151 = x24 * x55;
//     real_t x152 = x19 * (x151 * x21 + x28 * x60);
//     real_t x153 = x19 * (-x151 * x34 - x28 * x76);
//     real_t x154 = x19 * (x140 * x24 - x37 * (-x139 - 2 * x140 - x141));
//     real_t x155 = pow(x21, 2);
//     real_t x156 = pow(x28, 2);
//     real_t x157 = pow(x34, 2);
//     real_t x158 = x21 * x38;
//     real_t x159 = x34 * x40;
//     real_t x160 = x158 * x159;
//     real_t x161 = x158 * x28 * x43;
//     real_t x162 = x159 * x28 * x38;
//     element_matrix[0] = x19 * (-x23 * x24 - x37 * (2 * x23 + x30 + x36));
//     element_matrix[1] = x42;
//     element_matrix[2] = x44;
//     element_matrix[3] = x48;
//     element_matrix[4] = x51;
//     element_matrix[5] = x52;
//     element_matrix[6] = x59;
//     element_matrix[7] = x61;
//     element_matrix[8] = x62;
//     element_matrix[9] = x66;
//     element_matrix[10] = x68;
//     element_matrix[11] = x69;
//     element_matrix[12] = x42;
//     element_matrix[13] = x19 * (-x24 * x36 - x37 * (x23 + x30 + 2 * x36));
//     element_matrix[14] = x70;
//     element_matrix[15] = x73;
//     element_matrix[16] = x74;
//     element_matrix[17] = x75;
//     element_matrix[18] = x77;
//     element_matrix[19] = x78;
//     element_matrix[20] = x79;
//     element_matrix[21] = x81;
//     element_matrix[22] = x82;
//     element_matrix[23] = x83;
//     element_matrix[24] = x44;
//     element_matrix[25] = x70;
//     element_matrix[26] = x19 * (-x24 * x30 - x37 * (x23 + 2 * x30 + x36));
//     element_matrix[27] = x86;
//     element_matrix[28] = x87;
//     element_matrix[29] = x88;
//     element_matrix[30] = x90;
//     element_matrix[31] = x91;
//     element_matrix[32] = x92;
//     element_matrix[33] = x94;
//     element_matrix[34] = x95;
//     element_matrix[35] = x96;
//     element_matrix[36] = x48;
//     element_matrix[37] = x73;
//     element_matrix[38] = x86;
//     element_matrix[39] = x19 * (-x24 * x97 - x37 * (2 * x97 + x98 + x99));
//     element_matrix[40] = x102;
//     element_matrix[41] = x103;
//     element_matrix[42] = x107;
//     element_matrix[43] = x109;
//     element_matrix[44] = x110;
//     element_matrix[45] = x114;
//     element_matrix[46] = x115;
//     element_matrix[47] = x116;
//     element_matrix[48] = x51;
//     element_matrix[49] = x74;
//     element_matrix[50] = x87;
//     element_matrix[51] = x102;
//     element_matrix[52] = x19 * (-x24 * x99 - x37 * (x97 + x98 + 2 * x99));
//     element_matrix[53] = x117;
//     element_matrix[54] = x119;
//     element_matrix[55] = x120;
//     element_matrix[56] = x121;
//     element_matrix[57] = x122;
//     element_matrix[58] = x123;
//     element_matrix[59] = x124;
//     element_matrix[60] = x52;
//     element_matrix[61] = x75;
//     element_matrix[62] = x88;
//     element_matrix[63] = x103;
//     element_matrix[64] = x117;
//     element_matrix[65] = x19 * (-x24 * x98 - x37 * (x97 + 2 * x98 + x99));
//     element_matrix[66] = x126;
//     element_matrix[67] = x127;
//     element_matrix[68] = x128;
//     element_matrix[69] = x129;
//     element_matrix[70] = x130;
//     element_matrix[71] = x131;
//     element_matrix[72] = x59;
//     element_matrix[73] = x77;
//     element_matrix[74] = x90;
//     element_matrix[75] = x107;
//     element_matrix[76] = x119;
//     element_matrix[77] = x126;
//     element_matrix[78] = x19 * (-x132 * x24 - x37 * (2 * x132 + x133 + x134));
//     element_matrix[79] = x137;
//     element_matrix[80] = x138;
//     element_matrix[81] = x142;
//     element_matrix[82] = x144;
//     element_matrix[83] = x145;
//     element_matrix[84] = x61;
//     element_matrix[85] = x78;
//     element_matrix[86] = x91;
//     element_matrix[87] = x109;
//     element_matrix[88] = x120;
//     element_matrix[89] = x127;
//     element_matrix[90] = x137;
//     element_matrix[91] = x19 * (-x134 * x24 - x37 * (x132 + x133 + 2 * x134));
//     element_matrix[92] = x146;
//     element_matrix[93] = x148;
//     element_matrix[94] = x149;
//     element_matrix[95] = x150;
//     element_matrix[96] = x62;
//     element_matrix[97] = x79;
//     element_matrix[98] = x92;
//     element_matrix[99] = x110;
//     element_matrix[100] = x121;
//     element_matrix[101] = x128;
//     element_matrix[102] = x138;
//     element_matrix[103] = x146;
//     element_matrix[104] = x19 * (-x133 * x24 - x37 * (x132 + 2 * x133 + x134));
//     element_matrix[105] = x152;
//     element_matrix[106] = x153;
//     element_matrix[107] = x154;
//     element_matrix[108] = x66;
//     element_matrix[109] = x81;
//     element_matrix[110] = x94;
//     element_matrix[111] = x114;
//     element_matrix[112] = x122;
//     element_matrix[113] = x129;
//     element_matrix[114] = x142;
//     element_matrix[115] = x148;
//     element_matrix[116] = x152;
//     element_matrix[117] = x19 * (-x155 * x24 - x37 * (2 * x155 + x156 + x157));
//     element_matrix[118] = x160;
//     element_matrix[119] = x161;
//     element_matrix[120] = x68;
//     element_matrix[121] = x82;
//     element_matrix[122] = x95;
//     element_matrix[123] = x115;
//     element_matrix[124] = x123;
//     element_matrix[125] = x130;
//     element_matrix[126] = x144;
//     element_matrix[127] = x149;
//     element_matrix[128] = x153;
//     element_matrix[129] = x160;
//     element_matrix[130] = x19 * (-x157 * x24 - x37 * (x155 + x156 + 2 * x157));
//     element_matrix[131] = x162;
//     element_matrix[132] = x69;
//     element_matrix[133] = x83;
//     element_matrix[134] = x96;
//     element_matrix[135] = x116;
//     element_matrix[136] = x124;
//     element_matrix[137] = x131;
//     element_matrix[138] = x145;
//     element_matrix[139] = x150;
//     element_matrix[140] = x154;
//     element_matrix[141] = x161;
//     element_matrix[142] = x162;
//     element_matrix[143] = x19 * (-x156 * x24 - x37 * (x155 + 2 * x156 + x157));
// }
