#include "tet4_stokes_mini.h"

#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <mpi.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_symmetric(int n, const real_t *const element_matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const real_t diff = element_matrix[i * n + j] - element_matrix[i + j * n];
            assert(diff < 1e-16);
            if (diff > 1e-16) {
                return 1;
            }

            // printf("%g ", element_matrix[i * n + j]);
        }

        // printf("\n");
    }

    // printf("\n");

    return 0;
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols4(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(4)
            for (int d = 0; d < 4; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static SFEM_INLINE void tet4_stokes_assemble_hessian_kernel(const real_t mu,
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
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = py0 - py3;
    const real_t x8 = x0 * x7;
    const real_t x9 = px0 - px3;
    const real_t x10 = x4 * x9;
    const real_t x11 = -x10 + x8;
    const real_t x12 = x11 * x6;
    const real_t x13 = x3 * x7;
    const real_t x14 = -x1 * x9 + x13;
    const real_t x15 = x11 * x14;
    const real_t x16 = pz0 - pz2;
    const real_t x17 = x0 * x16;
    const real_t x18 = pz0 - pz1;
    const real_t x19 = x18 * x3;
    const real_t x20 = x17 - x19;
    const real_t x21 = pz0 - pz3;
    const real_t x22 = x0 * x21;
    const real_t x23 = x18 * x9;
    const real_t x24 = x22 - x23;
    const real_t x25 = x20 * x24;
    const real_t x26 = x21 * x3;
    const real_t x27 = x16 * x9;
    const real_t x28 = x26 - x27;
    const real_t x29 = x24 * x28;
    const real_t x30 = x16 * x4;
    const real_t x31 = -x1 * x18 + x30;
    const real_t x32 = x21 * x4;
    const real_t x33 = x18 * x7;
    const real_t x34 = x32 - x33;
    const real_t x35 = x31 * x34;
    const real_t x36 = x1 * x21;
    const real_t x37 = -x16 * x7 + x36;
    const real_t x38 = x34 * x37;
    const real_t x39 = x14 * x6;
    const real_t x40 = x20 * x28;
    const real_t x41 = x31 * x37;
    const real_t x42 = pow(x14, 2) + pow(x37, 2);
    const real_t x43 = pow(x28, 2) + x42;
    const real_t x44 = pow(x24, 2);
    const real_t x45 = pow(x11, 2) + pow(x34, 2) + x44;
    const real_t x46 = pow(x31, 2) + pow(x6, 2);
    const real_t x47 = pow(x20, 2) + x46;
    const real_t x48 = x43 + x45 + x47;
    const real_t x49 = x2 * x21;
    const real_t x50 = x16 * x8;
    const real_t x51 = x21 * x5;
    const real_t x52 = x19 * x7;
    const real_t x53 = x10 * x16;
    const real_t x54 = x1 * x23;
    const real_t x55 = mu / (6 * x49 - 6 * x50 - 6 * x51 + 6 * x52 + 6 * x53 - 6 * x54);
    const real_t x56 = -x55 * (-2 * x12 - 2 * x15 - 2 * x25 - 2 * x29 - 2 * x35 - 2 * x38 +
                               2 * x39 + 2 * x40 + 2 * x41 + x48);
    const real_t x57 = -x3;
    const real_t x58 = -x21;
    const real_t x59 = -x9;
    const real_t x60 = -x16;
    const real_t x61 = x57 * x58 - x59 * x60;
    const real_t x62 = x24 * x61;
    const real_t x63 = -x0;
    const real_t x64 = -x7;
    const real_t x65 = -x4;
    const real_t x66 = -x59 * x65 + x63 * x64;
    const real_t x67 = x14 * x66;
    const real_t x68 = -x18;
    const real_t x69 = x58 * x65 - x64 * x68;
    const real_t x70 = x37 * x69;
    const real_t x71 = -x57 * x68 + x60 * x63;
    const real_t x72 = x39 + x41;
    const real_t x73 = x61 * x71 + x72;
    const real_t x74 = x49 - x50 - x51 + x52 + x53 - x54;
    const real_t x75 = 1.0 / x74;
    const real_t x76 = (1.0 / 6.0) * mu * x75;
    const real_t x77 = x76 * (x42 + pow(x61, 2) - x62 - x67 - x70 + x73);
    const real_t x78 = x62 + x67 + x70;
    const real_t x79 = x6 * x66;
    const real_t x80 = x24 * x71;
    const real_t x81 = x31 * x69;
    const real_t x82 = x79 + x80 + x81;
    const real_t x83 = x76 * (x44 + pow(x66, 2) + pow(x69, 2) - x78 - x82);
    const real_t x84 = x76 * (x46 + pow(x71, 2) + x73 - x79 - x80 - x81);
    const real_t x85 = (1.0 / 24.0) * py1;
    const real_t x86 = (1.0 / 24.0) * py2;
    const real_t x87 = (1.0 / 24.0) * py3;
    const real_t x88 = -pz1 * x86 + pz1 * x87 + pz2 * x85 - pz2 * x87 - pz3 * x85 + pz3 * x86;
    const real_t x89 = -x43 * x55;
    const real_t x90 = x76 * x78;
    const real_t x91 = x40 + x72;
    const real_t x92 = -x55 * x91;
    const real_t x93 = (1.0 / 24.0) * x16 * x7 - 1.0 / 24.0 * x36;
    const real_t x94 = -x45 * x55;
    const real_t x95 = x76 * x82;
    const real_t x96 = (1.0 / 24.0) * x32 - 1.0 / 24.0 * x33;
    const real_t x97 = -x47 * x55;
    const real_t x98 = (1.0 / 24.0) * x1 * x18 - 1.0 / 24.0 * x30;
    const real_t x99 = -1.0 / 24.0 * px1 * pz2 + (1.0 / 24.0) * px1 * pz3 +
                       (1.0 / 24.0) * px2 * pz1 - 1.0 / 24.0 * px2 * pz3 - 1.0 / 24.0 * px3 * pz1 +
                       (1.0 / 24.0) * px3 * pz2;
    const real_t x100 = (1.0 / 24.0) * x26 - 1.0 / 24.0 * x27;
    const real_t x101 = (1.0 / 24.0) * x18 * x9 - 1.0 / 24.0 * x22;
    const real_t x102 = (1.0 / 24.0) * x17 - 1.0 / 24.0 * x19;
    const real_t x103 = px1 * x86 - px1 * x87 - px2 * x85 + px2 * x87 + px3 * x85 - px3 * x86;
    const real_t x104 = (1.0 / 24.0) * x1 * x9 - 1.0 / 24.0 * x13;
    const real_t x105 = -1.0 / 24.0 * x10 + (1.0 / 24.0) * x8;
    const real_t x106 = -1.0 / 24.0 * x2 + (1.0 / 24.0) * x3 * x4;
    const real_t x107 = -x22 + x23 + x61 + x71;
    const real_t x108 = -x74;
    const real_t x109 = x0 * x7 - x10 - x14 - x6;
    const real_t x110 = x21 * x4 - x31 - x33 - x37;
    const real_t x111 = (1.0 / 560.0) / (mu * (-x12 - x15 - x25 - x29 - x35 - x38 + x48 + x91));
    const real_t x112 = x107 * x74;
    const real_t x113 = x111 * (-x108 * (x109 * x14 + x110 * x37) - x112 * x28);
    const real_t x114 = x111 * (x108 * (x109 * x11 + x110 * x34) + x112 * x24);
    const real_t x115 = x111 * (-x108 * (x109 * x6 + x110 * x31) - x112 * x20);
    const real_t x116 = x111 * x74;
    const real_t x117 = x116 * (-x15 - x29 - x38);
    const real_t x118 = x116 * x91;
    const real_t x119 = x116 * (-x12 - x25 - x35);
    element_matrix[0] = x56;
    element_matrix[1] = x77;
    element_matrix[2] = x83;
    element_matrix[3] = x84;
    element_matrix[4] = 0;
    element_matrix[5] = 0;
    element_matrix[6] = 0;
    element_matrix[7] = 0;
    element_matrix[8] = 0;
    element_matrix[9] = 0;
    element_matrix[10] = 0;
    element_matrix[11] = 0;
    element_matrix[12] = x88;
    element_matrix[13] = x88;
    element_matrix[14] = x88;
    element_matrix[15] = x88;
    element_matrix[16] = x77;
    element_matrix[17] = x89;
    element_matrix[18] = x90;
    element_matrix[19] = x92;
    element_matrix[20] = 0;
    element_matrix[21] = 0;
    element_matrix[22] = 0;
    element_matrix[23] = 0;
    element_matrix[24] = 0;
    element_matrix[25] = 0;
    element_matrix[26] = 0;
    element_matrix[27] = 0;
    element_matrix[28] = x93;
    element_matrix[29] = x93;
    element_matrix[30] = x93;
    element_matrix[31] = x93;
    element_matrix[32] = x83;
    element_matrix[33] = x90;
    element_matrix[34] = x94;
    element_matrix[35] = x95;
    element_matrix[36] = 0;
    element_matrix[37] = 0;
    element_matrix[38] = 0;
    element_matrix[39] = 0;
    element_matrix[40] = 0;
    element_matrix[41] = 0;
    element_matrix[42] = 0;
    element_matrix[43] = 0;
    element_matrix[44] = x96;
    element_matrix[45] = x96;
    element_matrix[46] = x96;
    element_matrix[47] = x96;
    element_matrix[48] = x84;
    element_matrix[49] = x92;
    element_matrix[50] = x95;
    element_matrix[51] = x97;
    element_matrix[52] = 0;
    element_matrix[53] = 0;
    element_matrix[54] = 0;
    element_matrix[55] = 0;
    element_matrix[56] = 0;
    element_matrix[57] = 0;
    element_matrix[58] = 0;
    element_matrix[59] = 0;
    element_matrix[60] = x98;
    element_matrix[61] = x98;
    element_matrix[62] = x98;
    element_matrix[63] = x98;
    element_matrix[64] = 0;
    element_matrix[65] = 0;
    element_matrix[66] = 0;
    element_matrix[67] = 0;
    element_matrix[68] = x56;
    element_matrix[69] = x77;
    element_matrix[70] = x83;
    element_matrix[71] = x84;
    element_matrix[72] = 0;
    element_matrix[73] = 0;
    element_matrix[74] = 0;
    element_matrix[75] = 0;
    element_matrix[76] = x99;
    element_matrix[77] = x99;
    element_matrix[78] = x99;
    element_matrix[79] = x99;
    element_matrix[80] = 0;
    element_matrix[81] = 0;
    element_matrix[82] = 0;
    element_matrix[83] = 0;
    element_matrix[84] = x77;
    element_matrix[85] = x89;
    element_matrix[86] = x90;
    element_matrix[87] = x92;
    element_matrix[88] = 0;
    element_matrix[89] = 0;
    element_matrix[90] = 0;
    element_matrix[91] = 0;
    element_matrix[92] = x100;
    element_matrix[93] = x100;
    element_matrix[94] = x100;
    element_matrix[95] = x100;
    element_matrix[96] = 0;
    element_matrix[97] = 0;
    element_matrix[98] = 0;
    element_matrix[99] = 0;
    element_matrix[100] = x83;
    element_matrix[101] = x90;
    element_matrix[102] = x94;
    element_matrix[103] = x95;
    element_matrix[104] = 0;
    element_matrix[105] = 0;
    element_matrix[106] = 0;
    element_matrix[107] = 0;
    element_matrix[108] = x101;
    element_matrix[109] = x101;
    element_matrix[110] = x101;
    element_matrix[111] = x101;
    element_matrix[112] = 0;
    element_matrix[113] = 0;
    element_matrix[114] = 0;
    element_matrix[115] = 0;
    element_matrix[116] = x84;
    element_matrix[117] = x92;
    element_matrix[118] = x95;
    element_matrix[119] = x97;
    element_matrix[120] = 0;
    element_matrix[121] = 0;
    element_matrix[122] = 0;
    element_matrix[123] = 0;
    element_matrix[124] = x102;
    element_matrix[125] = x102;
    element_matrix[126] = x102;
    element_matrix[127] = x102;
    element_matrix[128] = 0;
    element_matrix[129] = 0;
    element_matrix[130] = 0;
    element_matrix[131] = 0;
    element_matrix[132] = 0;
    element_matrix[133] = 0;
    element_matrix[134] = 0;
    element_matrix[135] = 0;
    element_matrix[136] = x56;
    element_matrix[137] = x77;
    element_matrix[138] = x83;
    element_matrix[139] = x84;
    element_matrix[140] = x103;
    element_matrix[141] = x103;
    element_matrix[142] = x103;
    element_matrix[143] = x103;
    element_matrix[144] = 0;
    element_matrix[145] = 0;
    element_matrix[146] = 0;
    element_matrix[147] = 0;
    element_matrix[148] = 0;
    element_matrix[149] = 0;
    element_matrix[150] = 0;
    element_matrix[151] = 0;
    element_matrix[152] = x77;
    element_matrix[153] = x89;
    element_matrix[154] = x90;
    element_matrix[155] = x92;
    element_matrix[156] = x104;
    element_matrix[157] = x104;
    element_matrix[158] = x104;
    element_matrix[159] = x104;
    element_matrix[160] = 0;
    element_matrix[161] = 0;
    element_matrix[162] = 0;
    element_matrix[163] = 0;
    element_matrix[164] = 0;
    element_matrix[165] = 0;
    element_matrix[166] = 0;
    element_matrix[167] = 0;
    element_matrix[168] = x83;
    element_matrix[169] = x90;
    element_matrix[170] = x94;
    element_matrix[171] = x95;
    element_matrix[172] = x105;
    element_matrix[173] = x105;
    element_matrix[174] = x105;
    element_matrix[175] = x105;
    element_matrix[176] = 0;
    element_matrix[177] = 0;
    element_matrix[178] = 0;
    element_matrix[179] = 0;
    element_matrix[180] = 0;
    element_matrix[181] = 0;
    element_matrix[182] = 0;
    element_matrix[183] = 0;
    element_matrix[184] = x84;
    element_matrix[185] = x92;
    element_matrix[186] = x95;
    element_matrix[187] = x97;
    element_matrix[188] = x106;
    element_matrix[189] = x106;
    element_matrix[190] = x106;
    element_matrix[191] = x106;
    element_matrix[192] = x88;
    element_matrix[193] = x93;
    element_matrix[194] = x96;
    element_matrix[195] = x98;
    element_matrix[196] = x99;
    element_matrix[197] = x100;
    element_matrix[198] = x101;
    element_matrix[199] = x102;
    element_matrix[200] = x103;
    element_matrix[201] = x104;
    element_matrix[202] = x105;
    element_matrix[203] = x106;
    element_matrix[204] =
        x111 * x75 * (pow(x107, 2) * pow(x74, 2) + pow(x108, 2) * (pow(x109, 2) + pow(x110, 2)));
    element_matrix[205] = x113;
    element_matrix[206] = x114;
    element_matrix[207] = x115;
    element_matrix[208] = x88;
    element_matrix[209] = x93;
    element_matrix[210] = x96;
    element_matrix[211] = x98;
    element_matrix[212] = x99;
    element_matrix[213] = x100;
    element_matrix[214] = x101;
    element_matrix[215] = x102;
    element_matrix[216] = x103;
    element_matrix[217] = x104;
    element_matrix[218] = x105;
    element_matrix[219] = x106;
    element_matrix[220] = x113;
    element_matrix[221] = x116 * x43;
    element_matrix[222] = x117;
    element_matrix[223] = x118;
    element_matrix[224] = x88;
    element_matrix[225] = x93;
    element_matrix[226] = x96;
    element_matrix[227] = x98;
    element_matrix[228] = x99;
    element_matrix[229] = x100;
    element_matrix[230] = x101;
    element_matrix[231] = x102;
    element_matrix[232] = x103;
    element_matrix[233] = x104;
    element_matrix[234] = x105;
    element_matrix[235] = x106;
    element_matrix[236] = x114;
    element_matrix[237] = x117;
    element_matrix[238] = x116 * x45;
    element_matrix[239] = x119;
    element_matrix[240] = x88;
    element_matrix[241] = x93;
    element_matrix[242] = x96;
    element_matrix[243] = x98;
    element_matrix[244] = x99;
    element_matrix[245] = x100;
    element_matrix[246] = x101;
    element_matrix[247] = x102;
    element_matrix[248] = x103;
    element_matrix[249] = x104;
    element_matrix[250] = x105;
    element_matrix[251] = x106;
    element_matrix[252] = x115;
    element_matrix[253] = x118;
    element_matrix[254] = x119;
    element_matrix[255] = x116 * x47;
}

SFEM_INLINE void tet4_stokes_mini_assemble_rhs_kernel(const real_t mu,
                                                      const real_t rho,
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
                                                      const real_t *const SFEM_RESTRICT u_rhs,
                                                      const real_t *const SFEM_RESTRICT p_rhs,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    //     const real_t x0 = u_rhs[3] + u_rhs[4];
    //     const real_t x1 = u_rhs[2] + x0;
    //     const real_t x2 = pz0 - pz3;
    //     const real_t x3 = px0 - px1;
    //     const real_t x4 = py0 - py2;
    //     const real_t x5 = x3 * x4;
    //     const real_t x6 = py0 - py3;
    //     const real_t x7 = px0 - px2;
    //     const real_t x8 = pz0 - pz1;
    //     const real_t x9 = x7 * x8;
    //     const real_t x10 = pz0 - pz2;
    //     const real_t x11 = px0 - px3;
    //     const real_t x12 = py0 - py1;
    //     const real_t x13 = x11 * x12;
    //     const real_t x14 = x3 * x6;
    //     const real_t x15 = x12 * x7;
    //     const real_t x16 = x11 * x8;
    //     const real_t x17 = x10 * x13 - x10 * x14 - x15 * x2 - x16 * x4 + x2 * x5 + x6 * x9;
    //     const real_t x18 = rho * x17;
    //     const real_t x19 = (1.0 / 120.0) * x18;
    //     const real_t x20 = u_rhs[1] + u_rhs[2];
    //     const real_t x21 = u_rhs[8] + u_rhs[9];
    //     const real_t x22 = u_rhs[7] + x21;
    //     const real_t x23 = u_rhs[6] + u_rhs[7];
    //     const real_t x24 = u_rhs[13] + u_rhs[14];
    //     const real_t x25 = u_rhs[12] + x24;
    //     const real_t x26 = u_rhs[11] + u_rhs[12];
    //     const real_t x27 = -x15 + x5;
    //     const real_t x28 = -x11 * x4 + x6 * x7;
    //     const real_t x29 = u_rhs[11] + x25;
    //     const real_t x30 = 3 * x17;
    //     const real_t x31 = x29 * x30;
    //     const real_t x32 = x6 * x8;
    //     const real_t x33 = x12 * x2;
    //     const real_t x34 = x10 * x12 - x4 * x8;
    //     const real_t x35 = -x10 * x6 + x2 * x4;
    //     const real_t x36 = u_rhs[1] + x1;
    //     const real_t x37 = x30 * x36;
    //     const real_t x38 = x10 * x3 - x9;
    //     const real_t x39 = -x10 * x11 + x2 * x7;
    //     const real_t x40 = u_rhs[6] + x22;
    //     const real_t x41 = x30 * x40;
    //     const real_t x42 = p_rhs[2] + p_rhs[3];
    //     const real_t x43 = -x13 + x14;
    //     const real_t x44 = -x16 + x2 * x3;
    //     const real_t x45 = -x32 + x33;
    //     const real_t x46 = pow(x27, 2) + x27 * x28 - x27 * x43 + pow(x28, 2) - x28 * x43 +
    //     pow(x34, 2) +
    //                        x34 * x35 - x34 * x45 + pow(x35, 2) - x35 * x45 + pow(x38, 2) + x38 *
    //                        x39 - x38 * x44 + pow(x39, 2) - x39 * x44 + pow(x43, 2) + pow(x44, 2)
    //                        + pow(x45, 2);
    //     const real_t x47 = 56 * mu * x46;
    //     const real_t x48 = (1.0 / 6720.0) * x18 / (mu * x46);
    //     const real_t x49 = p_rhs[0] + p_rhs[1];
    //     element_vector[0] = x19 * (-2 * u_rhs[1] - x1);
    //     element_vector[1] = x19 * (-u_rhs[1] - 2 * u_rhs[2] - x0);
    //     element_vector[2] = x19 * (-2 * u_rhs[3] - u_rhs[4] - x20);
    //     element_vector[3] = x19 * (-u_rhs[3] - 2 * u_rhs[4] - x20);
    //     element_vector[4] = x19 * (-2 * u_rhs[6] - x22);
    //     element_vector[5] = x19 * (-u_rhs[6] - 2 * u_rhs[7] - x21);
    //     element_vector[6] = x19 * (-2 * u_rhs[8] - u_rhs[9] - x23);
    //     element_vector[7] = x19 * (-u_rhs[8] - 2 * u_rhs[9] - x23);
    //     element_vector[8] = x19 * (-2 * u_rhs[11] - x25);
    //     element_vector[9] = x19 * (-u_rhs[11] - 2 * u_rhs[12] - x24);
    //     element_vector[10] = x19 * (-2 * u_rhs[13] - u_rhs[14] - x26);
    //     element_vector[11] = x19 * (-u_rhs[13] - 2 * u_rhs[14] - x26);
    //     element_vector[12] =
    //         x48 * (x31 * (x13 - x14 + x27 + x28) + x37 * (x32 - x33 + x34 + x35) +
    //                x41 * (-x16 + x2 * x3 - x38 - x39) - x47 * (2 * p_rhs[0] + p_rhs[1] + x42));
    //     element_vector[13] =
    //         x48 * (3 * x17 * x39 * x40 - x28 * x31 - x35 * x37 - x47 * (p_rhs[0] + 2 * p_rhs[1] +
    //         x42));
    //     element_vector[14] = x48 * (3 * x17 * x29 * x43 + 3 * x17 * x36 * x45 - x41 * x44 -
    //                                 x47 * (2 * p_rhs[2] + p_rhs[3] + x49));
    //     element_vector[15] =
    //         x48 * (3 * x17 * x38 * x40 - x27 * x31 - x34 * x37 - x47 * (p_rhs[2] + 2 * p_rhs[3] +
    //         x49));

    const real_t x0 = -px0 + px1;
    const real_t x1 = (1.0 / 60.0) * x0;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = -py0 + py2;
    const real_t x4 = x2 * x3;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz2;
    const real_t x7 = x5 * x6;
    const real_t x8 = -px0 + px2;
    const real_t x9 = (1.0 / 60.0) * x8;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x2;
    const real_t x12 = -pz0 + pz1;
    const real_t x13 = x12 * x5;
    const real_t x14 = -px0 + px3;
    const real_t x15 = (1.0 / 60.0) * x14;
    const real_t x16 = x10 * x6;
    const real_t x17 = x12 * x3;
    const real_t x18 = x1 * x4 - x1 * x7 - x11 * x9 + x13 * x9 + x15 * x16 - x15 * x17;
    const real_t x19 = rho * u_rhs[1];
    const real_t x20 = (1.0 / 120.0) * x0;
    const real_t x21 = (1.0 / 120.0) * x8;
    const real_t x22 = (1.0 / 120.0) * x14;
    const real_t x23 = -x11 * x21 + x13 * x21 + x16 * x22 - x17 * x22 + x20 * x4 - x20 * x7;
    const real_t x24 = rho * x23;
    const real_t x25 = u_rhs[2] * x24;
    const real_t x26 = u_rhs[3] * x24;
    const real_t x27 = u_rhs[4] * x24;
    const real_t x28 = x26 + x27;
    const real_t x29 = x19 * x23;
    const real_t x30 = rho * x18;
    const real_t x31 = x25 + x29;
    const real_t x32 = u_rhs[7] * x24;
    const real_t x33 = u_rhs[8] * x24;
    const real_t x34 = u_rhs[9] * x24;
    const real_t x35 = x33 + x34;
    const real_t x36 = u_rhs[6] * x24;
    const real_t x37 = x32 + x36;
    const real_t x38 = u_rhs[12] * x24;
    const real_t x39 = u_rhs[13] * x24;
    const real_t x40 = u_rhs[14] * x24;
    const real_t x41 = x39 + x40;
    const real_t x42 = u_rhs[11] * x24;
    const real_t x43 = x38 + x42;
    const real_t x44 = p_rhs[1] * x24;
    const real_t x45 = p_rhs[2] * x24;
    const real_t x46 = p_rhs[3] * x24;
    const real_t x47 = x45 + x46;
    const real_t x48 = p_rhs[0] * x24;
    const real_t x49 = x44 + x48;
    element_vector[0] = x18 * x19 + x25 + x28;
    element_vector[1] = u_rhs[2] * x30 + x28 + x29;
    element_vector[2] = u_rhs[3] * x30 + x27 + x31;
    element_vector[3] = u_rhs[4] * x30 + x26 + x31;
    element_vector[4] = u_rhs[6] * x30 + x32 + x35;
    element_vector[5] = u_rhs[7] * x30 + x35 + x36;
    element_vector[6] = u_rhs[8] * x30 + x34 + x37;
    element_vector[7] = u_rhs[9] * x30 + x33 + x37;
    element_vector[8] = u_rhs[11] * x30 + x38 + x41;
    element_vector[9] = u_rhs[12] * x30 + x41 + x42;
    element_vector[10] = u_rhs[13] * x30 + x40 + x43;
    element_vector[11] = u_rhs[14] * x30 + x39 + x43;
    element_vector[12] = p_rhs[0] * x30 + x44 + x47;
    element_vector[13] = p_rhs[1] * x30 + x47 + x48;
    element_vector[14] = p_rhs[2] * x30 + x46 + x49;
    element_vector[15] = p_rhs[3] * x30 + x45 + x49;
}

static SFEM_INLINE void tet4_stokes_mini_apply_kernel(const real_t mu,
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
                                               const real_t *const SFEM_RESTRICT increment,
                                               real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -py0 + py1;
    const real_t x1 = -pz0 + pz2;
    const real_t x2 = x0*x1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = x3*x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -px0 + px1;
    const real_t x8 = -pz0 + pz3;
    const real_t x9 = x3*x8;
    const real_t x10 = -px0 + px2;
    const real_t x11 = -py0 + py3;
    const real_t x12 = -px0 + px3;
    const real_t x13 = x1*x11;
    const real_t x14 = x0*x8;
    const real_t x15 = x10*x11*x4 - x10*x14 + x12*x2 - x12*x5 - x13*x7 + x7*x9;
    const real_t x16 = 1.0/x15;
    const real_t x17 = (1.0/24.0)*x16;
    const real_t x18 = x11*x4 - x14;
    const real_t x19 = -x13 + x9;
    const real_t x20 = x17*x18 + x17*x19 + x17*x6;
    const real_t x21 = x15*x20;
    const real_t x22 = x10*x11 - x12*x3;
    const real_t x23 = pow(x22, 2);
    const real_t x24 = pow(x15, -2);
    const real_t x25 = (1.0/6.0)*x24;
    const real_t x26 = x1*x12 - x10*x8;
    const real_t x27 = pow(x26, 2);
    const real_t x28 = pow(x19, 2);
    const real_t x29 = x23*x25 + x25*x27 + x25*x28;
    const real_t x30 = x0*x12 - x11*x7;
    const real_t x31 = x25*x30;
    const real_t x32 = -x12*x4 + x7*x8;
    const real_t x33 = x25*x32;
    const real_t x34 = x18*x19;
    const real_t x35 = x22*x31 + x25*x34 + x26*x33;
    const real_t x36 = -x0*x10 + x3*x7;
    const real_t x37 = x22*x36;
    const real_t x38 = -x1*x7 + x10*x4;
    const real_t x39 = x26*x38;
    const real_t x40 = x25*x6;
    const real_t x41 = x19*x40 + x25*x37 + x25*x39;
    const real_t x42 = -x29 - x35 - x41;
    const real_t x43 = mu*x15;
    const real_t x44 = increment[1]*x43;
    const real_t x45 = pow(x30, 2);
    const real_t x46 = pow(x32, 2);
    const real_t x47 = pow(x18, 2);
    const real_t x48 = x25*x45 + x25*x46 + x25*x47;
    const real_t x49 = x18*x40 + x31*x36 + x33*x38;
    const real_t x50 = -x35 - x48 - x49;
    const real_t x51 = increment[2]*x43;
    const real_t x52 = pow(x36, 2);
    const real_t x53 = pow(x38, 2);
    const real_t x54 = pow(x6, 2);
    const real_t x55 = x25*x52 + x25*x53 + x25*x54;
    const real_t x56 = -x41 - x49 - x55;
    const real_t x57 = increment[3]*x43;
    const real_t x58 = (1.0/3.0)*x24;
    const real_t x59 = x30*x58;
    const real_t x60 = x32*x58;
    const real_t x61 = x58*x6;
    const real_t x62 = x18*x61 + x19*x61 + x22*x59 + x26*x60 + x29 + x34*x58 + x36*x59 + x37*x58 + x38*x60 + x39*x58 + x48 + x55;
    const real_t x63 = increment[0]*x43;
    const real_t x64 = (1.0/24.0)*x19;
    const real_t x65 = (1.0/24.0)*x18;
    const real_t x66 = (1.0/24.0)*x6;
    const real_t x67 = x17*x26 + x17*x32 + x17*x38;
    const real_t x68 = x15*x67;
    const real_t x69 = (1.0/24.0)*x26;
    const real_t x70 = (1.0/24.0)*x32;
    const real_t x71 = (1.0/24.0)*x38;
    const real_t x72 = x17*x22 + x17*x30 + x17*x36;
    const real_t x73 = x15*x72;
    const real_t x74 = increment[10]*x43;
    const real_t x75 = increment[11]*x43;
    const real_t x76 = increment[9]*x43;
    const real_t x77 = increment[8]*x43;
    const real_t x78 = (1.0/24.0)*x22;
    const real_t x79 = (1.0/24.0)*x30;
    const real_t x80 = (1.0/24.0)*x36;
    const real_t x81 = (16.0/315.0)*x16;
    const real_t x82 = -x22*x81 - x30*x81 - x36*x81;
    const real_t x83 = (4096.0/2835.0)*x24;
    const real_t x84 = x30*x83;
    const real_t x85 = x32*x83;
    const real_t x86 = x6*x83;
    const real_t x87 = 1/(mu*(x18*x86 + x19*x86 + x22*x84 + x23*x83 + x26*x85 + x27*x83 + x28*x83 + x34*x83 + x36*x84 + x37*x83 + x38*x85 + x39*x83 + x45*x83 + x46*x83 + x47*x83 + x52*x83 + x53*x83 + x54*x83));
    const real_t x88 = x15*x87;
    const real_t x89 = -x26*x81 - x32*x81 - x38*x81;
    const real_t x90 = -x18*x81 - x19*x81 - x6*x81;
    const real_t x91 = (16.0/315.0)*x87;
    const real_t x92 = x82*x91;
    const real_t x93 = x89*x91;
    const real_t x94 = x90*x91;
    const real_t x95 = -x19*x94 - x22*x92 - x26*x93;
    const real_t x96 = -x18*x94 - x30*x92 - x32*x93;
    const real_t x97 = -x36*x92 - x38*x93 - x6*x94;
    const real_t x98 = -increment[0]*x15*x20 + increment[10]*x79 + increment[11]*x80 + increment[1]*x64 + increment[2]*x65 + increment[3]*x66 - increment[4]*x15*x67 + increment[5]*x69 + increment[6]*x70 + increment[7]*x71 - increment[8]*x15*x72 + increment[9]*x78;
    const real_t x99 = (256.0/99225.0)*x16*x87;
    const real_t x100 = x30*x99;
    const real_t x101 = x32*x99;
    const real_t x102 = -x100*x22 - x101*x26 - x34*x99;
    const real_t x103 = x6*x99;
    const real_t x104 = -x103*x19 - x37*x99 - x39*x99;
    const real_t x105 = -x100*x36 - x101*x38 - x103*x18;
    element_vector[0] = increment[12]*x21 + increment[13]*x21 + increment[14]*x21 + increment[15]*x21 + x42*x44 + x50*x51 + x56*x57 + x62*x63;
    element_vector[1] = -increment[12]*x64 - increment[13]*x64 - increment[14]*x64 - increment[15]*x64 + x29*x44 + x35*x51 + x41*x57 + x42*x63;
    element_vector[2] = -increment[12]*x65 - increment[13]*x65 - increment[14]*x65 - increment[15]*x65 + x35*x44 + x48*x51 + x49*x57 + x50*x63;
    element_vector[3] = -increment[12]*x66 - increment[13]*x66 - increment[14]*x66 - increment[15]*x66 + x41*x44 + x49*x51 + x55*x57 + x56*x63;
    element_vector[4] = increment[12]*x68 + increment[13]*x68 + increment[14]*x68 + increment[15]*x68 + increment[4]*x43*x62 + increment[5]*x42*x43 + increment[6]*x43*x50 + increment[7]*x43*x56;
    element_vector[5] = -increment[12]*x69 - increment[13]*x69 - increment[14]*x69 - increment[15]*x69 + increment[4]*mu*x15*x42 + increment[5]*mu*x15*x29 + increment[6]*mu*x15*x35 + increment[7]*mu*x15*x41;
    element_vector[6] = -increment[12]*x70 - increment[13]*x70 - increment[14]*x70 - increment[15]*x70 + increment[4]*mu*x15*x50 + increment[5]*mu*x15*x35 + increment[6]*mu*x15*x48 + increment[7]*mu*x15*x49;
    element_vector[7] = -increment[12]*x71 - increment[13]*x71 - increment[14]*x71 - increment[15]*x71 + increment[4]*mu*x15*x56 + increment[5]*mu*x15*x41 + increment[6]*mu*x15*x49 + increment[7]*mu*x15*x55;
    element_vector[8] = increment[12]*x73 + increment[13]*x73 + increment[14]*x73 + increment[15]*x73 + x42*x76 + x50*x74 + x56*x75 + x62*x77;
    element_vector[9] = -increment[12]*x78 - increment[13]*x78 - increment[14]*x78 - increment[15]*x78 + x29*x76 + x35*x74 + x41*x75 + x42*x77;
    element_vector[10] = -increment[12]*x79 - increment[13]*x79 - increment[14]*x79 - increment[15]*x79 + x35*x76 + x48*x74 + x49*x75 + x50*x77;
    element_vector[11] = -increment[12]*x80 - increment[13]*x80 - increment[14]*x80 - increment[15]*x80 + x41*x76 + x49*x74 + x55*x75 + x56*x77;
    element_vector[12] = increment[12]*(-pow(x82, 2)*x88 - x88*pow(x89, 2) - x88*pow(x90, 2)) + increment[13]*x95 + increment[14]*x96 + increment[15]*x97 - x98;
    element_vector[13] = increment[12]*x95 + increment[13]*(-x23*x99 - x27*x99 - x28*x99) + increment[14]*x102 + increment[15]*x104 - x98;
    element_vector[14] = increment[12]*x96 + increment[13]*x102 + increment[14]*(-x45*x99 - x46*x99 - x47*x99) + increment[15]*x105 - x98;
    element_vector[15] = increment[12]*x97 + increment[13]*x104 + increment[14]*x105 + increment[15]*(-x52*x99 - x53*x99 - x54*x99) - x98;
}

void tet4_stokes_mini_assemble_hessian_soa(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t **const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int n_vars = 4;
    static const int nxe = 4;
    static const int cols = 16;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4][4];
            real_t element_matrix[4 * 4 * 4 * 4];
#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet4_stokes_assemble_hessian_kernel(mu,
                                                // X-coordinates
                                                x0,
                                                x1,
                                                x2,
                                                x3,
                                                // Y-coordinates
                                                y0,
                                                y1,
                                                y2,
                                                y3,
                                                // Z-coordinates
                                                z0,
                                                z1,
                                                z2,
                                                z3,
                                                element_matrix);

            // find all indices
            for (int edof_i = 0; edof_i < nxe; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t r_begin = rowptr[dof_i];
                const idx_t lenrow = rowptr[dof_i + 1] - r_begin;
                const idx_t *row = &colidx[rowptr[dof_i]];
                find_cols4(ev, row, lenrow, ks[edof_i]);
            }

            for (int bi = 0; bi < n_vars; ++bi) {
                for (int bj = 0; bj < n_vars; ++bj) {
                    for (int edof_i = 0; edof_i < nxe; ++edof_i) {
                        const int ii = bi * nxe + edof_i;

                        const idx_t dof_i = elems[edof_i][i];
                        const idx_t r_begin = rowptr[dof_i];
                        const int bb = bi * n_vars + bj;

                        real_t *const row_values = &values[bb][r_begin];

                        for (int edof_j = 0; edof_j < nxe; ++edof_j) {
                            const int jj = bj * nxe + edof_j;
                            const real_t val = element_matrix[ii * cols + jj];

                            assert(val == val);
#pragma omp atomic update
                            row_values[ks[edof_i][edof_j]] += val;
                        }
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_stokes.c: tet4_stokes_assemble_hessian\t%g seconds\n", tock - tick);
}

void tet4_stokes_mini_assemble_hessian_aos(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t *const values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    static const int block_size = 4;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_matrix[(4 * 4) * (4 * 4)];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet4_stokes_assemble_hessian_kernel(
                // Model parameters
                mu,
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

            assert(!check_symmetric(16, element_matrix));

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                {
                    const idx_t *row = &colidx[rowptr[dof_i]];
                    find_cols4(ev, row, lenrow, ks);
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
                            const real_t val = element_matrix[ii * mat_block_size + jj];
                            assert(val == val);
#pragma omp atomic update
                            row[offset_j + bj] += val;
                        }
                    }
                }
            }
        }
    }
    const double tock = MPI_Wtime();
    printf("stokes.c: stokes_assemble_hessian_aos\t%g seconds\n", tock - tick);
}

void tet4_stokes_mini_assemble_rhs_soa(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t **const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 4;
    static const int ndofs = 4;
    static const int rows = 16;
    static const int cols = 16;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];
            real_t element_vector[4 * 4];
            real_t u_rhs[5 * 4];
            real_t p_rhs[4] = {0., 0., 0., 0.};

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            memset(u_rhs, 0, 5 * 3 * sizeof(real_t));

            for (int v = 0; v < 3; v++) {
                for (int ii = 0; ii < 4; ii++) {
                    if (forcing[v]) {
                        // Skip bubble dof
                        u_rhs[v * 5 + ii + 1] = forcing[v][ev[ii]];
                    }
                }
            }

            if (forcing[3]) {
                for (int ii = 0; ii < 4; ii++) {
                    p_rhs[ii] = forcing[3][ev[ii]];
                }
            }

            tet4_stokes_mini_assemble_rhs_kernel(mu,
                                                 rho,
                                                 // X coords
                                                 points[0][i0],
                                                 points[0][i1],
                                                 points[0][i2],
                                                 points[0][i3],
                                                 // Y coords
                                                 points[1][i0],
                                                 points[1][i1],
                                                 points[1][i2],
                                                 points[1][i3],
                                                 // Z coords
                                                 points[2][i0],
                                                 points[2][i1],
                                                 points[2][i2],
                                                 points[2][i3],
                                                 //  buffers
                                                 u_rhs,
                                                 p_rhs,
                                                 element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
                    real_t val = element_vector[d1 * 4 + edof_i];
                    assert(val == val);
#pragma omp atomic update
                    rhs[d1][dof_i] += val;
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_stokes.c: tet4_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

void tet4_stokes_mini_assemble_rhs_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t *const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 4;
    static const int ndofs = 4;
    static const int rows = 16;
    static const int cols = 16;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];
            real_t element_vector[4 * 4];
            real_t u_rhs[5 * 4];
            real_t p_rhs[4] = {0., 0., 0., 0.};

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            memset(u_rhs, 0, 5 * 3 * sizeof(real_t));

            for (int v = 0; v < 3; v++) {
                for (int ii = 0; ii < 4; ii++) {
                    if (forcing[v]) {
                        // Skip bubble dof
                        u_rhs[v * 5 + ii + 1] = forcing[v][ev[ii]];
                    }
                }
            }

            if (forcing[3]) {
                for (int ii = 0; ii < 4; ii++) {
                    p_rhs[ii] = forcing[3][ev[ii]];
                }
            }

            tet4_stokes_mini_assemble_rhs_kernel(mu,
                                                 rho,
                                                 // X coords
                                                 points[0][i0],
                                                 points[0][i1],
                                                 points[0][i2],
                                                 points[0][i3],
                                                 // Y coords
                                                 points[1][i0],
                                                 points[1][i1],
                                                 points[1][i2],
                                                 points[1][i3],
                                                 // Z coords
                                                 points[2][i0],
                                                 points[2][i1],
                                                 points[2][i2],
                                                 points[2][i3],
                                                 //  buffers
                                                 u_rhs,
                                                 p_rhs,
                                                 element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
                    real_t val = element_vector[d1 * 4 + edof_i];
                    assert(val == val);
#pragma omp atomic update
                    rhs[dof_i * n_vars + d1] += val;
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_stokes.c: tet4_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

void tet4_stokes_mini_apply_aos(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const elems,
                                geom_t **const points,
                                const real_t mu,
                                // const real_t rho,
                                const real_t *const SFEM_RESTRICT x,
                                real_t *const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 4;
    static const int ndofs = 4;
    static const int rows = 16;
    static const int cols = 16;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];
            real_t element_vector[4 * 4];
            real_t element_x[4 * 4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < 4; ++enode) {
                idx_t dof = ev[enode] * n_vars;

                for (int b = 0; b < n_vars; ++b) {
                    element_x[b * 4 + enode] = x[dof + b];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet4_stokes_mini_apply_kernel(mu,
                                          // rho,
                                          // X coords
                                          points[0][i0],
                                          points[0][i1],
                                          points[0][i2],
                                          points[0][i3],
                                          // Y coords
                                          points[1][i0],
                                          points[1][i1],
                                          points[1][i2],
                                          points[1][i3],
                                          // Z coords
                                          points[2][i0],
                                          points[2][i1],
                                          points[2][i2],
                                          points[2][i3],
                                          //  buffers
                                          element_x,
                                          element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
                    real_t val = element_vector[d1 * 4 + edof_i];
                    assert(val == val);
#pragma omp atomic update
                    rhs[dof_i * n_vars + d1] += val;
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_stokes.c: tet4_stokes_mini_apply\t%g seconds\n", tock - tick);
}

void tet4_stokes_mini_assemble_gradient_aos(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const elems,
                                            geom_t **const points,
                                            const real_t mu,
                                            const real_t *const SFEM_RESTRICT x,
                                            real_t *const SFEM_RESTRICT g) {
    tet4_stokes_mini_apply_aos(nelements, nnodes, elems, points, mu, x, g);
}
