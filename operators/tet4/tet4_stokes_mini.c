#include "tet4_stokes_mini.h"

#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <mpi.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>

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

SFEM_INLINE void tet4_stokes_assemble_hessian_kernel(const real_t mu,
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
                                                      real_t *const SFEM_RESTRICT element_vector) {}

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
#pragma omp for nowait
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
            const idx_t i3 = ev[2];

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
#pragma omp for nowait
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
