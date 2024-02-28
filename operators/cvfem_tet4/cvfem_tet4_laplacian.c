#include "cvfem_tet4_laplacian.h"

#include "sfem_base.h"
#include "sfem_vec.h"

#include "sortreduce.h"

#include <mpi.h>
#include <stdio.h>

static SFEM_INLINE void cvfem_tet4_laplacian_hessian_kernel(const real_t px0,
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
                                                            real_t *element_matrix) {
    const real_t x0 = (1.0 / 12.0) * px2;
    const real_t x1 = (1.0 / 24.0) * px1;
    const real_t x2 = pz3 * x1;
    const real_t x3 = (1.0 / 24.0) * pz1;
    const real_t x4 = px3 * x3;
    const real_t x5 = -x2 + x4;
    const real_t x6 = (1.0 / 24.0) * px2;
    const real_t x7 = pz0 * x6;
    const real_t x8 = (1.0 / 24.0) * pz2;
    const real_t x9 = px0 * x8;
    const real_t x10 = -x7 + x9;
    const real_t x11 = (1.0 / 24.0) * px0;
    const real_t x12 = pz3 * x11;
    const real_t x13 = (1.0 / 24.0) * px3;
    const real_t x14 = pz0 * x13;
    const real_t x15 = -x12 + x14;
    const real_t x16 = px1 * pz2;
    const real_t x17 = (1.0 / 24.0) * x16;
    const real_t x18 = -1.0 / 24.0 * px2 * pz1 + x17;
    const real_t x19 = x15 + x18;
    const real_t x20 = (1.0 / 12.0) * px3 * pz2 - pz3 * x0 - x10 - x19 - x5;
    const real_t x21 = -px2;
    const real_t x22 = -px1 - x21;
    const real_t x23 = -pz3;
    const real_t x24 = -pz1 - x23;
    const real_t x25 = -px3;
    const real_t x26 = -px1 - x25;
    const real_t x27 = -pz2;
    const real_t x28 = -pz1 - x27;
    const real_t x29 = -py0 + py1;
    const real_t x30 = -pz0 - x27;
    const real_t x31 = -py2;
    const real_t x32 = -py0 - x31;
    const real_t x33 = -pz0 + pz1;
    const real_t x34 = x29 * x30 - x32 * x33;
    const real_t x35 = -px0 + px1;
    const real_t x36 = -px0 - x21;
    const real_t x37 = -x30 * x35 + x33 * x36;
    const real_t x38 = -x29 * x36 + x32 * x35;
    const real_t x39 = (1.0 / 6.0) / (x34 * (-1.0 / 6.0 * px0 + (1.0 / 6.0) * px3) +
                                      x37 * (-1.0 / 6.0 * py0 + (1.0 / 6.0) * py3) +
                                      x38 * (-1.0 / 6.0 * pz0 + (1.0 / 6.0) * pz3));
    const real_t x40 = x39 * (x22 * x24 - x26 * x28);
    const real_t x41 = x20 * x40;
    const real_t x42 = (1.0 / 12.0) * px3;
    const real_t x43 = py3 * x1;
    const real_t x44 = py1 * x13;
    const real_t x45 = -x43 + x44;
    const real_t x46 = py0 * x6;
    const real_t x47 = py2 * x11;
    const real_t x48 = -x46 + x47;
    const real_t x49 = py3 * x11;
    const real_t x50 = py0 * x13;
    const real_t x51 = -x49 + x50;
    const real_t x52 = py1 * x6;
    const real_t x53 = py2 * x1 - x52;
    const real_t x54 = x51 + x53;
    const real_t x55 = -py2 * x42 + py3 * x0 + x45 + x48 + x54;
    const real_t x56 = -py3;
    const real_t x57 = -py1 - x56;
    const real_t x58 = -py1 - x31;
    const real_t x59 = x39 * (-x22 * x57 + x26 * x58);
    const real_t x60 = x55 * x59;
    const real_t x61 = (1.0 / 12.0) * pz2;
    const real_t x62 = (1.0 / 12.0) * pz3;
    const real_t x63 = py1 * pz3;
    const real_t x64 = (1.0 / 24.0) * x63;
    const real_t x65 = py3 * x3;
    const real_t x66 = -x64 + x65;
    const real_t x67 = (1.0 / 24.0) * pz0;
    const real_t x68 = py2 * x67;
    const real_t x69 = py0 * x8;
    const real_t x70 = -x68 + x69;
    const real_t x71 = (1.0 / 24.0) * pz3;
    const real_t x72 = py0 * x71;
    const real_t x73 = py3 * x67;
    const real_t x74 = -x72 + x73;
    const real_t x75 = py2 * x3;
    const real_t x76 = py1 * x8 - x75;
    const real_t x77 = x74 + x76;
    const real_t x78 = py2 * x62 - py3 * x61 + x66 + x70 + x77;
    const real_t x79 = x39 * (-x24 * x58 + x28 * x57);
    const real_t x80 = x78 * x79;
    const real_t x81 = (1.0 / 12.0) * px1;
    const real_t x82 = pz3 * x6;
    const real_t x83 = px3 * x8;
    const real_t x84 = px0 * x3 - pz0 * x1;
    const real_t x85 = -x82 + x83 + x84;
    const real_t x86 = px2 * x3 - pz1 * x42 + pz3 * x81 + x15 - x17 + x85;
    const real_t x87 = py3 * x6;
    const real_t x88 = py2 * x13;
    const real_t x89 = -py0 * x1 + py1 * x11;
    const real_t x90 = -x87 + x88 + x89;
    const real_t x91 =
        (1.0 / 24.0) * px1 * py2 + (1.0 / 12.0) * px3 * py1 - py3 * x81 - x51 - x52 - x90;
    const real_t x92 = py2 * x71;
    const real_t x93 = py3 * x8;
    const real_t x94 = py0 * x3 - py1 * x67;
    const real_t x95 = -x92 + x93 + x94;
    const real_t x96 =
        (1.0 / 24.0) * py1 * pz2 + (1.0 / 12.0) * py3 * pz1 - 1.0 / 12.0 * x63 - x74 - x75 - x95;
    const real_t x97 = x40 * x86 + x59 * x91 + x79 * x96;
    const real_t x98 = x7 - x9;
    const real_t x99 = x82 - x83 + x84;
    const real_t x100 = (1.0 / 12.0) * px2 * pz1 - 1.0 / 12.0 * x16 - x5 - x98 - x99;
    const real_t x101 = x46 - x47;
    const real_t x102 = x87 - x88 + x89;
    const real_t x103 = -py1 * x0 + py2 * x81 + x101 + x102 + x45;
    const real_t x104 = (1.0 / 12.0) * pz1;
    const real_t x105 = x68 - x69;
    const real_t x106 = x92 - x93 + x94;
    const real_t x107 = py1 * x61 - py2 * x104 + x105 + x106 + x66;
    const real_t x108 = x100 * x40 + x103 * x59 + x107 * x79;
    const real_t x109 = -py0 - x56;
    const real_t x110 = -px0 - x25;
    const real_t x111 = x39 * (x109 * x36 - x110 * x32);
    const real_t x112 = x111 * x55;
    const real_t x113 = -pz0 - x23;
    const real_t x114 = x39 * (x110 * x30 - x113 * x36);
    const real_t x115 = x114 * x20;
    const real_t x116 = x39 * (-x109 * x30 + x113 * x32);
    const real_t x117 = x116 * x78;
    const real_t x118 = x111 * x91 + x114 * x86 + x116 * x96;
    const real_t x119 = x100 * x114 + x103 * x111 + x107 * x116;
    const real_t x120 = x39 * (-x110 * x33 + x113 * x35);
    const real_t x121 = x120 * x20;
    const real_t x122 = x39 * (-x109 * x35 + x110 * x29);
    const real_t x123 = x122 * x55;
    const real_t x124 = x39 * (x109 * x33 - x113 * x29);
    const real_t x125 = x124 * x78;
    const real_t x126 = x120 * x86 + x122 * x91 + x124 * x96;
    const real_t x127 = x100 * x120 + x103 * x122 + x107 * x124;
    const real_t x128 = x38 * x39;
    const real_t x129 = x128 * x55;
    const real_t x130 = x37 * x39;
    const real_t x131 = x130 * x20;
    const real_t x132 = x34 * x39;
    const real_t x133 = x132 * x78;
    const real_t x134 = x128 * x91 + x130 * x86 + x132 * x96;
    const real_t x135 = x100 * x130 + x103 * x128 + x107 * x132;
    const real_t x136 = x2 - x4;
    const real_t x137 = -px0 * x62 + pz0 * x42 + x10 + x136 + x99;
    const real_t x138 = x43 - x44;
    const real_t x139 = (1.0 / 12.0) * px0 * py3 - py0 * x42 - x102 - x138 - x48;
    const real_t x140 = (1.0 / 12.0) * pz0;
    const real_t x141 = x64 - x65;
    const real_t x142 = (1.0 / 12.0) * py0 * pz3 - py3 * x140 - x106 - x141 - x70;
    const real_t x143 = x137 * x40 + x139 * x59 + x142 * x79;
    const real_t x144 = (1.0 / 12.0) * px0 * pz2 - pz0 * x0 - x12 + x14 - x18 - x85;
    const real_t x145 = (1.0 / 12.0) * px0;
    const real_t x146 = py0 * x0 - py2 * x145 + x49 - x50 + x53 + x90;
    const real_t x147 = -py0 * x61 + py2 * x140 + x72 - x73 + x76 + x95;
    const real_t x148 = x144 * x40 + x146 * x59 + x147 * x79;
    const real_t x149 = x111 * x139 + x114 * x137 + x116 * x142;
    const real_t x150 = x111 * x146 + x114 * x144 + x116 * x147;
    const real_t x151 = x120 * x137 + x122 * x139 + x124 * x142;
    const real_t x152 = x120 * x144 + x122 * x146 + x124 * x147;
    const real_t x153 = x128 * x139 + x130 * x137 + x132 * x142;
    const real_t x154 = x128 * x146 + x130 * x144 + x132 * x147;
    const real_t x155 = -px0 * x104 + (1.0 / 12.0) * px1 * pz0 - x136 - x19 - x98;
    const real_t x156 = x155 * x40;
    const real_t x157 = -py0 * x81 + py1 * x145 + x101 + x138 + x54;
    const real_t x158 = x157 * x59;
    const real_t x159 = py0 * x104 - py1 * x140 + x105 + x141 + x77;
    const real_t x160 = x159 * x79;
    const real_t x161 = x111 * x157;
    const real_t x162 = x114 * x155;
    const real_t x163 = x116 * x159;
    const real_t x164 = x120 * x155;
    const real_t x165 = x122 * x157;
    const real_t x166 = x124 * x159;
    const real_t x167 = x128 * x157;
    const real_t x168 = x130 * x155;
    const real_t x169 = x132 * x159;
    element_matrix[0] = -x108 - x41 - x60 - x80 - x97;
    element_matrix[1] = -x112 - x115 - x117 - x118 - x119;
    element_matrix[2] = -x121 - x123 - x125 - x126 - x127;
    element_matrix[3] = -x129 - x131 - x133 - x134 - x135;
    element_matrix[4] = -x143 - x148 + x41 + x60 + x80;
    element_matrix[5] = x112 + x115 + x117 - x149 - x150;
    element_matrix[6] = x121 + x123 + x125 - x151 - x152;
    element_matrix[7] = x129 + x131 + x133 - x153 - x154;
    element_matrix[8] = x143 - x156 - x158 - x160 + x97;
    element_matrix[9] = x118 + x149 - x161 - x162 - x163;
    element_matrix[10] = x126 + x151 - x164 - x165 - x166;
    element_matrix[11] = x134 + x153 - x167 - x168 - x169;
    element_matrix[12] = x108 + x148 + x156 + x158 + x160;
    element_matrix[13] = x119 + x150 + x161 + x162 + x163;
    element_matrix[14] = x127 + x152 + x164 + x165 + x166;
    element_matrix[15] = x135 + x154 + x167 + x168 + x169;
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

void cvfem_tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait

        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_matrix[4 * 4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            cvfem_tet4_laplacian_hessian_kernel(
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
                element_matrix);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols4(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 4];

#pragma unroll(4)
                for (int edof_j = 0; edof_j < 4; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("cvfem_tet4_laplacian.c: cvfem_tet4_laplacian_assemble_hessian\t%g seconds\n",
           tock - tick);
}
