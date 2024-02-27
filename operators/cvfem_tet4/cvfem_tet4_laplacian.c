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
   const real_t x0 = (1.0/12.0)*py2;
   const real_t x1 = (1.0/12.0)*px2;
   const real_t x2 = px1*py3;
   const real_t x3 = (1.0/24.0)*x2;
   const real_t x4 = (1.0/24.0)*py1;
   const real_t x5 = px3*x4;
   const real_t x6 = -x3 + x5;
   const real_t x7 = (1.0/24.0)*px2;
   const real_t x8 = py0*x7;
   const real_t x9 = (1.0/24.0)*px0;
   const real_t x10 = py2*x9;
   const real_t x11 = x10 - x8;
   const real_t x12 = (1.0/24.0)*px3;
   const real_t x13 = py0*x12;
   const real_t x14 = (1.0/24.0)*py3;
   const real_t x15 = px0*x14;
   const real_t x16 = px2*x4;
   const real_t x17 = (1.0/24.0)*px1;
   const real_t x18 = py2*x17 - x16;
   const real_t x19 = -x13 + x15 + x18;
   const real_t x20 = -px3*x0 + py3*x1 + x11 + x19 + x6;
   const real_t x21 = pz0 - pz3;
   const real_t x22 = px0 - px1;
   const real_t x23 = py0 - py2;
   const real_t x24 = x22*x23;
   const real_t x25 = pz0 - pz1;
   const real_t x26 = px0 - px2;
   const real_t x27 = py0 - py3;
   const real_t x28 = x26*x27;
   const real_t x29 = pz0 - pz2;
   const real_t x30 = px0 - px3;
   const real_t x31 = py0 - py1;
   const real_t x32 = x30*x31;
   const real_t x33 = x22*x27;
   const real_t x34 = x26*x31;
   const real_t x35 = x23*x30;
   const real_t x36 = 1.0/(x21*x24 - x21*x34 + x25*x28 - x25*x35 + x29*x32 - x29*x33);
   const real_t x37 = x28 - x35;
   const real_t x38 = x24 - x34;
   const real_t x39 = x36*(x32 - x33 + x37 + x38);
   const real_t x40 = x20*x39;
   const real_t x41 = pz3*x17;
   const real_t x42 = pz1*x12;
   const real_t x43 = -x41 + x42;
   const real_t x44 = pz0*x7;
   const real_t x45 = pz2*x9;
   const real_t x46 = -x44 + x45;
   const real_t x47 = pz0*x12;
   const real_t x48 = pz3*x9;
   const real_t x49 = pz2*x17;
   const real_t x50 = -1.0/24.0*px2*pz1 + x49;
   const real_t x51 = -x47 + x48 + x50;
   const real_t x52 = (1.0/12.0)*px3*pz2 - pz3*x1 - x43 - x46 - x51;
   const real_t x53 = x25*x30;
   const real_t x54 = x21*x22;
   const real_t x55 = x21*x26 - x29*x30;
   const real_t x56 = x22*x29 - x25*x26;
   const real_t x57 = x36*(-x53 + x54 - x55 - x56);
   const real_t x58 = x52*x57;
   const real_t x59 = (1.0/12.0)*pz2;
   const real_t x60 = pz3*x4;
   const real_t x61 = pz1*x14;
   const real_t x62 = -x60 + x61;
   const real_t x63 = (1.0/24.0)*py2*pz0;
   const real_t x64 = (1.0/24.0)*py0*pz2;
   const real_t x65 = -x63 + x64;
   const real_t x66 = pz0*x14;
   const real_t x67 = (1.0/24.0)*pz3;
   const real_t x68 = py0*x67;
   const real_t x69 = (1.0/24.0)*pz1;
   const real_t x70 = py2*x69;
   const real_t x71 = pz2*x4 - x70;
   const real_t x72 = -x66 + x68 + x71;
   const real_t x73 = -py3*x59 + pz3*x0 + x62 + x65 + x72;
   const real_t x74 = x25*x27;
   const real_t x75 = x21*x31;
   const real_t x76 = x21*x23 - x27*x29;
   const real_t x77 = -x23*x25 + x29*x31;
   const real_t x78 = x36*(x74 - x75 + x76 + x77);
   const real_t x79 = x73*x78;
   const real_t x80 = x13 - x15;
   const real_t x81 = px2*x14;
   const real_t x82 = py2*x12;
   const real_t x83 = px0*x4 - py0*x17;
   const real_t x84 = -x81 + x82 + x83;
   const real_t x85 = (1.0/24.0)*px1*py2 + (1.0/12.0)*px3*py1 - x16 - 1.0/12.0*x2 - x80 - x84;
   const real_t x86 = (1.0/12.0)*pz1;
   const real_t x87 = (1.0/12.0)*pz3;
   const real_t x88 = x47 - x48;
   const real_t x89 = pz3*x7;
   const real_t x90 = pz2*x12;
   const real_t x91 = -pz0*x17 + pz1*x9;
   const real_t x92 = -x89 + x90 + x91;
   const real_t x93 = px1*x87 - px3*x86 + pz1*x7 - x49 + x88 + x92;
   const real_t x94 = (1.0/12.0)*py1;
   const real_t x95 = x66 - x68;
   const real_t x96 = py2*x67;
   const real_t x97 = pz2*x14;
   const real_t x98 = py0*x69 - pz0*x4;
   const real_t x99 = -x96 + x97 + x98;
   const real_t x100 = (1.0/24.0)*py1*pz2 + (1.0/12.0)*py3*pz1 - pz3*x94 - x70 - x95 - x99;
   const real_t x101 = x100*x78 + x39*x85 + x57*x93;
   const real_t x102 = -x10 + x8;
   const real_t x103 = x81 - x82 + x83;
   const real_t x104 = px1*x0 - px2*x94 + x102 + x103 + x6;
   const real_t x105 = x44 - x45;
   const real_t x106 = x89 - x90 + x91;
   const real_t x107 = -px1*x59 + (1.0/12.0)*px2*pz1 - x105 - x106 - x43;
   const real_t x108 = x63 - x64;
   const real_t x109 = x96 - x97 + x98;
   const real_t x110 = -pz1*x0 + pz2*x94 + x108 + x109 + x62;
   const real_t x111 = x104*x39 + x107*x57 + x110*x78;
   const real_t x112 = x36*x55;
   const real_t x113 = x112*x52;
   const real_t x114 = -x36*x37;
   const real_t x115 = x114*x20;
   const real_t x116 = -x36*x76;
   const real_t x117 = x116*x73;
   const real_t x118 = x100*x116 + x112*x93 + x114*x85;
   const real_t x119 = x104*x114 + x107*x112 + x110*x116;
   const real_t x120 = x36*(-x32 + x33);
   const real_t x121 = x120*x20;
   const real_t x122 = x36*(x53 - x54);
   const real_t x123 = x122*x52;
   const real_t x124 = x36*(-x74 + x75);
   const real_t x125 = x124*x73;
   const real_t x126 = x100*x124 + x120*x85 + x122*x93;
   const real_t x127 = x104*x120 + x107*x122 + x110*x124;
   const real_t x128 = x36*x56;
   const real_t x129 = x128*x52;
   const real_t x130 = -x36*x38;
   const real_t x131 = x130*x20;
   const real_t x132 = -x36*x77;
   const real_t x133 = x132*x73;
   const real_t x134 = x100*x132 + x128*x93 + x130*x85;
   const real_t x135 = x104*x130 + x107*x128 + x110*x132;
   const real_t x136 = (1.0/12.0)*px3;
   const real_t x137 = x3 - x5;
   const real_t x138 = (1.0/12.0)*px0*py3 - py0*x136 - x103 - x11 - x137;
   const real_t x139 = x41 - x42;
   const real_t x140 = -px0*x87 + pz0*x136 + x106 + x139 + x46;
   const real_t x141 = x60 - x61;
   const real_t x142 = (1.0/12.0)*py0*pz3 - 1.0/12.0*py3*pz0 - x109 - x141 - x65;
   const real_t x143 = x138*x39 + x140*x57 + x142*x78;
   const real_t x144 = -px0*x0 + py0*x1 + x19 + x84;
   const real_t x145 = (1.0/12.0)*px0*pz2 - pz0*x1 - x51 - x92;
   const real_t x146 = -py0*x59 + pz0*x0 + x72 + x99;
   const real_t x147 = x144*x39 + x145*x57 + x146*x78;
   const real_t x148 = x112*x140 + x114*x138 + x116*x142;
   const real_t x149 = x112*x145 + x114*x144 + x116*x146;
   const real_t x150 = x120*x138 + x122*x140 + x124*x142;
   const real_t x151 = x120*x144 + x122*x145 + x124*x146;
   const real_t x152 = x128*x140 + x130*x138 + x132*x142;
   const real_t x153 = x128*x145 + x130*x144 + x132*x146;
   const real_t x154 = px0*x94 - 1.0/12.0*px1*py0 + x102 + x137 + x18 + x80;
   const real_t x155 = x154*x39;
   const real_t x156 = -px0*x86 + (1.0/12.0)*px1*pz0 - x105 - x139 - x50 - x88;
   const real_t x157 = x156*x57;
   const real_t x158 = py0*x86 - pz0*x94 + x108 + x141 + x71 + x95;
   const real_t x159 = x158*x78;
   const real_t x160 = x112*x156;
   const real_t x161 = x114*x154;
   const real_t x162 = x116*x158;
   const real_t x163 = x120*x154;
   const real_t x164 = x122*x156;
   const real_t x165 = x124*x158;
   const real_t x166 = x128*x156;
   const real_t x167 = x130*x154;
   const real_t x168 = x132*x158;
   element_matrix[0] = -x101 - x111 - x40 - x58 - x79;
   element_matrix[1] = -x113 - x115 - x117 - x118 - x119;
   element_matrix[2] = -x121 - x123 - x125 - x126 - x127;
   element_matrix[3] = -x129 - x131 - x133 - x134 - x135;
   element_matrix[4] = -x143 - x147 + x40 + x58 + x79;
   element_matrix[5] = x113 + x115 + x117 - x148 - x149;
   element_matrix[6] = x121 + x123 + x125 - x150 - x151;
   element_matrix[7] = x129 + x131 + x133 - x152 - x153;
   element_matrix[8] = x101 + x143 - x155 - x157 - x159;
   element_matrix[9] = x118 + x148 - x160 - x161 - x162;
   element_matrix[10] = x126 + x150 - x163 - x164 - x165;
   element_matrix[11] = x134 + x152 - x166 - x167 - x168;
   element_matrix[12] = x111 + x147 + x155 + x157 + x159;
   element_matrix[13] = x119 + x149 + x160 + x161 + x162;
   element_matrix[14] = x127 + x151 + x163 + x164 + x165;
   element_matrix[15] = x135 + x153 + x166 + x167 + x168;
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
