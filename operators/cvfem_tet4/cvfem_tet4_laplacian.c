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
    const real_t x0 = (1.0/12.0)*py0;
    const real_t x1 = (1.0/24.0)*px3*py1;
    const real_t x2 = px1*py3;
    const real_t x3 = (1.0/24.0)*x2;
    const real_t x4 = -x1 + x3;
    const real_t x5 = (1.0/24.0)*py2;
    const real_t x6 = px3*x5;
    const real_t x7 = (1.0/24.0)*px2;
    const real_t x8 = py3*x7;
    const real_t x9 = (1.0/24.0)*py0;
    const real_t x10 = (1.0/24.0)*px0;
    const real_t x11 = -px1*x9 + py1*x10;
    const real_t x12 = x11 - x6 + x8;
    const real_t x13 = px2*x9;
    const real_t x14 = px0*py2;
    const real_t x15 = (1.0/24.0)*x14;
    const real_t x16 = -x13 + x15;
    const real_t x17 = (1.0/12.0)*px0*py3 - px3*x0 - x12 - x16 - x4;
    const real_t x18 = pz0 - pz3;
    const real_t x19 = px0 - px1;
    const real_t x20 = py0 - py2;
    const real_t x21 = x19*x20;
    const real_t x22 = pz0 - pz1;
    const real_t x23 = px0 - px2;
    const real_t x24 = py0 - py3;
    const real_t x25 = x23*x24;
    const real_t x26 = pz0 - pz2;
    const real_t x27 = px0 - px3;
    const real_t x28 = py0 - py1;
    const real_t x29 = x27*x28;
    const real_t x30 = x19*x24;
    const real_t x31 = x23*x28;
    const real_t x32 = x20*x27;
    const real_t x33 = 1.0/(x18*x21 - x18*x31 + x22*x25 - x22*x32 + x26*x29 - x26*x30);
    const real_t x34 = x25 - x32;
    const real_t x35 = x21 - x31;
    const real_t x36 = x33*(x29 - x30 + x34 + x35);
    const real_t x37 = py1*x7;
    const real_t x38 = x11 + x6 - x8;
    const real_t x39 = py3*x10;
    const real_t x40 = px3*x9;
    const real_t x41 = -x39 + x40;
    const real_t x42 = (1.0/24.0)*px1*py2 + (1.0/12.0)*px3*py1 - 1.0/12.0*x2 - x37 - x38 - x41;
    const real_t x43 = px1*x5 - x37;
    const real_t x44 = x39 - x40 + x43;
    const real_t x45 = px2*x0 - 1.0/12.0*x14 + x38 + x44;
    const real_t x46 = (1.0/12.0)*py1;
    const real_t x47 = (1.0/12.0)*py2;
    const real_t x48 = x13 - x15;
    const real_t x49 = x1 - x3;
    const real_t x50 = px1*x47 - px2*x46 + x12 + x48 + x49;
    const real_t x51 = px0*x46 - px1*x0 + x4 + x41 + x43 + x48;
    const real_t x52 = (1.0/12.0)*py3;
    const real_t x53 = px2*x52 - px3*x47 + x16 + x44 + x49;
    const real_t x54 = (1.0/12.0)*px0;
    const real_t x55 = (1.0/24.0)*px3;
    const real_t x56 = pz1*x55;
    const real_t x57 = (1.0/24.0)*px1;
    const real_t x58 = pz3*x57;
    const real_t x59 = -x56 + x58;
    const real_t x60 = pz2*x10;
    const real_t x61 = pz0*x7;
    const real_t x62 = -x60 + x61;
    const real_t x63 = pz2*x57;
    const real_t x64 = -1.0/24.0*px2*pz1 + x63;
    const real_t x65 = pz3*x10;
    const real_t x66 = pz0*x55;
    const real_t x67 = -x65 + x66;
    const real_t x68 = (1.0/12.0)*px1*pz0 - pz1*x54 - x59 - x62 - x64 - x67;
    const real_t x69 = x22*x27;
    const real_t x70 = x18*x19;
    const real_t x71 = x18*x23 - x26*x27;
    const real_t x72 = x19*x26 - x22*x23;
    const real_t x73 = x33*(-x69 + x70 - x71 - x72);
    const real_t x74 = pz2*x55;
    const real_t x75 = pz3*x7;
    const real_t x76 = -pz0*x57 + pz1*x10;
    const real_t x77 = -x74 + x75 + x76;
    const real_t x78 = x56 - x58;
    const real_t x79 = -1.0/12.0*px1*pz2 + (1.0/12.0)*px2*pz1 - x62 - x77 - x78;
    const real_t x80 = (1.0/12.0)*pz0;
    const real_t x81 = x64 + x65 - x66;
    const real_t x82 = x74 - x75 + x76;
    const real_t x83 = (1.0/12.0)*px0*pz2 - px2*x80 - x81 - x82;
    const real_t x84 = x60 - x61;
    const real_t x85 = px3*x80 - pz3*x54 + x59 + x77 + x84;
    const real_t x86 = (1.0/12.0)*px1*pz3 - 1.0/12.0*px3*pz1 + pz1*x7 - x63 + x67 + x82;
    const real_t x87 = -1.0/12.0*px2*pz3 + (1.0/12.0)*px3*pz2 - x78 - x81 - x84;
    const real_t x88 = (1.0/24.0)*py3*pz1;
    const real_t x89 = (1.0/24.0)*py1*pz3;
    const real_t x90 = -x88 + x89;
    const real_t x91 = (1.0/24.0)*pz2;
    const real_t x92 = py3*x91;
    const real_t x93 = pz3*x5;
    const real_t x94 = (1.0/24.0)*pz0;
    const real_t x95 = -py1*x94 + pz1*x9;
    const real_t x96 = -x92 + x93 + x95;
    const real_t x97 = pz0*x5;
    const real_t x98 = pz2*x9;
    const real_t x99 = -x97 + x98;
    const real_t x100 = (1.0/12.0)*py0*pz3 - pz0*x52 - x90 - x96 - x99;
    const real_t x101 = x22*x24;
    const real_t x102 = x18*x28;
    const real_t x103 = x18*x20 - x24*x26;
    const real_t x104 = -x20*x22 + x26*x28;
    const real_t x105 = x33*(x101 - x102 + x103 + x104);
    const real_t x106 = pz1*x5;
    const real_t x107 = x92 - x93 + x95;
    const real_t x108 = pz3*x9;
    const real_t x109 = py3*x94;
    const real_t x110 = -x108 + x109;
    const real_t x111 = (1.0/24.0)*py1*pz2 + (1.0/12.0)*py3*pz1 - pz3*x46 - x106 - x107 - x110;
    const real_t x112 = py1*x91 - x106;
    const real_t x113 = x108 - x109 + x112;
    const real_t x114 = pz0*x47 - pz2*x0 + x107 + x113;
    const real_t x115 = x97 - x98;
    const real_t x116 = x88 - x89;
    const real_t x117 = -pz1*x47 + pz2*x46 + x115 + x116 + x96;
    const real_t x118 = -pz0*x46 + pz1*x0 + x110 + x112 + x115 + x90;
    const real_t x119 = -pz2*x52 + pz3*x47 + x113 + x116 + x99;
    const real_t x120 = x100*x105 + x105*x111 + x105*x114 + x105*x117 + x105*x118 + x105*x119 + x17*x36 + x36*x42 + x36*x45 + x36*x50 + x36*x51 + x36*x53 + x68*x73 + x73*x79 + x73*x83 + x73*x85 + x73*x86 + 
    x73*x87;
    const real_t x121 = -x33*x34;
    const real_t x122 = x33*x71;
    const real_t x123 = -x103*x33;
    const real_t x124 = x100*x123 + x111*x123 + x114*x123 + x117*x123 + x118*x123 + x119*x123 + x121*x17 + x121*x42 + x121*x45 + x121*x50 + x121*x51 + x121*x53 + x122*x68 + x122*x79 + x122*x83 + x122*x85 + 
    x122*x86 + x122*x87;
    const real_t x125 = x33*(-x29 + x30);
    const real_t x126 = x33*(x69 - x70);
    const real_t x127 = x33*(-x101 + x102);
    const real_t x128 = x100*x127 + x111*x127 + x114*x127 + x117*x127 + x118*x127 + x119*x127 + x125*x17 + x125*x42 + x125*x45 + x125*x50 + x125*x51 + x125*x53 + x126*x68 + x126*x79 + x126*x83 + x126*x85 + 
    x126*x86 + x126*x87;
    const real_t x129 = -x33*x35;
    const real_t x130 = x33*x72;
    const real_t x131 = -x104*x33;
    const real_t x132 = x100*x131 + x111*x131 + x114*x131 + x117*x131 + x118*x131 + x119*x131 + x129*x17 + x129*x42 + x129*x45 + x129*x50 + x129*x51 + x129*x53 + x130*x68 + x130*x79 + x130*x83 + x130*x85 + 
    x130*x86 + x130*x87;
    element_matrix[0] = x120;
    element_matrix[1] = x124;
    element_matrix[2] = x128;
    element_matrix[3] = x132;
    element_matrix[4] = x120;
    element_matrix[5] = x124;
    element_matrix[6] = x128;
    element_matrix[7] = x132;
    element_matrix[8] = x120;
    element_matrix[9] = x124;
    element_matrix[10] = x128;
    element_matrix[11] = x132;
    element_matrix[12] = x120;
    element_matrix[13] = x124;
    element_matrix[14] = x128;
    element_matrix[15] = x132;
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
#pragma omp for //nowait

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
