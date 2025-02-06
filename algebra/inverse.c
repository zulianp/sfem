// Auto-generated
#include <assert.h>
#include <stddef.h>
#include "sfem_base.h"
#include "sfem_vec.h"

static SFEM_INLINE void inverse1(
    // Input
    const real_t mat_0,
    // Output
    real_t *const SFEM_RESTRICT mat_inv_0) {
    *mat_inv_0 = 1.0 / mat_0;
}

void dinvert1(const ptrdiff_t nnodes,
              const count_t *const SFEM_RESTRICT rowptr,
              const idx_t *const SFEM_RESTRICT colidx,
              real_t **const SFEM_RESTRICT values,
              real_t **const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const idx_t *cols = &colidx[r_begin];

        count_t diag_idx = SFEM_COUNT_INVALID;
        for (count_t k = 0; k < r_extent; k++) {
            if (cols[k] == i) {
                diag_idx = r_begin + k;
                break;
            }

        }  // end for

        assert(diag_idx != SFEM_COUNT_INVALID);

        inverse1(values[0][diag_idx], &inv_diag[0][i]);
    }
}

static SFEM_INLINE void inverse2(
    // Input
    const real_t mat_0,
    const real_t mat_1,
    const real_t mat_2,
    const real_t mat_3,
    // Output
    real_t *const SFEM_RESTRICT mat_inv_0,
    real_t *const SFEM_RESTRICT mat_inv_1,
    real_t *const SFEM_RESTRICT mat_inv_2,
    real_t *const SFEM_RESTRICT mat_inv_3) {
    const real_t x0 = 1.0 / (mat_0 * mat_3 - mat_1 * mat_2);
    *mat_inv_0 = mat_3 * x0;
    *mat_inv_1 = -mat_1 * x0;
    *mat_inv_2 = -mat_2 * x0;
    *mat_inv_3 = mat_0 * x0;
}

void dinvert2(const ptrdiff_t nnodes,
              const count_t *const SFEM_RESTRICT rowptr,
              const idx_t *const SFEM_RESTRICT colidx,
              real_t **const SFEM_RESTRICT values,
              real_t **const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const idx_t *cols = &colidx[r_begin];

        count_t diag_idx = SFEM_COUNT_INVALID;
        for (count_t k = 0; k < r_extent; k++) {
            if (cols[k] == i) {
                diag_idx = r_begin + k;
                break;
            }

        }  // end for

        assert(diag_idx != SFEM_COUNT_INVALID);

        inverse2(values[0][diag_idx],
                 values[1][diag_idx],
                 values[2][diag_idx],
                 values[3][diag_idx],
                 &inv_diag[0][i],
                 &inv_diag[1][i],
                 &inv_diag[2][i],
                 &inv_diag[3][i]);

        assert(inv_diag[0][i] == inv_diag[0][i]);
        assert(inv_diag[1][i] == inv_diag[1][i]);
        assert(inv_diag[2][i] == inv_diag[2][i]);
        assert(inv_diag[3][i] == inv_diag[3][i]);

    }
}

static SFEM_INLINE void inverse3(
    // Input
    const real_t mat_0,
    const real_t mat_1,
    const real_t mat_2,
    const real_t mat_3,
    const real_t mat_4,
    const real_t mat_5,
    const real_t mat_6,
    const real_t mat_7,
    const real_t mat_8,
    // Output
    real_t *const SFEM_RESTRICT mat_inv_0,
    real_t *const SFEM_RESTRICT mat_inv_1,
    real_t *const SFEM_RESTRICT mat_inv_2,
    real_t *const SFEM_RESTRICT mat_inv_3,
    real_t *const SFEM_RESTRICT mat_inv_4,
    real_t *const SFEM_RESTRICT mat_inv_5,
    real_t *const SFEM_RESTRICT mat_inv_6,
    real_t *const SFEM_RESTRICT mat_inv_7,
    real_t *const SFEM_RESTRICT mat_inv_8) {
    const real_t x0 = mat_4 * mat_8;
    const real_t x1 = mat_5 * mat_7;
    const real_t x2 = mat_1 * mat_5;
    const real_t x3 = mat_1 * mat_8;
    const real_t x4 = mat_2 * mat_4;
    const real_t x5 = 1.0 / (mat_0 * x0 - mat_0 * x1 + mat_2 * mat_3 * mat_7 - mat_3 * x3 +
                             mat_6 * x2 - mat_6 * x4);
    *mat_inv_0 = x5 * (x0 - x1);
    *mat_inv_1 = x5 * (mat_2 * mat_7 - x3);
    *mat_inv_2 = x5 * (x2 - x4);
    *mat_inv_3 = x5 * (-mat_3 * mat_8 + mat_5 * mat_6);
    *mat_inv_4 = x5 * (mat_0 * mat_8 - mat_2 * mat_6);
    *mat_inv_5 = x5 * (-mat_0 * mat_5 + mat_2 * mat_3);
    *mat_inv_6 = x5 * (mat_3 * mat_7 - mat_4 * mat_6);
    *mat_inv_7 = x5 * (-mat_0 * mat_7 + mat_1 * mat_6);
    *mat_inv_8 = x5 * (mat_0 * mat_4 - mat_1 * mat_3);
}

void dinvert3(const ptrdiff_t nnodes,
              const count_t *const SFEM_RESTRICT rowptr,
              const idx_t *const SFEM_RESTRICT colidx,
              real_t **const SFEM_RESTRICT values,
              real_t **const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const idx_t *cols = &colidx[r_begin];

        count_t diag_idx = SFEM_COUNT_INVALID;
        for (count_t k = 0; k < r_extent; k++) {
            if (cols[k] == i) {
                diag_idx = r_begin + k;
                break;
            }

        }  // end for

        assert(diag_idx != SFEM_COUNT_INVALID);

        inverse3(values[0][diag_idx],
                 values[1][diag_idx],
                 values[2][diag_idx],
                 values[3][diag_idx],
                 values[4][diag_idx],
                 values[5][diag_idx],
                 values[6][diag_idx],
                 values[7][diag_idx],
                 values[8][diag_idx],
                 &inv_diag[0][i],
                 &inv_diag[1][i],
                 &inv_diag[2][i],
                 &inv_diag[3][i],
                 &inv_diag[4][i],
                 &inv_diag[5][i],
                 &inv_diag[6][i],
                 &inv_diag[7][i],
                 &inv_diag[8][i]);
    }
}

static SFEM_INLINE void inverse4(
    // Input
    const real_t mat_0,
    const real_t mat_1,
    const real_t mat_2,
    const real_t mat_3,
    const real_t mat_4,
    const real_t mat_5,
    const real_t mat_6,
    const real_t mat_7,
    const real_t mat_8,
    const real_t mat_9,
    const real_t mat_10,
    const real_t mat_11,
    const real_t mat_12,
    const real_t mat_13,
    const real_t mat_14,
    const real_t mat_15,
    // Output
    real_t *const SFEM_RESTRICT mat_inv_0,
    real_t *const SFEM_RESTRICT mat_inv_1,
    real_t *const SFEM_RESTRICT mat_inv_2,
    real_t *const SFEM_RESTRICT mat_inv_3,
    real_t *const SFEM_RESTRICT mat_inv_4,
    real_t *const SFEM_RESTRICT mat_inv_5,
    real_t *const SFEM_RESTRICT mat_inv_6,
    real_t *const SFEM_RESTRICT mat_inv_7,
    real_t *const SFEM_RESTRICT mat_inv_8,
    real_t *const SFEM_RESTRICT mat_inv_9,
    real_t *const SFEM_RESTRICT mat_inv_10,
    real_t *const SFEM_RESTRICT mat_inv_11,
    real_t *const SFEM_RESTRICT mat_inv_12,
    real_t *const SFEM_RESTRICT mat_inv_13,
    real_t *const SFEM_RESTRICT mat_inv_14,
    real_t *const SFEM_RESTRICT mat_inv_15) {
    const real_t x0 = mat_10 * mat_13;
    const real_t x1 = mat_7 * x0;
    const real_t x2 = mat_11 * mat_14;
    const real_t x3 = mat_5 * x2;
    const real_t x4 = mat_15 * mat_9;
    const real_t x5 = mat_6 * x4;
    const real_t x6 = mat_10 * mat_15;
    const real_t x7 = mat_1 * x6;
    const real_t x8 = mat_1 * mat_6;
    const real_t x9 = mat_11 * x8;
    const real_t x10 = mat_1 * mat_7;
    const real_t x11 = mat_14 * x10;
    const real_t x12 = mat_3 * mat_5;
    const real_t x13 = mat_10 * x12;
    const real_t x14 = mat_11 * mat_13;
    const real_t x15 = mat_2 * x14;
    const real_t x16 = mat_2 * mat_7;
    const real_t x17 = mat_9 * x16;
    const real_t x18 = mat_3 * mat_6;
    const real_t x19 = mat_13 * x18;
    const real_t x20 = mat_14 * mat_9;
    const real_t x21 = mat_3 * x20;
    const real_t x22 = mat_2 * mat_5;
    const real_t x23 = mat_15 * x22;
    const real_t x24 =
        1.0 / (mat_0 * mat_10 * mat_15 * mat_5 + mat_0 * mat_11 * mat_13 * mat_6 +
               mat_0 * mat_14 * mat_7 * mat_9 - mat_0 * x1 - mat_0 * x3 - mat_0 * x5 +
               mat_1 * mat_10 * mat_12 * mat_7 + mat_1 * mat_11 * mat_14 * mat_4 +
               mat_1 * mat_15 * mat_6 * mat_8 + mat_10 * mat_13 * mat_3 * mat_4 +
               mat_11 * mat_12 * mat_2 * mat_5 + mat_12 * mat_3 * mat_6 * mat_9 - mat_12 * x13 -
               mat_12 * x17 - mat_12 * x9 + mat_13 * mat_2 * mat_7 * mat_8 +
               mat_14 * mat_3 * mat_5 * mat_8 + mat_15 * mat_2 * mat_4 * mat_9 - mat_4 * x15 -
               mat_4 * x21 - mat_4 * x7 - mat_8 * x11 - mat_8 * x19 - mat_8 * x23);
    const real_t x25 = mat_10 * mat_12;
    const real_t x26 = mat_15 * mat_8;
    const real_t x27 = mat_11 * mat_12;
    const real_t x28 = mat_14 * mat_8;
    const real_t x29 = mat_0 * mat_7;
    const real_t x30 = mat_2 * mat_4;
    const real_t x31 = mat_0 * mat_6;
    const real_t x32 = mat_3 * mat_4;
    const real_t x33 = mat_13 * mat_8;
    const real_t x34 = mat_12 * mat_9;
    const real_t x35 = mat_1 * mat_4;
    const real_t x36 = mat_0 * mat_5;
    *mat_inv_0 = x24 * (mat_10 * mat_15 * mat_5 + mat_11 * mat_13 * mat_6 + mat_14 * mat_7 * mat_9 -
                        x1 - x3 - x5);
    *mat_inv_1 = x24 * (mat_1 * mat_11 * mat_14 + mat_10 * mat_13 * mat_3 + mat_15 * mat_2 * mat_9 -
                        x15 - x21 - x7);
    *mat_inv_2 = x24 * (mat_1 * mat_15 * mat_6 + mat_13 * mat_2 * mat_7 + mat_14 * mat_3 * mat_5 -
                        x11 - x19 - x23);
    *mat_inv_3 = x24 * (mat_10 * x10 + mat_11 * x22 + mat_9 * x18 - x13 - x17 - x9);
    *mat_inv_4 =
        x24 * (mat_4 * x2 - mat_4 * x6 + mat_6 * x26 - mat_6 * x27 + mat_7 * x25 - mat_7 * x28);
    *mat_inv_5 =
        x24 * (-mat_0 * x2 + mat_0 * x6 - mat_2 * x26 + mat_2 * x27 - mat_3 * x25 + mat_3 * x28);
    *mat_inv_6 = x24 * (-mat_12 * x16 + mat_12 * x18 + mat_14 * x29 - mat_14 * x32 + mat_15 * x30 -
                        mat_15 * x31);
    *mat_inv_7 = x24 * (mat_0 * mat_11 * mat_6 + mat_10 * mat_3 * mat_4 - mat_10 * x29 -
                        mat_11 * x30 + mat_2 * mat_7 * mat_8 - mat_8 * x18);
    *mat_inv_8 =
        x24 * (-mat_4 * x14 + mat_4 * x4 - mat_5 * x26 + mat_5 * x27 + mat_7 * x33 - mat_7 * x34);
    *mat_inv_9 =
        x24 * (mat_0 * x14 - mat_0 * x4 + mat_1 * x26 - mat_1 * x27 - mat_3 * x33 + mat_3 * x34);
    *mat_inv_10 = x24 * (mat_0 * mat_15 * mat_5 + mat_1 * mat_12 * mat_7 - mat_12 * x12 +
                         mat_13 * mat_3 * mat_4 - mat_13 * x29 - mat_15 * x35);
    *mat_inv_11 = x24 * (mat_0 * mat_7 * mat_9 + mat_1 * mat_11 * mat_4 - mat_11 * x36 +
                         mat_3 * mat_5 * mat_8 - mat_8 * x10 - mat_9 * x32);
    *mat_inv_12 = x24 * (mat_10 * mat_13 * mat_4 + mat_12 * mat_6 * mat_9 + mat_14 * mat_5 * mat_8 -
                         mat_4 * x20 - mat_5 * x25 - mat_6 * x33);
    *mat_inv_13 = x24 * (mat_0 * mat_14 * mat_9 - mat_0 * x0 + mat_1 * mat_10 * mat_12 -
                         mat_1 * x28 + mat_13 * mat_2 * mat_8 - mat_2 * x34);
    *mat_inv_14 = x24 * (mat_12 * x22 - mat_12 * x8 - mat_13 * x30 + mat_13 * x31 + mat_14 * x35 -
                         mat_14 * x36);
    *mat_inv_15 =
        x24 * (-mat_10 * x35 + mat_10 * x36 - mat_8 * x22 + mat_8 * x8 + mat_9 * x30 - mat_9 * x31);
}

void dinvert4(const ptrdiff_t nnodes,
              const count_t *const SFEM_RESTRICT rowptr,
              const idx_t *const SFEM_RESTRICT colidx,
              real_t **const SFEM_RESTRICT values,
              real_t **const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const idx_t *cols = &colidx[r_begin];

        count_t diag_idx = SFEM_COUNT_INVALID;
        for (count_t k = 0; k < r_extent; k++) {
            if (cols[k] == i) {
                diag_idx = r_begin + k;
                break;
            }

        }  // end for

        assert(diag_idx != SFEM_COUNT_INVALID);

        inverse4(values[0][diag_idx],
                 values[1][diag_idx],
                 values[2][diag_idx],
                 values[3][diag_idx],
                 values[4][diag_idx],
                 values[5][diag_idx],
                 values[6][diag_idx],
                 values[7][diag_idx],
                 values[8][diag_idx],
                 values[9][diag_idx],
                 values[10][diag_idx],
                 values[11][diag_idx],
                 values[12][diag_idx],
                 values[13][diag_idx],
                 values[14][diag_idx],
                 values[15][diag_idx],
                 &inv_diag[0][i],
                 &inv_diag[1][i],
                 &inv_diag[2][i],
                 &inv_diag[3][i],
                 &inv_diag[4][i],
                 &inv_diag[5][i],
                 &inv_diag[6][i],
                 &inv_diag[7][i],
                 &inv_diag[8][i],
                 &inv_diag[9][i],
                 &inv_diag[10][i],
                 &inv_diag[11][i],
                 &inv_diag[12][i],
                 &inv_diag[13][i],
                 &inv_diag[14][i],
                 &inv_diag[15][i]);
    }
}
