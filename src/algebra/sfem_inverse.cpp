// Auto-generated
#include <assert.h>
#include <stddef.h>
#include "sfem_base.hpp"
#include "sfem_vec.hpp"
#include "sfem_inverse.hpp"

namespace sfem {

void invert_node_blocks_1(const ptrdiff_t nnodes,
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

void invert_node_blocks_2(const ptrdiff_t nnodes,
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

void invert_node_blocks_3(const ptrdiff_t nnodes,
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

void invert_node_blocks_4(const ptrdiff_t nnodes,
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

}  // namespace sfem
