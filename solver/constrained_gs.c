#include "constrained_gs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "dirichlet.h"

#include "inverse.c"
#include "sfem_base.h"

static SFEM_INLINE real_t atomic_read(const real_t *p) {
    real_t value;
#pragma omp atomic read
    value = *p;
    return value;
}

static SFEM_INLINE void atomic_write(real_t *p, const real_t value) {
#pragma omp atomic write
    *p = value;
}

static SFEM_INLINE void atomic_add(real_t *p, const real_t value) {
#pragma omp atomic update
    *p += value;
}

static void dinvert(const ptrdiff_t nnodes,
                    const count_t *const SFEM_RESTRICT rowptr,
                    const idx_t *const SFEM_RESTRICT colidx,
                    real_t *const SFEM_RESTRICT values,
                    real_t *const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const idx_t *cols = &colidx[r_begin];

        count_t diag_idx = -1;
        for (count_t k = 0; k < r_extent; k++) {
            if (cols[k] == i) {
                diag_idx = r_begin + k;
                break;
            }
        }  // end for

        assert(diag_idx != -1);
        inverse1(values[diag_idx], &inv_diag[i]);
    }
}

static void l1_dinvert(const ptrdiff_t nnodes,
                       const count_t *const SFEM_RESTRICT rowptr,
                       const idx_t *const SFEM_RESTRICT colidx,
                       real_t *const SFEM_RESTRICT values,
                       real_t *const SFEM_RESTRICT off,
                       real_t *const SFEM_RESTRICT inv_diag) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t r_begin = rowptr[i];
        const count_t r_end = rowptr[i + 1];
        const count_t r_extent = r_end - r_begin;
        const real_t *vals = &values[r_begin];

        real_t sum_abs = fabs(off[i]);
        for (count_t k = 0; k < r_extent; k++) {
            sum_abs += fabs(vals[k]);
        }  // end for

        inv_diag[i] = 1. / sum_abs;
    }
}

int constrained_gs_init(const ptrdiff_t nnodes,
                        const count_t *const SFEM_RESTRICT rowptr,
                        const idx_t *const SFEM_RESTRICT colidx,
                        real_t *const SFEM_RESTRICT values,
                        real_t *const SFEM_RESTRICT off,
                        real_t *const inv_diag) {
    // dinvert(nnodes, rowptr, colidx, values, inv_diag);
    l1_dinvert(nnodes, rowptr, colidx, values, off, inv_diag);
    return 0;
}

int constrained_gs_residual(const ptrdiff_t nnodes,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values,
                            real_t *const SFEM_RESTRICT rhs,
                            real_t *const SFEM_RESTRICT x,
                            real_t *res) {
    *res = 0;
#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t r_begin = rowptr[i];
            const count_t r_end = rowptr[i + 1];
            const count_t r_extent = r_end - r_begin;
            const idx_t *const r_colidx = &colidx[r_begin];

            real_t r = rhs[i];
            for (count_t k = 0; k < r_extent; k++) {
                const idx_t col = r_colidx[k];
                r -= values[r_begin + k] * x[col];
            }

            atomic_add(res, r * r);
        }
    }

    *res = sqrt(*res);
    return 0;
}

static int constrained_gs_forward(const ptrdiff_t nnodes,
                                  const count_t *const SFEM_RESTRICT rowptr,
                                  const idx_t *const SFEM_RESTRICT colidx,
                                  real_t *const SFEM_RESTRICT values,
                                  real_t *const SFEM_RESTRICT inv_diag,
                                  real_t *const SFEM_RESTRICT rhs,
                                  real_t *const SFEM_RESTRICT x,
                                  const real_t *const SFEM_RESTRICT weights,
                                  const real_t sum_weights,
                                  real_t *const SFEM_RESTRICT lagrange_multiplier) {
    real_t sum_values = 0;
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        sum_values += x[i] * weights[i];
    }

    *lagrange_multiplier = sum_values / sum_weights;

    for (int ii = 0; ii < 10; ii++) {
#pragma omp parallel
        {
#pragma omp for
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                const count_t r_begin = rowptr[i];
                const count_t r_end = rowptr[i + 1];
                const count_t r_extent = r_end - r_begin;
                const idx_t *const r_colidx = &colidx[r_begin];

                real_t r = rhs[i] - weights[i] * (*lagrange_multiplier);
                for (count_t k = 0; k < r_extent; k++) {
                    const idx_t col = r_colidx[k];
                    if (col == i) continue;
                    r -= values[r_begin + k] * atomic_read(&x[col]);
                }

                const real_t val = inv_diag[i] * r;
                atomic_write(&x[i], val);
            }
        }
    }

    return 0;
}

int constrained_gs(const ptrdiff_t nnodes,
                   const count_t *const SFEM_RESTRICT rowptr,
                   const idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values,
                   real_t *const SFEM_RESTRICT inv_diag,
                   real_t *const SFEM_RESTRICT rhs,
                   real_t *const SFEM_RESTRICT x,
                   const real_t *const SFEM_RESTRICT weights,
                   const real_t sum_weights,
                   real_t *const SFEM_RESTRICT lagrange_multiplier,
                   const int num_sweeps) {
    for (int s = 0; s < num_sweeps; s++) {
        constrained_gs_forward(nnodes,
                               rowptr,
                               colidx,
                               values,
                               inv_diag,
                               rhs,
                               x,
                               weights,
                               sum_weights,
                               lagrange_multiplier);
    }

    return 0;
}
