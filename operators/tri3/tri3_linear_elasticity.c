#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "tri3_linear_elasticity_inline_cpu.h"

int tri3_linear_elasticity_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

    real_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        scalar_t element_ux[3];
        scalar_t element_uy[3];

#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 3; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
        }

        real_t element_scalar = 0;
        tri3_linear_elasticity_value_points(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                element_ux,
                element_uy,
                // output matrix
                &element_scalar);

        acc += element_scalar;
    }

    *value += acc;
    return 0;
}

int tri3_linear_elasticity_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const ptrdiff_t out_stride,
                                 real_t *const outx,
                                 real_t *const outy) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        scalar_t element_ux[3];
        scalar_t element_uy[3];

        accumulator_t element_outx[3];
        accumulator_t element_outy[3];

#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 3; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_linear_elasticity_apply_points(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                element_ux,
                element_uy,
                // output matrix
                element_outx,
                element_outy);

        for (int edof_i = 0; edof_i < 3; edof_i++) {
            ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];
        }
    }

    return 0;
}

int tri3_linear_elasticity_crs_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t mu,
                                       const real_t lambda,
                                       const count_t *const SFEM_RESTRICT rowptr,
                                       const idx_t *const SFEM_RESTRICT colidx,
                                       real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

    static const int block_size = 2;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        idx_t ks[3];

        real_t element_matrix[(3 * 2) * (3 * 2)];

#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_linear_elasticity_crs_points(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                // output matrix
                element_matrix);

        // assert(!check_symmetric(3 * block_size, element_matrix));

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elements[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            {
                const idx_t *row = &colidx[rowptr[dof_i]];
                tri3_find_cols(ev, row, lenrow, ks);
            }

            // Blocks for row
            real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < 3; ++edof_j) {
                const idx_t offset_j = ks[edof_j] * block_size;

                for (int bi = 0; bi < block_size; ++bi) {
                    const int ii = bi * 3 + edof_i;

                    // Jump rows (including the block-size for the columns)
                    real_t *row = &block_start[bi * lenrow * block_size];

                    for (int bj = 0; bj < block_size; ++bj) {
                        const int jj = bj * 3 + edof_j;
                        const real_t val = element_matrix[ii * 6 + jj];
#pragma omp atomic update
                        row[offset_j + bj] += val;
                    }
                }
            }
        }
    }

    return 0;
}

int tri3_linear_elasticity_crs_soa(const ptrdiff_t nelements,
                                                const ptrdiff_t nnodes,
                                                idx_t **const SFEM_RESTRICT elements,
                                                geom_t **const SFEM_RESTRICT points,
                                                const real_t mu,
                                                const real_t lambda,
                                                const count_t *const SFEM_RESTRICT rowptr,
                                                const idx_t *const SFEM_RESTRICT colidx,
                                                real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

    static const int block_size = 2;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        idx_t ks[3][3];
        real_t element_matrix[(3 * 2) * (3 * 2)];

        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_linear_elasticity_crs_points(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                // output matrix
                element_matrix);

        // find all indices
        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elements[edof_i][i];
            const idx_t r_begin = rowptr[dof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - r_begin;
            const idx_t *row = &colidx[rowptr[dof_i]];
            tri3_find_cols(ev, row, lenrow, ks[edof_i]);
        }

        for (int bi = 0; bi < block_size; ++bi) {
            for (int bj = 0; bj < block_size; ++bj) {
                for (int edof_i = 0; edof_i < 3; ++edof_i) {
                    const int ii = bi * 3 + edof_i;

                    const idx_t dof_i = elements[edof_i][i];
                    const idx_t r_begin = rowptr[dof_i];
                    const int bb = bi * block_size + bj;

                    real_t *const row_values = &values[bb][r_begin];

                    for (int edof_j = 0; edof_j < 3; ++edof_j) {
                        const int jj = bj * 3 + edof_j;
                        const real_t val = element_matrix[ii * 6 + jj];

                        assert(val == val);
#pragma omp atomic update
                        row_values[ks[edof_i][edof_j]] += val;
                    }
                }
            }
        }
    }

    return 0;
}
