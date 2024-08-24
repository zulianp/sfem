#include "tet10_linear_elasticity.h"
#include "sfem_base.h"

#include "tet10_inline_cpu.h"
#include "tet10_linear_elasticity_inline_cpu.h"

#include <stddef.h>

static const int n_qp = 8;
static const scalar_t qx[8] =
        {0.0, 1.0, 0.0, 0.0, 0.333333333333, 0.333333333333, 0.0, 0.333333333333};

static const scalar_t qy[8] =
        {0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.0, 0.333333333333, 0.333333333333};

static const scalar_t qz[8] =
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.333333333333, 0.333333333333};

static const scalar_t qw[8] = {0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225};

int tet10_linear_elasticity_apply(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t mu,
                                  const real_t lambda,
                                  const ptrdiff_t u_stride,
                                  const real_t *const SFEM_RESTRICT ux,
                                  const real_t *const SFEM_RESTRICT uy,
                                  const real_t *const SFEM_RESTRICT uz,
                                  const ptrdiff_t out_stride,
                                  real_t *const SFEM_RESTRICT outx,
                                  real_t *const SFEM_RESTRICT outy,
                                  real_t *const SFEM_RESTRICT outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                              x[ev[1]],
                              x[ev[2]],
                              x[ev[3]],
                              // Y-coordinates
                              y[ev[0]],
                              y[ev[1]],
                              y[ev[2]],
                              y[ev[3]],
                              // Z-coordinates
                              z[ev[0]],
                              z[ev[1]],
                              z[ev[2]],
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_apply_adj(mu,
                                              lambda,
                                              jacobian_adjugate,
                                              jacobian_determinant,
                                              qx[k],
                                              qy[k],
                                              qz[k],
                                              qw[k],
                                              element_ux,
                                              element_uy,
                                              element_uz,
                                              element_outx,
                                              element_outy,
                                              element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_diag(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t out_stride,
                                 real_t *const SFEM_RESTRICT outx,
                                 real_t *const SFEM_RESTRICT outy,
                                 real_t *const SFEM_RESTRICT outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                              x[ev[1]],
                              x[ev[2]],
                              x[ev[3]],
                              // Y-coordinates
                              y[ev[0]],
                              y[ev[1]],
                              y[ev[2]],
                              y[ev[3]],
                              // Z-coordinates
                              z[ev[0]],
                              z[ev[1]],
                              z[ev[2]],
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_diag_adj(jacobian_adjugate,
                                             jacobian_determinant,
                                             mu,
                                             lambda,
                                             qx[k],
                                             qy[k],
                                             qz[k],
                                             qw[k],
                                             element_outx,
                                             element_outy,
                                             element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }
    return 0;
}

int tet10_linear_elasticity_apply_opt(const ptrdiff_t nelements,
                                      idx_t **const SFEM_RESTRICT elements,
                                      const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                      const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                      const real_t mu,
                                      const real_t lambda,
                                      const ptrdiff_t u_stride,
                                      const real_t *const SFEM_RESTRICT ux,
                                      const real_t *const SFEM_RESTRICT uy,
                                      const real_t *const SFEM_RESTRICT uz,
                                      const ptrdiff_t out_stride,
                                      real_t *const SFEM_RESTRICT outx,
                                      real_t *const SFEM_RESTRICT outy,
                                      real_t *const SFEM_RESTRICT outz) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        const scalar_t jacobian_determinant = g_jacobian_determinant[i];
        scalar_t jacobian_adjugate[9];
        for(int k = 0; k < 9; k++) {
            jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_apply_adj(mu,
                                              lambda,
                                              jacobian_adjugate,
                                              jacobian_determinant,
                                              qx[k],
                                              qy[k],
                                              qz[k],
                                              qw[k],
                                              element_ux,
                                              element_uy,
                                              element_uz,
                                              element_outx,
                                              element_outy,
                                              element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_diag_opt(const ptrdiff_t nelements,
                                     idx_t **const SFEM_RESTRICT elements,
                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                     const real_t mu,
                                     const real_t lambda,
                                     const ptrdiff_t out_stride,
                                     real_t *const SFEM_RESTRICT outx,
                                     real_t *const SFEM_RESTRICT outy,
                                     real_t *const SFEM_RESTRICT outz) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        accumulator_t element_outx[10] = {0};
        accumulator_t element_outy[10] = {0};
        accumulator_t element_outz[10] = {0};

        const scalar_t jacobian_determinant = g_jacobian_determinant[i];
        scalar_t jacobian_adjugate[9];
        for(int k = 0; k < 9; k++) {
            jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_diag_adj(jacobian_adjugate,
                                             jacobian_determinant,
                                             mu,
                                             lambda,
                                             qx[k],
                                             qy[k],
                                             qz[k],
                                             qw[k],
                                             element_outx,
                                             element_outy,
                                             element_outz);
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outx[ev[v] * out_stride] += element_outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outy[ev[v] * out_stride] += element_outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            outz[ev[v] * out_stride] += element_outz[v];
        }
    }

    return 0;
}

int tet10_linear_elasticity_crs(const ptrdiff_t nelements,
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
    const geom_t *const z = points[2];

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        idx_t ks[10];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        accumulator_t element_matrix[(10 * 3) * (10 * 3)] = {0};

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                              x[ev[1]],
                              x[ev[2]],
                              x[ev[3]],
                              // Y-coordinates
                              y[ev[0]],
                              y[ev[1]],
                              y[ev[2]],
                              y[ev[3]],
                              // Z-coordinates
                              z[ev[0]],
                              z[ev[1]],
                              z[ev[2]],
                              z[ev[3]],
                              // Output
                              jacobian_adjugate,
                              &jacobian_determinant);

        for (int k = 0; k < n_qp; k++) {
            tet10_linear_elasticity_crs_adj(qx[k],
                                                qy[k],
                                                qz[k],
                                                qw[k],
                                                jacobian_adjugate,
                                                jacobian_determinant,
                                                mu,
                                                lambda,
                                                element_matrix);
        }

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = elements[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            {
                const idx_t *row = &colidx[rowptr[dof_i]];
                tet10_find_cols(ev, row, lenrow, ks);
            }

            // Blocks for row
            real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < 10; ++edof_j) {
                const idx_t offset_j = ks[edof_j] * block_size;

                for (int bi = 0; bi < block_size; ++bi) {
                    const int ii = bi * 10 + edof_i;

                    // Jump rows (including the block-size for the columns)
                    real_t *row = &block_start[bi * lenrow * block_size];

                    for (int bj = 0; bj < block_size; ++bj) {
                        const int jj = bj * 10 + edof_j;
                        const real_t val = element_matrix[ii * 30 + jj];

#pragma omp atomic update
                        row[offset_j + bj] += val;
                    }
                }
            }
        }
    }

    return 0;
}
