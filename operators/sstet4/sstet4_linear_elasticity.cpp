#include "sstet4_linear_elasticity.hpp"

#include "hierarchy/sstet4/smesh_sstet4_transfer.impl.hpp"
#include "smesh_sstet4.hpp"
#include "tet4_inline_cpu.hpp"
#include "tet4_linear_elasticity_inline_cpu.hpp"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define SSTET4_LINEAR_ELASTICITY_MAX_STENCIL 32
#define SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE 9
#define SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE 16

struct sstet4_linear_elasticity_stencil {
    int       level;
    int       nrows;
    int       max_row_len;
    int       n_unique_stencils;
    ptrdiff_t nelements;
    int      *row_len;
    int      *cols;
    int      *element_stencil;
    scalar_t *unique_jacobian;
    scalar_t *weights;
};

static SFEM_INLINE void sstet4_linear_elasticity_macro_adjugate_and_det(
        const int                         level,
        idx_t **const SFEM_RESTRICT       elements,
        geom_t **const SFEM_RESTRICT      points,
        const ptrdiff_t                   e,
        scalar_t *const SFEM_RESTRICT     adjugate,
        scalar_t *const SFEM_RESTRICT     determinant) {
    const idx_t ev0 = elements[smesh::sstet4_lidx(level, 0, 0, 0)][e];
    const idx_t ev1 = elements[smesh::sstet4_lidx(level, level, 0, 0)][e];
    const idx_t ev2 = elements[smesh::sstet4_lidx(level, 0, level, 0)][e];
    const idx_t ev3 = elements[smesh::sstet4_lidx(level, 0, 0, level)][e];

    tet4_adjugate_and_det_s(points[0][ev0],
                            points[0][ev1],
                            points[0][ev2],
                            points[0][ev3],
                            points[1][ev0],
                            points[1][ev1],
                            points[1][ev2],
                            points[1][ev3],
                            points[2][ev0],
                            points[2][ev1],
                            points[2][ev2],
                            points[2][ev3],
                            adjugate,
                            determinant);
}

static SFEM_INLINE int sstet4_linear_elasticity_stencil_slot(
        const sstet4_linear_elasticity_stencil *const stencil,
        const int                                      row,
        const int                                      col) {
    const int row_offset = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int len        = stencil->row_len[row];
    for (int i = 0; i < len; ++i) {
        if (stencil->cols[row_offset + i] == col) {
            return i;
        }
    }

    return SFEM_IDX_INVALID;
}

static int sstet4_linear_elasticity_stencil_add_edge(sstet4_linear_elasticity_stencil *const stencil,
                                                     const int                                row,
                                                     const int                                col) {
    int slot = sstet4_linear_elasticity_stencil_slot(stencil, row, col);
    if (slot != SFEM_IDX_INVALID) {
        return SFEM_SUCCESS;
    }

    slot = stencil->row_len[row]++;
    if (slot >= SSTET4_LINEAR_ELASTICITY_MAX_STENCIL) {
        return SFEM_FAILURE;
    }

    stencil->cols[row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL + slot] = col;
    stencil->max_row_len =
            stencil->max_row_len > stencil->row_len[row] ? stencil->max_row_len : stencil->row_len[row];

    return SFEM_SUCCESS;
}

static int sstet4_linear_elasticity_stencil_add_microtet_topology(
        sstet4_linear_elasticity_stencil *const stencil,
        const int *const                         ev) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (sstet4_linear_elasticity_stencil_add_edge(stencil, ev[i], ev[j]) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }
    }

    return SFEM_SUCCESS;
}

static int sstet4_linear_elasticity_build_transfer_topology(sstet4_linear_elasticity_stencil *const stencil) {
    int err = SFEM_SUCCESS;
    smesh::sstet4_transfer::for_each_microtet(stencil->level, [&](const int *const ev) {
        if (err == SFEM_SUCCESS &&
            sstet4_linear_elasticity_stencil_add_microtet_topology(stencil, ev) != SFEM_SUCCESS) {
            err = SFEM_FAILURE;
        }
    });

    return err;
}

static bool sstet4_linear_elasticity_same_macro_jacobian(const scalar_t *const SFEM_RESTRICT a,
                                                         const scalar_t *const SFEM_RESTRICT b) {
    for (int i = 0; i < 10; ++i) {
        const real_t abs_a = fabs((real_t)a[i]);
        const real_t abs_b = fabs((real_t)b[i]);
        const real_t scale = abs_a > abs_b ? abs_a : abs_b;
        const real_t tol   = 1e-12 * (1 + scale);
        if (fabs((real_t)(a[i] - b[i])) > tol) {
            return false;
        }
    }

    return true;
}

static int sstet4_linear_elasticity_stencil_reserve_unique(
        sstet4_linear_elasticity_stencil *const stencil,
        int *const                               capacity,
        const int                                required) {
    if (required <= *capacity) {
        return SFEM_SUCCESS;
    }

    int new_capacity = *capacity > 0 ? *capacity : 4;
    while (new_capacity < required) {
        new_capacity *= 2;
    }

    const int entries = stencil->nrows * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;

    scalar_t *const unique_jacobian =
            (scalar_t *)realloc(stencil->unique_jacobian, new_capacity * 10 * sizeof(scalar_t));
    if (!unique_jacobian) {
        return SFEM_FAILURE;
    }

    stencil->unique_jacobian = unique_jacobian;

    scalar_t *const weights =
            (scalar_t *)realloc(stencil->weights,
                                new_capacity * entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE * sizeof(scalar_t));
    if (!weights) {
        return SFEM_FAILURE;
    }

    stencil->weights = weights;
    *capacity        = new_capacity;

    return SFEM_SUCCESS;
}

static int sstet4_linear_elasticity_build_point_weighted_stencils(
        sstet4_linear_elasticity_stencil *const stencil,
        idx_t **const SFEM_RESTRICT             elements,
        geom_t **const SFEM_RESTRICT            points,
        const real_t                            mu,
        const real_t                            lambda) {
    int       capacity = 0;
    const int entries  = stencil->nrows * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int stride   = entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE;

    for (ptrdiff_t e = 0; e < stencil->nelements; ++e) {
        scalar_t macro_jacobian[10];
        sstet4_linear_elasticity_macro_adjugate_and_det(
                stencil->level, elements, points, e, macro_jacobian, &macro_jacobian[9]);

        int stencil_id = SFEM_IDX_INVALID;
        for (int s = 0; s < stencil->n_unique_stencils; ++s) {
            if (sstet4_linear_elasticity_same_macro_jacobian(macro_jacobian, &stencil->unique_jacobian[s * 10])) {
                stencil_id = s;
                break;
            }
        }

        if (stencil_id == SFEM_IDX_INVALID) {
            if (sstet4_linear_elasticity_stencil_reserve_unique(stencil, &capacity, stencil->n_unique_stencils + 1) !=
                SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            stencil_id = stencil->n_unique_stencils++;
            for (int i = 0; i < 10; ++i) {
                stencil->unique_jacobian[stencil_id * 10 + i] = macro_jacobian[i];
            }

            scalar_t *const weights = &stencil->weights[stencil_id * stride];
            for (int i = 0; i < stride; ++i) {
                weights[i] = 0;
            }

            int err = SFEM_SUCCESS;
            smesh::sstet4_transfer::for_each_microtet(stencil->level, [&](const int *const lev) {
                if (err != SFEM_SUCCESS) {
                    return;
                }

                const idx_t gv[4] = {elements[lev[0]][e], elements[lev[1]][e], elements[lev[2]][e], elements[lev[3]][e]};
                scalar_t jacobian_adjugate[9];
                scalar_t jacobian_determinant = 0;
                accumulator_t element_matrix[12 * 12];

                tet4_adjugate_and_det_s(points[0][gv[0]],
                                        points[0][gv[1]],
                                        points[0][gv[2]],
                                        points[0][gv[3]],
                                        points[1][gv[0]],
                                        points[1][gv[1]],
                                        points[1][gv[2]],
                                        points[1][gv[3]],
                                        points[2][gv[0]],
                                        points[2][gv[1]],
                                        points[2][gv[2]],
                                        points[2][gv[3]],
                                        jacobian_adjugate,
                                        &jacobian_determinant);

                tet4_linear_elasticity_crs_adj(mu, lambda, jacobian_adjugate, jacobian_determinant, element_matrix);

                for (int i = 0; i < 4; ++i) {
                    const int row        = lev[i];
                    const int row_offset = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
                    for (int j = 0; j < 4; ++j) {
                        const int slot = sstet4_linear_elasticity_stencil_slot(stencil, row, lev[j]);
                        if (slot == SFEM_IDX_INVALID) {
                            err = SFEM_FAILURE;
                            return;
                        }

                        scalar_t *const block = &weights[(row_offset + slot) * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];
                        for (int bi = 0; bi < 3; ++bi) {
                            const int ii = bi * 4 + i;
                            for (int bj = 0; bj < 3; ++bj) {
                                const int jj = bj * 4 + j;
                                block[bi * 3 + bj] += element_matrix[ii * 12 + jj];
                            }
                        }
                    }
                }
            });

            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        stencil->element_stencil[e] = stencil_id;
    }

    return SFEM_SUCCESS;
}

static void sstet4_linear_elasticity_stencil_free_members(sstet4_linear_elasticity_stencil *const stencil) {
    if (!stencil) {
        return;
    }

    free(stencil->row_len);
    free(stencil->cols);
    free(stencil->element_stencil);
    free(stencil->unique_jacobian);
    free(stencil->weights);

    stencil->row_len         = nullptr;
    stencil->cols            = nullptr;
    stencil->element_stencil = nullptr;
    stencil->unique_jacobian = nullptr;
    stencil->weights         = nullptr;
}

int sstet4_linear_elasticity_apply_points(const int                         level,
                                          const ptrdiff_t                   nelements,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      mu,
                                          const real_t                      lambda,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT       values) {
    if (!elements || !points || !u || !values || level < 1 || nelements < 0) {
        return SFEM_FAILURE;
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        smesh::sstet4_transfer::for_each_microtet(level, [&](const int *const lev) {
            const idx_t gv[4] = {elements[lev[0]][e], elements[lev[1]][e], elements[lev[2]][e], elements[lev[3]][e]};
            scalar_t ux[4], uy[4], uz[4];
            accumulator_t outx[4], outy[4], outz[4];
            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant = 0;

            for (int i = 0; i < 4; ++i) {
                ux[i] = u[gv[i] * 3 + 0];
                uy[i] = u[gv[i] * 3 + 1];
                uz[i] = u[gv[i] * 3 + 2];
            }

            tet4_adjugate_and_det_s(points[0][gv[0]],
                                    points[0][gv[1]],
                                    points[0][gv[2]],
                                    points[0][gv[3]],
                                    points[1][gv[0]],
                                    points[1][gv[1]],
                                    points[1][gv[2]],
                                    points[1][gv[3]],
                                    points[2][gv[0]],
                                    points[2][gv[1]],
                                    points[2][gv[2]],
                                    points[2][gv[3]],
                                    jacobian_adjugate,
                                    &jacobian_determinant);

            tet4_linear_elasticity_apply_adj(
                    jacobian_adjugate, jacobian_determinant, mu, lambda, ux, uy, uz, outx, outy, outz);

            for (int i = 0; i < 4; ++i) {
#pragma omp atomic update
                values[gv[i] * 3 + 0] += outx[i];
#pragma omp atomic update
                values[gv[i] * 3 + 1] += outy[i];
#pragma omp atomic update
                values[gv[i] * 3 + 2] += outz[i];
            }
        });
    }

    return SFEM_SUCCESS;
}

int sstet4_linear_elasticity_stencil_create_from_points(const int                                  level,
                                                        const ptrdiff_t                            nelements,
                                                        idx_t **const SFEM_RESTRICT                elements,
                                                        geom_t **const SFEM_RESTRICT               points,
                                                        const real_t                               mu,
                                                        const real_t                               lambda,
                                                        sstet4_linear_elasticity_stencil_t **const stencil_out) {
    if (!elements || !points || !stencil_out || level < 1 || nelements < 0) {
        return SFEM_FAILURE;
    }

    *stencil_out = nullptr;

    sstet4_linear_elasticity_stencil *const stencil =
            (sstet4_linear_elasticity_stencil *)calloc(1, sizeof(sstet4_linear_elasticity_stencil));
    if (!stencil) {
        return SFEM_FAILURE;
    }

    stencil->level     = level;
    stencil->nrows     = smesh::sstet4_nxe(level);
    stencil->nelements = nelements;

    const int nrows        = stencil->nrows;
    const int stencil_size = nrows * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;

    stencil->row_len         = (int *)calloc(nrows, sizeof(int));
    stencil->cols            = (int *)malloc(stencil_size * sizeof(int));
    stencil->element_stencil = (int *)malloc((nelements > 0 ? nelements : 1) * sizeof(int));

    if (!stencil->row_len || !stencil->cols || !stencil->element_stencil) {
        sstet4_linear_elasticity_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    for (int i = 0; i < stencil_size; ++i) {
        stencil->cols[i] = SFEM_IDX_INVALID;
    }

    if (sstet4_linear_elasticity_build_transfer_topology(stencil) != SFEM_SUCCESS ||
        sstet4_linear_elasticity_build_point_weighted_stencils(stencil, elements, points, mu, lambda) != SFEM_SUCCESS) {
        sstet4_linear_elasticity_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    *stencil_out = stencil;
    return SFEM_SUCCESS;
}

void sstet4_linear_elasticity_stencil_destroy(sstet4_linear_elasticity_stencil_t *stencil) {
    if (!stencil) {
        return;
    }

    sstet4_linear_elasticity_stencil_free_members(stencil);
    free(stencil);
}

int sstet4_linear_elasticity_stencil_nrows(const sstet4_linear_elasticity_stencil_t *const stencil) {
    return stencil ? stencil->nrows : 0;
}

int sstet4_linear_elasticity_stencil_max_row_len(const sstet4_linear_elasticity_stencil_t *const stencil) {
    return stencil ? stencil->max_row_len : 0;
}

int sstet4_linear_elasticity_stencil_n_unique_stencils(const sstet4_linear_elasticity_stencil_t *const stencil) {
    return stencil ? stencil->n_unique_stencils : 0;
}

int sstet4_linear_elasticity_apply_stencil(const sstet4_linear_elasticity_stencil_t *const stencil,
                                           const ptrdiff_t                                  nelements,
                                           const real_t *const SFEM_RESTRICT                u,
                                           real_t *const SFEM_RESTRICT                      values) {
    if (!stencil || !u || !values || nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe          = stencil->nrows;
    const int entries      = nxe * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int weight_stride = entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const real_t *const SFEM_RESTRICT  element_u      = &u[e * nxe * 3];
        real_t *const SFEM_RESTRICT        element_vector = &values[e * nxe * 3];
        const scalar_t *const SFEM_RESTRICT weights =
                &stencil->weights[stencil->element_stencil[e] * weight_stride];

        for (int row = 0; row < nxe; ++row) {
            accumulator_t acc0 = 0;
            accumulator_t acc1 = 0;
            accumulator_t acc2 = 0;
            const int row_offset = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
            const int row_len    = stencil->row_len[row];

            for (int s = 0; s < row_len; ++s) {
                const int entry = row_offset + s;
                const int col   = stencil->cols[entry];
                const scalar_t *const block = &weights[entry * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];
                const scalar_t u0 = element_u[col * 3 + 0];
                const scalar_t u1 = element_u[col * 3 + 1];
                const scalar_t u2 = element_u[col * 3 + 2];

                acc0 += block[0] * u0 + block[1] * u1 + block[2] * u2;
                acc1 += block[3] * u0 + block[4] * u1 + block[5] * u2;
                acc2 += block[6] * u0 + block[7] * u1 + block[8] * u2;
            }

            element_vector[row * 3 + 0] += acc0;
            element_vector[row * 3 + 1] += acc1;
            element_vector[row * 3 + 2] += acc2;
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_linear_elasticity_apply_stencil_global_vectorized(
        const sstet4_linear_elasticity_stencil_t *const stencil,
        const ptrdiff_t                                  nelements,
        idx_t **const SFEM_RESTRICT                      elements,
        const real_t *const SFEM_RESTRICT                u,
        real_t *const SFEM_RESTRICT                      values) {
    if (!stencil || !elements || !u || !values || nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe          = stencil->nrows;
    const int entries      = nxe * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int weight_stride = entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE) {
        const int nelems = (int)((nelements - evbegin) < SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE
                                         ? (nelements - evbegin)
                                         : SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE);
        int stencil_id[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];

        const int first_stencil_id = stencil->element_stencil[evbegin];
        int same_stencil = 1;

#pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            stencil_id[lane] = stencil->element_stencil[evbegin + lane];
        }

        for (int lane = 0; lane < nelems; ++lane) {
            same_stencil &= stencil_id[lane] == first_stencil_id;
        }

        if (same_stencil) {
            const scalar_t *const SFEM_RESTRICT weights = &stencil->weights[first_stencil_id * weight_stride];

            for (int row = 0; row < nxe; ++row) {
                accumulator_t acc0[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                accumulator_t acc1[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                accumulator_t acc2[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                const int row_offset = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
                const int row_len    = stencil->row_len[row];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    acc0[lane] = 0;
                    acc1[lane] = 0;
                    acc2[lane] = 0;
                }

                for (int s = 0; s < row_len; ++s) {
                    const int entry = row_offset + s;
                    const idx_t *const element_col = elements[stencil->cols[entry]];
                    const scalar_t *const block = &weights[entry * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        const idx_t node = element_col[evbegin + lane];
                        const scalar_t u0 = u[node * 3 + 0];
                        const scalar_t u1 = u[node * 3 + 1];
                        const scalar_t u2 = u[node * 3 + 2];

                        acc0[lane] += block[0] * u0 + block[1] * u1 + block[2] * u2;
                        acc1[lane] += block[3] * u0 + block[4] * u1 + block[5] * u2;
                        acc2[lane] += block[6] * u0 + block[7] * u1 + block[8] * u2;
                    }
                }

                const idx_t *const element_row = elements[row];
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_row[evbegin + lane];
#pragma omp atomic update
                    values[node * 3 + 0] += acc0[lane];
#pragma omp atomic update
                    values[node * 3 + 1] += acc1[lane];
#pragma omp atomic update
                    values[node * 3 + 2] += acc2[lane];
                }
            }
        } else {
            for (int row = 0; row < nxe; ++row) {
                accumulator_t acc0[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                accumulator_t acc1[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                accumulator_t acc2[SSTET4_LINEAR_ELASTICITY_VECTOR_SIZE];
                const int row_offset = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
                const int row_len    = stencil->row_len[row];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    acc0[lane] = 0;
                    acc1[lane] = 0;
                    acc2[lane] = 0;
                }

                for (int s = 0; s < row_len; ++s) {
                    const int entry = row_offset + s;
                    const idx_t *const element_col = elements[stencil->cols[entry]];

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        const scalar_t *const block =
                                &stencil->weights[stencil_id[lane] * weight_stride +
                                                  entry * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];
                        const idx_t node = element_col[evbegin + lane];
                        const scalar_t u0 = u[node * 3 + 0];
                        const scalar_t u1 = u[node * 3 + 1];
                        const scalar_t u2 = u[node * 3 + 2];

                        acc0[lane] += block[0] * u0 + block[1] * u1 + block[2] * u2;
                        acc1[lane] += block[3] * u0 + block[4] * u1 + block[5] * u2;
                        acc2[lane] += block[6] * u0 + block[7] * u1 + block[8] * u2;
                    }
                }

                const idx_t *const element_row = elements[row];
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_row[evbegin + lane];
#pragma omp atomic update
                    values[node * 3 + 0] += acc0[lane];
#pragma omp atomic update
                    values[node * 3 + 1] += acc1[lane];
#pragma omp atomic update
                    values[node * 3 + 2] += acc2[lane];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_linear_elasticity_block_diag_sym_stencil(
        const sstet4_linear_elasticity_stencil_t *const stencil,
        const ptrdiff_t                                  nelements,
        idx_t **const SFEM_RESTRICT                      elements,
        const ptrdiff_t                                  out_stride,
        real_t *const SFEM_RESTRICT                      out0,
        real_t *const SFEM_RESTRICT                      out1,
        real_t *const SFEM_RESTRICT                      out2,
        real_t *const SFEM_RESTRICT                      out3,
        real_t *const SFEM_RESTRICT                      out4,
        real_t *const SFEM_RESTRICT                      out5) {
    if (!stencil || !elements || !out0 || !out1 || !out2 || !out3 || !out4 || !out5 ||
        nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe           = stencil->nrows;
    const int entries       = nxe * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int weight_stride = entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const scalar_t *const SFEM_RESTRICT weights =
                &stencil->weights[stencil->element_stencil[e] * weight_stride];

        for (int row = 0; row < nxe; ++row) {
            const int slot = sstet4_linear_elasticity_stencil_slot(stencil, row, row);
            if (slot == SFEM_IDX_INVALID) {
                continue;
            }

            const idx_t           node  = elements[row][e];
            const int             entry = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL + slot;
            const scalar_t *const block = &weights[entry * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];

#pragma omp atomic update
            out0[node * out_stride] += block[0];
#pragma omp atomic update
            out1[node * out_stride] += block[1];
#pragma omp atomic update
            out2[node * out_stride] += block[2];
#pragma omp atomic update
            out3[node * out_stride] += block[4];
#pragma omp atomic update
            out4[node * out_stride] += block[5];
#pragma omp atomic update
            out5[node * out_stride] += block[8];
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_linear_elasticity_diag_stencil(const sstet4_linear_elasticity_stencil_t *const stencil,
                                          const ptrdiff_t                                  nelements,
                                          idx_t **const SFEM_RESTRICT                      elements,
                                          const ptrdiff_t                                  out_stride,
                                          real_t *const SFEM_RESTRICT                      outx,
                                          real_t *const SFEM_RESTRICT                      outy,
                                          real_t *const SFEM_RESTRICT                      outz) {
    if (!stencil || !elements || !outx || !outy || !outz || nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe           = stencil->nrows;
    const int entries       = nxe * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL;
    const int weight_stride = entries * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const scalar_t *const SFEM_RESTRICT weights =
                &stencil->weights[stencil->element_stencil[e] * weight_stride];

        for (int row = 0; row < nxe; ++row) {
            const int slot = sstet4_linear_elasticity_stencil_slot(stencil, row, row);
            if (slot == SFEM_IDX_INVALID) {
                continue;
            }

            const idx_t           node  = elements[row][e];
            const int             entry = row * SSTET4_LINEAR_ELASTICITY_MAX_STENCIL + slot;
            const scalar_t *const block = &weights[entry * SSTET4_LINEAR_ELASTICITY_BLOCK_SIZE];

#pragma omp atomic update
            outx[node * out_stride] += block[0];
#pragma omp atomic update
            outy[node * out_stride] += block[4];
#pragma omp atomic update
            outz[node * out_stride] += block[8];
        }
    }

    return SFEM_SUCCESS;
}
