#include "sshex8_stencil_element_matrix_apply.hpp"

#include "packed_elements.hpp"
#include "sshex8.hpp"
#include "sshex8_skeleton_stencil.hpp"
#include "stencil3.hpp"

#include <math.h>
#include <string.h>

#define SSHEX8_LINEAR_ELASTICITY_TENSOR_COEFFS 81

int sshex8_stencil_element_matrix_apply(const int                           level,
                                        const ptrdiff_t                     nelements,
                                        idx_t **const SFEM_RESTRICT         elements,
                                        const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                        const real_t *const SFEM_RESTRICT   u,
                                        real_t *const SFEM_RESTRICT         values) {
    const int nxe  = sshex8_nxe(level);
    const int txe  = sshex8_txe(level);
    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = (accumulator_t *)malloc(nxe * sizeof(accumulator_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t *const element_matrix = &g_element_matrix[e * 64];

            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(element_matrix, laplacian_stencil);
            sshex8_stencil(
                // count
                level + 1, level + 1, level + 1, 
                // buffers
                laplacian_stencil, eu, v);
            
            sshex8_surface_stencil(
                    // count
                    level + 1, level + 1, level + 1, 
                    // stide
                    1, level + 1, (level + 1) * (level + 1), 
                    // buffers
                    element_matrix, eu, v);

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    values[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply_hyteg(const int                           level,
                                              const ptrdiff_t                     nelements,
                                              idx_t **const SFEM_RESTRICT         elements,
                                              const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                              const scalar_t *const SFEM_RESTRICT g_stencil,
                                              const real_t *const SFEM_RESTRICT   u,
                                              real_t *const SFEM_RESTRICT         values) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t      *eu = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = (accumulator_t *)malloc(nxe * sizeof(accumulator_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                eu[d] = u[ev[d]];
                assert(eu[d] == eu[d]);
            }

            memset(v, 0, nxe * sizeof(accumulator_t));

            const scalar_t *const element_matrix = &g_element_matrix[e * 64];
            const scalar_t *const stencil        = &g_stencil[e * 27];
            scalar_t             laplacian_stencil[3 * 3 * 3];
            for (int d = 0; d < 3 * 3 * 3; ++d) {
                laplacian_stencil[d] = stencil[d];
            }

            sshex8_stencil(
                    level + 1,
                    level + 1,
                    level + 1,
                    laplacian_stencil,
                    eu,
                    v);

            sshex8_surface_stencil(
                    level + 1,
                    level + 1,
                    level + 1,
                    1,
                    level + 1,
                    (level + 1) * (level + 1),
                    element_matrix,
                    eu,
                    v);

            for (int d = 0; d < nxe; d++) {
                assert(v[d] == v[d]);
#pragma omp atomic update
                values[ev[d]] += v[d];
            }
        }

        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply3(const int                           level,
                                         const ptrdiff_t                     nelements,
                                         idx_t **const SFEM_RESTRICT         elements,
                                         const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                         const ptrdiff_t                     u_stride,
                                         const real_t *const SFEM_RESTRICT   ux,
                                         const real_t *const SFEM_RESTRICT   uy,
                                         const real_t *const SFEM_RESTRICT   uz,
                                         const ptrdiff_t                     out_stride,
                                         real_t *const SFEM_RESTRICT         outx,
                                         real_t *const SFEM_RESTRICT         outy,
                                         real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t    *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        scalar_t *X  = (scalar_t *)malloc(txe * 24 * sizeof(scalar_t));
        scalar_t *Y  = (scalar_t *)malloc(txe * 24 * sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            sshex8_SoA_pack_elements(level, eu, X);
            packed_elements_matmul(24, txe, 24, &g_element_matrix[e * 24 * 24], X, Y);

            for (int d = 0; d < 3; d++) {
                memset(v[d], 0, nxe * sizeof(scalar_t));
            }

            sshex8_SoA_unpack_add_elements(level, Y, v);

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * out_stride;

#pragma omp atomic update
                outx[idx] += v[0][d];

#pragma omp atomic update
                outy[idx] += v[1][d];

#pragma omp atomic update
                outz[idx] += v[2][d];
            }
        }

        free(ev);
        free(X);
        free(Y);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

static SFEM_INLINE int sshex8_hex8_node(const int x, const int y, const int z) {
    return x + 2 * y + 4 * z;
}

static SFEM_INLINE int sshex8_category(const int cx, const int cy, const int cz) { return cx + 3 * cy + 9 * cz; }

static SFEM_INLINE int sshex8_offset(const int dx, const int dy, const int dz) {
    return (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1);
}

static SFEM_INLINE scalar_t sshex8_tp_m(const int d) {
    return d == 0 ? (scalar_t)(2.0 / 3.0) : (scalar_t)(1.0 / 6.0);
}

static SFEM_INLINE scalar_t sshex8_tp_k(const int d) {
    return d == 0 ? (scalar_t)2 : (scalar_t)-1;
}

static SFEM_INLINE scalar_t sshex8_tp_p(const int d) {
    return d < 0 ? (scalar_t)0.5 : (d > 0 ? (scalar_t)-0.5 : (scalar_t)0);
}

static SFEM_INLINE scalar_t sshex8_tp_q(const int d) {
    return d < 0 ? (scalar_t)-0.5 : (d > 0 ? (scalar_t)0.5 : (scalar_t)0);
}

static SFEM_INLINE scalar_t sshex8_tp_basis(const int basis, const int dx, const int dy, const int dz) {
    switch (basis) {
        case 0: return sshex8_tp_k(dx) * sshex8_tp_m(dy) * sshex8_tp_m(dz);
        case 1: return sshex8_tp_m(dx) * sshex8_tp_k(dy) * sshex8_tp_m(dz);
        case 2: return sshex8_tp_m(dx) * sshex8_tp_m(dy) * sshex8_tp_k(dz);
        case 3: return sshex8_tp_p(dx) * sshex8_tp_q(dy) * sshex8_tp_m(dz);
        case 4: return sshex8_tp_p(dx) * sshex8_tp_m(dy) * sshex8_tp_q(dz);
        default: return sshex8_tp_m(dx) * sshex8_tp_p(dy) * sshex8_tp_q(dz);
    }
}

static SFEM_INLINE scalar_t sshex8_tp_pair_basis(const int p, const int q, const int dx, const int dy, const int dz) {
    scalar_t ret = 1;
    const int d[3] = {dx, dy, dz};

    for (int axis = 0; axis < 3; ++axis) {
        if (axis == p && axis == q) {
            ret *= sshex8_tp_k(d[axis]);
        } else if (axis == p) {
            ret *= sshex8_tp_p(d[axis]);
        } else if (axis == q) {
            ret *= sshex8_tp_q(d[axis]);
        } else {
            ret *= sshex8_tp_m(d[axis]);
        }
    }

    return ret;
}

static SFEM_INLINE scalar_t sshex8_tp_local_m(const int row, const int col) {
    return row == col ? (scalar_t)(1.0 / 3.0) : (scalar_t)(1.0 / 6.0);
}

static SFEM_INLINE scalar_t sshex8_tp_local_k(const int row, const int col) {
    return row == col ? (scalar_t)1 : (scalar_t)-1;
}

static SFEM_INLINE scalar_t sshex8_tp_local_p(const int row, const int) {
    return row == 0 ? (scalar_t)-0.5 : (scalar_t)0.5;
}

static SFEM_INLINE scalar_t sshex8_tp_local_q(const int, const int col) {
    return col == 0 ? (scalar_t)-0.5 : (scalar_t)0.5;
}

static SFEM_INLINE scalar_t sshex8_tp_local_pair_basis(const int p,
                                                       const int q,
                                                       const int row_node,
                                                       const int col_node) {
    scalar_t ret = 1;
    const int row[3] = {row_node & 1, (row_node >> 1) & 1, (row_node >> 2) & 1};
    const int col[3] = {col_node & 1, (col_node >> 1) & 1, (col_node >> 2) & 1};

    for (int axis = 0; axis < 3; ++axis) {
        if (axis == p && axis == q) {
            ret *= sshex8_tp_local_k(row[axis], col[axis]);
        } else if (axis == p) {
            ret *= sshex8_tp_local_p(row[axis], col[axis]);
        } else if (axis == q) {
            ret *= sshex8_tp_local_q(row[axis], col[axis]);
        } else {
            ret *= sshex8_tp_local_m(row[axis], col[axis]);
        }
    }

    return ret;
}

static void sshex8_elasticity_solve_9x9(const scalar_t *const SFEM_RESTRICT A,
                                        const scalar_t *const SFEM_RESTRICT b,
                                        scalar_t *const SFEM_RESTRICT       x) {
    scalar_t M[9][10];

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            M[i][j] = A[i * 9 + j];
        }

        M[i][9] = b[i];
    }

    for (int k = 0; k < 9; ++k) {
        int      pivot     = k;
        scalar_t pivot_abs = fabs(M[k][k]);
        for (int i = k + 1; i < 9; ++i) {
            const scalar_t candidate = fabs(M[i][k]);
            if (candidate > pivot_abs) {
                pivot     = i;
                pivot_abs = candidate;
            }
        }

        if (pivot != k) {
            for (int j = k; j < 10; ++j) {
                const scalar_t tmp = M[k][j];
                M[k][j]           = M[pivot][j];
                M[pivot][j]       = tmp;
            }
        }

        const scalar_t inv_pivot = (scalar_t)1 / M[k][k];
        for (int i = k + 1; i < 9; ++i) {
            const scalar_t factor = M[i][k] * inv_pivot;
            M[i][k]               = 0;
            for (int j = k + 1; j < 10; ++j) {
                M[i][j] -= factor * M[k][j];
            }
        }
    }

    for (int i = 8; i >= 0; --i) {
        scalar_t rhs = M[i][9];
        for (int j = i + 1; j < 9; ++j) {
            rhs -= M[i][j] * x[j];
        }

        x[i] = rhs / M[i][i];
    }
}

static void sshex8_linear_elasticity_matrix_to_category_stencils(
        const scalar_t *const SFEM_RESTRICT element_matrix,
        scalar_t *const SFEM_RESTRICT       category_stencils) {
    memset(category_stencils, 0, 27 * 27 * 9 * sizeof(scalar_t));

    for (int cz = 0; cz < 3; ++cz) {
        for (int cy = 0; cy < 3; ++cy) {
            for (int cx = 0; cx < 3; ++cx) {
                const int cat = sshex8_category(cx, cy, cz);

                for (int sz = -1; sz <= 0; ++sz) {
                    if ((cz == 0 && sz == -1) || (cz == 2 && sz == 0)) continue;

                    for (int sy = -1; sy <= 0; ++sy) {
                        if ((cy == 0 && sy == -1) || (cy == 2 && sy == 0)) continue;

                        for (int sx = -1; sx <= 0; ++sx) {
                            if ((cx == 0 && sx == -1) || (cx == 2 && sx == 0)) continue;

                            const int row_node = sshex8_hex8_node(-sx, -sy, -sz);

                            for (int vz = 0; vz <= 1; ++vz) {
                                const int dz = sz + vz;
                                for (int vy = 0; vy <= 1; ++vy) {
                                    const int dy = sy + vy;
                                    for (int vx = 0; vx <= 1; ++vx) {
                                        const int dx       = sx + vx;
                                        const int off      = sshex8_offset(dx, dy, dz);
                                        const int col_node = sshex8_hex8_node(vx, vy, vz);
                                        scalar_t *const SFEM_RESTRICT block = &category_stencils[(cat * 27 + off) * 9];

                                        for (int co = 0; co < 3; ++co) {
                                            const scalar_t *const SFEM_RESTRICT row =
                                                    &element_matrix[(co * 8 + row_node) * 24];
                                            for (int ci = 0; ci < 3; ++ci) {
                                                block[co * 3 + ci] += row[ci * 8 + col_node];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int sshex8_linear_elasticity_element_matrix_to_category_stencils(
        const ptrdiff_t                     nelements,
        const scalar_t *const SFEM_RESTRICT g_element_matrix,
        scalar_t *const SFEM_RESTRICT       g_category_stencils) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        sshex8_linear_elasticity_matrix_to_category_stencils(&g_element_matrix[e * 24 * 24],
                                                             &g_category_stencils[e * 27 * 27 * 9]);
    }

    return SFEM_SUCCESS;
}

static void sshex8_linear_elasticity_category_stencil_to_tensor_coeffs(
        const scalar_t *const SFEM_RESTRICT category_stencils,
        scalar_t *const SFEM_RESTRICT       tensor_coeffs) {
    const scalar_t *const SFEM_RESTRICT stencil = &category_stencils[sshex8_category(1, 1, 1) * 27 * 9];
    scalar_t normal[9 * 9];
    memset(normal, 0, 9 * 9 * sizeof(scalar_t));

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                scalar_t basis[9];
                for (int p = 0; p < 3; ++p) {
                    for (int q = 0; q < 3; ++q) {
                        basis[p * 3 + q] = sshex8_tp_pair_basis(p, q, dx, dy, dz);
                    }
                }

                for (int i = 0; i < 9; ++i) {
                    for (int j = 0; j < 9; ++j) {
                        normal[i * 9 + j] += basis[i] * basis[j];
                    }
                }
            }
        }
    }

    for (int co = 0; co < 3; ++co) {
        for (int ci = 0; ci < 3; ++ci) {
            scalar_t rhs[9];
            memset(rhs, 0, 9 * sizeof(scalar_t));

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const scalar_t block_value = stencil[sshex8_offset(dx, dy, dz) * 9 + co * 3 + ci];
                        for (int p = 0; p < 3; ++p) {
                            for (int q = 0; q < 3; ++q) {
                                rhs[p * 3 + q] += sshex8_tp_pair_basis(p, q, dx, dy, dz) * block_value;
                            }
                        }
                    }
                }
            }

            sshex8_elasticity_solve_9x9(normal, rhs, &tensor_coeffs[(co * 3 + ci) * 9]);
        }
    }
}

static void sshex8_linear_elasticity_element_matrix_to_tensor_coeffs_one(
        const scalar_t *const SFEM_RESTRICT element_matrix,
        scalar_t *const SFEM_RESTRICT       tensor_coeffs) {
    scalar_t normal[9 * 9];
    memset(normal, 0, 9 * 9 * sizeof(scalar_t));

    for (int row_node = 0; row_node < 8; ++row_node) {
        for (int col_node = 0; col_node < 8; ++col_node) {
            scalar_t basis[9];

            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    basis[p * 3 + q] = sshex8_tp_local_pair_basis(p, q, row_node, col_node);
                }
            }

            for (int i = 0; i < 9; ++i) {
                for (int j = 0; j < 9; ++j) {
                    normal[i * 9 + j] += basis[i] * basis[j];
                }
            }
        }
    }

    for (int co = 0; co < 3; ++co) {
        for (int ci = 0; ci < 3; ++ci) {
            scalar_t rhs[9];
            memset(rhs, 0, 9 * sizeof(scalar_t));

            for (int row_node = 0; row_node < 8; ++row_node) {
                for (int col_node = 0; col_node < 8; ++col_node) {
                    scalar_t basis[9];
                    for (int p = 0; p < 3; ++p) {
                        for (int q = 0; q < 3; ++q) {
                            basis[p * 3 + q] = sshex8_tp_local_pair_basis(p, q, row_node, col_node);
                        }
                    }

                    const scalar_t value = element_matrix[(co * 8 + row_node) * 24 + ci * 8 + col_node];
                    for (int i = 0; i < 9; ++i) {
                        rhs[i] += basis[i] * value;
                    }
                }
            }

            sshex8_elasticity_solve_9x9(normal, rhs, &tensor_coeffs[(co * 3 + ci) * 9]);
        }
    }
}

int sshex8_linear_elasticity_category_stencils_to_tensor_coeffs(
        const ptrdiff_t                     nelements,
        const scalar_t *const SFEM_RESTRICT g_category_stencils,
        scalar_t *const SFEM_RESTRICT       g_tensor_coeffs) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        sshex8_linear_elasticity_category_stencil_to_tensor_coeffs(&g_category_stencils[e * 27 * 27 * 9],
                                                                   &g_tensor_coeffs[e * SSHEX8_LINEAR_ELASTICITY_TENSOR_COEFFS]);
    }

    return SFEM_SUCCESS;
}

int sshex8_linear_elasticity_element_matrix_to_tensor_coeffs(
        const ptrdiff_t                     nelements,
        const scalar_t *const SFEM_RESTRICT g_element_matrix,
        scalar_t *const SFEM_RESTRICT       g_tensor_coeffs) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        sshex8_linear_elasticity_element_matrix_to_tensor_coeffs_one(&g_element_matrix[e * 24 * 24],
                                                                     &g_tensor_coeffs[e * SSHEX8_LINEAR_ELASTICITY_TENSOR_COEFFS]);
    }

    return SFEM_SUCCESS;
}

static SFEM_INLINE void sshex8_tensor_terms_at(const int                           idx,
                                               const int                           Lp1,
                                               const int                           zstride,
                                               const scalar_t *const SFEM_RESTRICT u,
                                               scalar_t *const SFEM_RESTRICT       t) {
    scalar_t kmm[3];
    scalar_t mkm[3];
    scalar_t pqm[3];
    scalar_t pmm[3];
    scalar_t mpm[3];
    scalar_t mmm[3];

    for (int lz = 0; lz < 3; ++lz) {
        const int dz = lz - 1;

        scalar_t xm[3];
        scalar_t xk[3];
        scalar_t xp[3];

        for (int ly = 0; ly < 3; ++ly) {
            const int base = idx + dz * zstride + (ly - 1) * Lp1;
            const scalar_t um = u[base - 1];
            const scalar_t uc = u[base];
            const scalar_t up = u[base + 1];

            xm[ly] = (scalar_t)(1.0 / 6.0) * (um + up) + (scalar_t)(2.0 / 3.0) * uc;
            xk[ly] = -um + (scalar_t)2 * uc - up;
            xp[ly] = (scalar_t)0.5 * (um - up);
        }

        kmm[lz] = (scalar_t)(1.0 / 6.0) * (xk[0] + xk[2]) + (scalar_t)(2.0 / 3.0) * xk[1];
        mkm[lz] = -xm[0] + (scalar_t)2 * xm[1] - xm[2];
        pqm[lz] = (scalar_t)0.5 * (xp[2] - xp[0]);
        pmm[lz] = (scalar_t)(1.0 / 6.0) * (xp[0] + xp[2]) + (scalar_t)(2.0 / 3.0) * xp[1];
        mpm[lz] = (scalar_t)0.5 * (xm[0] - xm[2]);
        mmm[lz] = (scalar_t)(1.0 / 6.0) * (xm[0] + xm[2]) + (scalar_t)(2.0 / 3.0) * xm[1];
    }

    t[0] = (scalar_t)(1.0 / 6.0) * (kmm[0] + kmm[2]) + (scalar_t)(2.0 / 3.0) * kmm[1];
    t[1] = (scalar_t)(1.0 / 6.0) * (mkm[0] + mkm[2]) + (scalar_t)(2.0 / 3.0) * mkm[1];
    t[2] = -mmm[0] + (scalar_t)2 * mmm[1] - mmm[2];
    t[3] = (scalar_t)(1.0 / 6.0) * (pqm[0] + pqm[2]) + (scalar_t)(2.0 / 3.0) * pqm[1];
    t[4] = (scalar_t)0.5 * (pmm[2] - pmm[0]);
    t[5] = (scalar_t)0.5 * (mpm[2] - mpm[0]);
}

static SFEM_INLINE scalar_t sshex8_tp_row_m(const int cat, const int d) {
    if (cat == 0) {
        return d == 0 ? (scalar_t)(1.0 / 3.0) : (d == 1 ? (scalar_t)(1.0 / 6.0) : (scalar_t)0);
    }

    if (cat == 2) {
        return d == 0 ? (scalar_t)(1.0 / 3.0) : (d == -1 ? (scalar_t)(1.0 / 6.0) : (scalar_t)0);
    }

    return sshex8_tp_m(d);
}

static SFEM_INLINE scalar_t sshex8_tp_row_k(const int cat, const int d) {
    if (cat == 0) {
        return d == 0 ? (scalar_t)1 : (d == 1 ? (scalar_t)-1 : (scalar_t)0);
    }

    if (cat == 2) {
        return d == 0 ? (scalar_t)1 : (d == -1 ? (scalar_t)-1 : (scalar_t)0);
    }

    return sshex8_tp_k(d);
}

static SFEM_INLINE scalar_t sshex8_tp_row_p(const int cat, const int d) {
    if (cat == 0) {
        return d == 0 ? (scalar_t)-0.5 : (d == 1 ? (scalar_t)-0.5 : (scalar_t)0);
    }

    if (cat == 2) {
        return d == -1 ? (scalar_t)0.5 : (d == 0 ? (scalar_t)0.5 : (scalar_t)0);
    }

    return sshex8_tp_p(d);
}

static SFEM_INLINE scalar_t sshex8_tp_row_q(const int cat, const int d) {
    if (cat == 0) {
        return d == 0 ? (scalar_t)-0.5 : (d == 1 ? (scalar_t)0.5 : (scalar_t)0);
    }

    if (cat == 2) {
        return d == -1 ? (scalar_t)-0.5 : (d == 0 ? (scalar_t)0.5 : (scalar_t)0);
    }

    return sshex8_tp_q(d);
}

static SFEM_INLINE scalar_t sshex8_tp_row_basis(
        const int basis, const int cx, const int cy, const int cz, const int dx, const int dy, const int dz) {
    switch (basis) {
        case 0: return sshex8_tp_row_k(cx, dx) * sshex8_tp_row_m(cy, dy) * sshex8_tp_row_m(cz, dz);
        case 1: return sshex8_tp_row_m(cx, dx) * sshex8_tp_row_k(cy, dy) * sshex8_tp_row_m(cz, dz);
        case 2: return sshex8_tp_row_m(cx, dx) * sshex8_tp_row_m(cy, dy) * sshex8_tp_row_k(cz, dz);
        case 3: return sshex8_tp_row_p(cx, dx) * sshex8_tp_row_q(cy, dy) * sshex8_tp_row_m(cz, dz);
        case 4: return sshex8_tp_row_p(cx, dx) * sshex8_tp_row_m(cy, dy) * sshex8_tp_row_q(cz, dz);
        default: return sshex8_tp_row_m(cx, dx) * sshex8_tp_row_p(cy, dy) * sshex8_tp_row_q(cz, dz);
    }
}

static SFEM_INLINE scalar_t sshex8_tp_row_pair_basis(
        const int p, const int q, const int cx, const int cy, const int cz, const int dx, const int dy, const int dz) {
    scalar_t ret = 1;
    const int d[3]   = {dx, dy, dz};
    const int cat[3] = {cx, cy, cz};

    for (int axis = 0; axis < 3; ++axis) {
        if (axis == p && axis == q) {
            ret *= sshex8_tp_row_k(cat[axis], d[axis]);
        } else if (axis == p) {
            ret *= sshex8_tp_row_p(cat[axis], d[axis]);
        } else if (axis == q) {
            ret *= sshex8_tp_row_q(cat[axis], d[axis]);
        } else {
            ret *= sshex8_tp_row_m(cat[axis], d[axis]);
        }
    }

    return ret;
}

static SFEM_INLINE void sshex8_tensor_terms_at_category(const int                           idx,
                                                        const int                           Lp1,
                                                        const int                           zstride,
                                                        const int                           cx,
                                                        const int                           cy,
                                                        const int                           cz,
                                                        const int                           dx_begin,
                                                        const int                           dx_end,
                                                        const int                           dy_begin,
                                                        const int                           dy_end,
                                                        const int                           dz_begin,
                                                        const int                           dz_end,
                                                        const scalar_t *const SFEM_RESTRICT u,
                                                        scalar_t *const SFEM_RESTRICT       t) {
    for (int i = 0; i < 9; ++i) {
        t[i] = 0;
    }

    for (int dz = dz_begin; dz <= dz_end; ++dz) {
        for (int dy = dy_begin; dy <= dy_end; ++dy) {
            const int base = idx + dy * Lp1 + dz * zstride;
            for (int dx = dx_begin; dx <= dx_end; ++dx) {
                const scalar_t ui = u[base + dx];

                for (int p = 0; p < 3; ++p) {
                    for (int q = 0; q < 3; ++q) {
                        t[p * 3 + q] += sshex8_tp_row_pair_basis(p, q, cx, cy, cz, dx, dy, dz) * ui;
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_tensor_coeffs_to_boundary_category_stencils(
        const scalar_t *const SFEM_RESTRICT tensor_coeffs,
        scalar_t *const SFEM_RESTRICT       category_stencils) {
    memset(category_stencils, 0, 27 * 27 * 9 * sizeof(scalar_t));

    for (int cz = 0; cz < 3; ++cz) {
        const int dz_begin = cz == 0 ? 0 : -1;
        const int dz_end   = cz == 2 ? 0 : 1;

        for (int cy = 0; cy < 3; ++cy) {
            const int dy_begin = cy == 0 ? 0 : -1;
            const int dy_end   = cy == 2 ? 0 : 1;

            for (int cx = 0; cx < 3; ++cx) {
                if (cx == 1 && cy == 1 && cz == 1) {
                    continue;
                }

                const int dx_begin = cx == 0 ? 0 : -1;
                const int dx_end   = cx == 2 ? 0 : 1;
                const int cat      = sshex8_category(cx, cy, cz);

                for (int dz = dz_begin; dz <= dz_end; ++dz) {
                    for (int dy = dy_begin; dy <= dy_end; ++dy) {
                        for (int dx = dx_begin; dx <= dx_end; ++dx) {
                            scalar_t basis[9];
                            for (int p = 0; p < 3; ++p) {
                                for (int q = 0; q < 3; ++q) {
                                    basis[p * 3 + q] = sshex8_tp_row_pair_basis(p, q, cx, cy, cz, dx, dy, dz);
                                }
                            }

                            scalar_t *const SFEM_RESTRICT block =
                                    &category_stencils[(cat * 27 + sshex8_offset(dx, dy, dz)) * 9];

                            for (int co = 0; co < 3; ++co) {
                                for (int ci = 0; ci < 3; ++ci) {
                                    const scalar_t *const SFEM_RESTRICT c = &tensor_coeffs[(co * 3 + ci) * 9];
                                    block[co * 3 + ci] = c[0] * basis[0] + c[1] * basis[1] + c[2] * basis[2] +
                                                         c[3] * basis[3] + c[4] * basis[4] + c[5] * basis[5] +
                                                         c[6] * basis[6] + c[7] * basis[7] + c[8] * basis[8];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_apply_interior_stencil(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT category_stencils,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    if (level <= 1) {
        return;
    }

    const int                     Lp1     = level + 1;
    const int                     zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT eu0     = eu[0];
    const scalar_t *const SFEM_RESTRICT eu1     = eu[1];
    const scalar_t *const SFEM_RESTRICT eu2     = eu[2];
    scalar_t *const SFEM_RESTRICT       v0      = v[0];
    scalar_t *const SFEM_RESTRICT       v1      = v[1];
    scalar_t *const SFEM_RESTRICT       v2      = v[2];
    const scalar_t *const SFEM_RESTRICT stencil = &category_stencils[sshex8_category(1, 1, 1) * 27 * 9];

#define SSHEX8_ELASTICITY_LOAD_BLOCK(name_, dx_, dy_, dz_)                                           \
    const int name_##_offset = (dx_) + (dy_) * Lp1 + (dz_) * zstride;                                \
    const scalar_t *const SFEM_RESTRICT name_##_block = &stencil[sshex8_offset(dx_, dy_, dz_) * 9];  \
    const scalar_t name_##_00 = name_##_block[0];                                                    \
    const scalar_t name_##_01 = name_##_block[1];                                                    \
    const scalar_t name_##_02 = name_##_block[2];                                                    \
    const scalar_t name_##_10 = name_##_block[3];                                                    \
    const scalar_t name_##_11 = name_##_block[4];                                                    \
    const scalar_t name_##_12 = name_##_block[5];                                                    \
    const scalar_t name_##_20 = name_##_block[6];                                                    \
    const scalar_t name_##_21 = name_##_block[7];                                                    \
    const scalar_t name_##_22 = name_##_block[8]

    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmm, -1, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmm, 0, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmm, 1, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcm, -1, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccm, 0, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcm, 1, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpm, -1, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpm, 0, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppm, 1, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmc, -1, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmc, 0, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmc, 1, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcc, -1, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccc, 0, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcc, 1, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpc, -1, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpc, 0, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppc, 1, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmp, -1, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmp, 0, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmp, 1, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcp, -1, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccp, 0, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcp, 1, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpp, -1, 1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpp, 0, 1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppp, 1, 1, 1);

#define SSHEX8_ELASTICITY_APPLY_BLOCK(name_)                    \
    do {                                                        \
        const int nidx = idx + name_##_offset;                  \
        const scalar_t u0 = eu0[nidx];                          \
        const scalar_t u1 = eu1[nidx];                          \
        const scalar_t u2 = eu2[nidx];                          \
        acc0 += name_##_00 * u0 + name_##_01 * u1 + name_##_02 * u2; \
        acc1 += name_##_10 * u0 + name_##_11 * u1 + name_##_12 * u2; \
        acc2 += name_##_20 * u0 + name_##_21 * u1 + name_##_22 * u2; \
    } while (0)

#define SSHEX8_ELASTICITY_APPLY_ALL(apply_) \
    apply_(bmmm);                           \
    apply_(bcmm);                           \
    apply_(bpmm);                           \
    apply_(bmcm);                           \
    apply_(bccm);                           \
    apply_(bpcm);                           \
    apply_(bmpm);                           \
    apply_(bcpm);                           \
    apply_(bppm);                           \
    apply_(bmmc);                           \
    apply_(bcmc);                           \
    apply_(bpmc);                           \
    apply_(bmcc);                           \
    apply_(bccc);                           \
    apply_(bpcc);                           \
    apply_(bmpc);                           \
    apply_(bcpc);                           \
    apply_(bppc);                           \
    apply_(bmmp);                           \
    apply_(bcmp);                           \
    apply_(bpmp);                           \
    apply_(bmcp);                           \
    apply_(bccp);                           \
    apply_(bpcp);                           \
    apply_(bmpp);                           \
    apply_(bcpp);                           \
    apply_(bppp)

    for (int zi = 1; zi < level; ++zi) {
        for (int yi = 1; yi < level; ++yi) {
            int idx = 1 + yi * Lp1 + zi * zstride;
            for (int xi = 1; xi < level; ++xi, ++idx) {
                scalar_t acc0 = 0, acc1 = 0, acc2 = 0;

                SSHEX8_ELASTICITY_APPLY_ALL(SSHEX8_ELASTICITY_APPLY_BLOCK);

                v0[idx] += acc0;
                v1[idx] += acc1;
                v2[idx] += acc2;
            }
        }
    }

#undef SSHEX8_ELASTICITY_APPLY_ALL
#undef SSHEX8_ELASTICITY_APPLY_BLOCK
#undef SSHEX8_ELASTICITY_LOAD_BLOCK
}

static void sshex8_linear_elasticity_apply_interior_stencil_global(
        const int                           level,
        const ptrdiff_t                     element,
        idx_t **const SFEM_RESTRICT         elements,
        const scalar_t *const SFEM_RESTRICT category_stencils,
        const ptrdiff_t                     u_stride,
        const real_t *const SFEM_RESTRICT   ux,
        const real_t *const SFEM_RESTRICT   uy,
        const real_t *const SFEM_RESTRICT   uz,
        const ptrdiff_t                     out_stride,
        real_t *const SFEM_RESTRICT         outx,
        real_t *const SFEM_RESTRICT         outy,
        real_t *const SFEM_RESTRICT         outz) {
    if (level <= 1) {
        return;
    }

    const int Lp1     = level + 1;
    const int zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT stencil = &category_stencils[sshex8_category(1, 1, 1) * 27 * 9];

#define SSHEX8_ELASTICITY_LOAD_BLOCK(name_, dx_, dy_, dz_)                                           \
    const int name_##_offset = (dx_) + (dy_) * Lp1 + (dz_) * zstride;                                \
    const scalar_t *const SFEM_RESTRICT name_##_block = &stencil[sshex8_offset(dx_, dy_, dz_) * 9];  \
    const scalar_t name_##_00 = name_##_block[0];                                                    \
    const scalar_t name_##_01 = name_##_block[1];                                                    \
    const scalar_t name_##_02 = name_##_block[2];                                                    \
    const scalar_t name_##_10 = name_##_block[3];                                                    \
    const scalar_t name_##_11 = name_##_block[4];                                                    \
    const scalar_t name_##_12 = name_##_block[5];                                                    \
    const scalar_t name_##_20 = name_##_block[6];                                                    \
    const scalar_t name_##_21 = name_##_block[7];                                                    \
    const scalar_t name_##_22 = name_##_block[8]

    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmm, -1, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmm, 0, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmm, 1, -1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcm, -1, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccm, 0, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcm, 1, 0, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpm, -1, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpm, 0, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppm, 1, 1, -1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmc, -1, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmc, 0, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmc, 1, -1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcc, -1, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccc, 0, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcc, 1, 0, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpc, -1, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpc, 0, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppc, 1, 1, 0);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmmp, -1, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcmp, 0, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpmp, 1, -1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmcp, -1, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bccp, 0, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bpcp, 1, 0, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bmpp, -1, 1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bcpp, 0, 1, 1);
    SSHEX8_ELASTICITY_LOAD_BLOCK(bppp, 1, 1, 1);

#define SSHEX8_ELASTICITY_APPLY_BLOCK(name_)                         \
    do {                                                             \
        const idx_t gid = elements[idx + name_##_offset][element];   \
        const ptrdiff_t uidx = gid * u_stride;                       \
        const scalar_t u0 = ux[uidx];                                \
        const scalar_t u1 = uy[uidx];                                \
        const scalar_t u2 = uz[uidx];                                \
        acc0 += name_##_00 * u0 + name_##_01 * u1 + name_##_02 * u2; \
        acc1 += name_##_10 * u0 + name_##_11 * u1 + name_##_12 * u2; \
        acc2 += name_##_20 * u0 + name_##_21 * u1 + name_##_22 * u2; \
    } while (0)

#define SSHEX8_ELASTICITY_APPLY_ALL(apply_) \
    apply_(bmmm);                           \
    apply_(bcmm);                           \
    apply_(bpmm);                           \
    apply_(bmcm);                           \
    apply_(bccm);                           \
    apply_(bpcm);                           \
    apply_(bmpm);                           \
    apply_(bcpm);                           \
    apply_(bppm);                           \
    apply_(bmmc);                           \
    apply_(bcmc);                           \
    apply_(bpmc);                           \
    apply_(bmcc);                           \
    apply_(bccc);                           \
    apply_(bpcc);                           \
    apply_(bmpc);                           \
    apply_(bcpc);                           \
    apply_(bppc);                           \
    apply_(bmmp);                           \
    apply_(bcmp);                           \
    apply_(bpmp);                           \
    apply_(bmcp);                           \
    apply_(bccp);                           \
    apply_(bpcp);                           \
    apply_(bmpp);                           \
    apply_(bcpp);                           \
    apply_(bppp)

    for (int zi = 1; zi < level; ++zi) {
        for (int yi = 1; yi < level; ++yi) {
            int idx = 1 + yi * Lp1 + zi * zstride;
            for (int xi = 1; xi < level; ++xi, ++idx) {
                scalar_t acc0 = 0, acc1 = 0, acc2 = 0;

                SSHEX8_ELASTICITY_APPLY_ALL(SSHEX8_ELASTICITY_APPLY_BLOCK);

                const idx_t gid = elements[idx][element];
                const ptrdiff_t outidx = gid * out_stride;
                outx[outidx] += acc0;
                outy[outidx] += acc1;
                outz[outidx] += acc2;
            }
        }
    }

#undef SSHEX8_ELASTICITY_APPLY_ALL
#undef SSHEX8_ELASTICITY_APPLY_BLOCK
#undef SSHEX8_ELASTICITY_LOAD_BLOCK
}

static void sshex8_linear_elasticity_apply_interior_tensor_stencil(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT tensor_coeffs,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    if (level <= 1) {
        return;
    }

    const int                     Lp1     = level + 1;
    const int                     zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT eu0 = eu[0];
    const scalar_t *const SFEM_RESTRICT eu1 = eu[1];
    const scalar_t *const SFEM_RESTRICT eu2 = eu[2];
    scalar_t *const SFEM_RESTRICT       v0  = v[0];
    scalar_t *const SFEM_RESTRICT       v1  = v[1];
    scalar_t *const SFEM_RESTRICT       v2  = v[2];

    for (int zi = 1; zi < level; ++zi) {
        for (int yi = 1; yi < level; ++yi) {
            int idx = 1 + yi * Lp1 + zi * zstride;
            for (int xi = 1; xi < level; ++xi, ++idx) {
                scalar_t acc0 = 0, acc1 = 0, acc2 = 0;
                scalar_t t[6];

#define SSHEX8_ELASTICITY_TENSOR_ACCUMULATE(ci_, input_)               \
    do {                                                               \
        sshex8_tensor_terms_at(idx, Lp1, zstride, input_, t);          \
        const scalar_t *const SFEM_RESTRICT c0 = &tensor_coeffs[(0 * 3 + (ci_)) * 9]; \
        const scalar_t *const SFEM_RESTRICT c1 = &tensor_coeffs[(1 * 3 + (ci_)) * 9]; \
        const scalar_t *const SFEM_RESTRICT c2 = &tensor_coeffs[(2 * 3 + (ci_)) * 9]; \
        acc0 += c0[0] * t[0] + c0[4] * t[1] + c0[8] * t[2] + (c0[1] + c0[3]) * t[3] + (c0[2] + c0[6]) * t[4] + (c0[5] + c0[7]) * t[5]; \
        acc1 += c1[0] * t[0] + c1[4] * t[1] + c1[8] * t[2] + (c1[1] + c1[3]) * t[3] + (c1[2] + c1[6]) * t[4] + (c1[5] + c1[7]) * t[5]; \
        acc2 += c2[0] * t[0] + c2[4] * t[1] + c2[8] * t[2] + (c2[1] + c2[3]) * t[3] + (c2[2] + c2[6]) * t[4] + (c2[5] + c2[7]) * t[5]; \
    } while (0)

                SSHEX8_ELASTICITY_TENSOR_ACCUMULATE(0, eu0);
                SSHEX8_ELASTICITY_TENSOR_ACCUMULATE(1, eu1);
                SSHEX8_ELASTICITY_TENSOR_ACCUMULATE(2, eu2);

#undef SSHEX8_ELASTICITY_TENSOR_ACCUMULATE

                v0[idx] += acc0;
                v1[idx] += acc1;
                v2[idx] += acc2;
            }
        }
    }
}

static void sshex8_linear_elasticity_apply_boundary_stencils(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT category_stencils,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    const int                     Lp1     = level + 1;
    const int                     zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT eu0     = eu[0];
    const scalar_t *const SFEM_RESTRICT eu1     = eu[1];
    const scalar_t *const SFEM_RESTRICT eu2     = eu[2];
    scalar_t *const SFEM_RESTRICT       v0      = v[0];
    scalar_t *const SFEM_RESTRICT       v1      = v[1];
    scalar_t *const SFEM_RESTRICT       v2      = v[2];

    for (int cz = 0; cz < 3; ++cz) {
        const int zi_begin = cz == 0 ? 0 : (cz == 1 ? 1 : level);
        const int zi_end   = cz == 0 ? 0 : (cz == 1 ? level - 1 : level);
        const int dz_begin = cz == 0 ? 0 : -1;
        const int dz_end   = cz == 2 ? 0 : 1;

        for (int cy = 0; cy < 3; ++cy) {
            const int yi_begin = cy == 0 ? 0 : (cy == 1 ? 1 : level);
            const int yi_end   = cy == 0 ? 0 : (cy == 1 ? level - 1 : level);
            const int dy_begin = cy == 0 ? 0 : -1;
            const int dy_end   = cy == 2 ? 0 : 1;

            for (int cx = 0; cx < 3; ++cx) {
                if (cx == 1 && cy == 1 && cz == 1) {
                    continue;
                }

                const int xi_begin = cx == 0 ? 0 : (cx == 1 ? 1 : level);
                const int xi_end   = cx == 0 ? 0 : (cx == 1 ? level - 1 : level);
                const int dx_begin = cx == 0 ? 0 : -1;
                const int dx_end   = cx == 2 ? 0 : 1;
                const int cat      = sshex8_category(cx, cy, cz);

                for (int zi = zi_begin; zi <= zi_end; ++zi) {
                    for (int yi = yi_begin; yi <= yi_end; ++yi) {
                        for (int xi = xi_begin; xi <= xi_end; ++xi) {
                            const int idx = xi + yi * Lp1 + zi * zstride;
                            scalar_t  acc0 = 0, acc1 = 0, acc2 = 0;

                            for (int dz = dz_begin; dz <= dz_end; ++dz) {
                                for (int dy = dy_begin; dy <= dy_end; ++dy) {
                                    for (int dx = dx_begin; dx <= dx_end; ++dx) {
                                        const int nidx = (xi + dx) + (yi + dy) * Lp1 + (zi + dz) * zstride;
                                        const scalar_t *const SFEM_RESTRICT block =
                                                &category_stencils[(cat * 27 + sshex8_offset(dx, dy, dz)) * 9];
                                        const scalar_t u0 = eu0[nidx];
                                        const scalar_t u1 = eu1[nidx];
                                        const scalar_t u2 = eu2[nidx];

                                        acc0 += block[0] * u0 + block[1] * u1 + block[2] * u2;
                                        acc1 += block[3] * u0 + block[4] * u1 + block[5] * u2;
                                        acc2 += block[6] * u0 + block[7] * u1 + block[8] * u2;
                                    }
                                }
                            }

                            v0[idx] += acc0;
                            v1[idx] += acc1;
                            v2[idx] += acc2;
                        }
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_zero_boundary(
        const int                      level,
        scalar_t **const SFEM_RESTRICT v) {
    const int               Lp1     = level + 1;
    const int               zstride = Lp1 * Lp1;
    scalar_t *const SFEM_RESTRICT v0 = v[0];
    scalar_t *const SFEM_RESTRICT v1 = v[1];
    scalar_t *const SFEM_RESTRICT v2 = v[2];

    for (int cz = 0; cz < 3; ++cz) {
        const int zi_begin = cz == 0 ? 0 : (cz == 1 ? 1 : level);
        const int zi_end   = cz == 0 ? 0 : (cz == 1 ? level - 1 : level);

        for (int cy = 0; cy < 3; ++cy) {
            const int yi_begin = cy == 0 ? 0 : (cy == 1 ? 1 : level);
            const int yi_end   = cy == 0 ? 0 : (cy == 1 ? level - 1 : level);

            for (int cx = 0; cx < 3; ++cx) {
                if (cx == 1 && cy == 1 && cz == 1) {
                    continue;
                }

                const int xi_begin = cx == 0 ? 0 : (cx == 1 ? 1 : level);
                const int xi_end   = cx == 0 ? 0 : (cx == 1 ? level - 1 : level);

                for (int zi = zi_begin; zi <= zi_end; ++zi) {
                    for (int yi = yi_begin; yi <= yi_end; ++yi) {
                        for (int xi = xi_begin; xi <= xi_end; ++xi) {
                            const int idx = xi + yi * Lp1 + zi * zstride;
                            v0[idx] = 0;
                            v1[idx] = 0;
                            v2[idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_scatter_boundary(
        const int                   level,
        const ptrdiff_t             element,
        idx_t **const SFEM_RESTRICT elements,
        scalar_t **const SFEM_RESTRICT v,
        const ptrdiff_t             out_stride,
        real_t *const SFEM_RESTRICT outx,
        real_t *const SFEM_RESTRICT outy,
        real_t *const SFEM_RESTRICT outz) {
    const int                     Lp1     = level + 1;
    const int                     zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT v0 = v[0];
    const scalar_t *const SFEM_RESTRICT v1 = v[1];
    const scalar_t *const SFEM_RESTRICT v2 = v[2];

    for (int cz = 0; cz < 3; ++cz) {
        const int zi_begin = cz == 0 ? 0 : (cz == 1 ? 1 : level);
        const int zi_end   = cz == 0 ? 0 : (cz == 1 ? level - 1 : level);

        for (int cy = 0; cy < 3; ++cy) {
            const int yi_begin = cy == 0 ? 0 : (cy == 1 ? 1 : level);
            const int yi_end   = cy == 0 ? 0 : (cy == 1 ? level - 1 : level);

            for (int cx = 0; cx < 3; ++cx) {
                if (cx == 1 && cy == 1 && cz == 1) {
                    continue;
                }

                const int xi_begin = cx == 0 ? 0 : (cx == 1 ? 1 : level);
                const int xi_end   = cx == 0 ? 0 : (cx == 1 ? level - 1 : level);

                for (int zi = zi_begin; zi <= zi_end; ++zi) {
                    for (int yi = yi_begin; yi <= yi_end; ++yi) {
                        for (int xi = xi_begin; xi <= xi_end; ++xi) {
                            const int       local_idx = xi + yi * Lp1 + zi * zstride;
                            const ptrdiff_t idx       = elements[local_idx][element] * out_stride;

#pragma omp atomic update
                            outx[idx] += v0[local_idx];

#pragma omp atomic update
                            outy[idx] += v1[local_idx];

#pragma omp atomic update
                            outz[idx] += v2[local_idx];
                        }
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_apply_boundary_tensor_stencils(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT tensor_coeffs,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    const int                     Lp1     = level + 1;
    const int                     zstride = Lp1 * Lp1;
    const scalar_t *const SFEM_RESTRICT eu0 = eu[0];
    const scalar_t *const SFEM_RESTRICT eu1 = eu[1];
    const scalar_t *const SFEM_RESTRICT eu2 = eu[2];
    scalar_t *const SFEM_RESTRICT       v0  = v[0];
    scalar_t *const SFEM_RESTRICT       v1  = v[1];
    scalar_t *const SFEM_RESTRICT       v2  = v[2];

    for (int cz = 0; cz < 3; ++cz) {
        const int zi_begin = cz == 0 ? 0 : (cz == 1 ? 1 : level);
        const int zi_end   = cz == 0 ? 0 : (cz == 1 ? level - 1 : level);
        const int dz_begin = cz == 0 ? 0 : -1;
        const int dz_end   = cz == 2 ? 0 : 1;

        for (int cy = 0; cy < 3; ++cy) {
            const int yi_begin = cy == 0 ? 0 : (cy == 1 ? 1 : level);
            const int yi_end   = cy == 0 ? 0 : (cy == 1 ? level - 1 : level);
            const int dy_begin = cy == 0 ? 0 : -1;
            const int dy_end   = cy == 2 ? 0 : 1;

            for (int cx = 0; cx < 3; ++cx) {
                if (cx == 1 && cy == 1 && cz == 1) {
                    continue;
                }

                const int xi_begin = cx == 0 ? 0 : (cx == 1 ? 1 : level);
                const int xi_end   = cx == 0 ? 0 : (cx == 1 ? level - 1 : level);
                const int dx_begin = cx == 0 ? 0 : -1;
                const int dx_end   = cx == 2 ? 0 : 1;

                for (int zi = zi_begin; zi <= zi_end; ++zi) {
                    for (int yi = yi_begin; yi <= yi_end; ++yi) {
                        for (int xi = xi_begin; xi <= xi_end; ++xi) {
                            const int idx = xi + yi * Lp1 + zi * zstride;
                            scalar_t  acc0 = 0, acc1 = 0, acc2 = 0;
                            scalar_t  t[9];

#define SSHEX8_ELASTICITY_TENSOR_BOUNDARY_ACCUMULATE(ci_, input_)                                    \
    do {                                                                                             \
        sshex8_tensor_terms_at_category(idx,                                                         \
                                        Lp1,                                                         \
                                        zstride,                                                     \
                                        cx,                                                          \
                                        cy,                                                          \
                                        cz,                                                          \
                                        dx_begin,                                                    \
                                        dx_end,                                                      \
                                        dy_begin,                                                    \
                                        dy_end,                                                      \
                                        dz_begin,                                                    \
                                        dz_end,                                                      \
                                        input_,                                                      \
                                        t);                                                          \
        const scalar_t *const SFEM_RESTRICT c0 = &tensor_coeffs[(0 * 3 + (ci_)) * 9];                \
        const scalar_t *const SFEM_RESTRICT c1 = &tensor_coeffs[(1 * 3 + (ci_)) * 9];                \
        const scalar_t *const SFEM_RESTRICT c2 = &tensor_coeffs[(2 * 3 + (ci_)) * 9];                \
        acc0 += c0[0] * t[0] + c0[1] * t[1] + c0[2] * t[2] + c0[3] * t[3] + c0[4] * t[4] + c0[5] * t[5] + c0[6] * t[6] + c0[7] * t[7] + c0[8] * t[8]; \
        acc1 += c1[0] * t[0] + c1[1] * t[1] + c1[2] * t[2] + c1[3] * t[3] + c1[4] * t[4] + c1[5] * t[5] + c1[6] * t[6] + c1[7] * t[7] + c1[8] * t[8]; \
        acc2 += c2[0] * t[0] + c2[1] * t[1] + c2[2] * t[2] + c2[3] * t[3] + c2[4] * t[4] + c2[5] * t[5] + c2[6] * t[6] + c2[7] * t[7] + c2[8] * t[8]; \
    } while (0)

                            SSHEX8_ELASTICITY_TENSOR_BOUNDARY_ACCUMULATE(0, eu0);
                            SSHEX8_ELASTICITY_TENSOR_BOUNDARY_ACCUMULATE(1, eu1);
                            SSHEX8_ELASTICITY_TENSOR_BOUNDARY_ACCUMULATE(2, eu2);

#undef SSHEX8_ELASTICITY_TENSOR_BOUNDARY_ACCUMULATE

                            v0[idx] += acc0;
                            v1[idx] += acc1;
                            v2[idx] += acc2;
                        }
                    }
                }
            }
        }
    }
}

static void sshex8_linear_elasticity_apply_category_stencils(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT category_stencils,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    sshex8_linear_elasticity_apply_interior_stencil(level, category_stencils, eu, v);
    sshex8_linear_elasticity_apply_boundary_stencils(level, category_stencils, eu, v);
}

static void sshex8_linear_elasticity_apply_tensor_category_stencils(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT category_stencils,
        const scalar_t *const SFEM_RESTRICT tensor_coeffs,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    sshex8_linear_elasticity_apply_interior_tensor_stencil(level, tensor_coeffs, eu, v);
    sshex8_linear_elasticity_apply_boundary_stencils(level, category_stencils, eu, v);
}

static void sshex8_linear_elasticity_apply_tensor_stencils(
        const int                           level,
        const scalar_t *const SFEM_RESTRICT tensor_coeffs,
        scalar_t **const SFEM_RESTRICT      eu,
        scalar_t **const SFEM_RESTRICT      v) {
    sshex8_linear_elasticity_apply_interior_tensor_stencil(level, tensor_coeffs, eu, v);
    sshex8_linear_elasticity_apply_boundary_tensor_stencils(level, tensor_coeffs, eu, v);
}

int sshex8_stencil_element_matrix_apply3_hyteg(const int                           level,
                                               const ptrdiff_t                     nelements,
                                               idx_t **const SFEM_RESTRICT         elements,
                                               const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                               const ptrdiff_t                     u_stride,
                                               const real_t *const SFEM_RESTRICT   ux,
                                               const real_t *const SFEM_RESTRICT   uy,
                                               const real_t *const SFEM_RESTRICT   uz,
                                               const ptrdiff_t                     out_stride,
                                               real_t *const SFEM_RESTRICT         outx,
                                               real_t *const SFEM_RESTRICT         outy,
                                               real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t    *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        scalar_t *category_stencils = (scalar_t *)malloc(27 * 27 * 9 * sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            for (int d = 0; d < 3; d++) {
                memset(v[d], 0, nxe * sizeof(scalar_t));
            }

            sshex8_linear_elasticity_matrix_to_category_stencils(&g_element_matrix[e * 24 * 24], category_stencils);
            sshex8_linear_elasticity_apply_category_stencils(level, category_stencils, eu, v);

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * out_stride;

#pragma omp atomic update
                outx[idx] += v[0][d];

#pragma omp atomic update
                outy[idx] += v[1][d];

#pragma omp atomic update
                outz[idx] += v[2][d];
            }
        }

        free(ev);
        free(category_stencils);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply3_hyteg_tensor(
        const int                           level,
        const ptrdiff_t                     nelements,
        idx_t **const SFEM_RESTRICT         elements,
        const scalar_t *const SFEM_RESTRICT g_tensor_coeffs,
        const ptrdiff_t                     u_stride,
        const real_t *const SFEM_RESTRICT   ux,
        const real_t *const SFEM_RESTRICT   uy,
        const real_t *const SFEM_RESTRICT   uz,
        const ptrdiff_t                     out_stride,
        real_t *const SFEM_RESTRICT         outx,
        real_t *const SFEM_RESTRICT         outy,
        real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        scalar_t *category_stencils = (scalar_t *)malloc(27 * 27 * 9 * sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            for (int d = 0; d < 3; d++) {
                memset(v[d], 0, nxe * sizeof(scalar_t));
            }

            const scalar_t *const SFEM_RESTRICT tensor_coeffs =
                    &g_tensor_coeffs[e * SSHEX8_LINEAR_ELASTICITY_TENSOR_COEFFS];
            sshex8_linear_elasticity_tensor_coeffs_to_boundary_category_stencils(tensor_coeffs, category_stencils);
            sshex8_linear_elasticity_apply_tensor_category_stencils(level, category_stencils, tensor_coeffs, eu, v);

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * out_stride;

#pragma omp atomic update
                outx[idx] += v[0][d];

#pragma omp atomic update
                outy[idx] += v[1][d];

#pragma omp atomic update
                outz[idx] += v[2][d];
            }
        }

        free(ev);
        free(category_stencils);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply3_hyteg_tensor_stencil(
        const int                           level,
        const ptrdiff_t                     nelements,
        idx_t **const SFEM_RESTRICT         elements,
        const scalar_t *const SFEM_RESTRICT g_category_stencils,
        const scalar_t *const SFEM_RESTRICT g_tensor_coeffs,
        const ptrdiff_t                     u_stride,
        const real_t *const SFEM_RESTRICT   ux,
        const real_t *const SFEM_RESTRICT   uy,
        const real_t *const SFEM_RESTRICT   uz,
        const ptrdiff_t                     out_stride,
        real_t *const SFEM_RESTRICT         outx,
        real_t *const SFEM_RESTRICT         outy,
        real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = (idx_t *)malloc(nxe * sizeof(idx_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            for (int d = 0; d < 3; d++) {
                memset(v[d], 0, nxe * sizeof(scalar_t));
            }

            sshex8_linear_elasticity_apply_tensor_category_stencils(level,
                                                                    &g_category_stencils[e * 27 * 27 * 9],
                                                                    &g_tensor_coeffs[e * SSHEX8_LINEAR_ELASTICITY_TENSOR_COEFFS],
                                                                    eu,
                                                                    v);

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * out_stride;

#pragma omp atomic update
                outx[idx] += v[0][d];

#pragma omp atomic update
                outy[idx] += v[1][d];

#pragma omp atomic update
                outz[idx] += v[2][d];
            }
        }

        free(ev);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply3_hyteg_stencil(
        const int                           level,
        const ptrdiff_t                     nelements,
        idx_t **const SFEM_RESTRICT         elements,
        const scalar_t *const SFEM_RESTRICT g_category_stencils,
        const ptrdiff_t                     u_stride,
        const real_t *const SFEM_RESTRICT   ux,
        const real_t *const SFEM_RESTRICT   uy,
        const real_t *const SFEM_RESTRICT   uz,
        const ptrdiff_t                     out_stride,
        real_t *const SFEM_RESTRICT         outx,
        real_t *const SFEM_RESTRICT         outy,
        real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = (idx_t *)malloc(nxe * sizeof(idx_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const scalar_t *const SFEM_RESTRICT category_stencils = &g_category_stencils[e * 27 * 27 * 9];

            sshex8_linear_elasticity_apply_interior_stencil_global(level,
                                                                   e,
                                                                   elements,
                                                                   category_stencils,
                                                                   u_stride,
                                                                   ux,
                                                                   uy,
                                                                   uz,
                                                                   out_stride,
                                                                   outx,
                                                                   outy,
                                                                   outz);

            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            sshex8_linear_elasticity_zero_boundary(level, v);
            sshex8_linear_elasticity_apply_boundary_stencils(level, category_stencils, eu, v);
            sshex8_linear_elasticity_scatter_boundary(level, e, elements, v, out_stride, outx, outy, outz);
        }

        free(ev);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}
