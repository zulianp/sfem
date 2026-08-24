#include "sstet4_laplacian.hpp"

#include "hierarchy/sstet4/smesh_sstet4_transfer.impl.hpp"
#include "smesh_sstet4.hpp"
#include "tet4_inline_cpu.hpp"
#include "tet4_laplacian_inline_cpu.hpp"

#include <math.h>
#include <stdint.h>

#define POW3(x) ((x) * (x) * (x))
#define SSTET4_LAPLACIAN_MAX_STENCIL 32
#define SSTET4_LAPLACIAN_MAX_STENCIL_TERMS 32

int sstet4_nxe(int level) {
    int num_nodes = 0;
    if (level % 2 == 0) {
        for (int i = 0; i < floor(level / 2); i++) {
            num_nodes += (level - i + 1) * (i + 1) * 2;
        }
        num_nodes += (level / 2 + 1) * (level / 2 + 1);
    } else {
        for (int i = 0; i < floor(level / 2) + 1; i++) {
            num_nodes += (level - i + 1) * (i + 1) * 2;
        }
    }

    return num_nodes;
}

int sstet4_txe(int level) { return (int)pow(level, 3); }

static SFEM_INLINE void sstet4_sub_fff_0(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1.0 / L;
    sub_fff[0]        = fff[0] * x0;
    sub_fff[1]        = fff[1] * x0;
    sub_fff[2]        = fff[2] * x0;
    sub_fff[3]        = fff[3] * x0;
    sub_fff[4]        = fff[4] * x0;
    sub_fff[5]        = fff[5] * x0;
}

static SFEM_INLINE void sstet4_sub_fff_1(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[3] - x2;
    const scalar_t x5 = L * fff[2];
    const scalar_t x6 = L * fff[4];
    const scalar_t x7 = (1 / POW2(L));
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x0 * (L * (-x5 - x6) + x3);
    sub_fff[2]        = x4 * x7;
    sub_fff[3]        = x0 * (L * (x1 + x5) + L * (L * fff[5] + x5));
    sub_fff[4]        = x7 * (x2 + x6);
    sub_fff[5]        = fff[3] / L;
}

static SFEM_INLINE void sstet4_sub_fff_2(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = L * x3;
    const scalar_t x5 = POW2(L);
    const scalar_t x6 = L * fff[2];
    sub_fff[0]        = x0 * (L * (L * fff[3] + x2) + x4);
    sub_fff[1]        = -x3 / x5;
    sub_fff[2]        = x0 * (L * (L * fff[4] + x6) + x4);
    sub_fff[3]        = fff[0] / L;
    sub_fff[4]        = x0 * (-fff[0] * x5 - fff[2] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[5] + x6));
}

static SFEM_INLINE void sstet4_sub_fff_3(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0  = 1. / POW3(L);
    const scalar_t x1  = L * fff[0];
    const scalar_t x2  = L * fff[2];
    const scalar_t x3  = x1 + x2;
    const scalar_t x4  = -L * x3;
    const scalar_t x5  = L * fff[5] + x2;
    const scalar_t x6  = -L * x5 + x4;
    const scalar_t x7  = L * fff[1];
    const scalar_t x8  = L * fff[4];
    const scalar_t x9  = x7 + x8;
    const scalar_t x10 = -L * x9;
    const scalar_t x11 = L * fff[3];
    const scalar_t x12 = L * (-x1 - x7) + L * (-x11 - x7);
    sub_fff[0]         = -x0 * x6;
    sub_fff[1]         = x0 * (-x10 - x4);
    sub_fff[2]         = x0 * (x10 + x6);
    sub_fff[3]         = -x0 * x12;
    sub_fff[4]         = x0 * (L * (-x2 - x8) + x12);
    sub_fff[5]         = x0 * (L * (x11 + x9) + L * (x3 + x7) + L * (x5 + x8));
}

static SFEM_INLINE void sstet4_sub_fff_4(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1.0 / L;
    const scalar_t x1 = 1. / POW3(L);
    const scalar_t x2 = POW2(L);
    const scalar_t x3 = L * fff[1];
    const scalar_t x4 = L * fff[2];
    const scalar_t x5 = L * fff[0] + x3 + x4;
    const scalar_t x6 = L * fff[4];
    sub_fff[0]        = fff[3] * x0;
    sub_fff[1]        = x1 * (-fff[1] * x2 - fff[3] * x2 - fff[4] * x2);
    sub_fff[2]        = fff[1] * x0;
    sub_fff[3]        = x1 * (L * x5 + L * (L * fff[3] + x3 + x6) + L * (L * fff[5] + x4 + x6));
    sub_fff[4]        = -x5 / x2;
    sub_fff[5]        = fff[0] * x0;
}

static SFEM_INLINE void sstet4_sub_fff_5(const scalar_t                        L,
                                         const jacobian_t *const SFEM_RESTRICT fff,
                                         scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[2];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[5] - x2;
    const scalar_t x5 = POW2(L);
    const scalar_t x6 = L * fff[1];
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x4 / x5;
    sub_fff[2]        = x0 * (L * (-L * fff[4] - x6) + x3);
    sub_fff[3]        = fff[5] / L;
    sub_fff[4]        = x0 * (fff[2] * x5 + fff[4] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[3] + x6));
}

static SFEM_INLINE void sstet4_macro_fff(const int                    level,
                                         idx_t **const SFEM_RESTRICT  elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const ptrdiff_t              e,
                                         jacobian_t *const            fff) {
    const idx_t ev0 = elements[smesh::sstet4_lidx(level, 0, 0, 0)][e];
    const idx_t ev1 = elements[smesh::sstet4_lidx(level, level, 0, 0)][e];
    const idx_t ev2 = elements[smesh::sstet4_lidx(level, 0, level, 0)][e];
    const idx_t ev3 = elements[smesh::sstet4_lidx(level, 0, 0, level)][e];

    tet4_fff(points[0][ev0],
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
             fff);
}

struct sstet4_laplacian_stencil {
    int       level;
    int       nrows;
    int       max_row_len;
    int       max_slot_terms;
    int       n_unique_stencils;
    ptrdiff_t nelements;
    int      *row_len;
    int      *cols;
    int      *term_count;
    uint8_t  *terms;
    int      *element_stencil;
    jacobian_t *unique_fff;
    scalar_t *weights;
};

static SFEM_INLINE int sstet4_laplacian_stencil_slot(const sstet4_laplacian_stencil *const stencil,
                                                     const int                              row,
                                                     const int                              col) {
    const int row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
    const int len        = stencil->row_len[row];
    for (int i = 0; i < len; ++i) {
        if (stencil->cols[row_offset + i] == col) {
            return i;
        }
    }

    return SFEM_IDX_INVALID;
}

static int sstet4_laplacian_stencil_add_term(sstet4_laplacian_stencil *const stencil,
                                             const int                         row,
                                             const int                         col,
                                             const int                         category,
                                             const int                         local_row,
                                             const int                         local_col) {
    int slot = sstet4_laplacian_stencil_slot(stencil, row, col);
    if (slot == SFEM_IDX_INVALID) {
        slot = stencil->row_len[row]++;
        if (slot >= SSTET4_LAPLACIAN_MAX_STENCIL) {
            return SFEM_FAILURE;
        }

        stencil->cols[row * SSTET4_LAPLACIAN_MAX_STENCIL + slot] = col;
        stencil->max_row_len =
                stencil->max_row_len > stencil->row_len[row] ? stencil->max_row_len : stencil->row_len[row];
    }

    const int entry       = row * SSTET4_LAPLACIAN_MAX_STENCIL + slot;
    const int term_offset = entry * SSTET4_LAPLACIAN_MAX_STENCIL_TERMS;
    const int term        = stencil->term_count[entry]++;
    if (term >= SSTET4_LAPLACIAN_MAX_STENCIL_TERMS) {
        return SFEM_FAILURE;
    }

    stencil->terms[term_offset + term] = (uint8_t)(category * 16 + local_row * 4 + local_col);
    stencil->max_slot_terms =
            stencil->max_slot_terms > stencil->term_count[entry] ? stencil->max_slot_terms : stencil->term_count[entry];

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_stencil_add_edge(sstet4_laplacian_stencil *const stencil,
                                             const int                         row,
                                             const int                         col) {
    int slot = sstet4_laplacian_stencil_slot(stencil, row, col);
    if (slot != SFEM_IDX_INVALID) {
        return SFEM_SUCCESS;
    }

    slot = stencil->row_len[row]++;
    if (slot >= SSTET4_LAPLACIAN_MAX_STENCIL) {
        return SFEM_FAILURE;
    }

    stencil->cols[row * SSTET4_LAPLACIAN_MAX_STENCIL + slot] = col;
    stencil->max_row_len =
            stencil->max_row_len > stencil->row_len[row] ? stencil->max_row_len : stencil->row_len[row];

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_stencil_add_microtet(sstet4_laplacian_stencil *const stencil,
                                                const int                         category,
                                                const int *const                  ev) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (sstet4_laplacian_stencil_add_term(stencil, ev[i], ev[j], category, i, j) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }
    }

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_stencil_add_microtet_topology(sstet4_laplacian_stencil *const stencil,
                                                          const int *const                  ev) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (sstet4_laplacian_stencil_add_edge(stencil, ev[i], ev[j]) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
        }
    }

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_build_stencil_topology(sstet4_laplacian_stencil *const stencil) {
    const int level = stencil->level;
    int       ev[4];

    if (level == 1) {
        ev[0] = smesh::sstet4_lidx(1, 0, 0, 0);
        ev[1] = smesh::sstet4_lidx(1, 1, 0, 0);
        ev[2] = smesh::sstet4_lidx(1, 0, 1, 0);
        ev[3] = smesh::sstet4_lidx(1, 0, 0, 1);
        return sstet4_laplacian_stencil_add_microtet(stencil, 0, ev);
    }

    const int n = level + 1;

    {
        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i + 1) * (n - i) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                for (int k = 0; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + 1;
                    ev[2] = p + n - i - j;
                    ev[3] = p + layer_items - j;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 0, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + layer_items + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j;
                    ev[3] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 1, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j;
                    ev[2] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    ev[3] = p + layer_items + n - i - j;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 2, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j - 1;
                    ev[3] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 3, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        int p = 0;
        for (int i = 1; i < n - 1; i++) {
            p += n - i + 1;
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + layer_items + n - i;
                    ev[2] = p + layer_items + n - i - j + n - i;
                    ev[3] = p + layer_items + n - i - j + n - i - 1;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 4, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    ev[3] = p + n - i - j;
                    if (sstet4_laplacian_stencil_add_microtet(stencil, 5, ev) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_build_transfer_topology(sstet4_laplacian_stencil *const stencil) {
    int err = SFEM_SUCCESS;
    smesh::sstet4_transfer::for_each_microtet(stencil->level, [&](const int *const ev) {
        if (err == SFEM_SUCCESS &&
            sstet4_laplacian_stencil_add_microtet_topology(stencil, ev) != SFEM_SUCCESS) {
            err = SFEM_FAILURE;
        }
    });

    return err;
}

static SFEM_INLINE void sstet4_laplacian_category_matrices(const int                             level,
                                                           const jacobian_t *const SFEM_RESTRICT macro_fff,
                                                           scalar_t *const SFEM_RESTRICT         matrices) {
    scalar_t fff[6];

    sstet4_sub_fff_0(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[0 * 16]);

    sstet4_sub_fff_1(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[1 * 16]);

    sstet4_sub_fff_2(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[2 * 16]);

    sstet4_sub_fff_3(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[3 * 16]);

    sstet4_sub_fff_4(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[4 * 16]);

    sstet4_sub_fff_5(level, macro_fff, fff);
    tet4_laplacian_hessian_fff(fff, &matrices[5 * 16]);
}

static void sstet4_laplacian_stencil_weights(const sstet4_laplacian_stencil *const stencil,
                                             const jacobian_t *const SFEM_RESTRICT  macro_fff,
                                             scalar_t *const SFEM_RESTRICT          weights) {
    scalar_t matrices[6 * 16];
    sstet4_laplacian_category_matrices(stencil->level, macro_fff, matrices);

    for (int row = 0; row < stencil->nrows; ++row) {
        const int row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
        const int row_len    = stencil->row_len[row];
        for (int s = 0; s < row_len; ++s) {
            const int entry       = row_offset + s;
            const int term_offset = entry * SSTET4_LAPLACIAN_MAX_STENCIL_TERMS;
            const int nterms      = stencil->term_count[entry];
            scalar_t  weight      = 0;
            for (int t = 0; t < nterms; ++t) {
                weight += matrices[stencil->terms[term_offset + t]];
            }

            weights[entry] = weight;
        }
    }
}

static bool sstet4_laplacian_same_macro_fff(const jacobian_t *const SFEM_RESTRICT a,
                                            const jacobian_t *const SFEM_RESTRICT b) {
    for (int i = 0; i < 6; ++i) {
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

static int sstet4_laplacian_stencil_reserve_unique(sstet4_laplacian_stencil *const stencil,
                                                   int *const                              capacity,
                                                   const int                               required) {
    if (required <= *capacity) {
        return SFEM_SUCCESS;
    }

    int new_capacity = *capacity > 0 ? *capacity : 4;
    while (new_capacity < required) {
        new_capacity *= 2;
    }

    const int stencil_size = stencil->nrows * SSTET4_LAPLACIAN_MAX_STENCIL;
    jacobian_t *const unique_fff =
            (jacobian_t *)realloc(stencil->unique_fff, new_capacity * 6 * sizeof(jacobian_t));
    if (!unique_fff) {
        return SFEM_FAILURE;
    }

    stencil->unique_fff = unique_fff;

    scalar_t *const weights = (scalar_t *)realloc(stencil->weights, new_capacity * stencil_size * sizeof(scalar_t));
    if (!weights) {
        return SFEM_FAILURE;
    }

    stencil->weights = weights;
    *capacity        = new_capacity;

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_build_weighted_stencils(sstet4_laplacian_stencil *const stencil,
                                                    const jacobian_t *const SFEM_RESTRICT  g_fff) {
    int capacity = 0;
    const int stencil_size = stencil->nrows * SSTET4_LAPLACIAN_MAX_STENCIL;

    for (ptrdiff_t e = 0; e < stencil->nelements; ++e) {
        const jacobian_t *const macro_fff = &g_fff[e * 6];
        int                     stencil_id = SFEM_IDX_INVALID;

        for (int s = 0; s < stencil->n_unique_stencils; ++s) {
            if (sstet4_laplacian_same_macro_fff(macro_fff, &stencil->unique_fff[s * 6])) {
                stencil_id = s;
                break;
            }
        }

        if (stencil_id == SFEM_IDX_INVALID) {
            if (sstet4_laplacian_stencil_reserve_unique(stencil, &capacity, stencil->n_unique_stencils + 1) !=
                SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            stencil_id = stencil->n_unique_stencils++;
            for (int i = 0; i < 6; ++i) {
                stencil->unique_fff[stencil_id * 6 + i] = macro_fff[i];
            }

            scalar_t *const weights = &stencil->weights[stencil_id * stencil_size];
            for (int i = 0; i < stencil_size; ++i) {
                weights[i] = 0;
            }

            sstet4_laplacian_stencil_weights(stencil, macro_fff, weights);
        }

        stencil->element_stencil[e] = stencil_id;
    }

    return SFEM_SUCCESS;
}

static int sstet4_laplacian_build_point_weighted_stencils(sstet4_laplacian_stencil *const stencil,
                                                          idx_t **const SFEM_RESTRICT       elements,
                                                          geom_t **const SFEM_RESTRICT      points) {
    int       capacity     = 0;
    const int stencil_size = stencil->nrows * SSTET4_LAPLACIAN_MAX_STENCIL;

    for (ptrdiff_t e = 0; e < stencil->nelements; ++e) {
        jacobian_t macro_fff[6];
        sstet4_macro_fff(stencil->level, elements, points, e, macro_fff);

        int stencil_id = SFEM_IDX_INVALID;
        for (int s = 0; s < stencil->n_unique_stencils; ++s) {
            if (sstet4_laplacian_same_macro_fff(macro_fff, &stencil->unique_fff[s * 6])) {
                stencil_id = s;
                break;
            }
        }

        if (stencil_id == SFEM_IDX_INVALID) {
            if (sstet4_laplacian_stencil_reserve_unique(stencil, &capacity, stencil->n_unique_stencils + 1) !=
                SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            stencil_id = stencil->n_unique_stencils++;
            for (int i = 0; i < 6; ++i) {
                stencil->unique_fff[stencil_id * 6 + i] = macro_fff[i];
            }

            scalar_t *const weights = &stencil->weights[stencil_id * stencil_size];
            for (int i = 0; i < stencil_size; ++i) {
                weights[i] = 0;
            }

            int err = SFEM_SUCCESS;
            smesh::sstet4_transfer::for_each_microtet(stencil->level, [&](const int *const lev) {
                if (err != SFEM_SUCCESS) {
                    return;
                }

                const idx_t gv[4] = {elements[lev[0]][e], elements[lev[1]][e], elements[lev[2]][e], elements[lev[3]][e]};
                scalar_t    fff[6];
                scalar_t    element_matrix[16];
                tet4_fff_s(points[0][gv[0]],
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
                           fff);
                tet4_laplacian_hessian_fff(fff, element_matrix);

                for (int i = 0; i < 4; ++i) {
                    const int row        = lev[i];
                    const int row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
                    for (int j = 0; j < 4; ++j) {
                        const int slot = sstet4_laplacian_stencil_slot(stencil, row, lev[j]);
                        if (slot == SFEM_IDX_INVALID) {
                            err = SFEM_FAILURE;
                            return;
                        }

                        weights[row_offset + slot] += element_matrix[i * 4 + j];
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

static void sstet4_laplacian_stencil_free_members(sstet4_laplacian_stencil *const stencil) {
    if (!stencil) {
        return;
    }

    free(stencil->row_len);
    free(stencil->cols);
    free(stencil->term_count);
    free(stencil->terms);
    free(stencil->element_stencil);
    free(stencil->unique_fff);
    free(stencil->weights);

    stencil->row_len         = nullptr;
    stencil->cols            = nullptr;
    stencil->term_count      = nullptr;
    stencil->terms           = nullptr;
    stencil->element_stencil = nullptr;
    stencil->unique_fff      = nullptr;
    stencil->weights         = nullptr;
}

template <typename F>
static SFEM_INLINE void sstet4_for_each_microtet_fff(const int                             level,
                                                     const jacobian_t *const SFEM_RESTRICT macro_fff,
                                                     F &&                                 f) {
    int      ev[4];
    scalar_t fff[6];

    if (level == 1) {
        sstet4_sub_fff_0(1, macro_fff, fff);
        ev[0] = smesh::sstet4_lidx(1, 0, 0, 0);
        ev[1] = smesh::sstet4_lidx(1, 1, 0, 0);
        ev[2] = smesh::sstet4_lidx(1, 0, 1, 0);
        ev[3] = smesh::sstet4_lidx(1, 0, 0, 1);
        f(ev, fff);
        return;
    }

    const int n = level + 1;

    {
        sstet4_sub_fff_0(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i + 1) * (n - i) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                for (int k = 0; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + 1;
                    ev[2] = p + n - i - j;
                    ev[3] = p + layer_items - j;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        sstet4_sub_fff_1(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + layer_items + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j;
                    ev[3] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        sstet4_sub_fff_2(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j;
                    ev[2] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    ev[3] = p + layer_items + n - i - j;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        sstet4_sub_fff_3(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j - 1;
                    ev[3] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        sstet4_sub_fff_4(level, macro_fff, fff);

        int p = 0;
        for (int i = 1; i < n - 1; i++) {
            p += n - i + 1;
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + layer_items + n - i;
                    ev[2] = p + layer_items + n - i - j + n - i;
                    ev[3] = p + layer_items + n - i - j + n - i - 1;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    {
        sstet4_sub_fff_5(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            const int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j - 1;
                    ev[2] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    ev[3] = p + n - i - j;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

int sstet4_laplacian_apply_points(const int                         level,
                                  const ptrdiff_t                   nelements,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        smesh::sstet4_transfer::for_each_microtet(level, [&](const int *const lev) {
            idx_t gv[4] = {elements[lev[0]][e], elements[lev[1]][e], elements[lev[2]][e], elements[lev[3]][e]};
            scalar_t fff[6];
            tet4_fff_s(points[0][gv[0]],
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
                       fff);
            accumulator_t v[4];
            tet4_laplacian_apply_fff(fff, u[gv[0]], u[gv[1]], u[gv[2]], u[gv[3]], &v[0], &v[1], &v[2], &v[3]);
            for (int d = 0; d < 4; ++d) {
#pragma omp atomic update
                values[gv[d]] += v[d];
            }
        });
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_diag_points(const int                    level,
                                 const ptrdiff_t              nelements,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        smesh::sstet4_transfer::for_each_microtet(level, [&](const int *const lev) {
            idx_t gv[4] = {elements[lev[0]][e], elements[lev[1]][e], elements[lev[2]][e], elements[lev[3]][e]};
            scalar_t fff[6];
            tet4_fff_s(points[0][gv[0]],
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
                       fff);
            accumulator_t v[4];
            tet4_laplacian_diag_fff(fff, &v[0], &v[1], &v[2], &v[3]);
            for (int d = 0; d < 4; ++d) {
#pragma omp atomic update
                values[gv[d]] += v[d];
            }
        });
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_apply_elementwise(const int                             level,
                                       const ptrdiff_t                       nelements,
                                       const jacobian_t *const SFEM_RESTRICT g_fff,
                                       const real_t *const SFEM_RESTRICT     u,
                                       real_t *const SFEM_RESTRICT           values) {
    const int nxe = sstet4_nxe(level);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const real_t *const element_u      = &u[e * nxe];
        real_t *const       element_vector = &values[e * nxe];

        sstet4_for_each_microtet_fff(level, &g_fff[e * 6], [&](const int *const ev, const scalar_t *const fff) {
            accumulator_t v[4];
            tet4_laplacian_apply_fff(fff,
                                     element_u[ev[0]],
                                     element_u[ev[1]],
                                     element_u[ev[2]],
                                     element_u[ev[3]],
                                     &v[0],
                                     &v[1],
                                     &v[2],
                                     &v[3]);

            for (int d = 0; d < 4; ++d) {
                element_vector[ev[d]] += v[d];
            }
        });
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_apply(const int                             level,
                           const ptrdiff_t                       nelements,
                           const jacobian_t *const SFEM_RESTRICT g_fff,
                           const real_t *const SFEM_RESTRICT     u,
                           real_t *const SFEM_RESTRICT           values) {
    sstet4_laplacian_stencil_t *stencil = nullptr;
    int err = sstet4_laplacian_stencil_create(level, nelements, g_fff, &stencil);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    err = sstet4_laplacian_apply_stencil(stencil, nelements, u, values);
    sstet4_laplacian_stencil_destroy(stencil);

    return err;
}

int sstet4_laplacian_stencil_create(const int                             level,
                                    const ptrdiff_t                       nelements,
                                    const jacobian_t *const SFEM_RESTRICT g_fff,
                                    sstet4_laplacian_stencil_t **const    stencil_out) {
    if (!stencil_out || !g_fff || level < 1 || nelements < 0) {
        return SFEM_FAILURE;
    }

    *stencil_out = nullptr;

    sstet4_laplacian_stencil *const stencil =
            (sstet4_laplacian_stencil *)calloc(1, sizeof(sstet4_laplacian_stencil));
    if (!stencil) {
        return SFEM_FAILURE;
    }

    stencil->level     = level;
    stencil->nrows     = sstet4_nxe(level);
    stencil->nelements = nelements;

    const int nrows          = stencil->nrows;
    const int stencil_size   = nrows * SSTET4_LAPLACIAN_MAX_STENCIL;
    const int topology_terms = stencil_size * SSTET4_LAPLACIAN_MAX_STENCIL_TERMS;

    stencil->row_len         = (int *)calloc(nrows, sizeof(int));
    stencil->cols            = (int *)malloc(stencil_size * sizeof(int));
    stencil->term_count      = (int *)calloc(stencil_size, sizeof(int));
    stencil->terms           = (uint8_t *)calloc(topology_terms, sizeof(uint8_t));
    stencil->element_stencil = (int *)malloc((nelements > 0 ? nelements : 1) * sizeof(int));

    if (!stencil->row_len || !stencil->cols || !stencil->term_count || !stencil->terms || !stencil->element_stencil) {
        sstet4_laplacian_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    for (int i = 0; i < stencil_size; ++i) {
        stencil->cols[i] = SFEM_IDX_INVALID;
    }

    if (sstet4_laplacian_build_stencil_topology(stencil) != SFEM_SUCCESS ||
        sstet4_laplacian_build_weighted_stencils(stencil, g_fff) != SFEM_SUCCESS) {
        sstet4_laplacian_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    *stencil_out = stencil;
    return SFEM_SUCCESS;
}

int sstet4_laplacian_stencil_create_from_points(const int                         level,
                                                const ptrdiff_t                   nelements,
                                                idx_t **const SFEM_RESTRICT       elements,
                                                geom_t **const SFEM_RESTRICT      points,
                                                sstet4_laplacian_stencil_t **const stencil_out) {
    if (!elements || !points || !stencil_out || level < 1 || nelements < 0) {
        return SFEM_FAILURE;
    }

    *stencil_out = nullptr;

    sstet4_laplacian_stencil *const stencil =
            (sstet4_laplacian_stencil *)calloc(1, sizeof(sstet4_laplacian_stencil));
    if (!stencil) {
        return SFEM_FAILURE;
    }

    stencil->level     = level;
    stencil->nrows     = sstet4_nxe(level);
    stencil->nelements = nelements;

    const int nrows          = stencil->nrows;
    const int stencil_size   = nrows * SSTET4_LAPLACIAN_MAX_STENCIL;
    const int topology_terms = stencil_size * SSTET4_LAPLACIAN_MAX_STENCIL_TERMS;

    stencil->row_len         = (int *)calloc(nrows, sizeof(int));
    stencil->cols            = (int *)malloc(stencil_size * sizeof(int));
    stencil->term_count      = (int *)calloc(stencil_size, sizeof(int));
    stencil->terms           = (uint8_t *)calloc(topology_terms, sizeof(uint8_t));
    stencil->element_stencil = (int *)malloc((nelements > 0 ? nelements : 1) * sizeof(int));

    if (!stencil->row_len || !stencil->cols || !stencil->term_count || !stencil->terms || !stencil->element_stencil) {
        sstet4_laplacian_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    for (int i = 0; i < stencil_size; ++i) {
        stencil->cols[i] = SFEM_IDX_INVALID;
    }

    if (sstet4_laplacian_build_transfer_topology(stencil) != SFEM_SUCCESS ||
        sstet4_laplacian_build_point_weighted_stencils(stencil, elements, points) != SFEM_SUCCESS) {
        sstet4_laplacian_stencil_destroy(stencil);
        return SFEM_FAILURE;
    }

    *stencil_out = stencil;
    return SFEM_SUCCESS;
}

void sstet4_laplacian_stencil_destroy(sstet4_laplacian_stencil_t *stencil) {
    if (!stencil) {
        return;
    }

    sstet4_laplacian_stencil_free_members(stencil);
    free(stencil);
}

int sstet4_laplacian_stencil_nrows(const sstet4_laplacian_stencil_t *const stencil) {
    return stencil ? stencil->nrows : 0;
}

int sstet4_laplacian_stencil_max_row_len(const sstet4_laplacian_stencil_t *const stencil) {
    return stencil ? stencil->max_row_len : 0;
}

int sstet4_laplacian_stencil_max_slot_terms(const sstet4_laplacian_stencil_t *const stencil) {
    return stencil ? stencil->max_slot_terms : 0;
}

int sstet4_laplacian_stencil_n_unique_stencils(const sstet4_laplacian_stencil_t *const stencil) {
    return stencil ? stencil->n_unique_stencils : 0;
}

int sstet4_laplacian_apply_stencil(const sstet4_laplacian_stencil_t *const stencil,
                                   const ptrdiff_t                         nelements,
                                   const real_t *const SFEM_RESTRICT       u,
                                   real_t *const SFEM_RESTRICT             values) {
    if (!stencil || !u || !values || nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe          = stencil->nrows;
    const int stencil_size = nxe * SSTET4_LAPLACIAN_MAX_STENCIL;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const real_t *const  element_u       = &u[e * nxe];
        real_t *const        element_vector  = &values[e * nxe];
        const scalar_t *const stencil_weights = &stencil->weights[stencil->element_stencil[e] * stencil_size];

        for (int row = 0; row < nxe; ++row) {
            accumulator_t acc        = 0;
            const int     row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
            const int     row_len    = stencil->row_len[row];
            for (int s = 0; s < row_len; ++s) {
                const int entry = row_offset + s;
                acc += stencil_weights[entry] * element_u[stencil->cols[entry]];
            }

            element_vector[row] += acc;
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_apply_stencil_vectorized(const sstet4_laplacian_stencil_t *const stencil,
                                              const ptrdiff_t                         nelements,
                                              const real_t *const SFEM_RESTRICT       u,
                                              real_t *const SFEM_RESTRICT             values) {
    if (!stencil || !u || !values || nelements != stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe          = stencil->nrows;
    const int stencil_size = nxe * SSTET4_LAPLACIAN_MAX_STENCIL;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const real_t *const SFEM_RESTRICT  element_u       = &u[e * nxe];
        real_t *const SFEM_RESTRICT        element_vector  = &values[e * nxe];
        const scalar_t *const SFEM_RESTRICT stencil_weights = &stencil->weights[stencil->element_stencil[e] * stencil_size];

        for (int row = 0; row < nxe; ++row) {
            accumulator_t acc        = 0;
            const int     row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
            const int     row_len    = stencil->row_len[row];

#pragma omp simd reduction(+ : acc)
            for (int s = 0; s < row_len; ++s) {
                const int entry = row_offset + s;
                acc += stencil_weights[entry] * element_u[stencil->cols[entry]];
            }

            element_vector[row] += acc;
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_apply_stencil_global(const sstet4_laplacian_stencil_t *const stencil,
                                          const ptrdiff_t                         nelements,
                                          idx_t **const SFEM_RESTRICT             elements,
                                          const real_t *const SFEM_RESTRICT       u,
                                          real_t *const SFEM_RESTRICT             values) {
    return sstet4_laplacian_apply_stencil_global_range(stencil, 0, nelements, elements, u, values);
}

int sstet4_laplacian_apply_stencil_global_vectorized(const sstet4_laplacian_stencil_t *const stencil,
                                                     const ptrdiff_t                         nelements,
                                                     idx_t **const SFEM_RESTRICT             elements,
                                                     const real_t *const SFEM_RESTRICT       u,
                                                     real_t *const SFEM_RESTRICT             values) {
    return sstet4_laplacian_apply_stencil_global_range_vectorized(stencil, 0, nelements, elements, u, values);
}

int sstet4_laplacian_apply_stencil_global_range(const sstet4_laplacian_stencil_t *const stencil,
                                                const ptrdiff_t                         element_offset,
                                                const ptrdiff_t                         nelements,
                                                idx_t **const SFEM_RESTRICT             elements,
                                                const real_t *const SFEM_RESTRICT       u,
                                                real_t *const SFEM_RESTRICT             values) {
    if (!stencil || !elements || !u || !values || element_offset < 0 || nelements < 0 ||
        element_offset + nelements > stencil->nelements) {
        return SFEM_FAILURE;
    }

    const int nxe          = stencil->nrows;
    const int stencil_size = nxe * SSTET4_LAPLACIAN_MAX_STENCIL;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const scalar_t *const stencil_weights =
                &stencil->weights[stencil->element_stencil[element_offset + e] * stencil_size];

        for (int row = 0; row < nxe; ++row) {
            accumulator_t acc        = 0;
            const int     row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
            const int     row_len    = stencil->row_len[row];
            for (int s = 0; s < row_len; ++s) {
                const int entry = row_offset + s;
                acc += stencil_weights[entry] * u[elements[stencil->cols[entry]][e]];
            }

#pragma omp atomic update
            values[elements[row][e]] += acc;
        }
    }

    return SFEM_SUCCESS;
}

int sstet4_laplacian_apply_stencil_global_range_vectorized(const sstet4_laplacian_stencil_t *const stencil,
                                                           const ptrdiff_t                         element_offset,
                                                           const ptrdiff_t                         nelements,
                                                           idx_t **const SFEM_RESTRICT             elements,
                                                           const real_t *const SFEM_RESTRICT       u,
                                                           real_t *const SFEM_RESTRICT             values) {
    if (!stencil || !elements || !u || !values || element_offset < 0 || nelements < 0 ||
        element_offset + nelements > stencil->nelements) {
        return SFEM_FAILURE;
    }

    static const int VECTOR_SIZE = 16;

    const int nxe          = stencil->nrows;
    const int stencil_size = nxe * SSTET4_LAPLACIAN_MAX_STENCIL;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)((nelements - evbegin) < VECTOR_SIZE ? (nelements - evbegin) : VECTOR_SIZE);

        int stencil_id[VECTOR_SIZE];

        const int first_stencil_id = stencil->element_stencil[element_offset + evbegin];
        int       same_stencil     = 1;

#pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            stencil_id[lane] = stencil->element_stencil[element_offset + evbegin + lane];
        }

        for (int lane = 0; lane < nelems; ++lane) {
            same_stencil &= stencil_id[lane] == first_stencil_id;
        }

        if (same_stencil) {
            const scalar_t *const SFEM_RESTRICT stencil_weights = &stencil->weights[first_stencil_id * stencil_size];

            for (int row = 0; row < nxe; ++row) {
                accumulator_t acc[VECTOR_SIZE];
                const int     row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
                const int     row_len    = stencil->row_len[row];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    acc[lane] = 0;
                }

                for (int s = 0; s < row_len; ++s) {
                    const int          entry       = row_offset + s;
                    const scalar_t     w           = stencil_weights[entry];
                    const idx_t *const element_col = elements[stencil->cols[entry]];

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        acc[lane] += w * u[element_col[evbegin + lane]];
                    }
                }

                const idx_t *const element_row = elements[row];
                for (int lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
                    values[element_row[evbegin + lane]] += acc[lane];
                }
            }
        } else {
            for (int row = 0; row < nxe; ++row) {
                accumulator_t acc[VECTOR_SIZE];
                const int     row_offset = row * SSTET4_LAPLACIAN_MAX_STENCIL;
                const int     row_len    = stencil->row_len[row];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    acc[lane] = 0;
                }

                for (int s = 0; s < row_len; ++s) {
                    const int          entry       = row_offset + s;
                    const idx_t *const element_col = elements[stencil->cols[entry]];

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        acc[lane] += stencil->weights[stencil_id[lane] * stencil_size + entry] *
                                     u[element_col[evbegin + lane]];
                    }
                }

                const idx_t *const element_row = elements[row];
                for (int lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
                    values[element_row[evbegin + lane]] += acc[lane];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
