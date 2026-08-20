#include "sstet4_laplacian.hpp"

#include "hierarchy/sstet4/smesh_sstet4_transfer.impl.hpp"
#include "smesh_sstet4.hpp"
#include "tet4_inline_cpu.hpp"
#include "tet4_laplacian_inline_cpu.hpp"

#include <math.h>

#define POW3(x) ((x) * (x) * (x))

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

template <typename F>
static SFEM_INLINE void sstet4_for_each_microtet_fff(const int                             level,
                                                     const jacobian_t *const SFEM_RESTRICT macro_fff,
                                                     F &&                                 f) {
    int      ev[4];
    scalar_t fff[6];
    const int n = level + 1;

    if (level == 1) {
        sstet4_sub_fff_0(1, macro_fff, fff);
        ev[0] = 0;
        ev[1] = 1;
        ev[2] = 2;
        ev[3] = 3;
        f(ev, fff);
        return;
    }

    ///////////////////////////////////
    // Cat 0
    ///////////////////////////////////
    {
        sstet4_sub_fff_0(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            int layer_items = (n - i + 1) * (n - i) / 2;
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

    ///////////////////////////////////
    // Cat 1
    ///////////////////////////////////
    {
        sstet4_sub_fff_1(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            int layer_items = (n - i) * (n - i - 1) / 2;
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

    ///////////////////////////////////
    // Cat 2
    ///////////////////////////////////
    {
        sstet4_sub_fff_2(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            int layer_items = (n - i) * (n - i - 1) / 2;
            for (int j = 0; j < n - i - 1; j++) {
                p++;
                for (int k = 1; k < n - i - j - 1; k++) {
                    ev[0] = p;
                    ev[1] = p + n - i - j;
                    ev[3] = p + layer_items + n - i - j;
                    ev[2] = p + layer_items + n - i - j - 1 + n - i - j - 1;
                    f(ev, fff);
                    p++;
                }
                p++;
            }
            p++;
        }
    }

    ///////////////////////////////////
    // Cat 3
    ///////////////////////////////////
    {
        sstet4_sub_fff_3(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            int layer_items = (n - i) * (n - i - 1) / 2;
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

    ///////////////////////////////////
    // Cat 4
    ///////////////////////////////////
    {
        sstet4_sub_fff_4(level, macro_fff, fff);

        int p = 0;
        for (int i = 1; i < n - 1; i++) {
            p               = p + n - i + 1;
            int layer_items = (n - i) * (n - i - 1) / 2;
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

    ///////////////////////////////////
    // Cat 5
    ///////////////////////////////////
    {
        sstet4_sub_fff_5(level, macro_fff, fff);

        int p = 0;
        for (int i = 0; i < n - 1; i++) {
            int layer_items = (n - i) * (n - i - 1) / 2;
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

int sstet4_laplacian_apply(const int                             level,
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
