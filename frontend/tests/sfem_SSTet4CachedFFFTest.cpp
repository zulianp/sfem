#include <math.h>
#include <stdlib.h>

#include <memory>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"

#include "smesh_sstet4.hpp"
#include "sstet4_laplacian.hpp"
#include "tet4_inline_cpu.hpp"
#include "tet4_laplacian_inline_cpu.hpp"

#define TEST_POW2(x) ((x) * (x))
#define TEST_POW3(x) ((x) * (x) * (x))

static SFEM_INLINE void test_sstet4_sub_fff_0(const scalar_t                        L,
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

static SFEM_INLINE void test_sstet4_sub_fff_1(const scalar_t                        L,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / TEST_POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[3] - x2;
    const scalar_t x5 = L * fff[2];
    const scalar_t x6 = L * fff[4];
    const scalar_t x7 = (1 / TEST_POW2(L));
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x0 * (L * (-x5 - x6) + x3);
    sub_fff[2]        = x4 * x7;
    sub_fff[3]        = x0 * (L * (x1 + x5) + L * (L * fff[5] + x5));
    sub_fff[4]        = x7 * (x2 + x6);
    sub_fff[5]        = fff[3] / L;
}

static SFEM_INLINE void test_sstet4_sub_fff_2(const scalar_t                        L,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / TEST_POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = L * x3;
    const scalar_t x5 = TEST_POW2(L);
    const scalar_t x6 = L * fff[2];
    sub_fff[0]        = x0 * (L * (L * fff[3] + x2) + x4);
    sub_fff[1]        = -x3 / x5;
    sub_fff[2]        = x0 * (L * (L * fff[4] + x6) + x4);
    sub_fff[3]        = fff[0] / L;
    sub_fff[4]        = x0 * (-fff[0] * x5 - fff[2] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[5] + x6));
}

static SFEM_INLINE void test_sstet4_sub_fff_3(const scalar_t                        L,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0  = 1. / TEST_POW3(L);
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

static SFEM_INLINE void test_sstet4_sub_fff_4(const scalar_t                        L,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1.0 / L;
    const scalar_t x1 = 1. / TEST_POW3(L);
    const scalar_t x2 = TEST_POW2(L);
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

static SFEM_INLINE void test_sstet4_sub_fff_5(const scalar_t                        L,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              scalar_t *const SFEM_RESTRICT         sub_fff) {
    const scalar_t x0 = 1. / TEST_POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[2];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[5] - x2;
    const scalar_t x5 = TEST_POW2(L);
    const scalar_t x6 = L * fff[1];
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x4 / x5;
    sub_fff[2]        = x0 * (L * (-L * fff[4] - x6) + x3);
    sub_fff[3]        = fff[5] / L;
    sub_fff[4]        = x0 * (fff[2] * x5 + fff[4] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[3] + x6));
}

static SFEM_INLINE void test_apply_tet4_fff(const scalar_t *const SFEM_RESTRICT fff,
                                            const int *const SFEM_RESTRICT      ev,
                                            const real_t *const SFEM_RESTRICT   element_u,
                                            real_t *const SFEM_RESTRICT         element_vector) {
    accumulator_t v[4];
    tet4_laplacian_apply_fff(
            fff, element_u[ev[0]], element_u[ev[1]], element_u[ev[2]], element_u[ev[3]], &v[0], &v[1], &v[2], &v[3]);
    for (int d = 0; d < 4; ++d) {
        element_vector[ev[d]] += v[d];
    }
}

static int test_hyteg_logical_sstet4_laplacian_apply(const int                             level,
                                                     const ptrdiff_t                       nelements,
                                                     const jacobian_t *const SFEM_RESTRICT g_fff,
                                                     const real_t *const SFEM_RESTRICT     u,
                                                     real_t *const SFEM_RESTRICT           values) {
    const int nxe = smesh::sstet4_nxe(level);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const real_t *const element_u      = &u[e * nxe];
        real_t *const       element_vector = &values[e * nxe];
        const jacobian_t   *macro_fff      = &g_fff[e * 6];
        scalar_t            fff[6];
        int                 ev[4];

        test_sstet4_sub_fff_0(level, macro_fff, fff);
        for (int z = 0; z < level; ++z) {
            for (int y = 0; y < level - z; ++y) {
                for (int x = 0; x < level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x, y, z);
                    ev[1] = smesh::sstet4_lidx(level, x + 1, y, z);
                    ev[2] = smesh::sstet4_lidx(level, x, y + 1, z);
                    ev[3] = smesh::sstet4_lidx(level, x, y, z + 1);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }

        test_sstet4_sub_fff_1(level, macro_fff, fff);
        for (int z = 1; z < level; ++z) {
            for (int y = 0; y < level - z; ++y) {
                for (int x = 0; x < level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x + 1, y, z - 1);
                    ev[1] = smesh::sstet4_lidx(level, x, y, z);
                    ev[2] = smesh::sstet4_lidx(level, x + 1, y, z);
                    ev[3] = smesh::sstet4_lidx(level, x, y + 1, z);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }

        test_sstet4_sub_fff_2(level, macro_fff, fff);
        for (int z = 1; z < level; ++z) {
            for (int y = 1; y <= level - z; ++y) {
                for (int x = 0; x <= level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x + 1, y - 1, z - 1);
                    ev[1] = smesh::sstet4_lidx(level, x + 1, y, z - 1);
                    ev[2] = smesh::sstet4_lidx(level, x, y, z);
                    ev[3] = smesh::sstet4_lidx(level, x + 1, y - 1, z);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }

        test_sstet4_sub_fff_3(level, macro_fff, fff);
        for (int z = 1; z < level; ++z) {
            for (int y = 0; y < level - z; ++y) {
                for (int x = 0; x < level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x + 1, y, z - 1);
                    ev[1] = smesh::sstet4_lidx(level, x, y + 1, z - 1);
                    ev[2] = smesh::sstet4_lidx(level, x, y, z);
                    ev[3] = smesh::sstet4_lidx(level, x, y + 1, z);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }

        test_sstet4_sub_fff_4(level, macro_fff, fff);
        for (int z = 1; z < level; ++z) {
            for (int y = 1; y < level - z; ++y) {
                for (int x = 0; x < level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x + 1, y, z - 1);
                    ev[1] = smesh::sstet4_lidx(level, x + 1, y - 1, z);
                    ev[2] = smesh::sstet4_lidx(level, x + 1, y, z);
                    ev[3] = smesh::sstet4_lidx(level, x, y, z);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }

        test_sstet4_sub_fff_5(level, macro_fff, fff);
        for (int z = 0; z < level; ++z) {
            for (int y = 1; y < level - z; ++y) {
                for (int x = 0; x < level - y - z; ++x) {
                    ev[0] = smesh::sstet4_lidx(level, x + 1, y - 1, z);
                    ev[1] = smesh::sstet4_lidx(level, x, y, z);
                    ev[2] = smesh::sstet4_lidx(level, x, y, z + 1);
                    ev[3] = smesh::sstet4_lidx(level, x + 1, y, z);
                    test_apply_tet4_fff(fff, ev, element_u, element_vector);
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

template <typename F>
static void for_each_hyteg_microtet(const int level, F &&f) {
    int ev[4];

    if (level == 1) {
        ev[0] = smesh::sstet4_lidx(1, 0, 0, 0);
        ev[1] = smesh::sstet4_lidx(1, 1, 0, 0);
        ev[2] = smesh::sstet4_lidx(1, 0, 1, 0);
        ev[3] = smesh::sstet4_lidx(1, 0, 0, 1);
        f(ev);
        return;
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
                    f(ev);
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
                    f(ev);
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
                    f(ev);
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
                    f(ev);
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
                    f(ev);
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
                    f(ev);
                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

static int apply_hyteg_points(const int                         level,
                              const ptrdiff_t                   nelements,
                              idx_t **const SFEM_RESTRICT       elements,
                              geom_t **const SFEM_RESTRICT      points,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT       values) {
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for_each_hyteg_microtet(level, [&](const int *const lev) {
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
                values[gv[d]] += v[d];
            }
        });
    }

    return SFEM_SUCCESS;
}

static int test_sstet4_cached_fff_laplacian() {
    auto comm = sfem::Communicator::world();

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_VALIDATE = 1;
    SFEM_READ_ENV(SFEM_VALIDATE, atoi);

    int SFEM_THROUGHPUT_REPEAT = 10;
    SFEM_READ_ENV(SFEM_THROUGHPUT_REPEAT, atoi);

    int SFEM_COMPARE_HYTEG_TRAVERSAL = 1;
    SFEM_READ_ENV(SFEM_COMPARE_HYTEG_TRAVERSAL, atoi);

    int SFEM_COMPARE_HYTEG_STENCIL = 1;
    SFEM_READ_ENV(SFEM_COMPARE_HYTEG_STENCIL, atoi);

    auto mesh = sfem::Mesh::create_tet4_cube(
            comm, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);
    mesh = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, mesh, true, false);

    const int       level   = smesh::semistructured_level(*mesh);
    const int       nxe     = smesh::sstet4_nxe(level);
    const ptrdiff_t n_nodes = mesh->n_nodes();
    const auto      es      = sfem::EXECUTION_SPACE_HOST;

    auto input      = sfem::create_buffer<real_t>(n_nodes, es);
    auto point_out  = sfem::create_buffer<real_t>(n_nodes, es);
    auto transfer_point_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto cached_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto global_stencil_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto points_stencil_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto frontend_stencil_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto frontend_point_out   = sfem::create_buffer<real_t>(n_nodes, es);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const real_t x = points[0][i];
        const real_t y = points[1][i];
        const real_t z = points[2][i];
        input->data()[i]      = 0.17 + x * x + 0.25 * y - 0.125 * z + 0.5 * x * y +
                           0.03125 * ((i * 17 + 3) % 29);
        point_out->data()[i]  = 0;
        transfer_point_out->data()[i] = 0;
        cached_out->data()[i] = 0;
        global_stencil_out->data()[i] = 0;
        points_stencil_out->data()[i] = 0;
        frontend_stencil_out->data()[i] = 0;
        frontend_point_out->data()[i]   = 0;
    }

    const int corners[4] = {smesh::sstet4_lidx(level, 0, 0, 0),
                            smesh::sstet4_lidx(level, level, 0, 0),
                            smesh::sstet4_lidx(level, 0, level, 0),
                            smesh::sstet4_lidx(level, 0, 0, level)};

    std::vector<ptrdiff_t>                     block_ne;
    std::vector<sfem::SharedBuffer<jacobian_t>> block_g_fff;
    std::vector<sfem::SharedBuffer<real_t>>     block_u;
    std::vector<sfem::SharedBuffer<real_t>>     block_out;
    std::vector<sfem::SharedBuffer<real_t>>     block_hyteg_out;
    std::vector<sfem::SharedBuffer<real_t>>     block_stencil_out;
    std::vector<sstet4_laplacian_stencil_t *>   block_stencils;

    ptrdiff_t total_macro_elements = 0;
    ptrdiff_t total_local_dofs     = 0;
    ptrdiff_t total_microtets      = 0;
    ptrdiff_t total_unique_stencils = 0;
    real_t    hyteg_largest_diff   = 0;
    ptrdiff_t hyteg_arg            = SFEM_PTRDIFF_INVALID;
    real_t    stencil_largest_diff = 0;
    ptrdiff_t stencil_arg          = SFEM_PTRDIFF_INVALID;

    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        auto            block    = mesh->block(b);
        const ptrdiff_t ne       = block->n_elements();
        auto            elements = block->elements()->data();

        SFEM_TEST_ASSERT(block->n_nodes_per_element() == nxe);
        if (SFEM_VALIDATE) {
            SFEM_TEST_ASSERT(apply_hyteg_points(level, ne, elements, points, input->data(), point_out->data()) ==
                             SFEM_SUCCESS);
            SFEM_TEST_ASSERT(sstet4_laplacian_apply_points(
                                     level, ne, elements, points, input->data(), transfer_point_out->data()) ==
                             SFEM_SUCCESS);
        }

        auto g_fff     = sfem::create_buffer<jacobian_t>(ne * 6, es);
        auto local_u   = sfem::create_buffer<real_t>(ne * nxe, es);
        auto local_out = sfem::create_buffer<real_t>(ne * nxe, es);
        auto hyteg_out = sfem::create_buffer<real_t>(ne * nxe, es);
        auto stencil_out = sfem::create_buffer<real_t>(ne * nxe, es);
        sstet4_laplacian_stencil_t *stencil = nullptr;

        for (ptrdiff_t e = 0; e < ne; ++e) {
            const idx_t ev0 = elements[corners[0]][e];
            const idx_t ev1 = elements[corners[1]][e];
            const idx_t ev2 = elements[corners[2]][e];
            const idx_t ev3 = elements[corners[3]][e];
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
                     &g_fff->data()[e * 6]);

            for (int v = 0; v < nxe; ++v) {
                local_u->data()[e * nxe + v]   = input->data()[elements[v][e]];
                local_out->data()[e * nxe + v] = 0;
                hyteg_out->data()[e * nxe + v] = 0;
                stencil_out->data()[e * nxe + v] = 0;
            }
        }

        if (SFEM_COMPARE_HYTEG_STENCIL) {
            SFEM_TEST_ASSERT(sstet4_laplacian_stencil_create(level, ne, g_fff->data(), &stencil) == SFEM_SUCCESS);
            total_unique_stencils += sstet4_laplacian_stencil_n_unique_stencils(stencil);

            if (block_stencils.empty()) {
                printf("prebuilt stencil topology level=%d rows=%d max_row_len=%d max_slot_terms=%d\n",
                       level,
                       sstet4_laplacian_stencil_nrows(stencil),
                       sstet4_laplacian_stencil_max_row_len(stencil),
                       sstet4_laplacian_stencil_max_slot_terms(stencil));
            }
        }

        SFEM_TEST_ASSERT(sstet4_laplacian_apply(level, ne, g_fff->data(), local_u->data(), local_out->data()) ==
                         SFEM_SUCCESS);
        if (SFEM_COMPARE_HYTEG_TRAVERSAL) {
            SFEM_TEST_ASSERT(test_hyteg_logical_sstet4_laplacian_apply(
                                     level, ne, g_fff->data(), local_u->data(), hyteg_out->data()) == SFEM_SUCCESS);
        }
        if (SFEM_COMPARE_HYTEG_STENCIL) {
            SFEM_TEST_ASSERT(sstet4_laplacian_apply_stencil(stencil, ne, local_u->data(), stencil_out->data()) ==
                             SFEM_SUCCESS);
            SFEM_TEST_ASSERT(sstet4_laplacian_apply_stencil_global(
                                     stencil, ne, elements, input->data(), global_stencil_out->data()) == SFEM_SUCCESS);
        }

        if (SFEM_VALIDATE) {
            if (SFEM_COMPARE_HYTEG_TRAVERSAL) {
                for (ptrdiff_t i = 0; i < ne * nxe; ++i) {
                    const real_t diff = fabs(hyteg_out->data()[i] - local_out->data()[i]);
                    if (diff > hyteg_largest_diff || diff != diff) {
                        hyteg_largest_diff = diff;
                        hyteg_arg          = i;
                    }
                }
            }
            if (SFEM_COMPARE_HYTEG_STENCIL) {
                for (ptrdiff_t i = 0; i < ne * nxe; ++i) {
                    const real_t diff = fabs(stencil_out->data()[i] - local_out->data()[i]);
                    if (diff > stencil_largest_diff || diff != diff) {
                        stencil_largest_diff = diff;
                        stencil_arg          = i;
                    }
                }
            }

            for (ptrdiff_t e = 0; e < ne; ++e) {
                for (int v = 0; v < nxe; ++v) {
                    cached_out->data()[elements[v][e]] += local_out->data()[e * nxe + v];
                }
            }
        }

        block_ne.push_back(ne);
        block_g_fff.push_back(g_fff);
        block_u.push_back(local_u);
        block_out.push_back(local_out);
        block_hyteg_out.push_back(hyteg_out);
        block_stencil_out.push_back(stencil_out);
        block_stencils.push_back(stencil);

        total_macro_elements += ne;
        total_local_dofs += ne * nxe;
        total_microtets += ne * level * level * level;
    }

    if (SFEM_VALIDATE) {
        real_t    largest_diff = 0;
        ptrdiff_t arg          = SFEM_PTRDIFF_INVALID;
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const real_t diff = fabs(cached_out->data()[i] - point_out->data()[i]);
            if (diff > largest_diff || diff != diff) {
                largest_diff = diff;
                arg          = i;
            }
        }

        printf("default SSTET4 FFF check level=%d largest_diff(%ld) = %g\n", level, arg, (double)largest_diff);
        SFEM_TEST_ASSERT(largest_diff < 1e-7);
        if (SFEM_COMPARE_HYTEG_TRAVERSAL) {
            printf("HyTeG logical traversal check level=%d largest_diff(%ld) = %g\n",
                   level,
                   hyteg_arg,
                   (double)hyteg_largest_diff);
            SFEM_TEST_ASSERT(hyteg_largest_diff < 1e-7);
        }
        if (SFEM_COMPARE_HYTEG_STENCIL) {
            printf("prebuilt stencil check level=%d largest_diff(%ld) = %g\n",
                   level,
                   stencil_arg,
                   (double)stencil_largest_diff);
            SFEM_TEST_ASSERT(stencil_largest_diff < 1e-7);
            real_t    global_stencil_largest_diff = 0;
            ptrdiff_t global_stencil_arg          = SFEM_PTRDIFF_INVALID;
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                const real_t diff = fabs(global_stencil_out->data()[i] - point_out->data()[i]);
                if (diff > global_stencil_largest_diff || diff != diff) {
                    global_stencil_largest_diff = diff;
                    global_stencil_arg          = i;
                }
            }

            printf("global prebuilt stencil check level=%d largest_diff(%ld) = %g\n",
                   level,
                   global_stencil_arg,
                   (double)global_stencil_largest_diff);
            SFEM_TEST_ASSERT(global_stencil_largest_diff < 1e-7);
            printf("prebuilt stencil variants level=%d total_unique_stencils=%ld\n", level, total_unique_stencils);

            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto                        block          = mesh->block(b);
                sstet4_laplacian_stencil_t *points_stencil = nullptr;
                SFEM_TEST_ASSERT(sstet4_laplacian_stencil_create_from_points(level,
                                                                              block->n_elements(),
                                                                              block->elements()->data(),
                                                                              points,
                                                                              &points_stencil) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(sstet4_laplacian_apply_stencil_global(points_stencil,
                                                                        block->n_elements(),
                                                                        block->elements()->data(),
                                                                        input->data(),
                                                                        points_stencil_out->data()) == SFEM_SUCCESS);
                sstet4_laplacian_stencil_destroy(points_stencil);
            }

            real_t    points_stencil_largest_diff = 0;
            ptrdiff_t points_stencil_arg          = SFEM_PTRDIFF_INVALID;
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                const real_t diff = fabs(points_stencil_out->data()[i] - transfer_point_out->data()[i]);
                if (diff > points_stencil_largest_diff || diff != diff) {
                    points_stencil_largest_diff = diff;
                    points_stencil_arg          = i;
                }
            }

            printf("points-created stencil check level=%d largest_diff(%ld) = %g\n",
                   level,
                   points_stencil_arg,
                   (double)points_stencil_largest_diff);
            SFEM_TEST_ASSERT(points_stencil_largest_diff < 1e-7);
        }

        auto fs = sfem::FunctionSpace::create(mesh, 1);
        auto op = sfem::create_op(fs, "Laplacian", es);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);

        setenv("SFEM_SSTET4_LAPLACIAN_USE_STENCIL", "1", 1);
        SFEM_TEST_ASSERT(op->apply(nullptr, input->data(), frontend_stencil_out->data()) == SFEM_SUCCESS);
        setenv("SFEM_SSTET4_LAPLACIAN_USE_STENCIL", "0", 1);
        SFEM_TEST_ASSERT(op->apply(nullptr, input->data(), frontend_point_out->data()) == SFEM_SUCCESS);
        unsetenv("SFEM_SSTET4_LAPLACIAN_USE_STENCIL");

        real_t    frontend_largest_diff = 0;
        ptrdiff_t frontend_arg          = SFEM_PTRDIFF_INVALID;
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const real_t diff = fabs(frontend_stencil_out->data()[i] - frontend_point_out->data()[i]);
            if (diff > frontend_largest_diff || diff != diff) {
                frontend_largest_diff = diff;
                frontend_arg          = i;
            }
        }

        printf("frontend default stencil check level=%d largest_diff(%ld) = %g\n",
               level,
               frontend_arg,
               (double)frontend_largest_diff);
        SFEM_TEST_ASSERT(frontend_largest_diff < 1e-7);
    }

    if (SFEM_THROUGHPUT_REPEAT > 0) {
        if (SFEM_THROUGHPUT_REPEAT > 1) {
            for (size_t b = 0; b < block_ne.size(); ++b) {
                SFEM_TEST_ASSERT(sstet4_laplacian_apply(
                                         level, block_ne[b], block_g_fff[b]->data(), block_u[b]->data(), block_out[b]->data()) ==
                                 SFEM_SUCCESS);
            }
        }

        const double tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < block_ne.size(); ++b) {
                SFEM_TEST_ASSERT(sstet4_laplacian_apply(
                                         level, block_ne[b], block_g_fff[b]->data(), block_u[b]->data(), block_out[b]->data()) ==
                                 SFEM_SUCCESS);
            }
        }

        const double elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;
        printf("default SSTET4 FFF throughput level=%d base=%d macro_elements=%ld global_nodes=%ld local_dofs=%ld "
               "microtets=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrotet/s=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microtets,
               SFEM_THROUGHPUT_REPEAT,
               elapsed,
               1e-6 * total_local_dofs / elapsed,
               1e-6 * total_microtets / elapsed);
        fflush(stdout);

        if (SFEM_COMPARE_HYTEG_TRAVERSAL) {
            if (SFEM_THROUGHPUT_REPEAT > 1) {
                for (size_t b = 0; b < block_ne.size(); ++b) {
                    SFEM_TEST_ASSERT(test_hyteg_logical_sstet4_laplacian_apply(level,
                                                                               block_ne[b],
                                                                               block_g_fff[b]->data(),
                                                                               block_u[b]->data(),
                                                                               block_hyteg_out[b]->data()) ==
                                     SFEM_SUCCESS);
                }
            }

            const double hyteg_tick = smesh::time_seconds();
            for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
                for (size_t b = 0; b < block_ne.size(); ++b) {
                    SFEM_TEST_ASSERT(test_hyteg_logical_sstet4_laplacian_apply(level,
                                                                               block_ne[b],
                                                                               block_g_fff[b]->data(),
                                                                               block_u[b]->data(),
                                                                               block_hyteg_out[b]->data()) ==
                                     SFEM_SUCCESS);
                }
            }

            const double hyteg_elapsed = (smesh::time_seconds() - hyteg_tick) / SFEM_THROUGHPUT_REPEAT;
            printf("HyTeG logical SSTET4 FFF throughput level=%d base=%d macro_elements=%ld global_nodes=%ld local_dofs=%ld "
                   "microtets=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrotet/s=%g speedup_vs_hyteg=%g\n",
                   level,
                   SFEM_BASE_RESOLUTION,
                   total_macro_elements,
                   n_nodes,
                   total_local_dofs,
                   total_microtets,
                   SFEM_THROUGHPUT_REPEAT,
                   hyteg_elapsed,
                   1e-6 * total_local_dofs / hyteg_elapsed,
                   1e-6 * total_microtets / hyteg_elapsed,
                   hyteg_elapsed / elapsed);
            fflush(stdout);
        }

        if (SFEM_COMPARE_HYTEG_STENCIL) {
            if (SFEM_THROUGHPUT_REPEAT > 1) {
                for (size_t b = 0; b < block_ne.size(); ++b) {
                    SFEM_TEST_ASSERT(sstet4_laplacian_apply_stencil(block_stencils[b],
                                                                     block_ne[b],
                                                                     block_u[b]->data(),
                                                                     block_stencil_out[b]->data()) ==
                                     SFEM_SUCCESS);
                }
            }

            const double stencil_tick = smesh::time_seconds();
            for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
                for (size_t b = 0; b < block_ne.size(); ++b) {
                    SFEM_TEST_ASSERT(sstet4_laplacian_apply_stencil(block_stencils[b],
                                                                     block_ne[b],
                                                                     block_u[b]->data(),
                                                                     block_stencil_out[b]->data()) ==
                                     SFEM_SUCCESS);
                }
            }

            const double stencil_elapsed = (smesh::time_seconds() - stencil_tick) / SFEM_THROUGHPUT_REPEAT;
            printf("prebuilt stencil SSTET4 FFF throughput level=%d base=%d macro_elements=%ld global_nodes=%ld local_dofs=%ld "
                   "microtets=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrotet/s=%g speedup_vs_default=%g\n",
                   level,
                   SFEM_BASE_RESOLUTION,
                   total_macro_elements,
                   n_nodes,
                   total_local_dofs,
                   total_microtets,
                   SFEM_THROUGHPUT_REPEAT,
                   stencil_elapsed,
                   1e-6 * total_local_dofs / stencil_elapsed,
                   1e-6 * total_microtets / stencil_elapsed,
                   elapsed / stencil_elapsed);
            fflush(stdout);
        }
    }

    for (size_t b = 0; b < block_stencils.size(); ++b) {
        sstet4_laplacian_stencil_destroy(block_stencils[b]);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_sstet4_cached_fff_laplacian);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
