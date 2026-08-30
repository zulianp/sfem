#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace {

const real_t val_tol() { return sizeof(real_t) == sizeof(double) ? real_t(1e-12) : real_t(1e-5); }

const geom_t geom_tol() { return sizeof(geom_t) == sizeof(double) ? geom_t(1e-12) : geom_t(1e-5); }

void fill_ones(const sfem::SharedBuffer<real_t> &v) {
    auto            d = v->data();
    const ptrdiff_t n = (ptrdiff_t)v->size();
    for (ptrdiff_t i = 0; i < n; ++i) {
        d[i] = 1;
    }
}

void fill_linear(const smesh::Mesh &mesh, const sfem::SharedBuffer<real_t> &v) {
    auto            p = mesh.points()->data();
    auto            d = v->data();
    const ptrdiff_t n = mesh.n_nodes();
    const bool      has_z = mesh.spatial_dimension() > 2;
    for (ptrdiff_t i = 0; i < n; ++i) {
        d[i] = (real_t)p[0][i] + real_t(2) * (real_t)p[1][i] + (has_z ? real_t(3) * (real_t)p[2][i] : real_t(0));
    }
}

int map_nodes_by_xyz(const smesh::Mesh &from, const smesh::Mesh &to, std::vector<ptrdiff_t> &from_to_to) {
    SFEM_TEST_EQ(from.n_nodes(), to.n_nodes());
    const ptrdiff_t n   = from.n_nodes();
    const geom_t    tol = geom_tol();
    const bool      use_z = from.spatial_dimension() > 2 && to.spatial_dimension() > 2;
    from_to_to.assign((size_t)n, -1);

    auto pa = from.points()->data();
    auto pb = to.points()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t found = -1;
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (std::fabs(pa[0][i] - pb[0][j]) <= tol && std::fabs(pa[1][i] - pb[1][j]) <= tol &&
                (!use_z || std::fabs(pa[2][i] - pb[2][j]) <= tol)) {
                found = j;
                break;
            }
        }
        SFEM_TEST_ASSERT(found >= 0);
        from_to_to[(size_t)i] = found;
    }
    return SFEM_TEST_SUCCESS;
}

int compare_mapped(const sfem::SharedBuffer<real_t> &a,
                   const sfem::SharedBuffer<real_t> &b,
                   const std::vector<ptrdiff_t>     &a2b) {
    const ptrdiff_t n   = (ptrdiff_t)a->size();
    const real_t    tol = val_tol();
    SFEM_TEST_EQ(n, (ptrdiff_t)a2b.size());
    SFEM_TEST_EQ(n, (ptrdiff_t)b->size());
    auto da = a->data();
    auto db = b->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        SFEM_TEST_ASSERT(std::isfinite(da[i]));
        SFEM_TEST_ASSERT(std::isfinite(db[a2b[(size_t)i]]));
        SFEM_TEST_ASSERT(std::fabs(da[i] - db[a2b[(size_t)i]]) <= tol);
    }
    return SFEM_TEST_SUCCESS;
}

struct SSPair {
    std::shared_ptr<sfem::FunctionSpace> fine;
    std::shared_ptr<sfem::FunctionSpace> coarse;
};

SSPair make_ss_pair(const std::shared_ptr<smesh::Mesh> &mesh) {
    const int L  = 4;
    auto      ss = smesh::to_semistructured(L, mesh, true, false);
    auto      fine   = sfem::FunctionSpace::create(ss, 1);
    auto      coarse = fine->derefine(2);
    return {fine, coarse};
}

std::shared_ptr<smesh::Mesh> create_two_block_quad4_square(const ptrdiff_t nx, const ptrdiff_t ny) {
    auto base = sfem::Mesh::create_quad4_square(sfem::Communicator::self(), nx, ny, 0, 0, 1, 1);
    if (!base) {
        return nullptr;
    }

    auto elements = base->elements(0)->data();
    auto points   = base->points()->data();

    std::vector<ptrdiff_t> left_ids;
    std::vector<ptrdiff_t> right_ids;
    left_ids.reserve((size_t)base->n_elements());
    right_ids.reserve((size_t)base->n_elements());

    for (ptrdiff_t e = 0; e < base->n_elements(); ++e) {
        geom_t cx = 0;
        for (int d = 0; d < 4; ++d) {
            cx += points[0][elements[d][e]];
        }
        cx *= geom_t(0.25);
        if (cx < geom_t(0.5)) {
            left_ids.push_back(e);
        } else {
            right_ids.push_back(e);
        }
    }

    auto copy_block = [&](const std::vector<ptrdiff_t> &ids) {
        auto block_elements = sfem::create_host_buffer<idx_t>(4, ids.size());
        auto dst            = block_elements->data();
        for (ptrdiff_t i = 0; i < (ptrdiff_t)ids.size(); ++i) {
            const ptrdiff_t e = ids[(size_t)i];
            for (int d = 0; d < 4; ++d) {
                dst[d][i] = elements[d][e];
            }
        }
        return block_elements;
    };

    std::vector<std::shared_ptr<smesh::Mesh::Block>> blocks;
    auto                                             left_block = std::make_shared<smesh::Mesh::Block>();
    left_block->set_name("left");
    left_block->set_element_type(smesh::QUAD4);
    left_block->set_elements(copy_block(left_ids));
    blocks.push_back(left_block);

    auto right_block = std::make_shared<smesh::Mesh::Block>();
    right_block->set_name("right");
    right_block->set_element_type(smesh::QUAD4);
    right_block->set_elements(copy_block(right_ids));
    blocks.push_back(right_block);

    return std::make_shared<smesh::Mesh>(base->comm(), blocks, base->points());
}

std::shared_ptr<smesh::Mesh> create_two_block_tet4_cube(const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    auto base = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
    if (!base) {
        return nullptr;
    }

    auto elements = base->elements(0)->data();
    auto points   = base->points()->data();

    std::vector<ptrdiff_t> left_ids;
    std::vector<ptrdiff_t> right_ids;
    left_ids.reserve((size_t)base->n_elements());
    right_ids.reserve((size_t)base->n_elements());

    for (ptrdiff_t e = 0; e < base->n_elements(); ++e) {
        geom_t cx = 0;
        for (int d = 0; d < 4; ++d) {
            cx += points[0][elements[d][e]];
        }
        cx *= geom_t(0.25);
        if (cx < geom_t(0.5)) {
            left_ids.push_back(e);
        } else {
            right_ids.push_back(e);
        }
    }

    auto copy_block = [&](const std::vector<ptrdiff_t> &ids) {
        auto block_elements = sfem::create_host_buffer<idx_t>(4, ids.size());
        auto dst            = block_elements->data();
        for (ptrdiff_t i = 0; i < (ptrdiff_t)ids.size(); ++i) {
            const ptrdiff_t e = ids[(size_t)i];
            for (int d = 0; d < 4; ++d) {
                dst[d][i] = elements[d][e];
            }
        }
        return block_elements;
    };

    std::vector<std::shared_ptr<smesh::Mesh::Block>> blocks;
    auto                                             left_block = std::make_shared<smesh::Mesh::Block>();
    left_block->set_name("left");
    left_block->set_element_type(smesh::TET4);
    left_block->set_elements(copy_block(left_ids));
    blocks.push_back(left_block);

    auto right_block = std::make_shared<smesh::Mesh::Block>();
    right_block->set_name("right");
    right_block->set_element_type(smesh::TET4);
    right_block->set_elements(copy_block(right_ids));
    blocks.push_back(right_block);

    return std::make_shared<smesh::Mesh>(base->comm(), blocks, base->points());
}

std::shared_ptr<smesh::Mesh> mesh_from_single_block(const std::shared_ptr<smesh::Mesh> &mesh, const size_t b) {
    std::vector<std::shared_ptr<smesh::Mesh::Block>> blocks;
    blocks.push_back(mesh->block(b));
    return std::make_shared<smesh::Mesh>(mesh->comm(), blocks, mesh->points());
}

void mark_block_nodes(const std::shared_ptr<smesh::Mesh::Block> &block, std::vector<char> &mask) {
    const ptrdiff_t ne = block->n_elements();
    if (ne == 0) {
        return;
    }
    const int nxe = block->n_nodes_per_element();
    auto      els = block->elements()->data();
    for (int d = 0; d < nxe; ++d) {
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const idx_t n = els[d][e];
            if (n >= 0 && (size_t)n < mask.size()) {
                mask[(size_t)n] = 1;
            }
        }
    }
}

int compare_masked(const sfem::SharedBuffer<real_t> &a, const sfem::SharedBuffer<real_t> &b, const std::vector<char> &mask) {
    const ptrdiff_t n   = (ptrdiff_t)a->size();
    const real_t    tol = val_tol();
    SFEM_TEST_EQ(n, (ptrdiff_t)b->size());
    SFEM_TEST_EQ(n, (ptrdiff_t)mask.size());
    auto      da        = a->data();
    auto      db        = b->data();
    ptrdiff_t n_checked = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (!mask[(size_t)i]) {
            continue;
        }
        SFEM_TEST_ASSERT(std::isfinite(da[i]));
        SFEM_TEST_ASSERT(std::isfinite(db[i]));
        SFEM_TEST_ASSERT(std::fabs(da[i] - db[i]) <= tol);
        ++n_checked;
    }
    SFEM_TEST_ASSERT(n_checked > 0);
    return SFEM_TEST_SUCCESS;
}

}  // namespace

int test_checkerboard_prolong_ones() {
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));

    auto pair = make_ss_pair(mesh);
    SFEM_TEST_ASSERT(pair.fine->has_semi_structured_mesh());
    SFEM_TEST_EQ(pair.fine->mesh().n_blocks(), static_cast<size_t>(2));

    auto prolongation = sfem::create_hierarchical_prolongation(pair.coarse, pair.fine, sfem::EXECUTION_SPACE_HOST);
    auto coarse_field = sfem::create_host_buffer<real_t>(pair.coarse->n_dofs());
    auto fine_field   = sfem::create_host_buffer<real_t>(pair.fine->n_dofs());
    fill_ones(coarse_field);

    SFEM_TEST_ASSERT(prolongation->apply(coarse_field->data(), fine_field->data()) == SFEM_SUCCESS);

    const real_t tol = val_tol();
    auto         d   = fine_field->data();
    for (ptrdiff_t i = 0; i < pair.fine->n_dofs(); ++i) {
        SFEM_TEST_ASSERT(std::isfinite(d[i]));
        SFEM_TEST_ASSERT(std::fabs(d[i] - real_t(1)) <= tol);
    }
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_restrict_finite() {
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto pair = make_ss_pair(mesh);

    auto restriction  = sfem::create_hierarchical_restriction(pair.fine, pair.coarse, sfem::EXECUTION_SPACE_HOST);
    auto fine_field   = sfem::create_host_buffer<real_t>(pair.fine->n_dofs());
    auto coarse_field = sfem::create_host_buffer<real_t>(pair.coarse->n_dofs());
    fill_ones(fine_field);

    SFEM_TEST_ASSERT(restriction->apply(fine_field->data(), coarse_field->data()) == SFEM_SUCCESS);

    auto d = coarse_field->data();
    for (ptrdiff_t i = 0; i < pair.coarse->n_dofs(); ++i) {
        SFEM_TEST_ASSERT(std::isfinite(d[i]));
    }
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_vs_cube() {
    auto cb   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cube = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);

    auto cb_pair   = make_ss_pair(cb);
    auto cube_pair = make_ss_pair(cube);

    SFEM_TEST_EQ(cb_pair.fine->n_dofs(), cube_pair.fine->n_dofs());
    SFEM_TEST_EQ(cb_pair.coarse->n_dofs(), cube_pair.coarse->n_dofs());

    std::vector<ptrdiff_t> fine_cb2cube;
    std::vector<ptrdiff_t> coarse_cb2cube;
    SFEM_TEST_ASSERT(map_nodes_by_xyz(cb_pair.fine->mesh(), cube_pair.fine->mesh(), fine_cb2cube) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(map_nodes_by_xyz(cb_pair.coarse->mesh(), cube_pair.coarse->mesh(), coarse_cb2cube) ==
                     SFEM_TEST_SUCCESS);

    auto es = sfem::EXECUTION_SPACE_HOST;

    {
        auto p_cb   = sfem::create_hierarchical_prolongation(cb_pair.coarse, cb_pair.fine, es);
        auto p_cube = sfem::create_hierarchical_prolongation(cube_pair.coarse, cube_pair.fine, es);

        auto c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        auto c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        auto f_cb   = sfem::create_host_buffer<real_t>(cb_pair.fine->n_dofs());
        auto f_cube = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());

        fill_ones(c_cb);
        fill_ones(c_cube);
        SFEM_TEST_ASSERT(p_cb->apply(c_cb->data(), f_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_cb, f_cube, fine_cb2cube) == SFEM_TEST_SUCCESS);

        fill_linear(cb_pair.coarse->mesh(), c_cb);
        fill_linear(cube_pair.coarse->mesh(), c_cube);
        SFEM_TEST_ASSERT(p_cb->apply(c_cb->data(), f_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_cb, f_cube, fine_cb2cube) == SFEM_TEST_SUCCESS);
    }

    {
        auto r_cb   = sfem::create_hierarchical_restriction(cb_pair.fine, cb_pair.coarse, es);
        auto r_cube = sfem::create_hierarchical_restriction(cube_pair.fine, cube_pair.coarse, es);

        auto f_cb   = sfem::create_host_buffer<real_t>(cb_pair.fine->n_dofs());
        auto f_cube = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());
        auto c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        auto c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());

        fill_ones(f_cb);
        fill_ones(f_cube);
        SFEM_TEST_ASSERT(r_cb->apply(f_cb->data(), c_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_cb, c_cube, coarse_cb2cube) == SFEM_TEST_SUCCESS);

        fill_linear(cb_pair.fine->mesh(), f_cb);
        fill_linear(cube_pair.fine->mesh(), f_cube);
        c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        SFEM_TEST_ASSERT(r_cb->apply(f_cb->data(), c_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_cb, c_cube, coarse_cb2cube) == SFEM_TEST_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int test_two_block_quad4_vs_square() {
    auto split  = create_two_block_quad4_square(4, 4);
    auto square = sfem::Mesh::create_quad4_square(sfem::Communicator::self(), 4, 4, 0, 0, 1, 1);

    SFEM_TEST_ASSERT(split != nullptr);
    SFEM_TEST_ASSERT(square != nullptr);
    SFEM_TEST_EQ(split->n_blocks(), static_cast<size_t>(2));

    auto split_pair  = make_ss_pair(split);
    auto square_pair = make_ss_pair(square);

    SFEM_TEST_EQ(split_pair.fine->n_dofs(), square_pair.fine->n_dofs());
    SFEM_TEST_EQ(split_pair.coarse->n_dofs(), square_pair.coarse->n_dofs());

    std::vector<ptrdiff_t> fine_split2square;
    std::vector<ptrdiff_t> coarse_split2square;
    SFEM_TEST_ASSERT(
            map_nodes_by_xyz(split_pair.fine->mesh(), square_pair.fine->mesh(), fine_split2square) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(map_nodes_by_xyz(split_pair.coarse->mesh(), square_pair.coarse->mesh(), coarse_split2square) ==
                     SFEM_TEST_SUCCESS);

    auto es = sfem::EXECUTION_SPACE_HOST;

    {
        auto p_split  = sfem::create_hierarchical_prolongation(split_pair.coarse, split_pair.fine, es);
        auto p_square = sfem::create_hierarchical_prolongation(square_pair.coarse, square_pair.fine, es);

        auto c_split  = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        auto c_square = sfem::create_host_buffer<real_t>(square_pair.coarse->n_dofs());
        auto f_split  = sfem::create_host_buffer<real_t>(split_pair.fine->n_dofs());
        auto f_square = sfem::create_host_buffer<real_t>(square_pair.fine->n_dofs());

        fill_ones(c_split);
        fill_ones(c_square);
        SFEM_TEST_ASSERT(p_split->apply(c_split->data(), f_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_square->apply(c_square->data(), f_square->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_split, f_square, fine_split2square) == SFEM_TEST_SUCCESS);

        fill_linear(split_pair.coarse->mesh(), c_split);
        fill_linear(square_pair.coarse->mesh(), c_square);
        SFEM_TEST_ASSERT(p_split->apply(c_split->data(), f_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_square->apply(c_square->data(), f_square->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_split, f_square, fine_split2square) == SFEM_TEST_SUCCESS);
    }

    {
        auto r_split  = sfem::create_hierarchical_restriction(split_pair.fine, split_pair.coarse, es);
        auto r_square = sfem::create_hierarchical_restriction(square_pair.fine, square_pair.coarse, es);

        auto f_split  = sfem::create_host_buffer<real_t>(split_pair.fine->n_dofs());
        auto f_square = sfem::create_host_buffer<real_t>(square_pair.fine->n_dofs());
        auto c_split  = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        auto c_square = sfem::create_host_buffer<real_t>(square_pair.coarse->n_dofs());

        fill_ones(f_split);
        fill_ones(f_square);
        SFEM_TEST_ASSERT(r_split->apply(f_split->data(), c_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_square->apply(f_square->data(), c_square->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_split, c_square, coarse_split2square) == SFEM_TEST_SUCCESS);

        fill_linear(split_pair.fine->mesh(), f_split);
        fill_linear(square_pair.fine->mesh(), f_square);
        c_split  = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        c_square = sfem::create_host_buffer<real_t>(square_pair.coarse->n_dofs());
        SFEM_TEST_ASSERT(r_split->apply(f_split->data(), c_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_square->apply(f_square->data(), c_square->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_split, c_square, coarse_split2square) == SFEM_TEST_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int test_two_block_tet4_vs_cube() {
    auto split = create_two_block_tet4_cube(2, 2, 2);
    auto cube  = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2);

    SFEM_TEST_ASSERT(split != nullptr);
    SFEM_TEST_ASSERT(cube != nullptr);
    SFEM_TEST_EQ(split->n_blocks(), static_cast<size_t>(2));

    auto split_pair = make_ss_pair(split);
    auto cube_pair  = make_ss_pair(cube);

    SFEM_TEST_EQ(split_pair.fine->n_dofs(), cube_pair.fine->n_dofs());
    SFEM_TEST_EQ(split_pair.coarse->n_dofs(), cube_pair.coarse->n_dofs());

    std::vector<ptrdiff_t> fine_split2cube;
    std::vector<ptrdiff_t> coarse_split2cube;
    SFEM_TEST_ASSERT(map_nodes_by_xyz(split_pair.fine->mesh(), cube_pair.fine->mesh(), fine_split2cube) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(map_nodes_by_xyz(split_pair.coarse->mesh(), cube_pair.coarse->mesh(), coarse_split2cube) ==
                     SFEM_TEST_SUCCESS);

    auto es = sfem::EXECUTION_SPACE_HOST;

    {
        auto p_split = sfem::create_hierarchical_prolongation(split_pair.coarse, split_pair.fine, es);
        auto p_cube  = sfem::create_hierarchical_prolongation(cube_pair.coarse, cube_pair.fine, es);

        auto c_split = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        auto c_cube  = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        auto f_split = sfem::create_host_buffer<real_t>(split_pair.fine->n_dofs());
        auto f_cube  = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());

        fill_ones(c_split);
        fill_ones(c_cube);
        SFEM_TEST_ASSERT(p_split->apply(c_split->data(), f_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_split, f_cube, fine_split2cube) == SFEM_TEST_SUCCESS);

        fill_linear(split_pair.coarse->mesh(), c_split);
        fill_linear(cube_pair.coarse->mesh(), c_cube);
        SFEM_TEST_ASSERT(p_split->apply(c_split->data(), f_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_split, f_cube, fine_split2cube) == SFEM_TEST_SUCCESS);
    }

    {
        auto r_split = sfem::create_hierarchical_restriction(split_pair.fine, split_pair.coarse, es);
        auto r_cube  = sfem::create_hierarchical_restriction(cube_pair.fine, cube_pair.coarse, es);

        auto f_split = sfem::create_host_buffer<real_t>(split_pair.fine->n_dofs());
        auto f_cube  = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());
        auto c_split = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        auto c_cube  = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());

        fill_ones(f_split);
        fill_ones(f_cube);
        SFEM_TEST_ASSERT(r_split->apply(f_split->data(), c_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_split, c_cube, coarse_split2cube) == SFEM_TEST_SUCCESS);

        fill_linear(split_pair.fine->mesh(), f_split);
        fill_linear(cube_pair.fine->mesh(), f_cube);
        c_split = sfem::create_host_buffer<real_t>(split_pair.coarse->n_dofs());
        c_cube  = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        SFEM_TEST_ASSERT(r_split->apply(f_split->data(), c_split->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_split, c_cube, coarse_split2cube) == SFEM_TEST_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int test_hex8_tet4_ss_vs_split_blocks() {
    auto mesh = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), 2, 2, 2);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));
    SFEM_TEST_ASSERT(mesh->block(0)->name() == "hex");
    SFEM_TEST_ASSERT(mesh->block(1)->name() == "tet");

    auto mixed = make_ss_pair(mesh);
    SFEM_TEST_ASSERT(mixed.fine->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(mixed.coarse->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(smesh::is_hex_ss_family(mixed.fine->element_type(0)));
    SFEM_TEST_ASSERT(smesh::is_tet_ss_family(mixed.fine->element_type(1)));
    SFEM_TEST_ASSERT(smesh::is_hex_ss_family(mixed.coarse->element_type(0)));
    SFEM_TEST_ASSERT(smesh::is_tet_ss_family(mixed.coarse->element_type(1)));

    auto hex_fine   = sfem::FunctionSpace::create(mesh_from_single_block(mixed.fine->mesh_ptr(), 0), 1);
    auto hex_coarse = sfem::FunctionSpace::create(mesh_from_single_block(mixed.coarse->mesh_ptr(), 0), 1);
    auto tet_fine   = sfem::FunctionSpace::create(mesh_from_single_block(mixed.fine->mesh_ptr(), 1), 1);
    auto tet_coarse = sfem::FunctionSpace::create(mesh_from_single_block(mixed.coarse->mesh_ptr(), 1), 1);
    SFEM_TEST_EQ(hex_fine->n_dofs(), mixed.fine->n_dofs());
    SFEM_TEST_EQ(tet_fine->n_dofs(), mixed.fine->n_dofs());
    SFEM_TEST_EQ(hex_coarse->n_dofs(), mixed.coarse->n_dofs());
    SFEM_TEST_EQ(tet_coarse->n_dofs(), mixed.coarse->n_dofs());

    std::vector<char> hex_fine_nodes((size_t)mixed.fine->n_dofs(), 0);
    std::vector<char> tet_fine_nodes((size_t)mixed.fine->n_dofs(), 0);
    std::vector<char> hex_coarse_nodes((size_t)mixed.coarse->n_dofs(), 0);
    std::vector<char> tet_coarse_nodes((size_t)mixed.coarse->n_dofs(), 0);
    mark_block_nodes(mixed.fine->mesh().block(0), hex_fine_nodes);
    mark_block_nodes(mixed.fine->mesh().block(1), tet_fine_nodes);
    mark_block_nodes(mixed.coarse->mesh().block(0), hex_coarse_nodes);
    mark_block_nodes(mixed.coarse->mesh().block(1), tet_coarse_nodes);

    std::vector<char> hex_exclusive_coarse = hex_coarse_nodes;
    std::vector<char> tet_exclusive_coarse = tet_coarse_nodes;
    for (size_t i = 0; i < hex_exclusive_coarse.size(); ++i) {
        if (hex_coarse_nodes[i] && tet_coarse_nodes[i]) {
            hex_exclusive_coarse[i] = 0;
            tet_exclusive_coarse[i] = 0;
        }
    }

    const auto es = sfem::EXECUTION_SPACE_HOST;

    {
        auto p_mixed = sfem::create_hierarchical_prolongation(mixed.coarse, mixed.fine, es);
        auto p_hex   = sfem::create_hierarchical_prolongation(hex_coarse, hex_fine, es);
        auto p_tet   = sfem::create_hierarchical_prolongation(tet_coarse, tet_fine, es);

        auto c_mixed = sfem::create_host_buffer<real_t>(mixed.coarse->n_dofs());
        auto c_hex   = sfem::create_host_buffer<real_t>(hex_coarse->n_dofs());
        auto c_tet   = sfem::create_host_buffer<real_t>(tet_coarse->n_dofs());
        auto f_mixed = sfem::create_host_buffer<real_t>(mixed.fine->n_dofs());
        auto f_hex   = sfem::create_host_buffer<real_t>(hex_fine->n_dofs());
        auto f_tet   = sfem::create_host_buffer<real_t>(tet_fine->n_dofs());

        fill_ones(c_mixed);
        fill_ones(c_hex);
        fill_ones(c_tet);
        SFEM_TEST_ASSERT(p_mixed->apply(c_mixed->data(), f_mixed->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_hex->apply(c_hex->data(), f_hex->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_tet->apply(c_tet->data(), f_tet->data()) == SFEM_SUCCESS);

        const real_t tol = val_tol();
        auto         dm  = f_mixed->data();
        for (ptrdiff_t i = 0; i < mixed.fine->n_dofs(); ++i) {
            SFEM_TEST_ASSERT(std::isfinite(dm[i]));
            SFEM_TEST_ASSERT(std::fabs(dm[i] - real_t(1)) <= tol);
        }
        SFEM_TEST_ASSERT(compare_masked(f_mixed, f_hex, hex_fine_nodes) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(compare_masked(f_mixed, f_tet, tet_fine_nodes) == SFEM_TEST_SUCCESS);
    }

    {
        auto r_mixed = sfem::create_hierarchical_restriction(mixed.fine, mixed.coarse, es);
        auto r_hex   = sfem::create_hierarchical_restriction(hex_fine, hex_coarse, es);
        auto r_tet   = sfem::create_hierarchical_restriction(tet_fine, tet_coarse, es);

        auto f_mixed = sfem::create_host_buffer<real_t>(mixed.fine->n_dofs());
        auto f_hex   = sfem::create_host_buffer<real_t>(hex_fine->n_dofs());
        auto f_tet   = sfem::create_host_buffer<real_t>(tet_fine->n_dofs());
        auto c_mixed = sfem::create_host_buffer<real_t>(mixed.coarse->n_dofs());
        auto c_hex   = sfem::create_host_buffer<real_t>(hex_coarse->n_dofs());
        auto c_tet   = sfem::create_host_buffer<real_t>(tet_coarse->n_dofs());

        fill_ones(f_mixed);
        fill_ones(f_hex);
        fill_ones(f_tet);
        SFEM_TEST_ASSERT(r_mixed->apply(f_mixed->data(), c_mixed->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_hex->apply(f_hex->data(), c_hex->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_tet->apply(f_tet->data(), c_tet->data()) == SFEM_SUCCESS);

        auto d = c_mixed->data();
        for (ptrdiff_t i = 0; i < mixed.coarse->n_dofs(); ++i) {
            SFEM_TEST_ASSERT(std::isfinite(d[i]));
        }
        SFEM_TEST_ASSERT(compare_masked(c_mixed, c_hex, hex_exclusive_coarse) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(compare_masked(c_mixed, c_tet, tet_exclusive_coarse) == SFEM_TEST_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_checkerboard_prolong_ones);
    SFEM_RUN_TEST(test_checkerboard_restrict_finite);
    SFEM_RUN_TEST(test_checkerboard_vs_cube);
    SFEM_RUN_TEST(test_two_block_quad4_vs_square);
    SFEM_RUN_TEST(test_two_block_tet4_vs_cube);
    SFEM_RUN_TEST(test_hex8_tet4_ss_vs_split_blocks);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

