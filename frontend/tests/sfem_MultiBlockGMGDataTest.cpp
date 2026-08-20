#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_GeometricMultigrid.hpp"
#include "sfem_ssgmg.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

    constexpr real_t k_white = real_t(1);
    constexpr real_t k_black = real_t(10);

    struct SSGMGResidual {
        real_t abs_res{0};
        real_t rel_res{0};
    };

    void fill_ones(const sfem::SharedBuffer<real_t> &v) {
        auto            d = v->data();
        const ptrdiff_t n = (ptrdiff_t)v->size();
        for (ptrdiff_t i = 0; i < n; ++i) {
            d[i] = 1;
        }
    }

    int export_level_mesh(const std::shared_ptr<sfem::FunctionSpace> &fs, const smesh::Path &path) {
        if (fs->has_semi_structured_mesh()) {
            SFEM_TEST_ASSERT(smesh::semistructured_export_as_standard(fs->mesh_ptr(), path) == SFEM_SUCCESS);
        } else {
            SFEM_TEST_ASSERT(fs->mesh_ptr()->write(path) == SFEM_SUCCESS);
        }
        return SFEM_TEST_SUCCESS;
    }

    int assert_named_hex_blocks(const std::shared_ptr<smesh::Mesh> &mesh) {
        SFEM_TEST_ASSERT(mesh != nullptr);
        SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_ASSERT(mesh->block(0)->name() == "white");
        SFEM_TEST_ASSERT(mesh->block(1)->name() == "black");
        return SFEM_TEST_SUCCESS;
    }

    sfem::SharedBuffer<idx_t> nodeset_from_sidesets(const std::shared_ptr<smesh::Mesh>                 &mesh,
                                                    const std::vector<std::shared_ptr<smesh::Sideset>> &sidesets) {
        std::vector<idx_t> ids;
        for (const auto &ss : sidesets) {
            auto ns = smesh::create_nodeset_from_sideset(mesh, ss);
            if (!ns || ns->size() == 0) {
                continue;
            }
            auto d = ns->data();
            ids.insert(ids.end(), d, d + ns->size());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
        if (!ids.empty()) {
            std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
        }
        return out;
    }

    sfem::SharedBuffer<idx_t> nodeset_from_selector(const std::shared_ptr<smesh::Mesh> &mesh,
                                                    const std::function<bool(geom_t, geom_t, geom_t)> &selector) {
        std::vector<idx_t> ids;
        auto               p     = mesh->points()->data();
        const ptrdiff_t    n     = mesh->n_nodes();
        const bool         has_z = mesh->spatial_dimension() > 2;
        for (ptrdiff_t i = 0; i < n; ++i) {
            const geom_t z = has_z ? p[2][i] : geom_t(0);
            if (selector(p[0][i], p[1][i], z)) {
                ids.push_back((idx_t)i);
            }
        }

        auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
        if (!ids.empty()) {
            std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
        }
        return out;
    }

    void set_checkerboard_diffusion(const std::shared_ptr<sfem::Op> &op) {
        op->set_value_in_block("white", "k", k_white);
        op->set_value_in_block("black", "k", k_black);
    }

    void fill_x2(const smesh::Mesh &mesh, real_t *const v) {
        auto            p = mesh.points()->data();
        const ptrdiff_t n = mesh.n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            v[i] = real_t(p[0][i]) * real_t(p[0][i]);
        }
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

    int apply_laplacian(const std::shared_ptr<sfem::FunctionSpace> &fs,
                        const real_t                                k_w,
                        const real_t                                k_b,
                        const real_t *const                         x,
                        real_t *const                               y) {
        auto f  = sfem::Function::create(fs);
        auto op = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        op->set_value_in_block("white", "k", k_w);
        op->set_value_in_block("black", "k", k_b);
        f->add_operator(op);
        const ptrdiff_t n = fs->n_dofs();
        for (ptrdiff_t i = 0; i < n; ++i) {
            y[i] = 0;
        }
        SFEM_TEST_ASSERT(f->apply(nullptr, x, y) == SFEM_SUCCESS);
        return SFEM_TEST_SUCCESS;
    }

    void zero_buffer(const sfem::SharedBuffer<real_t> &v) { std::fill(v->data(), v->data() + v->size(), real_t(0)); }

    int assert_heterogeneous_apply(const std::shared_ptr<sfem::Function> &f) {
        auto fs = f->space();
        auto x  = sfem::create_host_buffer<real_t>(fs->n_dofs());
        fill_x2(*fs->mesh_ptr(), x->data());

        auto            y_het = sfem::create_host_buffer<real_t>(fs->n_dofs());
        auto            y_hom = sfem::create_host_buffer<real_t>(fs->n_dofs());
        const ptrdiff_t n     = fs->n_dofs();
        for (ptrdiff_t i = 0; i < n; ++i) {
            y_het->data()[i] = 0;
            y_hom->data()[i] = 0;
        }

        SFEM_TEST_ASSERT(f->apply(nullptr, x->data(), y_het->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(apply_laplacian(fs, real_t(1), real_t(1), x->data(), y_hom->data()) == SFEM_TEST_SUCCESS);

        real_t max_diff = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            SFEM_TEST_ASSERT(std::isfinite(y_het->data()[i]));
            max_diff = std::max(max_diff, std::fabs(y_het->data()[i] - y_hom->data()[i]));
        }
        SFEM_TEST_ASSERT(max_diff > real_t(1e-8));

        auto y_ref = sfem::create_host_buffer<real_t>(n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            y_ref->data()[i] = 0;
        }
        SFEM_TEST_ASSERT(apply_laplacian(fs, k_white, k_black, x->data(), y_ref->data()) == SFEM_TEST_SUCCESS);

        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-4);
        for (ptrdiff_t i = 0; i < n; ++i) {
            SFEM_TEST_ASSERT(std::fabs(y_het->data()[i] - y_ref->data()[i]) <= tol);
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_checkerboard_derefine_keeps_blocks() {
    auto hex = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    SFEM_TEST_ASSERT(assert_named_hex_blocks(hex) == SFEM_TEST_SUCCESS);

    auto ss   = smesh::to_semistructured(4, hex, true, false);
    auto fine = sfem::FunctionSpace::create(ss, 1);
    SFEM_TEST_ASSERT(fine->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(assert_named_hex_blocks(fine->mesh_ptr()) == SFEM_TEST_SUCCESS);

    auto mid = fine->derefine(2);
    SFEM_TEST_ASSERT(mid != nullptr);
    SFEM_TEST_ASSERT(mid->has_semi_structured_mesh());
    SFEM_TEST_EQ(mid->n_blocks(), static_cast<size_t>(2));
    SFEM_TEST_ASSERT(assert_named_hex_blocks(mid->mesh_ptr()) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(smesh::is_semistructured_type(mid->element_type(0)));
    SFEM_TEST_ASSERT(smesh::is_semistructured_type(mid->element_type(1)));

    auto coarse = fine->derefine(1);
    SFEM_TEST_ASSERT(coarse != nullptr);
    SFEM_TEST_ASSERT(!coarse->has_semi_structured_mesh());
    SFEM_TEST_EQ(coarse->n_blocks(), static_cast<size_t>(2));
    SFEM_TEST_ASSERT(assert_named_hex_blocks(coarse->mesh_ptr()) == SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ(coarse->element_type(0), smesh::HEX8);
    SFEM_TEST_EQ(coarse->element_type(1), smesh::HEX8);

    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_create_gmg_data() {
    auto hex = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto ss  = smesh::to_semistructured(4, hex, true, false);
    auto fs  = sfem::FunctionSpace::create(ss, 1);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    set_checkerboard_diffusion(op);
    f->add_operator(op);

    auto data = sfem::create_gmg_data(f);
    SFEM_TEST_ASSERT(data != nullptr);

    auto levels = smesh::derefinement_levels(fs->mesh());
    SFEM_TEST_EQ(data->functions.size(), levels.size());
    SFEM_TEST_EQ(data->restrictions.size(), levels.size() - 1);
    SFEM_TEST_EQ(data->prolongations.size(), levels.size());
    SFEM_TEST_ASSERT(data->prolongations.front() == nullptr);

    for (size_t i = 0; i < data->functions.size(); ++i) {
        auto space = data->functions[i]->space();
        SFEM_TEST_ASSERT(space != nullptr);
        SFEM_TEST_EQ(space->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_ASSERT(assert_named_hex_blocks(space->mesh_ptr()) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(assert_heterogeneous_apply(data->functions[i]) == SFEM_TEST_SUCCESS);
    }

    for (size_t i = 0; i < data->restrictions.size(); ++i) {
        SFEM_TEST_ASSERT(data->restrictions[i] != nullptr);
    }
    for (size_t i = 1; i < data->prolongations.size(); ++i) {
        SFEM_TEST_ASSERT(data->prolongations[i] != nullptr);
    }

    auto coarse = sfem::create_host_buffer<real_t>(data->functions[1]->space()->n_dofs());
    auto fine_v = sfem::create_host_buffer<real_t>(data->functions[0]->space()->n_dofs());
    fill_ones(coarse);
    SFEM_TEST_ASSERT(data->prolongations[1]->apply(coarse->data(), fine_v->data()) == SFEM_SUCCESS);

    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-12) : real_t(1e-5);
    auto         d   = fine_v->data();
    for (ptrdiff_t i = 0; i < data->functions[0]->space()->n_dofs(); ++i) {
        SFEM_TEST_ASSERT(std::isfinite(d[i]));
        SFEM_TEST_ASSERT(std::fabs(d[i] - real_t(1)) <= tol);
    }

    const bool enable_output = smesh::Env::read<bool>("SFEM_ENABLE_OUTPUT", false);
    if (enable_output) {
        const smesh::Path root("test_multiblock_gmg_data");
        smesh::create_directory(root);
        for (size_t i = 0; i < data->functions.size(); ++i) {
            auto level_dir = root / ("level_" + std::to_string(i));
            smesh::create_directory(level_dir);
            SFEM_TEST_ASSERT(export_level_mesh(data->functions[i]->space(), level_dir / "mesh") == SFEM_TEST_SUCCESS);
        }
    }

    return SFEM_TEST_SUCCESS;
}

std::shared_ptr<sfem::Function> make_ss_poisson(const std::shared_ptr<sfem::Mesh> &hex, const bool checkerboard_diffusion) {
    auto ss = smesh::to_semistructured(4, hex, true, false);
    auto fs = sfem::FunctionSpace::create(ss, 1);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    if (!op || op->initialize() != SFEM_SUCCESS) {
        return nullptr;
    }
    if (checkerboard_diffusion) {
        set_checkerboard_diffusion(op);
    }
    f->add_operator(op);

    auto bottom_ns = nodeset_from_selector(
            ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
    auto right_ns = nodeset_from_selector(
            ss, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 1 - 1e-5 && x < 1 + 1e-5; });

    sfem::DirichletConditions::Condition left{.nodeset = bottom_ns, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.nodeset = right_ns, .value = 1, .component = 0};
    f->add_constraint(sfem::create_dirichlet_conditions(fs, {left, right}, sfem::EXECUTION_SPACE_HOST));
    return f;
}

std::shared_ptr<sfem::Function> make_checkerboard_ss_poisson() {
    auto hex = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    return make_ss_poisson(hex, true);
}

int compute_ssgmg_residual(const std::shared_ptr<sfem::Function> &f, SSGMGResidual &residual) {
    auto fs = f->space();

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    zero_buffer(x);
    zero_buffer(rhs);
    SFEM_TEST_ASSERT(f->apply_constraints(x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(f->apply_constraints(rhs->data()) == SFEM_SUCCESS);

    auto mg = sfem::create_ssgmg(f, f->execution_space());
    SFEM_TEST_ASSERT(mg != nullptr);
    mg->verbose = false;
    mg->set_max_it(smesh::Env::read("SFEM_MG_MAX_IT", 40));
    mg->set_atol(smesh::Env::read("SFEM_MG_ATOL", real_t(1e-10)));
    SFEM_TEST_ASSERT(mg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);

    auto A  = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, f->execution_space());
    auto ax = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    SFEM_TEST_ASSERT(A->apply(x->data(), ax->data()) == SFEM_SUCCESS);
    mg->blas()->axpby(fs->n_dofs(), real_t(1), rhs->data(), real_t(-1), ax->data());

    residual.abs_res     = mg->blas()->norm2(fs->n_dofs(), ax->data());
    const real_t rhs_nrm = mg->blas()->norm2(fs->n_dofs(), rhs->data());
    residual.rel_res     = residual.abs_res / (rhs_nrm + real_t(1e-16));
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_dirichlet_hessian_diag() {
    auto f = make_checkerboard_ss_poisson();
    SFEM_TEST_ASSERT(f != nullptr);

    auto data = sfem::create_gmg_data(f);
    SFEM_TEST_ASSERT(data != nullptr);

    for (size_t i = 0; i < data->functions.size(); ++i) {
        auto fi   = data->functions[i];
        auto diag = sfem::create_host_buffer<real_t>(fi->space()->n_dofs());
        SFEM_TEST_ASSERT(fi->hessian_diag(nullptr, diag->data()) == SFEM_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_ssgmg_residual() {
    auto checkerboard_f = make_checkerboard_ss_poisson();
    SFEM_TEST_ASSERT(checkerboard_f != nullptr);

    auto cube_hex = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cube_f   = make_ss_poisson(cube_hex, false);
    SFEM_TEST_ASSERT(cube_f != nullptr);

    SSGMGResidual checkerboard;
    SSGMGResidual cube;
    SFEM_TEST_ASSERT(compute_ssgmg_residual(checkerboard_f, checkerboard) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compute_ssgmg_residual(cube_f, cube) == SFEM_TEST_SUCCESS);

    printf("checkerboard ssgmg residual abs %g rel %g; cube abs %g rel %g\n",
           (double)checkerboard.abs_res,
           (double)checkerboard.rel_res,
           (double)cube.abs_res,
           (double)cube.rel_res);

    const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    SFEM_TEST_ASSERT(checkerboard.abs_res < abs_tol || checkerboard.rel_res < abs_tol);
    SFEM_TEST_ASSERT(cube.abs_res < abs_tol || cube.rel_res < abs_tol);

    const real_t comparable_factor = real_t(100);
    SFEM_TEST_ASSERT(checkerboard.rel_res <= comparable_factor * cube.rel_res + abs_tol);
    return SFEM_TEST_SUCCESS;
}

int test_two_block_quad4_ssgmg_residual() {
    auto split  = create_two_block_quad4_square(4, 4);
    auto square = sfem::Mesh::create_quad4_square(sfem::Communicator::self(), 4, 4, 0, 0, 1, 1);
    SFEM_TEST_ASSERT(split != nullptr);
    SFEM_TEST_ASSERT(square != nullptr);

    auto split_f  = make_ss_poisson(split, false);
    auto square_f = make_ss_poisson(square, false);
    SFEM_TEST_ASSERT(split_f != nullptr);
    SFEM_TEST_ASSERT(square_f != nullptr);

    auto split_data = sfem::create_gmg_data(split_f);
    SFEM_TEST_ASSERT(split_data != nullptr);
    SFEM_TEST_EQ(split_data->functions.size(), smesh::derefinement_levels(split_f->space()->mesh()).size());

    SSGMGResidual split_res;
    SSGMGResidual square_res;
    SFEM_TEST_ASSERT(compute_ssgmg_residual(split_f, split_res) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compute_ssgmg_residual(square_f, square_res) == SFEM_TEST_SUCCESS);

    printf("two-block quad4 ssgmg residual abs %g rel %g; square abs %g rel %g\n",
           (double)split_res.abs_res,
           (double)split_res.rel_res,
           (double)square_res.abs_res,
           (double)square_res.rel_res);

    const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    SFEM_TEST_ASSERT(split_res.abs_res < abs_tol || split_res.rel_res < abs_tol);
    SFEM_TEST_ASSERT(square_res.abs_res < abs_tol || square_res.rel_res < abs_tol);

    const real_t comparable_factor = real_t(100);
    SFEM_TEST_ASSERT(split_res.rel_res <= comparable_factor * square_res.rel_res + abs_tol);
    return SFEM_TEST_SUCCESS;
}

int test_two_block_tet4_ssgmg_residual() {
    auto split = create_two_block_tet4_cube(2, 2, 2);
    auto cube  = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2);
    SFEM_TEST_ASSERT(split != nullptr);
    SFEM_TEST_ASSERT(cube != nullptr);

    auto split_f = make_ss_poisson(split, false);
    auto cube_f  = make_ss_poisson(cube, false);
    SFEM_TEST_ASSERT(split_f != nullptr);
    SFEM_TEST_ASSERT(cube_f != nullptr);

    auto split_data = sfem::create_gmg_data(split_f);
    SFEM_TEST_ASSERT(split_data != nullptr);
    SFEM_TEST_EQ(split_data->functions.size(), smesh::derefinement_levels(split_f->space()->mesh()).size());

    SSGMGResidual split_res;
    SSGMGResidual cube_res;
    SFEM_TEST_ASSERT(compute_ssgmg_residual(split_f, split_res) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compute_ssgmg_residual(cube_f, cube_res) == SFEM_TEST_SUCCESS);

    printf("two-block tet4 ssgmg residual abs %g rel %g; cube abs %g rel %g\n",
           (double)split_res.abs_res,
           (double)split_res.rel_res,
           (double)cube_res.abs_res,
           (double)cube_res.rel_res);

    const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    const real_t rel_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    SFEM_TEST_ASSERT(split_res.abs_res < abs_tol || split_res.rel_res < rel_tol);
    SFEM_TEST_ASSERT(cube_res.abs_res < abs_tol || cube_res.rel_res < rel_tol);

    const real_t comparable_factor = real_t(100);
    SFEM_TEST_ASSERT(split_res.rel_res <= comparable_factor * cube_res.rel_res + rel_tol);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_checkerboard_derefine_keeps_blocks);
    SFEM_RUN_TEST(test_checkerboard_create_gmg_data);
    SFEM_RUN_TEST(test_checkerboard_dirichlet_hessian_diag);
    SFEM_RUN_TEST(test_checkerboard_ssgmg_residual);
    SFEM_RUN_TEST(test_two_block_quad4_ssgmg_residual);
    SFEM_RUN_TEST(test_two_block_tet4_ssgmg_residual);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
