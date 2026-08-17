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

int assert_heterogeneous_apply(const std::shared_ptr<sfem::Function> &f) {
    auto fs = f->space();
    auto x  = sfem::create_host_buffer<real_t>(fs->n_dofs());
    fill_x2(*fs->mesh_ptr(), x->data());

    auto y_het = sfem::create_host_buffer<real_t>(fs->n_dofs());
    auto y_hom = sfem::create_host_buffer<real_t>(fs->n_dofs());
    const ptrdiff_t n = fs->n_dofs();
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

    auto ss    = smesh::to_semistructured(4, hex, true, false);
    auto fine  = sfem::FunctionSpace::create(ss, 1);
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

std::shared_ptr<sfem::Function> make_checkerboard_ss_poisson() {
    auto hex = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto ss  = smesh::to_semistructured(4, hex, true, false);
    auto fs  = sfem::FunctionSpace::create(ss, 1);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    if (!op || op->initialize() != SFEM_SUCCESS) {
        return nullptr;
    }
    set_checkerboard_diffusion(op);
    f->add_operator(op);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
    auto right_ss = sfem::Sideset::create_from_selector(
            ss, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 1 - 1e-5 && x < 1 + 1e-5; });

    sfem::DirichletConditions::Condition left{.nodeset = nodeset_from_sidesets(ss, bottom_ss), .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.nodeset = nodeset_from_sidesets(ss, right_ss), .value = 1, .component = 0};
    f->add_constraint(sfem::create_dirichlet_conditions(fs, {left, right}, sfem::EXECUTION_SPACE_HOST));
    return f;
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
    auto f = make_checkerboard_ss_poisson();
    SFEM_TEST_ASSERT(f != nullptr);
    auto fs = f->space();

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
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
    const real_t abs_res = mg->blas()->norm2(fs->n_dofs(), ax->data());
    const real_t rhs_nrm = mg->blas()->norm2(fs->n_dofs(), rhs->data());
    const real_t rel_res = abs_res / (rhs_nrm + real_t(1e-16));
    printf("checkerboard ssgmg residual abs %g rel %g\n", (double)abs_res, (double)rel_res);

    const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
    SFEM_TEST_ASSERT(abs_res < abs_tol || rel_res < abs_tol);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_checkerboard_derefine_keeps_blocks);
    SFEM_RUN_TEST(test_checkerboard_create_gmg_data);
    SFEM_RUN_TEST(test_checkerboard_dirichlet_hessian_diag);
    SFEM_RUN_TEST(test_checkerboard_ssgmg_residual);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

