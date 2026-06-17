#include <algorithm>
#include <cmath>
#include <memory>

#include "sfem_test.hpp"

#define private public
#include "../contact/sfem_MaMAL.hpp"
#undef private

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include "../contact/sfem_ContactSolveKernels.cpp"
#include "../contact/sfem_MaMAL.cpp"
#include "../contact/sfem_SelfContact.cpp"

static real_t contact_objective(const ptrdiff_t     nnodes,
                                const count_t*      rowptr,
                                const idx_t*        colidx,
                                const real_t*       vals,
                                const real_t*       distances,
                                const real_t*       agumentation,
                                const real_t* const normals[3],
                                const real_t*       mass,
                                const real_t        penalty,
                                const real_t*       x) {
    static const real_t zero_step[1] = {0};
    real_t              value[1]     = {0};

    sfem::contact_objective_steps(3,
                                  nnodes,
                                  rowptr,
                                  colidx,
                                  vals,
                                  distances,
                                  agumentation,
                                  normals,
                                  mass,
                                  penalty,
                                  x,
                                  x,
                                  1,
                                  zero_step,
                                  value);
    return value[0];
}

static void contact_gradient(const ptrdiff_t     nnodes,
                             const count_t*      rowptr,
                             const idx_t*        colidx,
                             const real_t*       vals,
                             const real_t*       distances,
                             const real_t*       agumentation,
                             const real_t* const normals[3],
                             const real_t*       mass,
                             const real_t        penalty,
                             const real_t*       x,
                             real_t*             macaulay,
                             real_t*             grad) {
    const real_t* const x_soa[3] = {x + 0, x + 1, x + 2};

    sfem::compute_macaulay_term(
            3, nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, 3, x_soa, macaulay);

    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) {
        grad[i] = 0;
    }

    sfem::assemble_contact_gradient(
            3, nnodes, penalty, rowptr, colidx, vals, distances, agumentation, normals, mass, macaulay, grad);
}

static void contact_hessian_apply_aos(const ptrdiff_t     nnodes,
                                      const count_t*      rowptr,
                                      const idx_t*        colidx,
                                      const real_t*       vals,
                                      const real_t* const normals[3],
                                      const real_t*       mass,
                                      const real_t        penalty,
                                      const real_t*       macaulay,
                                      const real_t*       x,
                                      real_t*             y) {
    const real_t* const x_soa[3] = {x + 0, x + 1, x + 2};
    real_t* const       y_soa[3] = {y + 0, y + 1, y + 2};

    for (ptrdiff_t i = 0; i < 3 * nnodes; ++i) {
        y[i] = 0;
    }

    sfem::contact_hessian_apply(3, nnodes, rowptr, colidx, vals, normals, mass, penalty, macaulay, 3, x_soa, 3, y_soa);
}

int test_contact_objective_gradient_hessian_finite_differences() {
    static const ptrdiff_t nnodes = 4;
    static const ptrdiff_t ndofs  = 3 * nnodes;

    const count_t rowptr[nnodes + 1] = {0, 2, 3, 4, 6};
    const idx_t   colidx[6]          = {1, 2, 2, 3, 0, 1};
    const real_t  vals[6]            = {0.35, 0.25, 0.40, 0.55, 0.20, 0.30};

    const real_t nx[nnodes] = {0.84, -0.21, 0.36, -0.48};
    const real_t ny[nnodes] = {0.28, 0.91, -0.48, 0.64};
    const real_t nz[nnodes] = {0.46, 0.35, 0.80, 0.60};

    const real_t* const normals[3] = {nx, ny, nz};
    const real_t        mass[nnodes] = {1.20, 0.85, 1.10, 0.95};
    const real_t        distances[nnodes] = {-0.35, -0.28, -0.31, -0.25};
    const real_t        agumentation[nnodes] = {0.40, -0.15, 0.20, -0.10};
    const real_t        penalty = 17.0;

    const real_t x[ndofs] = {0.12,  -0.08, 0.05,  -0.03, 0.11,  -0.06,
                             0.07,  0.04,  -0.02, -0.09, 0.06,  0.10};
    const real_t p[ndofs] = {0.31,  -0.17, 0.23,  -0.29, 0.19,  -0.11,
                             0.13,  0.07,  -0.37, 0.21,  -0.05, 0.09};

    real_t macaulay[nnodes];
    real_t grad[ndofs];
    real_t hp[ndofs];

    contact_gradient(nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, x, macaulay, grad);
    contact_hessian_apply_aos(nnodes, rowptr, colidx, vals, normals, mass, penalty, macaulay, p, hp);

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        SFEM_TEST_ASSERT(macaulay[i] > 1e-2);
    }

    const real_t eps = 1e-6;
    real_t       xp[ndofs];
    real_t       xm[ndofs];
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        xp[i] = x[i] + eps * p[i];
        xm[i] = x[i] - eps * p[i];
    }

    const real_t fp = contact_objective(nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, xp);
    const real_t fm = contact_objective(nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, xm);

    real_t grad_dot_p = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        grad_dot_p += grad[i] * p[i];
    }

    const real_t fd_grad_dot_p = (fp - fm) / (2 * eps);
    SFEM_TEST_ASSERT(std::fabs(fd_grad_dot_p - grad_dot_p) < 1e-7);

    real_t macaulay_p[nnodes];
    real_t macaulay_m[nnodes];
    real_t grad_p[ndofs];
    real_t grad_m[ndofs];
    contact_gradient(nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, xp, macaulay_p, grad_p);
    contact_gradient(nnodes, rowptr, colidx, vals, distances, agumentation, normals, mass, penalty, xm, macaulay_m, grad_m);

    real_t max_hessian_error = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        const real_t fd_hp = (grad_p[i] - grad_m[i]) / (2 * eps);
        max_hessian_error  = std::max(max_hessian_error, std::fabs(fd_hp - hp[i]));
    }

    SFEM_TEST_ASSERT(max_hessian_error < 1e-7);

    return SFEM_TEST_SUCCESS;
}

std::shared_ptr<sfem::Function> create_touching_two_body_function(const sfem::ExecutionSpace es) {
    const ptrdiff_t n  = smesh::Env::read("SFEM_BASE_RESOLUTION", 2);
    const ptrdiff_t nx = n;
    const ptrdiff_t ny = n;
    const ptrdiff_t nz = n;

    const geom_t poc = 1;  // Use 1 for actual contact

    auto lower = smesh::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, nx, nx, nx, 0, 0, 0, 1, 1, 1);
    auto upper = smesh::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, nx, ny, nz, 0.25, poc, 0.25, 0.75, 1.75, 0.75);
    auto mesh  = smesh::concatenate(upper, lower);

    mesh = smesh::to_semistructured(16, mesh, true, false);

    const int dim   = mesh->spatial_dimension();
    auto      space = sfem::FunctionSpace::create(mesh, dim);
    auto      f     = sfem::Function::create(space);
    auto      op    = sfem::create_op(space, "LinearElasticity", es);
    op->initialize();
    f->add_operator(op);

    auto top_ss = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t, const geom_t y, const geom_t) -> bool { return y > (1.75 - 1e-5) && y < (1.75 + 1e-5); });

    auto left_ss = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t z) -> bool { return x > -1e-5 && x < 1e-5; });

    auto right_ss = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t z) -> bool { return x > (1 - 1e-5) && x < (1 + 1e-5); });

    auto top_ns   = smesh::create_nodeset_from_sidesets(mesh, top_ss);
    auto left_ns  = smesh::create_nodeset_from_sidesets(mesh, left_ss);
    auto right_ns = smesh::create_nodeset_from_sidesets(mesh, right_ss);

    assert(top_ns && top_ns->size() > 0);
    assert(left_ns);
    assert(right_ns);

    sfem::DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = -0.3, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

    sfem::DirichletConditions::Condition xleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition yleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition zleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 2};

    sfem::DirichletConditions::Condition xright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition yright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition zright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 2};

    f->add_constraint(
            sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xleft, yleft, zleft, xright, yright, zright}, es));

    return f;
}

int test_mamal_nonlinear_cycle() {
    auto es    = sfem::EXECUTION_SPACE_HOST;
    auto f     = create_touching_two_body_function(es);
    auto space = f->space();

    auto x    = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto blas = sfem::blas<real_t>(es);
    blas->values(x->size(), 0, x->data());
    f->apply_constraints(x->data());

    auto mamal = sfem::MaMAL::create(f);
    auto impl  = mamal->impl_.get();

    SFEM_TEST_ASSERT(impl->n_levels() > 1);
    SFEM_TEST_ASSERT(impl->galerkin_restrictions.size() > 1);

    SFEM_TEST_ASSERT(mamal->solve(x) == SFEM_SUCCESS);

    SFEM_TEST_ASSERT(impl->coupling_matrices.size() == std::size_t(impl->n_levels()));
    SFEM_TEST_ASSERT(impl->contact_block_diag.size() == std::size_t(impl->n_levels()));

    for (int l = 0; l < impl->n_levels(); ++l) {
        const real_t* const sol = impl->memory[l]->solution->data();
        for (ptrdiff_t i = 0; i < impl->memory[l]->solution->size(); ++i) {
            SFEM_TEST_ASSERT(std::isfinite(sol[i]));
        }

        const real_t* const diag = impl->contact_block_diag_aos[l]->data();
        for (ptrdiff_t i = 0; i < impl->contact_block_diag_aos[l]->size(); ++i) {
            SFEM_TEST_ASSERT(std::isfinite(diag[i]));
        }
    }

    const smesh::Path output_dir("mamal_output");
    smesh::create_directory(output_dir);

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(output_dir);

    SFEM_TEST_ASSERT(smesh::semistructured_export_as_standard(space->mesh_ptr(), output_dir) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(impl->contact_eval_surface->write(smesh::Path("mamal_contact_surface")) == SFEM_SUCCESS);

    auto lagr_mult_normal = sfem::create_buffer<real_t>(space->n_dofs(), es);
    blas->values(lagr_mult_normal->size(), 0, lagr_mult_normal->data());

    {
        const idx_t* const   node_mapping          = impl->contact_eval_surface->node_mapping()->data();
        const real_t* const  lagr_mult             = impl->agumentation->data();
        const real_t* const  normal_x              = impl->contact->normals()->data()[0];
        const real_t* const  normal_y              = impl->contact->normals()->data()[1];
        const real_t* const  normal_z              = impl->contact->normals()->data()[2];
        real_t* const        lagr_mult_normal_data = lagr_mult_normal->data();
        const ptrdiff_t      n                     = impl->contact_eval_surface->node_mapping()->size();
        static constexpr int dim                   = 3;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t dof            = node_mapping[i] * dim;
            const real_t    lm             = lagr_mult[i];
            lagr_mult_normal_data[dof]     = lm * normal_x[i];
            lagr_mult_normal_data[dof + 1] = lm * normal_y[i];
            lagr_mult_normal_data[dof + 2] = lm * normal_z[i];
        }
    }

    SFEM_TEST_ASSERT(out->write("disp", x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("distance", impl->contact->distances_whole()->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("directors", impl->contact->directors()->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("lagr_mult_normal", lagr_mult_normal->data()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_contact_objective_gradient_hessian_finite_differences);
    SFEM_RUN_TEST(test_mamal_nonlinear_cycle);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
