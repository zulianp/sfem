#include <algorithm>
#include <cmath>
#include <memory>

#include "sfem_test.hpp"

#define private public
#include "../contact/sfem_MaMAL.hpp"
#undef private

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include "../contact/sfem_MaMAL.cpp"
#include "../contact/sfem_SelfContact.cpp"

std::shared_ptr<sfem::Function> create_touching_two_body_function(const sfem::ExecutionSpace es) {
    const ptrdiff_t n  = smesh::Env::read("SFEM_BASE_RESOLUTION", 2);
    const ptrdiff_t nx = n;
    const ptrdiff_t ny = n;
    const ptrdiff_t nz = n;

    const geom_t poc = 1;  // Use 1 for actual contact

    auto lower = smesh::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, nx, nx, nx, 0, 0, 0, 1, 1, 1);
    auto upper = smesh::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, nx, ny, nz, 0.25, poc, 0.25, 0.75, 1.75, 0.75);
    auto mesh  = smesh::concatenate(lower, upper);

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
    sfem::DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = -0.05, .component = 1};
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
    SFEM_RUN_TEST(test_mamal_nonlinear_cycle);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
