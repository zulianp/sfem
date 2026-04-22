#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_SelfCollisions.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include "sfem_API.hpp"

#include <algorithm>
#include <utility>
#include <vector>

using namespace sfem;

std::shared_ptr<Function> create_function(const ptrdiff_t nx, const ExecutionSpace es) {
    auto mesh1 = smesh::Mesh::create_tet4_cube(Communicator::self(), nx, nx, nx, 0, 0, 0, 1, 1, 1);
    auto mesh2 = smesh::Mesh::create_tet4_cube(Communicator::self(), nx, nx, nx, 0.1, 1.1, 0.1, 0.9, 1.9, 0.9);

    auto mesh = smesh::concatenate(mesh1, mesh2);

    printf("Bulk: #nodes %zu #elements %zu\n", mesh->n_nodes(), mesh->n_elements());

    mesh->write(smesh::Path("contact_mesh"));

    auto top_ss = sfem::Sideset::create_from_selector(mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
        return y > (1.9 - 1e-4) && y < (1.9 + 1e-4);
    });

    auto bottom_ss = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > (-1e-4) && y < (1e-4); });

    const int dim   = mesh->spatial_dimension();
    auto      space = FunctionSpace::create(mesh, dim);

    auto op = create_op(space, "LinearElasticity", es);
    op->initialize();

    auto f = Function::create(space);
    f->add_operator(op);

    auto top_ns    = smesh::create_nodeset_from_sidesets(mesh, top_ss);
    auto bottom_ns = smesh::create_nodeset_from_sidesets(mesh, bottom_ss);

    assert(top_ns != nullptr);
    assert(bottom_ns != nullptr);
    assert(top_ns->size() > 0);
    assert(bottom_ns->size() > 0);

    DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
    DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = -0.2, .component = 1};
    DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

    DirichletConditions::Condition xbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 0};
    DirichletConditions::Condition ybottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 1};
    DirichletConditions::Condition zbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xbottom, ybottom, zbottom}, es);
    f->add_constraint(conds);

    return f;
}

int test_toi() {
    ptrdiff_t nx = 4;

    auto f       = create_function(nx, ExecutionSpace::EXECUTION_SPACE_HOST);
    auto space   = f->space();
    auto es      = f->execution_space();
    auto mesh    = space->mesh_ptr();
    auto surface = skin(mesh);

    const int dim = mesh->spatial_dimension();

    surface->write(smesh::Path("contact_surface"));

    printf("Surf: #nodes %zu #elements %zu\n", surface->n_nodes(), surface->n_elements());

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(1000);
    solver->set_rtol(1e-6);
    solver->set_verbose(false);

    auto previous_displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto displacement          = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs                   = sfem::create_buffer<real_t>(space->n_dofs(), es);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), displacement->data());

    auto prev_disp3 = convert_host_buffer_to_fake_SoA(dim, previous_displacement);
    auto disp3      = convert_host_buffer_to_fake_SoA(dim, displacement);

    auto collisions = SelfCollisions::create(surface);
    collisions->find(dim, prev_disp3->data(), disp3->data());
    real_t toi = collisions->time_of_impact();

    SFEM_TEST_APPROXEQ(toi, 0.5, 1e-2);
    printf("TOI: %g\n", toi);

    auto blas = sfem::blas<real_t>(es);
    blas->scal(space->n_dofs(), toi, displacement->data());

    auto d       = sfem::create_host_buffer<real_t>(surface->n_nodes());
    auto normals = sfem::create_host_buffer<real_t>(dim, surface->n_nodes());
    collisions->distance_and_normal(toi, d->data(), dim, normals->data());

    {
        Output out(FunctionSpace::create(surface, 1));
        out.enable_AoS_to_SoA(true);
        out.set_output_dir(smesh::Path("contac_surface_output"));
        out.write("d", d->data());
        out.write("nx", normals->data()[0]);
        out.write("ny", normals->data()[1]);
        out.write("nz", normals->data()[2]);
    }

    // TODO find actual collisions using scaled displacement (compute penalizer)
    // 0) Envelope or penetration ?
    // 1) VF distances and penalization
    // 2) EE distances and penalization

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));
    out->write("disp", displacement->data());

    return SFEM_TEST_SUCCESS;
}

int test_two_body_contact() {
    ptrdiff_t nx = 4;

    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    auto f     = create_function(nx, es);
    auto space = f->space();
    auto mesh  = space->mesh_ptr();

    auto pen = SelfContactPenalty::create(space);

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(1000);
    solver->set_rtol(1e-6);
    solver->set_verbose(false);

    auto previous_displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto displacement_old      = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto displacement          = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs                   = sfem::create_buffer<real_t>(space->n_dofs(), es);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), displacement->data());

    pen->update(displacement_old->data(), displacement->data());

    auto alpha = pen->max_step_size();
    blas->scal(space->n_dofs(), alpha, displacement->data());

    auto g = sfem::create_buffer<real_t>(space->n_dofs(), es);
    pen->gradient(displacement->data(), g->data());

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));
    out->write("g", g->data());

    // blas->norm(space->n_dofs(), g->data());
    printf("Gradient norm: %g\n", blas->norm2(space->n_dofs(), g->data()));

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_toi);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
