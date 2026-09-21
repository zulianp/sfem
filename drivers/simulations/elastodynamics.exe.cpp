#include <memory>

#include "sfem_test.hpp"

#include "sfem_Function.hpp"

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_CRS.hpp"


#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "smesh_env.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_NewmarkScheme.hpp"
#include "sfem_ssmgc.hpp"

int solve_obstacle_problem(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_obstacle_problem");

    if (argc != 6) {
        fprintf(stderr, "usage: %s <mesh> <sdf> <dirichlet_conditions> <contact_boundary> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    smesh::Path mesh_path{argv[1]};
    smesh::Path sdf_path{argv[2]};
    smesh::Path dirichlet_path{argv[3]};
    smesh::Path contact_boundary_path{argv[4]};
    smesh::Path output_path{argv[5]};

    int SFEM_ELEMENT_REFINE_LEVEL = 2;

    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    const char *SFEM_OPERATOR = "KelvinVoigtNewmark";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;
    const char          *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const bool verbose = smesh::Env::read("SFEM_VERBOSE", false);

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(mesh_path));
    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        mesh = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, mesh, true, false);
    }
    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);

// FIXME
#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
    auto f                    = sfem::Function::create(fs);
    auto kv_op                = sfem::create_op(fs, SFEM_OPERATOR, es);
    kv_op->initialize();
    f->add_operator(kv_op);

#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        f->add_constraint(sfem::to_device(dirichlet_conditions));
    } else
#endif  // SFEM_ENABLE_CUDA
    {
        f->add_constraint(dirichlet_conditions);
    }

    auto sdf              = smesh::Grid<geom_t>::create_from_file(comm, sdf_path);
    auto contact_boundary = smesh::Sideset::create_from_file(comm, smesh::Path(contact_boundary_path));
    auto contact_conds    = sfem::ContactConditions::create(fs, sdf, {contact_boundary}, es);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(es);

    // Newmark with beta = 1/4 and gamma = 1/2, which is what the predictor and
    // corrector written out below came to.  The scheme carries the state, so
    // `displacement`, `velocity` and `acceleration` are its buffers.
    //
    // Its `InertiaPotential` is not added to the function here and stays
    // unused: `KelvinVoigtNewmark` takes the acceleration as a field and
    // carries the inertia itself.
    auto scheme = std::make_shared<sfem::NewmarkScheme>(fs, es);
    if (scheme->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    auto displacement = scheme->state();
    auto velocity     = scheme->velocity();
    auto acceleration = scheme->acceleration();
    auto increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto temp_vel     = sfem::create_buffer<real_t>(ndofs, es);
    auto solution     = sfem::create_buffer<real_t>(ndofs, es);
    auto g            = sfem::create_buffer<real_t>(ndofs, es);
    auto gap          = sfem::create_buffer<real_t>(ndofs, es);

    // Initialize all buffers to zero
    blas->zeros(ndofs, displacement->data());
    blas->zeros(ndofs, velocity->data());
    {
        // Built on the host and copied, because `velocity` is allocated in
        // `es` and the workflow for this driver runs on the device.
        auto      nnodes = fs->mesh().n_nodes();
        auto      dims   = fs->mesh_ptr()->spatial_dimension();
        auto      host_v = sfem::create_host_buffer<real_t>(ndofs);
        for (int i = 0; i < nnodes; i++) {
            host_v->data()[i * dims + 1] = 0.1;
        }
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            host_v = smesh::to_device(host_v);
        }
#endif
        blas->copy(ndofs, host_v->data(), velocity->data());
    }

    blas->zeros(ndofs, acceleration->data());
    blas->zeros(ndofs, solution->data());
    blas->zeros(ndofs, increment->data());
    blas->zeros(ndofs, temp_vel->data());
    blas->zeros(ndofs, g->data());

    // Time integration parameters
    real_t dt          = smesh::Env::read("SFEM_DT", 0.1);
    real_t T           = smesh::Env::read("SFEM_T_END", 5.0);
    size_t export_freq = smesh::Env::read("SFEM_EXPORT_FREQ", 1);
    int    nliter      = smesh::Env::read("SFEM_NLITER", 1);

    if (!comm->rank() && verbose) {
        printf("\n=== Elastodynamics with Contact Time Integration ===\n");
        printf("Number of DOFs: %td\n", ndofs);
        printf("Execution space: %s\n", SFEM_EXECUTION_SPACE ? SFEM_EXECUTION_SPACE : "HOST");
        printf("Refine level: %d\n", SFEM_ELEMENT_REFINE_LEVEL);
        printf("Time step: %g\n", dt);
        printf("End time: %g\n", T);
        printf("Export frequency: %zu\n", export_freq);
        printf("Nonlinear iterations per step: %d\n", nliter);
    }

    f->apply_constraints(solution->data());
    f->apply_constraints(displacement->data());
    contact_conds->init();

    // Set velocity and acceleration fields for KV operator
    kv_op->set_field("velocity", temp_vel, 0);
    kv_op->set_field("acceleration", increment, 0);

    int SFEM_USE_SPMG = 1;
    SFEM_READ_ENV(SFEM_USE_SPMG, atoi);

    std::shared_ptr<sfem::Input> in;
    const char                  *SFEM_SSMGC_YAML{nullptr};
    SFEM_READ_ENV(SFEM_SSMGC_YAML, );

    if (SFEM_SSMGC_YAML) {
        in = sfem::YAMLNoIndent::create_from_file(SFEM_SSMGC_YAML);
    }

    auto solver = sfem::create_ssmgc(f, contact_conds, in);

    // Setup output
    smesh::create_directory(output_path);
    smesh::create_directory(output_path / "out");

    fs->mesh_ptr()->write(smesh::Path(output_path / "coarse_mesh"));
    smesh::semistructured_export_as_standard(fs->mesh_ptr(), output_path / "mesh");

    auto out = f->output();
    out->set_output_dir(output_path / "out");
    out->enable_AoS_to_SoA(true);

    // Time variables
    real_t t     = 0.0;
    size_t steps = 0;

    bool SFEM_NEWMARK_ENABLE_OUTPUT = true;
    SFEM_READ_ENV(SFEM_NEWMARK_ENABLE_OUTPUT, atoi);

    // Write initial condition
    if (SFEM_NEWMARK_ENABLE_OUTPUT) {
        auto u = displacement;
        auto v = velocity;
        auto a = acceleration;
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            u = smesh::to_host(u);
            v = smesh::to_host(v);
            a = smesh::to_host(a);
        }
#endif
        out->write_time_step("disp", t, u->data());
        out->write_time_step("velocity", t, v->data());
        out->write_time_step("acceleration", t, a->data());
        out->log_time(t);

        // if (es != sfem::EXECUTION_SPACE_DEVICE) {
        contact_conds->update(displacement->data());
        contact_conds->signed_distance_for_mesh_viz(displacement->data(), gap->data());
        out->write_time_step("gap", t, gap->data());
        // }
    }

    // Time loop
    while (t < T) {
        // Opens the step: the shift, the history and the predictor are all
        // built from the state carried out of the last step, never from the
        // iterate the nonlinear loop below is moving.
        scheme->begin_step(t, dt);

        for (int k = 0; k < nliter; k++) {
            // The velocity and acceleration the method implies at the current
            // iterate.  These are the operator's fields, so they are refreshed
            // every nonlinear iteration rather than once per step.
            scheme->reconstruct(solution->data(), temp_vel->data(), increment->data());

            // Update contact conditions based on current solution
            // solver->update(solution->data());

            blas->zeros(ndofs, g->data());
            f->gradient(solution->data(), g->data());
            blas->scal(ndofs, -1.0, g->data());

            // Hack for linear functions
            f->apply(nullptr, solution->data(), g->data());

            // Solve for increment
            // blas->zeros(ndofs, increment->data());
            solver->apply(g->data(), solution->data());
            // blas->axpy(ndofs, -1, increment->data(), solution->data());
        }

        scheme->advance(solution->data());

        t += dt;
        if (++steps % export_freq == 0 && SFEM_NEWMARK_ENABLE_OUTPUT) {
            if (!comm->rank()) {
                printf("%g/%g\n", double(t), double(T));
            }

            auto u = displacement;
            auto v = velocity;
            auto a = acceleration;
#ifdef SFEM_ENABLE_CUDA
            if (es == sfem::EXECUTION_SPACE_DEVICE) {
                u = smesh::to_host(u);
                v = smesh::to_host(v);
                a = smesh::to_host(a);
            }
#endif
            // Write to disk
            out->write_time_step("disp", t, u->data());
            out->write_time_step("velocity", t, v->data());
            out->write_time_step("acceleration", t, a->data());

            // if (es != sfem::EXECUTION_SPACE_DEVICE) {
            contact_conds->update(displacement->data());
            contact_conds->signed_distance_for_mesh_viz(displacement->data(), gap->data());
            out->write_time_step("gap", t, gap->data());

            blas->zeros(ndofs, g->data());
            f->gradient(displacement->data(), g->data());

            auto contact_stress = sfem::create_buffer<real_t>(ndofs, es);
            blas->zeros(ndofs, contact_stress->data());
            contact_conds->full_apply_boundary_mass_inverse(g->data(), contact_stress->data());
            out->write_time_step("contact_stress", t, contact_stress->data());
            // }

            out->log_time(t);
        }
    }

    if (!comm->rank() && verbose) {
        printf("\n=== Simulation Complete ===\n");
        printf("Total steps: %zu\n", steps);
        printf("Final time: %g\n", t);
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_obstacle_problem(ctx->communicator(), argc, argv);
}
