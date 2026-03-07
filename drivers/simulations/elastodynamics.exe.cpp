#include <memory>

#include "sfem_test.hpp"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.hpp"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "smesh_env.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

int solve_obstacle_problem(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_obstacle_problem");

    if (argc != 6) {
        fprintf(stderr, "usage: %s <mesh> <sdf> <dirichlet_conditions> <contact_boundary> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    const char *mesh_path             = argv[1];
    const char *sdf_path              = argv[2];
    const char *dirichlet_path        = argv[3];
    const char *contact_boundary_path = argv[4];
    std::string output_path           = argv[5];

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

    auto      mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(mesh_path));
    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);

    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

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

    auto sdf              = sfem::Grid<geom_t>::create_from_file(comm, sdf_path);
    auto contact_boundary = sfem::Sideset::create_from_file(comm, contact_boundary_path);
    auto contact_conds    = sfem::ContactConditions::create(fs, sdf, {contact_boundary}, es);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(es);

    // Create state vectors for time integration
    auto displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto velocity     = sfem::create_buffer<real_t>(ndofs, es);
    auto acceleration = sfem::create_buffer<real_t>(ndofs, es);
    auto increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto temp_vel     = sfem::create_buffer<real_t>(ndofs, es);
    auto solution     = sfem::create_buffer<real_t>(ndofs, es);
    auto g            = sfem::create_buffer<real_t>(ndofs, es);
    auto gap          = sfem::create_buffer<real_t>(ndofs, es);

    // Initialize all buffers to zero
    blas->zeros(ndofs, displacement->data());
    blas->zeros(ndofs, velocity->data());
    {
        auto nnodes = fs->semi_structured_mesh().n_nodes();
        auto dims = fs->mesh_ptr()->spatial_dimension();
        auto v = velocity->data();
        for (int i = 0; i < nnodes; i++) {
            v[i * dims + 1] = 0.1;
        }
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
    sfem::create_directory(output_path.c_str());
    sfem::create_directory((output_path + "/out").c_str());

    fs->mesh_ptr()->write(smesh::Path((output_path + "/coarse_mesh")));
    fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());

    auto out = f->output();
    out->set_output_dir((output_path + "/out").c_str());
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
            u = sfem::to_host(u);
            v = sfem::to_host(v);
            a = sfem::to_host(a);
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
        for (int k = 0; k < nliter; k++) {
            // Use increment as temp buffer for acceleration prediction
            blas->zeros(ndofs, increment->data());
            blas->zaxpby(ndofs, 1, solution->data(), -1, displacement->data(), increment->data());
            blas->axpy(ndofs, -dt, velocity->data(), increment->data());
            blas->scal(ndofs, 4 / (dt * dt), increment->data());
            blas->axpy(ndofs, -1, acceleration->data(), increment->data());

            // Compute velocity prediction
            blas->zeros(ndofs, temp_vel->data());
            blas->copy(ndofs, increment->data(), temp_vel->data());
            blas->zaxpby(ndofs, dt / 2, temp_vel->data(), dt / 2, acceleration->data(), temp_vel->data());
            blas->axpy(ndofs, 1, velocity->data(), temp_vel->data());

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

        // Update all quantities
        // acceleration
        blas->axpby(ndofs, -4 / (dt * dt), displacement->data(), -1, acceleration->data());
        blas->axpy(ndofs, 4 / (dt * dt), solution->data(), acceleration->data());
        blas->axpy(ndofs, -4 / dt, velocity->data(), acceleration->data());

        // velocity
        blas->axpby(ndofs, -2 / dt, displacement->data(), -1, velocity->data());
        blas->axpy(ndofs, 2 / dt, solution->data(), velocity->data());

        // displacement
        blas->copy(ndofs, solution->data(), displacement->data());

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
                u = sfem::to_host(u);
                v = sfem::to_host(v);
                a = sfem::to_host(a);
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
