#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Env.hpp"
#include "sfem_Function.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "sfem_mesh_write.h"
#include "sfem_ssgmg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

int solve_kelvin_voigt_newmark(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    // Read environment variables
    auto        es                   = sfem::EXECUTION_SPACE_HOST;
    const char *SFEM_EXECUTION_SPACE = nullptr;
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    const bool verbose = sfem::Env::read("SFEM_VERBOSE", false);

    // Parse command line arguments
    if (argc != 5) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <mesh> <dirichlet_conditions> <output> <neumann_conditions>\n", argv[0]);
        }
        return 1;
    }

    const char *mesh_path      = argv[1];
    const char *dirichlet_path = argv[2];
    std::string output_path    = argv[3];
    const char *neumann_path   = argv[4];

    auto m = sfem::Mesh::create_from_file(comm, mesh_path);

    // Create function space
    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
        fs->semi_structured_mesh().apply_hierarchical_renumbering();
    }

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

    auto f = sfem::Function::create(fs);

    // Load boundary conditions
    auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);

    auto dirichlet_conditions_gpu = sfem::create_dirichlet_conditions(fs, dirichlet_conditions->conditions(), es);

    f->add_constraint(dirichlet_conditions_gpu);

    // Create Neumann conditions from environment variables (returns empty if unset)
    auto neumann_conditions = sfem::NeumannConditions::create_from_env(fs);

    auto neumann_conditions_gpu = sfem::create_neumann_conditions(fs, neumann_conditions->conditions(), es);

    f->add_operator(neumann_conditions_gpu);

    if (!comm->rank() && verbose) {
        printf("Loaded boundary conditions from: %s\n", dirichlet_path);
        printf("Loaded Neumann conditions from: %s\n", neumann_path);
    }

    // Create Kelvin-Voigt-Newmark operator
    auto kv_op = sfem::create_op(fs, "KelvinVoigtNewmark", es);
    kv_op->initialize();
    f->add_operator(kv_op);

    // Get problem size
    auto            blas  = sfem::blas<real_t>(es);
    const ptrdiff_t ndofs = fs->n_dofs();

    if (!comm->rank() && verbose) {
        printf("\n=== Kelvin-Voigt Newmark Time Integration ===\n");
        printf("Number of DOFs: %td\n", ndofs);
        printf("Execution space: %s\n", SFEM_EXECUTION_SPACE ? SFEM_EXECUTION_SPACE : "HOST");
        printf("Refine level: %d\n", SFEM_ELEMENT_REFINE_LEVEL);
    }

    // Create state vectors
    auto displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto velocity     = sfem::create_buffer<real_t>(ndofs, es);
    auto acceleration = sfem::create_buffer<real_t>(ndofs, es);
    auto increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto temp_vel     = sfem::create_buffer<real_t>(ndofs, es);
    auto solution     = sfem::create_buffer<real_t>(ndofs, es);
    auto g            = sfem::create_buffer<real_t>(ndofs, es);

    // Initialize all buffers to zero
    blas->zeros(ndofs, displacement->data());
    blas->zeros(ndofs, velocity->data());
    blas->zeros(ndofs, acceleration->data());
    blas->zeros(ndofs, solution->data());
    blas->zeros(ndofs, increment->data());
    blas->zeros(ndofs, temp_vel->data());
    blas->zeros(ndofs, g->data());

    // Time integration parameters
    real_t dt          = sfem::Env::read("SFEM_DT", 0.1);
    real_t T           = sfem::Env::read("SFEM_T_END", 5.0);
    size_t export_freq = sfem::Env::read("SFEM_EXPORT_FREQ", 1);
    int    nliter      = sfem::Env::read("SFEM_NLITER", 1);

    if (!comm->rank() && verbose) {
        printf("Time step: %g\n", dt);
        printf("End time: %g\n", T);
        printf("Export frequency: %zu\n", export_freq);
        printf("Nonlinear iterations per step: %d\n", nliter);
    }

    // Setup output
    auto out = f->output();
    out->set_output_dir((output_path + "/out").c_str());
    out->enable_AoS_to_SoA(true);

    // Write mesh
    sfem::create_directory(output_path.c_str());
    sfem::create_directory((output_path + "/out").c_str());
    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());
        fs->mesh_ptr()->write((output_path + "/coarse_mesh").c_str());
    } else {
        fs->mesh_ptr()->write((output_path + "/mesh").c_str());
    }

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
    }

    // Set velocity and acceleration fields for KV operator
    kv_op->set_field("velocity", temp_vel, 0);
    kv_op->set_field("acceleration", increment, 0);

    // Create solver (SSGMG|CG)
    std::shared_ptr<sfem::Operator<real_t>> solver = nullptr;
    if (!sfem::Env::read("SFEM_USE_SSGMG", true) || SFEM_ELEMENT_REFINE_LEVEL <= 1) {
        // This could be put out of the loop since the operator is linear.
        // We will do nonlinear materials next, so we keep it here.
        auto material_op = sfem::create_linear_operator("MF", f, solution, es);
        auto cg          = sfem::create_cg<real_t>(material_op, es);

        if (sfem::Env::read("SFEM_USE_BJACOBI", false)) {
            int  block_size = fs->block_size();
            auto diag       = sfem::create_buffer<real_t>((fs->n_dofs() / block_size) * (block_size == 3 ? 6 : 3), es);
            auto mask       = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
            f->constaints_mask(mask->data());
            f->hessian_block_diag_sym(nullptr, diag->data());
            auto jacobi = sfem::create_shiftable_block_sym_jacobi(fs->block_size(), diag, mask, es);
            cg->set_preconditioner_op(jacobi);
        } else {
            auto diag = sfem::create_buffer<real_t>(ndofs, es);
            f->hessian_diag(nullptr, diag->data());
            f->set_value_to_constrained_dofs(1, diag->data());
            auto jacobi = sfem::create_shiftable_jacobi(diag, es);
            cg->set_preconditioner_op(jacobi);
        }

        cg->verbose = verbose;
        solver      = cg;
    } else {
        auto mg = sfem::create_ssgmg(f, es);
        solver  = mg;
    }

    // Time loop
    while (t < T) {
        for (int k = 0; k < nliter; k++) {
            // Use increment as temp buffer
            blas->zeros(ndofs, increment->data());
            blas->zaxpby(ndofs, 1, solution->data(), -1, displacement->data(), increment->data());
            blas->axpy(ndofs, -dt, velocity->data(), increment->data());
            blas->scal(ndofs, 4 / (dt * dt), increment->data());
            blas->axpy(ndofs, -1, acceleration->data(), increment->data());

            blas->zeros(ndofs, temp_vel->data());
            blas->copy(ndofs, increment->data(), temp_vel->data());
            blas->zaxpby(ndofs, dt / 2, temp_vel->data(), dt / 2, acceleration->data(), temp_vel->data());
            blas->axpy(ndofs, 1, velocity->data(), temp_vel->data());

            blas->zeros(ndofs, g->data());
            // Adds material gradient computation to g
            f->gradient(solution->data(), g->data());

            blas->zeros(ndofs, increment->data());
            solver->apply(g->data(), increment->data());
            blas->axpy(ndofs, -1, increment->data(), solution->data());
        }

        ////////////////////////////////
        // Update all quantities
        ////////////////////////////////

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
    return solve_kelvin_voigt_newmark(ctx->communicator(), argc, argv);
}
