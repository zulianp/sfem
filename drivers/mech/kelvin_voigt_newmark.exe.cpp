#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "smesh_env.hpp"

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

    const bool verbose = smesh::Env::read("SFEM_VERBOSE", false);

    // Parse command line arguments
    if (argc != 5) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <mesh> <dirichlet_conditions> <neumann_conditions> <output> \n", argv[0]);
        }
        return 1;
    }

    smesh::Path mesh_path{argv[1]};
    smesh::Path dirichlet_path{argv[2]};
    smesh::Path neumann_path{argv[3]};
    smesh::Path output_path{argv[4]};

    auto m = sfem::Mesh::create_from_file(comm, mesh_path);
    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        m = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);
    }

    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());
    auto f  = sfem::Function::create(fs);

    if (dirichlet_path.to_string() != "NONE") {
        auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            f->add_constraint(sfem::to_device(dirichlet_conditions));
        } else {
            f->add_constraint(dirichlet_conditions);
        }
    }

    if (neumann_path.to_string() != "NONE") {
        auto neumann_conditions = sfem::NeumannConditions::create_from_file(fs, neumann_path);
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            f->add_operator(sfem::to_device(neumann_conditions));
        } else {
            f->add_operator(neumann_conditions);
        }
    }

    auto kv_op = sfem::create_op(fs, "KelvinVoigtNewmark", es);
    kv_op->initialize();
    f->add_operator(kv_op);

    // Get problem size
    auto            blas       = sfem::blas<real_t>(es);
    const ptrdiff_t ndofs      = fs->n_dofs();
    const int       block_size = fs->block_size();

    // Body force parameters. This driver targets the regular HEX8 workflow; keep
    // semistructured/body-force behavior out of scope unless explicitly added.
    real_t SFEM_BODY_FORCE_X = smesh::Env::read("SFEM_BODY_FORCE_X", real_t(0));
    real_t SFEM_BODY_FORCE_Y = smesh::Env::read("SFEM_BODY_FORCE_Y", real_t(0));
    real_t SFEM_BODY_FORCE_Z = smesh::Env::read("SFEM_BODY_FORCE_Z", real_t(0));
    const real_t body_force[3] = {SFEM_BODY_FORCE_X, SFEM_BODY_FORCE_Y, SFEM_BODY_FORCE_Z};
    const real_t body_force_norm = sqrt(SFEM_BODY_FORCE_X * SFEM_BODY_FORCE_X +
                                        SFEM_BODY_FORCE_Y * SFEM_BODY_FORCE_Y +
                                        SFEM_BODY_FORCE_Z * SFEM_BODY_FORCE_Z);

    if (body_force_norm > 0 && es != sfem::EXECUTION_SPACE_HOST) {
        if (!comm->rank()) {
            fprintf(stderr, "SFEM_BODY_FORCE_* is currently supported by this driver only on host execution space.\n");
        }
        return SFEM_FAILURE;
    }

    if (!comm->rank() && verbose) {
        printf("\n=== Kelvin-Voigt Newmark Time Integration ===\n");
        printf("Number of DOFs: %td\n", ndofs);
        printf("Execution space: %s\n", SFEM_EXECUTION_SPACE ? SFEM_EXECUTION_SPACE : "HOST");
        printf("Refine level: %d\n", SFEM_ELEMENT_REFINE_LEVEL);
        printf("Body force: [%g, %g, %g]\n", SFEM_BODY_FORCE_X, SFEM_BODY_FORCE_Y, SFEM_BODY_FORCE_Z);
    }

    // Create state vectors
    auto displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto velocity     = sfem::create_buffer<real_t>(ndofs, es);
    auto acceleration = sfem::create_buffer<real_t>(ndofs, es);
    auto increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto temp_vel     = sfem::create_buffer<real_t>(ndofs, es);
    auto solution     = sfem::create_buffer<real_t>(ndofs, es);
    auto g            = sfem::create_buffer<real_t>(ndofs, es);
    auto body_rhs     = sfem::create_buffer<real_t>(ndofs, es);

    // Initialize all buffers to zero
    blas->zeros(ndofs, displacement->data());
    blas->zeros(ndofs, velocity->data());
    blas->zeros(ndofs, acceleration->data());
    blas->zeros(ndofs, solution->data());
    blas->zeros(ndofs, increment->data());
    blas->zeros(ndofs, temp_vel->data());
    blas->zeros(ndofs, g->data());
    blas->zeros(ndofs, body_rhs->data());

    if (body_force_norm > 0) {
        auto mass_op = sfem::create_op(fs, "LumpedMass", es);
        if (!mass_op) {
            if (!comm->rank()) {
                fprintf(stderr, "Failed to create LumpedMass operator for Kelvin-Voigt body force.\n");
            }
            return SFEM_FAILURE;
        }

        int err = mass_op->initialize();
        if (err != SFEM_SUCCESS) {
            if (!comm->rank()) {
                fprintf(stderr, "Failed to initialize LumpedMass operator for Kelvin-Voigt body force.\n");
            }
            return err;
        }

        auto mass_diag = sfem::create_buffer<real_t>(ndofs, es);
        err            = mass_op->hessian_diag(nullptr, mass_diag->data());
        if (err != SFEM_SUCCESS) {
            if (!comm->rank()) {
                fprintf(stderr, "Failed to compute LumpedMass diagonal for Kelvin-Voigt body force.\n");
            }
            return err;
        }

        const ptrdiff_t n_nodes = m->n_nodes();
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            for (int d = 0; d < block_size && d < 3; ++d) {
                const ptrdiff_t dof = node * block_size + d;
                body_rhs->data()[dof] = mass_diag->data()[dof] * body_force[d];
            }
        }

        f->set_value_to_constrained_dofs(0, body_rhs->data());
    }

    // Time integration parameters
    real_t dt          = smesh::Env::read("SFEM_DT", 0.1);
    real_t T           = smesh::Env::read("SFEM_T_END", 5.0);
    size_t export_freq = smesh::Env::read("SFEM_EXPORT_FREQ", 1);
    int    nliter      = smesh::Env::read("SFEM_NLITER", 1);

    if (!comm->rank() && verbose) {
        printf("Time step: %g\n", dt);
        printf("End time: %g\n", T);
        printf("Export frequency: %zu\n", export_freq);
        printf("Nonlinear iterations per step: %d\n", nliter);
    }

    // Setup output
    auto out = f->output();
    out->set_output_dir(output_path / "out");
    out->enable_AoS_to_SoA(true);

    smesh::create_directory(output_path);
    smesh::create_directory(output_path / "out");

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        smesh::semistructured_export_as_standard(fs->mesh_ptr(), output_path / "mesh");
        fs->mesh_ptr()->write(output_path / "coarse_mesh");
    } else {
        fs->mesh_ptr()->write(output_path / "mesh");
    }

    // Time variables
    real_t t     = 0.0;
    size_t steps = 0;

    bool SFEM_NEWMARK_ENABLE_OUTPUT = true;
    SFEM_READ_ENV(SFEM_NEWMARK_ENABLE_OUTPUT, atoi);
    // Write initial condition
    if (SFEM_NEWMARK_ENABLE_OUTPUT) {
        auto u = smesh::to_host(displacement);
        auto v = smesh::to_host(velocity);
        auto a = smesh::to_host(acceleration);

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
    if (!smesh::Env::read("SFEM_USE_SSGMG", true) || SFEM_ELEMENT_REFINE_LEVEL <= 1) {
        // This could be put out of the loop since the operator is linear.
        // We will do nonlinear materials next, so we keep it here.
        auto material_op = sfem::create_linear_operator("MF", f, solution, es);
        auto cg          = sfem::create_cg<real_t>(material_op, es);

        if (smesh::Env::read("SFEM_USE_BJACOBI", false)) {
            int  block_size = fs->block_size();
            auto diag       = sfem::create_buffer<real_t>((fs->n_dofs() / block_size) * (block_size == 3 ? 6 : 3), es);
            auto mask       = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
            f->constraints_mask(mask->data());
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
            if (body_force_norm > 0) {
                blas->axpy(ndofs, 1.0, body_rhs->data(), g->data());
            }

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

            auto u = smesh::to_host(displacement);
            auto v = smesh::to_host(velocity);
            auto a = smesh::to_host(acceleration);
            // Write to disk
            out->write_time_step("disp", t, u->data());
            out->write_time_step("velocity", t, v->data());
            out->write_time_step("acceleration", t, a->data());
            out->log_time(t);
        }
    }

    const real_t displacement_norm = blas->norm2(ndofs, displacement->data());
    const real_t velocity_norm     = blas->norm2(ndofs, velocity->data());
    const real_t acceleration_norm = blas->norm2(ndofs, acceleration->data());

    if (!comm->rank()) {
        printf("Final norms: displacement=%g velocity=%g acceleration=%g\n",
               displacement_norm,
               velocity_norm,
               acceleration_norm);
    }

    if (!isfinite(displacement_norm) || !isfinite(velocity_norm) || !isfinite(acceleration_norm)) {
        if (!comm->rank()) {
            fprintf(stderr, "Kelvin-Voigt Newmark produced non-finite final state.\n");
        }
        return SFEM_FAILURE;
    }

    if (body_force_norm > 0 && T > 0 && displacement_norm <= 0) {
        if (!comm->rank()) {
            fprintf(stderr, "Kelvin-Voigt body-force run produced zero final displacement.\n");
        }
        return SFEM_FAILURE;
    }

    if (!comm->rank() && verbose) {
        printf("\n=== Simulation Complete ===\n");
        printf("Total steps: %zu\n", steps);
        printf("Final time: %g\n", t);
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize_serial(argc, argv);
    return solve_kelvin_voigt_newmark(ctx->communicator(), argc, argv);
}
