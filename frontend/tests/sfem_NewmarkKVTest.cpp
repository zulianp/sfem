#include <stdio.h>

#include "hex8_jacobian.h"
#include "kelvin_voigt_newmark.h"
#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "sfem_ssgmg.hpp"
#include "sfem_test.h"

static void compute_contact_lower_bound(
    real_t* lower_bound,
    const real_t* displacement,
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t contact_plane,
    const int contact_dir,
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        const real_t current_disp = displacement[dof_idx];
        lower_bound[dof_idx] = contact_plane - current_disp;
    }
}

// Return: function, operator, mesh, left_sideset, right_sideset
struct KVFunctionBundle {
    std::shared_ptr<sfem::Function> f;
    std::shared_ptr<sfem::Op> kelvin_voigt_newmark;
    std::shared_ptr<sfem::Mesh> mesh;
    std::vector<std::shared_ptr<sfem::Sideset>> left_sideset;
    std::vector<std::shared_ptr<sfem::Sideset>> right_sideset;
};

KVFunctionBundle create_kelvin_voigt_newmark_function(bool enable_contact = false) {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_BASE_RESOLUTION = 4;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    auto m = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm),
                                          // Grid
                                          SFEM_BASE_RESOLUTION * 2,
                                          SFEM_BASE_RESOLUTION,
                                          SFEM_BASE_RESOLUTION,
                                          // Geometry
                                          0.,
                                          0.,
                                          0.,
                                          2.,
                                          1.,
                                          1.);

    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
        fs->semi_structured_mesh().apply_hierarchical_renumbering();
    }

    auto f = sfem::Function::create(fs);

    // Same setup as Mooney-Rivlin: fix left, apply force on right, contact on left
    auto left_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x < 1e-5; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 2 - 1e-5; });

    // Left side boundary conditions
    if (!enable_contact) {
        // Standard mode: fix left side completely (x, y, z)
        sfem::DirichletConditions::Condition left0{.sidesets = left_sideset, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition left1{.sidesets = left_sideset, .value = 0, .component = 1};
        sfem::DirichletConditions::Condition left2{.sidesets = left_sideset, .value = 0, .component = 2};
        auto d_conds = sfem::create_dirichlet_conditions(fs, {left0, left1, left2}, es);
        f->add_constraint(d_conds);
    }
    // In contact mode: no Dirichlet BC on left, x is constrained by contact lower bound
    // y and z can move freely (Poisson effect)

    // Neumann force on right side (negative = compression, same as Mooney-Rivlin)
    real_t SFEM_NEUMANN_FORCE = -0.5;
    SFEM_READ_ENV(SFEM_NEUMANN_FORCE, atof);
    sfem::NeumannConditions::Condition nc_right{.sidesets = right_sideset, .value = SFEM_NEUMANN_FORCE, .component = 0};
    auto                               n_conds = sfem::create_neumann_conditions(fs, {nc_right}, es);
    f->add_operator(n_conds);

    auto kelvin_voigt_newmark = sfem::create_op(fs, "KelvinVoigtNewmark", es);
    kelvin_voigt_newmark->initialize();
    f->add_operator(kelvin_voigt_newmark);
    
    return KVFunctionBundle{f, kelvin_voigt_newmark, m, left_sideset, right_sideset};
}

// std::shared_ptr<sfem::Buffer<real_t>> create_inverse_mass_vector(const std::shared_ptr<sfem::Function> &f) {
//     auto fs = f->space();
//     auto es = f->execution_space();

//     auto blas = sfem::blas<real_t>(es);

//     auto inv_mass_vector = sfem::create_buffer<real_t>(fs->n_dofs(), es);
//     auto mass            = sfem::create_op(fs, "LumpedMass", es);
//     mass->initialize();
//     mass->hessian_diag(nullptr, inv_mass_vector->data());
//     f->set_value_to_constrained_dofs(1, inv_mass_vector->data());
//     blas->reciprocal(inv_mass_vector->size(), 1, inv_mass_vector->data());
//     return inv_mass_vector;
// }

// std::shared_ptr<sfem::Buffer<real_t>> create_mass_vector(const std::shared_ptr<sfem::Function> &f) {
//     auto fs = f->space();
//     auto es = f->execution_space();

//     auto blas = sfem::blas<real_t>(es);

//     auto mass_vector = sfem::create_buffer<real_t>(fs->n_dofs(), es);
//     auto mass        = sfem::create_op(fs, "LumpedMass", es);
//     mass->initialize();
//     mass->hessian_diag(nullptr, mass_vector->data());
//     f->set_value_to_constrained_dofs(1, mass_vector->data());
//     return mass_vector;
// }

std::shared_ptr<sfem::Output> create_output(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto fs = f->space();

    sfem::create_directory(output_dir.c_str());
    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir.c_str());

    if (fs->has_semi_structured_mesh()) {
        fs->semi_structured_mesh().export_as_standard(output_dir.c_str());
    } else {
        fs->mesh_ptr()->write(output_dir.c_str());
    }
    return output;
}

int test_newmark_kv() {
    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        // Semi-structured mesh path (no contact support in this branch)
        auto bundle = create_kelvin_voigt_newmark_function(false);
        auto f = bundle.f;
        auto kelvin_voigt_newmark = bundle.kelvin_voigt_newmark;
        
        auto output = create_output(f, "test_newmark_kv");

        auto fs = f->space();
        auto m  = fs->mesh_ptr();
        auto es = f->execution_space();

        auto blas = sfem::blas<real_t>(es);

        const ptrdiff_t ndofs = fs->n_dofs();
        auto displacement = sfem::create_buffer<real_t>(ndofs, es);
        auto velocity     = sfem::create_buffer<real_t>(ndofs, es);
        auto acceleration = sfem::create_buffer<real_t>(ndofs, es);

        auto increment = sfem::create_buffer<real_t>(ndofs, es);
        auto temp_vel  = sfem::create_buffer<real_t>(ndofs, es);
        auto solution  = sfem::create_buffer<real_t>(ndofs, es);
        auto g         = sfem::create_buffer<real_t>(ndofs, es);

        // Ensure device buffers are initialized to zero to avoid NaNs
        blas->zeros(ndofs, displacement->data());
        blas->zeros(ndofs, velocity->data());
        blas->zeros(ndofs, acceleration->data());
        blas->zeros(ndofs, solution->data());
        blas->zeros(ndofs, increment->data());
        blas->zeros(ndofs, temp_vel->data());
        blas->zeros(ndofs, g->data());

        // Read time parameters from environment
        real_t SFEM_DT = 0.1;
        real_t SFEM_T  = 5;
        SFEM_READ_ENV(SFEM_DT, atof);
        SFEM_READ_ENV(SFEM_T, atof);
        
        real_t dt          = SFEM_DT;
        real_t T           = SFEM_T;
        size_t export_freq = 1;
        size_t steps       = 0;
        real_t t           = 0;
        int    nliter      = 1;

        bool SFEM_NEWMARK_ENABLE_OUTPUT = false;
        SFEM_READ_ENV(SFEM_NEWMARK_ENABLE_OUTPUT, atoi);

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
            output->write_time_step("disp", t, u->data());
            output->write_time_step("velocity", t, v->data());
            output->write_time_step("acceleration", t, a->data());

            // If no issues encountered we log the time
            output->log_time(t);
        }

        kelvin_voigt_newmark->set_field("velocity", temp_vel, 0);
        kelvin_voigt_newmark->set_field("acceleration", increment, 0);

        std::shared_ptr<sfem::Operator<real_t>> solver    = nullptr;
        bool                                    use_ssgmg = true;
        auto                                    mg        = sfem::create_ssgmg(f, es);
        solver                                            = mg;

        auto diag = sfem::create_buffer<real_t>(ndofs, es);
        f->hessian_diag(nullptr, diag->data());
        f->set_value_to_constrained_dofs(1, diag->data());
        auto jacobi = sfem::create_shiftable_jacobi(diag, es);

        if (!use_ssgmg) {
            // This could be put out of the loop since the operator is linear.
            // We will do nonlinear materials next, so we keep it here.
            auto material_op = sfem::create_linear_operator("MF", f, solution, es);
            auto cg = sfem::create_cg<real_t>(material_op, es);
            cg->set_preconditioner_op(jacobi);
            cg->verbose = false;
            solver = cg;
        }

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
                printf("%g/%g\n", double(t), double(T));

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
                output->write_time_step("disp", t, u->data());
                output->write_time_step("velocity", t, v->data());
                output->write_time_step("acceleration", t, a->data());

                // If no issues encountered we log the time
                output->log_time(t);
            }
        }

    } else {
        // Read contact setting
        int SFEM_ENABLE_CONTACT = 0;
        SFEM_READ_ENV(SFEM_ENABLE_CONTACT, atoi);
        
        auto bundle = create_kelvin_voigt_newmark_function(SFEM_ENABLE_CONTACT != 0);
        auto f = bundle.f;
        auto kelvin_voigt_newmark = bundle.kelvin_voigt_newmark;
        auto left_sideset = bundle.left_sideset;
        
        auto output = create_output(f, "test_newmark_kv");

        auto fs = f->space();
        auto m  = fs->mesh_ptr();
        auto es = f->execution_space();

        auto blas = sfem::blas<real_t>(es);

        const ptrdiff_t ndofs        = fs->n_dofs();
        const int       block_size   = 3;
        auto            displacement = sfem::create_buffer<real_t>(ndofs, es);
        auto            velocity     = sfem::create_buffer<real_t>(ndofs, es);
        auto            acceleration = sfem::create_buffer<real_t>(ndofs, es);

        auto increment = sfem::create_buffer<real_t>(ndofs, es);
        auto temp_vel  = sfem::create_buffer<real_t>(ndofs, es);
        auto solution  = sfem::create_buffer<real_t>(ndofs, es);
        auto g         = sfem::create_buffer<real_t>(ndofs, es);

        // Read time parameters from environment
        real_t SFEM_DT = 0.1;
        real_t SFEM_T  = 5;
        SFEM_READ_ENV(SFEM_DT, atof);
        SFEM_READ_ENV(SFEM_T, atof);
        
        real_t dt          = SFEM_DT;
        real_t T           = SFEM_T;
        size_t export_freq = 1;
        size_t steps       = 0;
        real_t t           = 0;
        int    nliter      = 1;

        bool SFEM_NEWMARK_ENABLE_OUTPUT = false;
        SFEM_READ_ENV(SFEM_NEWMARK_ENABLE_OUTPUT, atoi);

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
            output->write_time_step("disp", t, u->data());
            output->write_time_step("velocity", t, v->data());
            output->write_time_step("acceleration", t, a->data());

            // If no issues encountered we log the time
            output->log_time(t);
        }

        kelvin_voigt_newmark->set_field("velocity", temp_vel, 0);
        kelvin_voigt_newmark->set_field("acceleration", increment, 0);

        // Ensure all state vectors are initialized to zero to avoid NaNs
        blas->zeros(ndofs, displacement->data());
        blas->zeros(ndofs, velocity->data());
        blas->zeros(ndofs, acceleration->data());
        blas->zeros(ndofs, solution->data());
        blas->zeros(ndofs, temp_vel->data());
        blas->zeros(ndofs, increment->data());
        blas->zeros(ndofs, g->data());

        // Create linear operator
        auto material_op = sfem::create_linear_operator("MF", f, solution, es);
        auto linear_op   = sfem::make_op<real_t>(
                material_op->rows(),
                material_op->cols(),
                [=](const real_t *const x, real_t *const y) { material_op->apply(x, y); },
                es);

        // Solver setup - CG or MPRGP depending on contact mode
        std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
        sfem::SharedBuffer<real_t> lower_bound;
        
        // Contact parameters
        real_t SFEM_CONTACT_PLANE = -0.1;  // Position of contact plane
        int SFEM_CONTACT_DIR = 0;          // Contact direction: 0=x, 1=y, 2=z
        real_t SFEM_CONTACT_RESTITUTION = 0.0;  // Coefficient of restitution (0 = fully damped)
        SFEM_READ_ENV(SFEM_CONTACT_PLANE, atof);
        SFEM_READ_ENV(SFEM_CONTACT_DIR, atoi);
        SFEM_READ_ENV(SFEM_CONTACT_RESTITUTION, atof);
        
        // Store contact node indices for use in the function
        std::vector<ptrdiff_t> contact_node_indices;
        
        if (!SFEM_ENABLE_CONTACT) {
            printf("Using CG solver (no contact)\n");
            auto cg = sfem::create_cg<real_t>(linear_op, es);
            cg->verbose = false;
            solver = cg;
        } else {
            printf("Using MPRGP solver (contact enabled)\n");
            auto mprgp = sfem::create_mprgp(linear_op, es);
            lower_bound = sfem::create_buffer<real_t>(ndofs, es);
            
            // Get contact boundary nodes (left boundary for compression from right)
            auto lnodes = sfem::create_nodeset_from_sideset(fs, left_sideset[0]);
            const ptrdiff_t nbnodes = lnodes->size();
            contact_node_indices.resize(nbnodes);
            for (ptrdiff_t i = 0; i < nbnodes; i++) {
                contact_node_indices[i] = lnodes->data()[i];
            }
            
            // Initial lower bound computation (displacement = 0 at start)
            compute_contact_lower_bound(
                lower_bound->data(),
                displacement->data(),  // All zeros at start
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                SFEM_CONTACT_PLANE,
                SFEM_CONTACT_DIR,
                block_size,
                ndofs);
            
            mprgp->verbose = false;
            mprgp->set_lower_bound(lower_bound);
            solver = mprgp;
            
            printf("Contact plane: %.3f, direction: %d, restitution: %.2f\n", 
                   SFEM_CONTACT_PLANE, SFEM_CONTACT_DIR, SFEM_CONTACT_RESTITUTION);
        }

        printf("===== Kelvin-Voigt Newmark Test =====\n");
        printf("Mesh: %ld nodes, %ld DOFs\n", (long)m->n_nodes(), (long)ndofs);
        printf("Contact: %s\n", SFEM_ENABLE_CONTACT ? "ENABLED" : "DISABLED");
        printf("Time: T=%.2f, dt=%.3f\n", T, dt);

        // Get a left boundary node for debugging
        ptrdiff_t debug_node = -1;
        if (SFEM_ENABLE_CONTACT && !contact_node_indices.empty()) {
            debug_node = contact_node_indices[0];
            printf("Debug node %ld (contact boundary)\n", (long)debug_node);
        }

        while (t < T) {
            // Compute lower_bound based on current displacement (before solve)
            // This is the function requested: compute current lower bound based on 
            // current displacement, change from left to right
            if (SFEM_ENABLE_CONTACT && lower_bound && !contact_node_indices.empty()) {
                compute_contact_lower_bound(
                    lower_bound->data(),
                    displacement->data(),
                    contact_node_indices.data(),
                    (ptrdiff_t)contact_node_indices.size(),
                    SFEM_CONTACT_PLANE,
                    SFEM_CONTACT_DIR,
                    block_size,
                    ndofs);
            }

            // Debug: print displacement and lower_bound at debug node
            if (SFEM_ENABLE_CONTACT && debug_node >= 0) {
                real_t disp_x = displacement->data()[debug_node * block_size + SFEM_CONTACT_DIR];
                real_t sol_x = solution->data()[debug_node * block_size + SFEM_CONTACT_DIR];
                real_t lb_x = lower_bound->data()[debug_node * block_size + SFEM_CONTACT_DIR];
                printf("t=%.2f: disp=%.4f, sol=%.4f, lb=%.4f\n", t, disp_x, sol_x, lb_x);
            }

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
                
                // Negate gradient so that solution += increment (same sign convention as MR)
                blas->scal(ndofs, -1.0, g->data());

                blas->zeros(ndofs, increment->data());
                solver->apply(g->data(), increment->data());
                blas->axpy(ndofs, 1.0, increment->data(), solution->data());  // solution += increment
                
                // Update lower_bound for contact (same as MR: lb -= delta_x)
                if (SFEM_ENABLE_CONTACT && lower_bound) {
                    blas->axpy(ndofs, -1.0, increment->data(), lower_bound->data());
                }
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
                printf("%g/%g\n", double(t), double(T));

                // Write to disk
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
                output->write_time_step("disp", t, u->data());
                output->write_time_step("velocity", t, v->data());
                output->write_time_step("acceleration", t, a->data());

                // If no issues encountered we log the time
                output->log_time(t);
            }
        }
        
        printf("===== Test Completed =====\n");
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif
    SFEM_RUN_TEST(test_newmark_kv);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
