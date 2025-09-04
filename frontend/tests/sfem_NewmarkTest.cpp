#include <stdio.h>

#include "sfem_test.h"
#include "sfem_ssgmg.hpp"
#include "sfem_API.hpp"
#include "sfem_Function.hpp"

std::shared_ptr<sfem::Function> create_elasticity_function() {
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
        // fs->semi_structured_mesh().apply_hierarchical_renumbering();
    }

    auto f = sfem::Function::create(fs);
    
    auto left_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 2 - 1e-5; });

    sfem::DirichletConditions::Condition right0{.sidesets = right_sideset, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition right1{.sidesets = right_sideset, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition right2{.sidesets = right_sideset, .value = 0, .component = 2};

#if 1
    auto d_conds = sfem::create_dirichlet_conditions(fs, {right0, right1, right2}, es);
    f->add_constraint(d_conds);

    sfem::NeumannConditions::Condition nc_left{.sidesets = left_sideset, .value = 0.5, .component = 0};
    auto                               n_conds = sfem::create_neumann_conditions(fs, {nc_left}, es);
    f->add_operator(n_conds);
#else  // Test with Dirichlet only (in case diable test_newmark)
    sfem::DirichletConditions::Condition left0{.sidesets = left_sideset, .value = 0.2, .component = 0};
    sfem::DirichletConditions::Condition left1{.sidesets = left_sideset, .value = 0.2, .component = 1};
    sfem::DirichletConditions::Condition left2{.sidesets = left_sideset, .value = 0.2, .component = 2};
    auto d_conds = sfem::create_dirichlet_conditions(fs, {left0, left1, left2, right0, right1, right2}, es);
    f->add_constraint(d_conds);
#endif

    auto linear_elasticity = sfem::create_op(fs, "LinearElasticity", es);
    linear_elasticity->initialize();
    f->add_operator(linear_elasticity);
    return f;
}

std::shared_ptr<sfem::Buffer<real_t>> create_inverse_mass_vector(const std::shared_ptr<sfem::Function> &f) {
    auto fs = f->space();
    auto es = f->execution_space();

    auto blas = sfem::blas<real_t>(es);

    auto inv_mass_vector = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto mass            = sfem::create_op(fs, "LumpedMass", es);
    mass->initialize();
    mass->hessian_diag(nullptr, inv_mass_vector->data());
    f->set_value_to_constrained_dofs(1, inv_mass_vector->data());
    blas->reciprocal(inv_mass_vector->size(), 1, inv_mass_vector->data());
    return inv_mass_vector;
}

std::shared_ptr<sfem::Buffer<real_t>> create_mass_vector(const std::shared_ptr<sfem::Function> &f) {
    auto fs = f->space();
    auto es = f->execution_space();

    auto blas = sfem::blas<real_t>(es);

    auto mass_vector = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto mass        = sfem::create_op(fs, "LumpedMass", es);
    mass->initialize();
    mass->hessian_diag(nullptr, mass_vector->data());
    f->set_value_to_constrained_dofs(1, mass_vector->data());
    return mass_vector;
}

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

int test_explicit_euler() {
    auto f               = create_elasticity_function();
    auto inv_mass_vector = create_inverse_mass_vector(f);
    auto output          = create_output(f, "explicit_euler");

    auto fs = f->space();
    auto m  = fs->mesh_ptr();
    auto es = f->execution_space();

    auto blas = sfem::blas<real_t>(es);

    auto displacement = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto g            = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    real_t dt          = 0.1;
    real_t T           = 50 * dt;
    size_t export_freq = 1;
    size_t steps       = 0;
    real_t t           = 0;

    bool SFEM_EXPLICIT_EULER_ENABLE_OUTPUT = false;
    SFEM_READ_ENV(SFEM_EXPLICIT_EULER_ENABLE_OUTPUT, atoi);

    if (SFEM_EXPLICIT_EULER_ENABLE_OUTPUT) {
        output->write_time_step("disp", t, displacement->data());

        blas->zeros(g->size(), g->data());
        f->gradient(displacement->data(), g->data());
        output->write_time_step("g", t, g->data());

        output->log_time(t);
    }

    while (t < T) {
        blas->zeros(g->size(), g->data());
        f->gradient(displacement->data(), g->data());
        blas->scal(g->size(), -dt, g->data());
        blas->xypaz(g->size(), g->data(), inv_mass_vector->data(), 1, displacement->data());

        t += dt;

        // Output
        if (++steps % export_freq == 0 && SFEM_EXPLICIT_EULER_ENABLE_OUTPUT) {
            // Write to disk
            output->write_time_step("disp", t, displacement->data());

            blas->zeros(g->size(), g->data());
            f->gradient(displacement->data(), g->data());
            output->write_time_step("g", t, g->data());

            output->log_time(t);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_newmark() {
    auto f           = create_elasticity_function();
    auto mass_vector = create_mass_vector(f);
    auto output      = create_output(f, "test_newmark");

    auto fs = f->space();
    auto m  = fs->mesh_ptr();
    auto es = f->execution_space();

    auto blas = sfem::blas<real_t>(es);

    const ptrdiff_t ndofs        = fs->n_dofs();
    auto            displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto            velocity     = sfem::create_buffer<real_t>(ndofs, es);
    auto            acceleration = sfem::create_buffer<real_t>(ndofs, es);

    auto increment = sfem::create_buffer<real_t>(ndofs, es);
    auto solution  = sfem::create_buffer<real_t>(ndofs, es);
    auto g         = sfem::create_buffer<real_t>(ndofs, es);

    real_t dt          = 0.1;
    real_t T           = 5;
    size_t export_freq = 1;
    size_t steps       = 0;
    real_t t           = 0;
    int    nliter      = 1;

    bool SFEM_NEWMARK_ENABLE_OUTPUT = true;
    SFEM_READ_ENV(SFEM_NEWMARK_ENABLE_OUTPUT, atoi);

    if (SFEM_NEWMARK_ENABLE_OUTPUT) {
        output->write_time_step("disp", t, displacement->data());
        output->write_time_step("velocity", t, velocity->data());
        output->write_time_step("acceleration", t, acceleration->data());

        // If no issues encountered we log the time
        output->log_time(t);
    }

    auto solver     =  sfem::create_ssgmg(f, es);

    while (t < T) {
        for (int k = 0; k < nliter; k++) {
            // This could be put out of the loop since the operator is linear.
            // We will do nonlinear materials next, so we keep it here.
            auto material_op = sfem::create_linear_operator(MATRIX_FREE, f, solution, es);
            auto linear_op   = sfem::make_op<real_t>(
                    material_op->rows(),
                    material_op->cols(),
                    [=](const real_t *const x, real_t *const y) {
                        {
                            SFEM_TRACE_SCOPE("Newmark::hessian_apply_integr");
                            blas->xypaz(ndofs, x, mass_vector->data(), 0, y);
                            blas->scal(ndofs, 4 / (dt * dt), y);
                        }
                        material_op->apply(x, y);
                    },
                    es);

            auto solver     = sfem::create_cg<real_t>(linear_op, es);
            solver->verbose = false;

            // Use increment as temp buffer
            blas->zeros(ndofs, increment->data());
            blas->zaxpby(ndofs, 1, solution->data(), -1, displacement->data(), increment->data());
            blas->axpy(ndofs, -dt, velocity->data(), increment->data());
            blas->scal(ndofs, 4 / (dt * dt), increment->data());
            blas->axpy(ndofs, -1, acceleration->data(), increment->data());
            blas->xypaz(ndofs, increment->data(), mass_vector->data(), 0, g->data());

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

            // Write to disk
            output->write_time_step("disp", t, displacement->data());
            output->write_time_step("velocity", t, velocity->data());
            output->write_time_step("acceleration", t, acceleration->data());

            // If no issues encountered we log the time
            output->log_time(t);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_explicit_euler);
    SFEM_RUN_TEST(test_newmark);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}