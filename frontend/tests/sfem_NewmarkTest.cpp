#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

int test_explicit_euler() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    auto blas = sfem::blas<real_t>(es);

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    // SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    auto m = sfem::Mesh::create_hex8_cube(comm,
                                          // Grid
                                          10,
                                          10,
                                          10,
                                          // Geometry
                                          0.,
                                          0.,
                                          0.,
                                          2.,
                                          1.,
                                          1.);

    auto fs = sfem::FunctionSpace::create(m, 3);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);

    auto f = sfem::Function::create(fs);

    auto left_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 2 - 1e-5; });

    sfem::DirichletConditions::Condition left0{.sideset = left_sideset, .value = 0.5, .component = 0};
    // sfem::DirichletConditions::Condition left1{.sideset = left_sideset, .value = 0, .component = 1};
    // sfem::DirichletConditions::Condition left2{.sideset = left_sideset, .value = 0, .component = 2};

    sfem::DirichletConditions::Condition right0{.sideset = right_sideset, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition right1{.sideset = right_sideset, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition right2{.sideset = right_sideset, .value = 0, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left0, right0, right1, right2}, es);
    f->add_constraint(conds);

    auto linear_elasticity = sfem::create_op(fs, "LinearElasticity", es);
    linear_elasticity->initialize();
    f->add_operator(linear_elasticity);

    auto displacement = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto g = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    
    real_t dt = 0.001;
    real_t T = 5;
    size_t export_freq = 20;

    auto inv_mass_vector = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto mass        = sfem::create_op(fs, "LumpedMass", es);
    mass->initialize();
    mass->hessian_diag(nullptr, inv_mass_vector->data());
    f->set_value_to_constrained_dofs(1, inv_mass_vector->data());
    blas->reciprocal(inv_mass_vector->size(), 1, inv_mass_vector->data());

    size_t steps = 0;
    real_t t = 0;

    std::string output_dir = "explicit_euler";
    sfem::create_directory(output_dir.c_str());
    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir.c_str());

    SFEM_TEST_ASSERT(m->write(output_dir.c_str()) == SFEM_SUCCESS);
    
    output->write_time_step("disp", t, displacement->data());
    

    while(t < T) {
    	f->gradient(displacement->data(), g->data());
        blas->scal(g->size(), -dt, g->data());
    	blas->xypaz(g->size(), g->data(), inv_mass_vector->data(), 1, displacement->data());

    	t += dt;
    	if(++steps % export_freq == 0) {
    		// Write to disk
    		output->write_time_step("disp", t, displacement->data());
    	}
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif
    SFEM_RUN_TEST(test_explicit_euler);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

