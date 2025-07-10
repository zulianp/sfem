#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

// FIXME
#include "hex8_fff.h"
#include "sshex8_laplacian.h"

int test_linear_function_0(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);

    std::shared_ptr<sfem::Operator<real_t>> bjacobi;
    auto                                    diag = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    std::shared_ptr<sfem::Buffer<uint16_t>> count;

    if (fs->has_semi_structured_mesh()) {
        auto fff = sfem::create_host_buffer<jacobian_t>(fs->mesh_ptr()->n_elements() * 6);

        if (SFEM_SUCCESS != hex8_fff_fill(fs->mesh_ptr()->n_elements(),
                                          fs->mesh_ptr()->elements()->data(),
                                          fs->mesh_ptr()->points()->data(),
                                          fff->data())) {
            SFEM_ERROR("Unable to create fff");
        }

        count = sfem::create_buffer<uint16_t>(fs->semi_structured_mesh().n_nodes(), es);
        {
            auto buff     = count->data();
            auto elements = fs->semi_structured_mesh().element_data();

            const int nxe = fs->semi_structured_mesh().n_nodes_per_element();
            printf("nxe = %d\n", nxe);

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < fs->semi_structured_mesh().n_elements(); ++i) {
                    buff[elements[d][i]]++;
                }
            }
        }

        // count->print(std::cout);

        f->hessian_diag(nullptr, diag->data());
        f->set_value_to_constrained_dofs(1, diag->data());

        auto constraints_mask = sfem::create_buffer<mask_t>(fs->n_dofs(), es);
        f->constaints_mask(constraints_mask->data());

        bjacobi = sfem::make_op<real_t>(
                fs->n_dofs(),
                fs->n_dofs(),
                [=](const real_t *x, real_t *y) {
                    SFEM_TRACE_SCOPE("affine_sshex8_laplacian_bjacobi_fff");
                    affine_sshex8_laplacian_bjacobi_fff(fs->semi_structured_mesh().level(),
                                                        fs->semi_structured_mesh().n_elements(),
                                                        fs->semi_structured_mesh().elements()->data(),
                                                        fff->data(),
                                                        count->data(),
                                                        constraints_mask->data(),
                                                        diag->data(),
                                                        x,
                                                        y);
                },
                es);
    }

    // bjacobi = sfem::h_shiftable_jacobi(diag);

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    double tick = MPI_Wtime();

    auto solver = sfem::create_cg(linear_op, es);
    // auto preconditioner = sfem::h_stationary(linear_op, bjacobi);
    // preconditioner->set_max_it(1);

    auto preconditioner = bjacobi;

    solver->set_preconditioner_op(preconditioner);

    int max_it = 4000;

#if 0
    {
        max_it      = 80;
        auto output = f->output();
        sfem::create_directory(output_dir.c_str());
        std::string dbg_dir = output_dir + "/dbg";
        sfem::create_directory(dbg_dir.c_str());
        output->set_output_dir(dbg_dir.c_str());

        solver->interceptor = [=](real_t *x) {
            static int iter = 0;
            output->write_time_step("x", iter++, x);
            output->log_time(iter);
        };
    }

#endif

    solver->verbose = true;
    solver->set_max_it(max_it);
    solver->set_rtol(0);
    solver->set_atol(1e-8);
    solver->apply(rhs->data(), x->data());

    double tock = MPI_Wtime();

    auto g = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    f->gradient(x->data(), g->data());
    real_t rnorm = sfem::blas<real_t>(es)->norm2(g->size(), g->data());

    int SFEM_VERBOSE = 0;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);

    if (SFEM_VERBOSE) {
        printf("---------------------\n");
        printf("%s #dofs %ld (%g seconds), rnorm %g\n", output_dir.c_str(), fs->n_dofs(), tock - tick, rnorm);
        printf("---------------------\n");
    }

#if 1
    sfem::create_directory(output_dir.c_str());

    if (fs->has_semi_structured_mesh()) {
        SFEM_TEST_ASSERT(m->write((output_dir + "/coarse_mesh").c_str()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((output_dir + "/mesh").c_str()) == SFEM_SUCCESS);
    } else {
        SFEM_TEST_ASSERT(m->write((output_dir + "/mesh").c_str()) == SFEM_SUCCESS);
    }

    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir.c_str());

#ifdef SFEM_ENABLE_CUDA
    if (x->mem_space() == sfem::MEMORY_SPACE_DEVICE) {
        SFEM_TEST_ASSERT(output->write("x", sfem::to_host(x)->data()) == SFEM_SUCCESS);
    } else
#endif
    {

        SFEM_TEST_ASSERT(output->write("x", x->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(output->write("rhs", rhs->data()) == SFEM_SUCCESS);

        if (count) {
            auto rcount = sfem::astype<real_t>(count);
            SFEM_TEST_ASSERT(output->write("count", rcount->data()) == SFEM_SUCCESS);
        }
    }
#endif

    return SFEM_TEST_SUCCESS;
}

int test_linear_function(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = true;

    int SFEM_MAX_IT = 20000;
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);

    cg->set_max_it(SFEM_MAX_IT);
    cg->set_op(linear_op);
    cg->set_rtol(1e-9);

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    double tick = MPI_Wtime();
    SFEM_TEST_ASSERT(cg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);
    double tock = MPI_Wtime();

    int SFEM_VERBOSE = 0;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);

    if (SFEM_VERBOSE) {
        printf("---------------------\n");
        printf("%s #dofs %ld (%g seconds)\n", output_dir.c_str(), fs->n_dofs(), tock - tick);
        printf("---------------------\n");
    }

    int SFEM_ENABLE_OUTPUT = 0;
    SFEM_READ_ENV(SFEM_ENABLE_OUTPUT, atoi);

    if (SFEM_ENABLE_OUTPUT) {
        if (SFEM_VERBOSE) {
            printf("Writing output in %s\n", output_dir.c_str());
        }

        sfem::create_directory(output_dir.c_str());

        if (fs->has_semi_structured_mesh()) {
            SFEM_TEST_ASSERT(m->write((output_dir + "/coarse_mesh").c_str()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((output_dir + "/mesh").c_str()) == SFEM_SUCCESS);
        } else {
            SFEM_TEST_ASSERT(m->write((output_dir + "/mesh").c_str()) == SFEM_SUCCESS);
        }

        auto output = f->output();
        output->enable_AoS_to_SoA(fs->block_size() > 1);
        output->set_output_dir(output_dir.c_str());

#ifdef SFEM_ENABLE_CUDA
        if (x->mem_space() == sfem::MEMORY_SPACE_DEVICE) {
            SFEM_TEST_ASSERT(output->write("x", sfem::to_host(x)->data()) == SFEM_SUCCESS);
        } else
#endif
        {
            SFEM_TEST_ASSERT(output->write("x", x->data()) == SFEM_SUCCESS);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_poisson() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 1;
    // SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);
    auto fs = sfem::FunctionSpace::create(m, 1);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);

    auto f = sfem::Function::create(fs);

    auto left_parent  = sfem::create_host_buffer<element_idx_t>(1);
    auto right_parent = sfem::create_host_buffer<element_idx_t>(1);
    auto left_lfi     = sfem::create_host_buffer<int16_t>(1);
    auto right_lfi    = sfem::create_host_buffer<int16_t>(1);

    left_parent->data()[0]  = 0;
    left_lfi->data()[0]     = HEX8_LEFT;
    right_parent->data()[0] = 0;
    right_lfi->data()[0]    = HEX8_RIGHT;

    auto left_sideset  = std::make_shared<sfem::Sideset>(sfem::Communicator::wrap(comm), left_parent, left_lfi);
    auto right_sideset = std::make_shared<sfem::Sideset>(sfem::Communicator::wrap(comm), right_parent, right_lfi);

    sfem::DirichletConditions::Condition left{.sidesets = {left_sideset}, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.sidesets = {right_sideset}, .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right}, es);
    f->add_constraint(conds);

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_operator(op);
    return test_linear_function(f, "test_poisson");
}

int test_poisson_and_boundary_selector_aux(const char *test_name,
                                          const std::shared_ptr<sfem::Mesh> &m, 
                                          const char *operator_name,
                                          int block_size,
                                          sfem::ExecutionSpace es,
                                          std::vector<std::string> block_names = {}) {
    auto fs = sfem::FunctionSpace::create(m, block_size);

    int SFEM_ELEMENT_REFINE_LEVEL = 1;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);

    auto f = sfem::Function::create(fs);

    auto aabb = m->compute_bounding_box();
    
    const geom_t x_min = aabb.first->data()[0];   
    const geom_t y_min = aabb.first->data()[1];
    const geom_t z_min = aabb.first->data()[2];
    const geom_t x_max = aabb.second->data()[0];
    const geom_t y_max = aabb.second->data()[1];
    const geom_t z_max = aabb.second->data()[2];
    
    auto left_sideset = sfem::Sideset::create_from_selector(
            m, [x_min](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { 
                return x > (x_min - 1e-5) && x < (x_min + 1e-5); 
            });

    auto right_sideset = sfem::Sideset::create_from_selector(
            m, [x_max](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
                return x > (x_max - 1e-5) && x < (x_max + 1e-5);
            });

    auto top_sideset = sfem::Sideset::create_from_selector(
            m, [z_max](const geom_t /*x*/, const geom_t /*y*/, const geom_t z) -> bool { 
                return z > (z_max - 1e-5) && z < (z_max + 1e-5); 
            });

    auto bottom_sideset = sfem::Sideset::create_from_selector(
            m, [z_min](const geom_t /*x*/, const geom_t /*y*/, const geom_t z) -> bool { 
                return z > (z_min - 1e-5) && z < (z_min + 1e-5); 
            });

    auto front_sideset = sfem::Sideset::create_from_selector(
            m, [y_max](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { 
                return y > (y_max - 1e-5) && y < (y_max + 1e-5); 
            });

    auto back_sideset = sfem::Sideset::create_from_selector(
            m, [y_min](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { 
                return y > (y_min - 1e-5) && y < (y_min + 1e-5); 
            });

    sfem::DirichletConditions::Condition left{.sidesets = left_sideset, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.sidesets = right_sideset, .value = 1, .component = 0};
    sfem::DirichletConditions::Condition top{.sidesets = top_sideset, .value = 1, .component = 0};
    sfem::DirichletConditions::Condition bottom{.sidesets = bottom_sideset, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition front{.sidesets = front_sideset, .value = 1, .component = 0};
    sfem::DirichletConditions::Condition back{.sidesets = back_sideset, .value = -1, .component = 0};

    if (block_size == 1) {
        auto conds = sfem::create_dirichlet_conditions(fs, {left, right, bottom, top, front, back}, es);
        f->add_constraint(conds);
    } else {
        sfem::DirichletConditions::Condition left1{.sidesets = left_sideset, .value = -1, .component = 1};
        sfem::DirichletConditions::Condition right1{.sidesets = right_sideset, .value = 1, .component = 1};
        sfem::DirichletConditions::Condition left2{.sidesets = left_sideset, .value = -1, .component = 2};
        sfem::DirichletConditions::Condition right2{.sidesets = right_sideset, .value = 1, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {left, left1, left2, top, right1, right2}, es);
        f->add_constraint(conds);
    }

    auto op = sfem::create_op(fs, operator_name, es);
    op->initialize(block_names);
    f->add_operator(op);
    return test_linear_function(f, test_name);
}

int test_poisson_and_boundary_selector() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR = "Laplacian";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    int SFEM_BLOCK_SIZE = 1;
    if (strcmp(SFEM_OPERATOR, "VectorLaplacian") == 0) {
        int SFEM_ELEMENT_REFINE_LEVEL = 1;
        SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
        assert(SFEM_ELEMENT_REFINE_LEVEL <= 1);
        SFEM_BLOCK_SIZE = 3;
    }

    int SFEM_BASE_RESOLUTION = 6;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int x_dim = 1;

    auto m = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 
                                          SFEM_BASE_RESOLUTION * x_dim, 
                                          SFEM_BASE_RESOLUTION * 1, 
                                          SFEM_BASE_RESOLUTION * 1, 
                                          0, 0, 0, x_dim, 1, 1);
    
    return test_poisson_and_boundary_selector_aux("test_poisson_and_boundary_selector", m, SFEM_OPERATOR, SFEM_BLOCK_SIZE, es);
}

int test_poisson_and_boundary_selector_checkerboard() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR = "Laplacian";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    int SFEM_BLOCK_SIZE = 1;
    if (strcmp(SFEM_OPERATOR, "VectorLaplacian") == 0) {
        int SFEM_ELEMENT_REFINE_LEVEL = 1;
        SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
        assert(SFEM_ELEMENT_REFINE_LEVEL <= 1);
        SFEM_BLOCK_SIZE = 3;
    }

    int SFEM_BASE_RESOLUTION = 6;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);
    auto m = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::wrap(comm), 
                                                       SFEM_BASE_RESOLUTION, 
                                                       SFEM_BASE_RESOLUTION, 
                                                       SFEM_BASE_RESOLUTION, 
                                                       0, 0, 0, 2, 2, 2);
    
    return test_poisson_and_boundary_selector_aux("test_poisson_and_boundary_selector_checkerboard", m, SFEM_OPERATOR, SFEM_BLOCK_SIZE, es);
}

int test_poisson_and_boundary_selector_bidomain() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR = "Laplacian";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    int SFEM_BLOCK_SIZE = 1;
    if (strcmp(SFEM_OPERATOR, "VectorLaplacian") == 0) {
        int SFEM_ELEMENT_REFINE_LEVEL = 1;
        SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
        assert(SFEM_ELEMENT_REFINE_LEVEL <= 1);
        SFEM_BLOCK_SIZE = 3;
    }

    int SFEM_BASE_RESOLUTION = 6;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);
    auto m = sfem::Mesh::create_hex8_bidomain_cube(sfem::Communicator::wrap(comm), 
                                                       SFEM_BASE_RESOLUTION, 
                                                       SFEM_BASE_RESOLUTION, 
                                                       SFEM_BASE_RESOLUTION, 
                                                       0, 0, 0, 2, 2, 2);
    
    return test_poisson_and_boundary_selector_aux("test_poisson_and_boundary_selector_bidomain", m, SFEM_OPERATOR, SFEM_BLOCK_SIZE, es, {"left"});
}

int test_linear_elasticity() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    // SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    auto m  = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm));
    auto fs = sfem::FunctionSpace::create(m, 3);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    auto f = sfem::Function::create(fs);

    auto left_parent  = sfem::create_host_buffer<element_idx_t>(1);
    auto right_parent = sfem::create_host_buffer<element_idx_t>(1);
    auto left_lfi     = sfem::create_host_buffer<int16_t>(1);
    auto right_lfi    = sfem::create_host_buffer<int16_t>(1);

    left_parent->data()[0]  = 0;
    left_lfi->data()[0]     = HEX8_LEFT;
    right_parent->data()[0] = 0;
    right_lfi->data()[0]    = HEX8_RIGHT;

    auto left_sideset  = std::make_shared<sfem::Sideset>(sfem::Communicator::wrap(comm), left_parent, left_lfi);
    auto right_sideset = std::make_shared<sfem::Sideset>(sfem::Communicator::wrap(comm), right_parent, right_lfi);

    sfem::DirichletConditions::Condition left0{.sidesets = {left_sideset}, .value = -2, .component = 0};
    sfem::DirichletConditions::Condition left1{.sidesets = {left_sideset}, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition left2{.sidesets = {left_sideset}, .value = 0, .component = 2};

    sfem::DirichletConditions::Condition right{.sidesets = {right_sideset}, .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left0, left1, left2, right}, es);
    f->add_constraint(conds);

    auto op = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();
    f->add_operator(op);
    return test_linear_function(f, "test_linear_elasticity");
}

#ifdef SFEM_ENABLE_RYAML

std::string yaml =
        R"(
dirichlet_conditions:
- name: right
  type: sideset
  format: expr
  parent: [0]
  lfi: [3]
  value: 1
  component: 0
- name: left
  type: sideset
  format: expr
  parent: [0]
  lfi: [1]
  value: -1
  component: 0
)";

int test_poisson_yaml() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    auto m  = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm));
    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(16);
    auto f = sfem::Function::create(fs);

    auto conds = sfem::DirichletConditions::create_from_yaml(fs, yaml);
    f->add_constraint(conds);

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_operator(op);
    return test_linear_function(f, "test_poisson_yaml");
}

#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    // SFEM_RUN_TEST(test_poisson);
    // SFEM_RUN_TEST(test_linear_elasticity);
    SFEM_RUN_TEST(test_poisson_and_boundary_selector);
    SFEM_RUN_TEST(test_poisson_and_boundary_selector_checkerboard);
    SFEM_RUN_TEST(test_poisson_and_boundary_selector_bidomain);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_poisson_yaml);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
