#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

int test_linear_function(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = true;
    cg->set_max_it(1000);
    cg->set_op(linear_op);
    cg->set_rtol(1e-8);

    auto diag = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    f->hessian_diag(nullptr, diag->data());
    cg->set_preconditioner_op(create_shiftable_jacobi(diag, es));

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    SFEM_TEST_ASSERT(cg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);

#if 1
    sfem::create_directory(output_dir.c_str());
    SFEM_TEST_ASSERT(m->write((output_dir + "/coarse_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((output_dir + "/mesh").c_str()) == SFEM_SUCCESS);
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
#endif

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
    // SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    auto m  = sfem::Mesh::create_hex8_cube(comm);
    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    auto f = sfem::Function::create(fs);

    auto left_parent  = sfem::create_host_buffer<element_idx_t>(1);
    auto right_parent = sfem::create_host_buffer<element_idx_t>(1);
    auto left_lfi     = sfem::create_host_buffer<int16_t>(1);
    auto right_lfi    = sfem::create_host_buffer<int16_t>(1);

    left_parent->data()[0]  = 0;
    left_lfi->data()[0]     = HEX8_LEFT;
    right_parent->data()[0] = 0;
    right_lfi->data()[0]    = HEX8_RIGHT;

    auto left_sideset  = std::make_shared<sfem::Sideset>(comm, left_parent, left_lfi);
    auto right_sideset = std::make_shared<sfem::Sideset>(comm, right_parent, right_lfi);

    sfem::DirichletConditions::Condition left{.sideset = left_sideset, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.sideset = right_sideset, .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right}, es);
    f->add_constraint(conds);

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_operator(op);
    return test_linear_function(f, "test_poisson");
}

int test_poisson_and_boundary_selector() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 6;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 2, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 2, 1, 1);
    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    auto f = sfem::Function::create(fs);

    auto left_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > (2 - 1e-5) && x < (2 + 1e-5); });

    sfem::DirichletConditions::Condition left{.sideset = left_sideset, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.sideset = right_sideset, .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right}, es);
    f->add_constraint(conds);

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_operator(op);
    return test_linear_function(f, "test_poisson_and_boundary_selector");
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

    auto m  = sfem::Mesh::create_hex8_cube(comm);
    auto fs = sfem::FunctionSpace::create(m, 3);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    auto f = sfem::Function::create(fs);

    auto left_parent  = sfem::create_host_buffer<element_idx_t>(1);
    auto right_parent = sfem::create_host_buffer<element_idx_t>(1);
    auto left_lfi     = sfem::create_host_buffer<int16_t>(1);
    auto right_lfi    = sfem::create_host_buffer<int16_t>(1);

    left_parent->data()[0]  = 0;
    left_lfi->data()[0]     = HEX8_LEFT;
    right_parent->data()[0] = 0;
    right_lfi->data()[0]    = HEX8_RIGHT;

    auto left_sideset  = std::make_shared<sfem::Sideset>(comm, left_parent, left_lfi);
    auto right_sideset = std::make_shared<sfem::Sideset>(comm, right_parent, right_lfi);

    sfem::DirichletConditions::Condition left0{.sideset = left_sideset, .value = -2, .component = 0};
    sfem::DirichletConditions::Condition left1{.sideset = left_sideset, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition left2{.sideset = left_sideset, .value = 0, .component = 2};

    sfem::DirichletConditions::Condition right{.sideset = right_sideset, .value = 1, .component = 0};

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

    auto m  = sfem::Mesh::create_hex8_cube(comm);
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

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    SFEM_RUN_TEST(test_poisson);
    SFEM_RUN_TEST(test_linear_elasticity);
    SFEM_RUN_TEST(test_poisson_and_boundary_selector);

#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_poisson_yaml);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
