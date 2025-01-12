#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

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

int test_poisson() {
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

    auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = true;
    cg->set_max_it(1000);
    cg->set_op(linear_op);

    auto diag = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    f->hessian_diag(nullptr, diag->data());
    cg->set_preconditioner_op(create_shiftable_jacobi(diag, es));

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    SFEM_TEST_ASSERT(cg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);

#if 1
    SFEM_TEST_ASSERT(m->write("test_coarse_mesh") == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard("test_mesh") == SFEM_SUCCESS);
    auto output = f->output();
    output->set_output_dir("test_output");
    SFEM_TEST_ASSERT(output->write("x", x->data()) == SFEM_SUCCESS);
#endif

    return SFEM_TEST_SUCCESS;
}

#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_poisson);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
