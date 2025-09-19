#include <stdio.h>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#ifdef SFEM_ENABLE_RYAML

int test_dirichlet_conditions_read_yaml() {
    std::string yaml =
            R"(
    dirichlet_conditions:
    - name: right
      type: sideset
      format: expr
      parent: [0]
      lfi: [2]
      value: [-0.6, 0, 0]
      component: [0, 1, 2]
    - name: left
      type: nodeset
      format: expr
      nodes: [0, 1, 2]
      value: 1
      component: 0
    )";

    auto comm = sfem::Communicator::world();
    auto     m    = sfem::Mesh::create_hex8_cube(comm);
    auto     fs   = sfem::FunctionSpace::create(m, 3);

    // Conditions for standard mesh
    auto conds = sfem::DirichletConditions::create_from_yaml(fs, yaml);

    // Conditions for semi-structured mesh
    fs->promote_to_semi_structured(32);
    conds = sfem::DirichletConditions::create_from_yaml(fs, yaml);

    return SFEM_TEST_SUCCESS;
}

#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_dirichlet_conditions_read_yaml);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
