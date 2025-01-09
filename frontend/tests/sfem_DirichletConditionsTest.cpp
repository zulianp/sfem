#include <stdio.h>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#ifdef SFEM_ENABLE_RYAML

int test_dirichlet_conditions_read_yaml() {
    std::string yaml = 
    R"(
    dirichlet_conditions:
    - name: right
      sideset: mesh/sidesets/right
      value: [-0.6, 0, 0]
      component: [0, 1, 2]
    - name: left
      nodeset: mesh/boundary_nodes/left.int32.raw
      value: [0.6, 0, 0]
      component: [0, 1, 2]
    )";

    auto conds = sfem::DirichletConditions::create_from_yaml(nullptr, yaml);
    return SFEM_TEST_SUCCESS;
}

#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_dirichlet_conditions_read_yaml);
#endif
    return SFEM_UNIT_TEST_FINALIZE();
}
