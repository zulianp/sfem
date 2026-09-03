#include "sfem_test.hpp"

#include "sfem_API.hpp"

int test_generated_saint_venant_kirchhoff_factory() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);
    auto op    = sfem::create_op(space, "GeneratedSaintVenantKirchhoff", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_RYAML
int test_yaml_operator_factory() {
    auto        mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto        space = sfem::FunctionSpace::create(mesh, 3);
    std::string yaml =
            "operator:\n"
            "  type: GeneratedSaintVenantKirchhoff\n"
            "  mu: 2\n"
            "  lmbda: 3\n"
            "  blocks:\n"
            "    - name: default\n"
            "      mu: 4\n"
            "      lmbda: 6\n";
    auto op = sfem::create_op_from_yaml(space, yaml, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    return SFEM_TEST_SUCCESS;
}
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_saint_venant_kirchhoff_factory);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_yaml_operator_factory);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
