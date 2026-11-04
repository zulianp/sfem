#include "sfem_test.hpp"

#include "sfem_API.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

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

// The TET4 patch gradient computes each node's row from its own patch (no atomics): its team form
// gives the same bits on one thread as on all of them, and every node's row is written.
int test_gradient_factory() {
    auto mesh  = sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::TET4, 6, 5, 4, 0, 0, 0, 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    auto op    = sfem::create_op(space, "Gradient", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);

    const ptrdiff_t n      = mesh->n_nodes();
    const ptrdiff_t m      = op->n_dofs_image();
    auto            u      = sfem::create_host_buffer<real_t>(n);
    auto            one    = sfem::create_host_buffer<real_t>(m);
    auto            all    = sfem::create_host_buffer<real_t>(m);
    auto            points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        u->data()[i] = points[0][i] * points[0][i] + 2 * points[1][i] - 3 * points[2][i] * points[0][i];
    }
    for (ptrdiff_t i = 0; i < m; ++i) {
        one->data()[i] = 0;
        all->data()[i] = 0;
    }

#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    SFEM_TEST_ASSERT(op->apply(nullptr, u->data(), one->data()) == SFEM_SUCCESS);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    SFEM_TEST_ASSERT(op->apply(nullptr, u->data(), all->data()) == SFEM_SUCCESS);

    for (ptrdiff_t i = 0; i < n; ++i) {
        const real_t *g = &all->data()[3 * i];
        SFEM_TEST_ASSERT(g[0] != 0 || g[1] != 0 || g[2] != 0);
    }
    SFEM_TEST_ASSERT(std::memcmp(one->data(), all->data(), m * sizeof(real_t)) == 0);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_saint_venant_kirchhoff_factory);
    SFEM_RUN_TEST(test_gradient_factory);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_yaml_operator_factory);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
