#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_GeneratedNavierStokes.hpp"

#include <cmath>
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

// The Taylor-Hood Navier-Stokes Jacobian action in team form (mixed velocity-pressure kernels with
// an element range): one thread and all of them agree to round-off, and the action is not zero.
// Constructed directly: the host factory does not register GeneratedNavierStokes.
int test_generated_navier_stokes_team() {
    auto mesh = smesh::Mesh::create_wall_mounted_hump(sfem::Communicator::self(), smesh::HEX27, 6, 3, 2);
    SFEM_TEST_ASSERT(mesh != nullptr);
    auto space = sfem::FunctionSpace::create(mesh, 4, smesh::HEX27);
    std::shared_ptr<sfem::Op> op = sfem::GeneratedNavierStokes::create(space);
    SFEM_TEST_ASSERT(op != nullptr);
    op->set_value_in_block("fluid", "rho", 1);
    op->set_value_in_block("fluid", "nu", 0.01);
    op->set_value_in_block("fluid", "dt", 0.1);
    op->set_value_in_block("fluid", "convection_scale", 1);
    SFEM_TEST_ASSERT(op->team_supported());

    const ptrdiff_t n        = space->n_dofs();
    auto            previous = sfem::create_host_buffer<real_t>(n);
    auto            h        = sfem::create_host_buffer<real_t>(n);
    auto            one      = sfem::create_host_buffer<real_t>(n);
    auto            all      = sfem::create_host_buffer<real_t>(n);
    auto            points   = mesh->points()->data();
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        const real_t x = points[0][node], y = points[1][node], z = points[2][node];
        previous->data()[4 * node + 0] = y * (3 - y);
        previous->data()[4 * node + 1] = 0.1 * x * z;
        previous->data()[4 * node + 2] = 0.05 * y;
        previous->data()[4 * node + 3] = x;
        h->data()[4 * node + 0]        = std::sin(x + z);
        h->data()[4 * node + 1]        = std::cos(y);
        h->data()[4 * node + 2]        = x * y;
        h->data()[4 * node + 3]        = z - y;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        one->data()[i] = 0;
        all->data()[i] = 0;
    }
    op->set_field("previous", previous, 0);

#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    SFEM_TEST_ASSERT(op->apply(previous->data(), h->data(), one->data()) == SFEM_SUCCESS);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    SFEM_TEST_ASSERT(op->apply(previous->data(), h->data(), all->data()) == SFEM_SUCCESS);

    real_t norm = 0, diff = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        norm += one->data()[i] * one->data()[i];
        diff += (one->data()[i] - all->data()[i]) * (one->data()[i] - all->data()[i]);
    }
    SFEM_TEST_ASSERT(norm > 0);
    SFEM_TEST_ASSERT(std::sqrt(diff) <= 1e-12 * std::sqrt(norm));
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_saint_venant_kirchhoff_factory);
    SFEM_RUN_TEST(test_gradient_factory);
    SFEM_RUN_TEST(test_generated_navier_stokes_team);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_yaml_operator_factory);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
