// MPRGP on a singular operator: the Laplacian of a free chain (no Dirichlet node), whose constant
// vector is a null direction, as a free body's rigid motions are of its tangent. With a consistent
// right-hand side the bounded problem has solutions and MPRGP finds one; a net load toward the bound
// moves the chain along its free direction until the bound holds it, as the wall holds a free valve;
// when the energy decreases without end along the free direction it fails, and returns no NaN.

#include <cmath>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_test.hpp"

namespace {
    const ptrdiff_t n = 64;

    // y = A x, A the chain's Laplacian with free ends: every row sums to zero.
    void laplacian(const real_t *const x, real_t *const y) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            real_t v = 0;
            if (i > 0) v += x[i] - x[i - 1];
            if (i < n - 1) v += x[i] - x[i + 1];
            y[i] += v;
        }
    }

    // Solves min 1/2 x.Ax - b.x subject to x <= ub from zero; returns the status, x in `x`.
    int solve(const std::vector<real_t> &b, const real_t ub, std::vector<real_t> &x) {
        auto op = sfem::make_op<real_t>(
                n, n, [](const real_t *const in, real_t *const out) { laplacian(in, out); }, sfem::EXECUTION_SPACE_HOST);
        auto qp = sfem::create_mprgp(op, sfem::EXECUTION_SPACE_HOST);
        auto upper = sfem::create_host_buffer<real_t>(n);
        for (ptrdiff_t i = 0; i < n; ++i) upper->data()[i] = ub;
        qp->verbose    = false;
        qp->check_each = 1 << 30;
        qp->set_upper_bound(upper);
        qp->set_max_eig(4);
        qp->set_max_it(10 * n);
        qp->set_rtol(1e-12);
        qp->set_atol(1e-14);
        x.assign(n, 0);
        return qp->apply(b.data(), x.data());
    }
}  // namespace

// A consistent load (it sums to zero): pushing the chain's two halves apart against the bound.
int test_consistent_load_converges() {
    std::vector<real_t> b(n, 0), x;
    b[0]     = -1;
    b[n - 1] = 1;
    SFEM_TEST_ASSERT(solve(b, real_t(0.5), x) == SFEM_SUCCESS);
    for (const real_t v : x) SFEM_TEST_ASSERT(std::isfinite(v) && v <= real_t(0.5) + 1e-12);
    return SFEM_TEST_SUCCESS;
}

// A net push up the free direction, which the bound stops: the chain rests against it, x = ub. The
// first search direction is the constant vector, which the operator does not see.
int test_bound_holds_the_free_direction() {
    std::vector<real_t> b(n, real_t(0.01)), x;
    SFEM_TEST_ASSERT(solve(b, real_t(0.5), x) == SFEM_SUCCESS);
    for (const real_t v : x) SFEM_TEST_ASSERT(std::abs(v - real_t(0.5)) < 1e-12);
    return SFEM_TEST_SUCCESS;
}

// A net pull down the free direction, which no bound stops: no solution. MPRGP says so.
int test_unbounded_direction_fails_cleanly() {
    std::vector<real_t> b(n, real_t(-0.01)), x;
    SFEM_TEST_ASSERT(solve(b, real_t(0.5), x) != SFEM_SUCCESS);
    for (const real_t v : x) SFEM_TEST_ASSERT(std::isfinite(v));
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_consistent_load_converges);
    SFEM_RUN_TEST(test_bound_holds_the_free_direction);
    SFEM_RUN_TEST(test_unbounded_direction_fails_cleanly);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
