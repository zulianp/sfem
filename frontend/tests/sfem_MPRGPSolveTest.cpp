// A bound-constrained solve on the host and, with CUDA, on the device: a chain with a mass term
// pressed between two wavy obstacles, so that the active set changes often and the iteration takes
// CG, expansion and proportioning steps. The solution is checked by its KKT conditions; the
// iteration count, the steps of each kind and a checksum of the solution's bits are printed, so that
// two builds of MPRGP can be compared bit for bit on the same operator. On the host the solution's
// bits do not depend on the number of threads.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_API.hpp"
#include "sfem_mprgp.hpp"
#include "sfem_test.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_crs_SpMV.hpp"
#endif

namespace {
    using V = std::vector<real_t>;

    constexpr ptrdiff_t m    = 200003;
    constexpr real_t    mass = real_t(0.05);

    real_t upper_bound(const ptrdiff_t i) {
        const real_t s = real_t(i) / real_t(m - 1);
        return real_t(0.02) + real_t(0.5) * (s - real_t(0.5)) * (s - real_t(0.5)) + real_t(0.004) * std::sin(real_t(733) * s);
    }

    real_t lower_bound(const ptrdiff_t i) {
        const real_t s = real_t(i) / real_t(m - 1);
        return real_t(-0.03) - real_t(0.2) * s + real_t(0.004) * std::sin(real_t(517) * s);
    }

    real_t rhs(const ptrdiff_t i) {
        const real_t s = real_t(i) / real_t(m - 1);
        return real_t(0.004) * std::sin(real_t(9) * s) + real_t(0.003) * std::sin(real_t(1201) * s);
    }

    // (2 + mass) on the diagonal, -1 next to it.
    void chain(sfem::SharedBuffer<count_t> &rowptr, sfem::SharedBuffer<idx_t> &colidx, sfem::SharedBuffer<real_t> &values) {
        rowptr = sfem::create_host_buffer<count_t>(m + 1);
        colidx = sfem::create_host_buffer<idx_t>(3 * m - 2);
        values = sfem::create_host_buffer<real_t>(3 * m - 2);
        count_t k = 0;
        for (ptrdiff_t i = 0; i < m; ++i) {
            rowptr->data()[i] = k;
            if (i > 0) {
                colidx->data()[k] = idx_t(i - 1);
                values->data()[k++] = -1;
            }
            colidx->data()[k]   = idx_t(i);
            values->data()[k++] = 2 + mass;
            if (i < m - 1) {
                colidx->data()[k]   = idx_t(i + 1);
                values->data()[k++] = -1;
            }
        }
        rowptr->data()[m] = k;
    }

    uint64_t checksum(const V &x) {
        uint64_t h = 1469598103934665603ull;
        for (const real_t v : x) {
            uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(v) < sizeof(bits) ? sizeof(v) : sizeof(bits));
            h = (h ^ bits) * 1099511628211ull;
        }
        return h;
    }

    int solve(const sfem::ExecutionSpace es, uint64_t *const sum = nullptr) {
        sfem::SharedBuffer<count_t> rowptr;
        sfem::SharedBuffer<idx_t>   colidx;
        sfem::SharedBuffer<real_t>  values;
        chain(rowptr, colidx, values);

        auto lower = sfem::create_host_buffer<real_t>(m), upper = sfem::create_host_buffer<real_t>(m);
        auto b = sfem::create_host_buffer<real_t>(m), x = sfem::create_host_buffer<real_t>(m);
        for (ptrdiff_t i = 0; i < m; ++i) {
            lower->data()[i] = lower_bound(i);
            upper->data()[i] = upper_bound(i);
            b->data()[i]     = rhs(i);
            x->data()[i]     = 0;
        }

        std::shared_ptr<sfem::Operator<real_t>> op;
        if (es == sfem::EXECUTION_SPACE_HOST) {
            op = sfem::h_crs_spmv<count_t, idx_t, real_t, real_t>(m, m, rowptr, colidx, values, real_t(1));
        }
#ifdef SFEM_ENABLE_CUDA
        else {
            // The library's own BSR kernel (1 x 1 blocks): row by row, a fixed summation order.
            op    = sfem::d_bsr_spmv(m, m, 1, smesh::to_device(rowptr), smesh::to_device(colidx), smesh::to_device(values), real_t(1));
            lower = smesh::to_device(lower);
            upper = smesh::to_device(upper);
            b     = smesh::to_device(b);
            x     = smesh::to_device(x);
        }
#endif

        auto qp        = sfem::create_mprgp(op, es);
        qp->verbose    = true;
        qp->check_each = 1 << 30;
        qp->set_upper_bound(upper);
        qp->set_lower_bound(lower);
        qp->set_max_eig(4 + mass);
        qp->set_max_it(20000);
        qp->set_rtol(1e-12);
        qp->set_atol(1e-14);
        SFEM_TEST_ASSERT(qp->apply(b->data(), x->data()) == SFEM_SUCCESS);

#ifdef SFEM_ENABLE_CUDA
        if (es != sfem::EXECUTION_SPACE_HOST) x = smesh::to_host(x);
#endif
        V xs(x->data(), x->data() + m);

        real_t    kkt = 0, scale = 0;
        ptrdiff_t on_lb = 0, on_ub = 0;
        for (ptrdiff_t i = 0; i < m; ++i) {
            real_t g = -rhs(i) + (2 + mass) * xs[i];
            if (i > 0) g -= xs[i - 1];
            if (i < m - 1) g -= xs[i + 1];
            const real_t l = lower_bound(i), u = upper_bound(i);
            SFEM_TEST_ASSERT(xs[i] >= l && xs[i] <= u);
            real_t r = g;
            if (xs[i] - l < 1e-12) {
                r = std::min(g, real_t(0));
                on_lb++;
            } else if (u - xs[i] < 1e-12) {
                r = std::max(g, real_t(0));
                on_ub++;
            }
            kkt += r * r;
            scale += rhs(i) * rhs(i);
        }
        printf("  %s: KKT %.3g of |b| %.3g, %ld on the lower bound, %ld on the upper, %d iterations, checksum %016llx\n",
               es == sfem::EXECUTION_SPACE_HOST ? "host" : "device",
               (double)std::sqrt(kkt),
               (double)std::sqrt(scale),
               (long)on_lb,
               (long)on_ub,
               qp->iterations(),
               (unsigned long long)checksum(xs));
        SFEM_TEST_ASSERT(on_lb > 0 && on_ub > 0);
        SFEM_TEST_ASSERT(std::sqrt(kkt) < 1e-10 * std::sqrt(scale));
        if (sum) *sum = checksum(xs);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

// The host passes sum over chunks fixed by the length (sfem_fixed_chunks.hpp): one thread and all of
// them take the same steps.
int test_host_solve() {
    uint64_t all = 0;
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_HOST, &all) == SFEM_TEST_SUCCESS);
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    uint64_t  one     = 0;
    omp_set_num_threads(1);
    const int err = solve(sfem::EXECUTION_SPACE_HOST, &one);
    omp_set_num_threads(threads);
    SFEM_TEST_ASSERT(err == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(one == all);
#endif
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
int test_device_solve() { return solve(sfem::EXECUTION_SPACE_DEVICE); }
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_host_solve);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_solve);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
