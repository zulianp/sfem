// A bound-constrained solve on the host and, with CUDA, on the device: a chain with a mass term
// pressed between two wavy obstacles, so that the active set changes often and the iteration takes
// CG, expansion and proportioning steps. The solution is checked by its KKT conditions; the
// iteration count, the steps of each kind and a checksum of the solution's bits are printed, so that
// two builds of MPRGP can be compared bit for bit on the same operator. On the host the solution's
// bits do not depend on the number of threads, nor on whether the CG steps run pass by pass or in
// one region (an operator in team form).

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
#include <cuda_runtime.h>

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

    enum class HostOp { LIBRARY, ROWS, TEAM };

    // On the device, how the CG steps run: the product adds into its output and each step returns
    // to the host (ADDS); the product writes it whole and the steps return to the host one by one
    // (STEPS) or run in the execution space until the run ends (RUNS, MPRGP_Tpl::cg_run).
    enum class DeviceCg { ADDS, STEPS, RUNS };

    int solve(const sfem::ExecutionSpace es,
              uint64_t *const            sum        = nullptr,
              const HostOp               host_op    = HostOp::LIBRARY,
              const DeviceCg             device     = DeviceCg::ADDS,
              int *const                 iterations = nullptr,
              int *const                 runs       = nullptr,
              void *const                op_stream  = nullptr,
              void *const                qp_stream  = nullptr) {
        SFEM_UNUSED(op_stream);  // the device product's
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
        if (es == sfem::EXECUTION_SPACE_HOST && host_op == HostOp::LIBRARY) {
            op = sfem::h_crs_spmv<count_t, idx_t, real_t, real_t>(m, m, rowptr, colidx, values, real_t(1));
        } else if (es == sfem::EXECUTION_SPACE_HOST) {
            // One row function, either in a region of its own per product or in the caller's.
            auto row = [rowptr, colidx, values](const ptrdiff_t i, const real_t *const x) {
                real_t s = 0;
                for (count_t k = rowptr->data()[i]; k < rowptr->data()[i + 1]; ++k) s += values->data()[k] * x[colidx->data()[k]];
                return s;
            };
            if (host_op == HostOp::ROWS) {
                op = sfem::make_op<real_t>(
                        m,
                        m,
                        [row](const real_t *const x, real_t *const y) {
#pragma omp parallel for
                            for (ptrdiff_t i = 0; i < m; ++i) y[i] = row(i, x);
                        },
                        sfem::EXECUTION_SPACE_HOST);
            } else {
                op = sfem::make_team_op<real_t>(sfem::Communicator::world(), m, m, m, m, false, [row](const sfem::Team &team, const real_t *const x, real_t *const y) {
                    for (ptrdiff_t i = team.begin; i < team.end; ++i) y[i] = row(i, x);
                });
            }
        }
#ifdef SFEM_ENABLE_CUDA
        else {
            // The library's own BSR kernel (1 x 1 blocks): row by row, a fixed summation order.
            op    = sfem::d_bsr_spmv(m,
                                  m,
                                  1,
                                  smesh::to_device(rowptr),
                                  smesh::to_device(colidx),
                                  smesh::to_device(values),
                                  device == DeviceCg::ADDS ? real_t(1) : real_t(0),
                                  sfem::ExecutionContext::device(op_stream));
            lower = smesh::to_device(lower);
            upper = smesh::to_device(upper);
            b     = smesh::to_device(b);
            x     = smesh::to_device(x);
        }
#endif

        auto qp = sfem::create_mprgp(
                op, es == sfem::EXECUTION_SPACE_DEVICE ? sfem::ExecutionContext::device(qp_stream) : sfem::ExecutionContext(es));
        qp->verbose    = true;
        qp->set_op_overwrites(device != DeviceCg::ADDS);
        qp->set_device_runs(device == DeviceCg::RUNS);
        qp->check_each = 1 << 30;
        qp->set_upper_bound(upper);
        qp->set_lower_bound(lower);
        qp->set_max_eig(4 + mass);
        qp->set_max_it(20000);
        qp->set_rtol(1e-12);
        qp->set_atol(1e-14);
        SFEM_TEST_ASSERT(qp->apply(b->data(), x->data()) == SFEM_SUCCESS);

#ifdef SFEM_ENABLE_CUDA
        if (es != sfem::EXECUTION_SPACE_HOST) {
            cudaDeviceSynchronize();  // the solve's stream may not be the default one
            x = smesh::to_host(x);
        }
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
        if (iterations) *iterations = qp->iterations();
        if (runs) *runs = qp->device_run_count();
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

// The CG steps in one region (MPRGP::team_cg_steps) against the same steps pass by pass, on one
// row function: the same bits, at all threads and at one.
int test_host_team_solve() {
    uint64_t rows = 0, team = 0;
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_HOST, &rows, HostOp::ROWS) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_HOST, &team, HostOp::TEAM) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(rows == team);
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    uint64_t  one     = 0;
    omp_set_num_threads(1);
    const int err = solve(sfem::EXECUTION_SPACE_HOST, &one, HostOp::TEAM);
    omp_set_num_threads(threads);
    SFEM_TEST_ASSERT(err == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(one == team);
#endif
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
int test_device_solve() { return solve(sfem::EXECUTION_SPACE_DEVICE); }

// The CG steps in runs on the device, without returning to the host between them (a CUDA graph
// whose loop the device's decision continues), against the same steps returning to it one by one:
// the same bits, the same iterations, and the runs taken there.
int test_device_runs() {
    uint64_t steps = 0, runs = 0;
    int      it_steps = 0, it_runs = 0, n_steps = 0, n_runs = 0;
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_DEVICE, &steps, HostOp::LIBRARY, DeviceCg::STEPS, &it_steps, &n_steps) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_DEVICE, &runs, HostOp::LIBRARY, DeviceCg::RUNS, &it_runs, &n_runs) == SFEM_TEST_SUCCESS);
    printf("  steps one by one: %d iterations; in runs: %d iterations in %d runs\n", it_steps, it_runs, n_runs);
    SFEM_TEST_ASSERT(n_steps == 0 && n_runs > 0);
    SFEM_TEST_ASSERT(it_runs == it_steps);
    SFEM_TEST_ASSERT(runs == steps);
    return SFEM_TEST_SUCCESS;
}

// The product on a stream of its own, the solver on another (the default) or on the same: MPRGP
// orders the two, so each way the runs are taken and give the same bits as the steps one by one.
int test_device_runs_across_streams() {
    cudaStream_t s;
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    uint64_t steps = 0, apart = 0, together = 0;
    int      it = 0, n_steps = 0, n_apart = 0, n_together = 0;
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_DEVICE, &steps, HostOp::LIBRARY, DeviceCg::STEPS, &it, &n_steps) ==
                     SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(
            solve(sfem::EXECUTION_SPACE_DEVICE, &apart, HostOp::LIBRARY, DeviceCg::RUNS, &it, &n_apart, (void *)s, nullptr) ==
            SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(solve(sfem::EXECUTION_SPACE_DEVICE,
                           &together,
                           HostOp::LIBRARY,
                           DeviceCg::RUNS,
                           &it,
                           &n_together,
                           (void *)s,
                           (void *)s) == SFEM_TEST_SUCCESS);
    printf("  product on its own stream: %d runs; product and solver on one: %d runs\n", n_apart, n_together);
    cudaStreamDestroy(s);
    SFEM_TEST_ASSERT(n_apart > 0 && n_together > 0);
    SFEM_TEST_ASSERT(apart == steps && together == steps);
    return SFEM_TEST_SUCCESS;
}
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_host_solve);
    SFEM_RUN_TEST(test_host_team_solve);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_solve);
    SFEM_RUN_TEST(test_device_runs);
    SFEM_RUN_TEST(test_device_runs_across_streams);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
