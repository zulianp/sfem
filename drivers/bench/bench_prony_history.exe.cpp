// The Prony history's storage formats: bytes, the cost of a step's two passes, and what the format
// does to the history, on the host or the device (SFEM_EXECUTION_SPACE=device).
//
//   bench_prony_history <n_elements> <n_qp> [steps] [elements_per_mean]
//
// The series is the LDP valves' EPDM (15 terms, workflows/ldp_containment/transient) at their step,
// dt = 1e-4 s: four terms are stored and eleven relax within the step and share the increment D.
// The relaxed stress is synthetic and smooth: per component a level, varying along the elements
// (as a field on a mesh ordered along a space-filling curve does) and by 1% over an element's
// points, driven by a ramp and an oscillation in time.  Every row reports the elements, points and
// threads, the stored terms, the bytes of H, D and the frozen history, begin_step and advance per
// call (synchronized), advance per stored value, and the frozen history's largest difference from
// full storage's after the last step, relative to its largest entry.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_PronyRelaxation.hpp"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
    using clock_type = std::chrono::steady_clock;

    void synchronize(const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) cudaDeviceSynchronize();
#else
        (void)es;
#endif
    }

    sfem::PronySeries epdm() {
        return sfem::PronySeries({2.83e-04, 3.12e-04, 4.45e-04, 1.03e-03, 6.09e-03, 6.74e-02, 2.22e-01, 1.90e-01, 1.12e-01,
                                  5.07e-02, 6.91e-02, 4.59e-02, 8.62e-02, 6.76e-02, 7.87e-02},
                                 {1.61e+03, 1.29e+01, 1.53e-01, 9.89e-04, 1.18e-06, 8.51e-09, 1.42e-10, 1.80e-12, 1.27e-14,
                                  2.00e-16, 1.10e-19, 1.38e-22, 1.62e-24, 6.62e-26, 1.82e-27});
    }

    // The relaxed stress at time t, laid out [c][q][e]: in Pa, as the valves' (1e5 to 1e7).
    void relaxed_stress(const int nc, const int nq, const ptrdiff_t ne, const real_t t, real_t *const out) {
        const real_t drive = real_t(1e5) * (t / real_t(0.01) + std::sin(real_t(2 * M_PI) * t / real_t(0.045)));
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < ne; e++) {
            const real_t x = real_t(e) / real_t(ne);
            for (int c = 0; c < nc; c++) {
                const real_t level = (1 + c) * (real_t(1.5) + std::sin(real_t(2 * M_PI) * 7 * x + c));
                for (int q = 0; q < nq; q++) {
                    const real_t point = 1 + real_t(0.01) * (nq > 1 ? real_t(q) / (nq - 1) - real_t(0.5) : 0);
                    out[((ptrdiff_t)c * nq + q) * ne + e] = drive * level * point;
                }
            }
        }
    }

    // The relaxed stress at time t into `out`, in its space; untimed.
    void stress_at(const int nc, const int nq, const ptrdiff_t ne, const real_t t, const sfem::SharedBuffer<real_t> &host,
                   sfem::SharedBuffer<real_t> &out, const sfem::ExecutionSpace es) {
        relaxed_stress(nc, nq, ne, t, host->data());
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            out = smesh::to_device(host);
        } else {
            if (!out) out = sfem::create_host_buffer<real_t>(host->size());
            std::copy(host->data(), host->data() + host->size(), out->data());
        }
    }

    std::vector<real_t> to_host_vector(const real_t *const data, const ptrdiff_t n, const sfem::ExecutionSpace es) {
        auto view = sfem::Buffer<real_t>::wrap(n, const_cast<real_t *>(data), es == sfem::EXECUTION_SPACE_DEVICE ? sfem::MEMORY_SPACE_DEVICE : sfem::MEMORY_SPACE_HOST);
        auto host = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_host(view) : view;
        return std::vector<real_t>(host->data(), host->data() + n);
    }
}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "usage: %s <n_elements> <n_qp> [steps] [elements_per_mean]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const ptrdiff_t ne       = atol(argv[1]);
    const int       nq       = atoi(argv[2]);
    const int       steps    = argc > 3 ? atoi(argv[3]) : 50;
    const int       per_mean = argc > 4 ? atoi(argv[4]) : 0;
    const int       nc       = 6;
    const real_t    dt       = real_t(1e-4);

    const char *space = getenv("SFEM_EXECUTION_SPACE");
    const auto  es    = space && std::string(space) == "device" ? sfem::EXECUTION_SPACE_DEVICE : sfem::EXECUTION_SPACE_HOST;
    int         threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    const ptrdiff_t n    = (ptrdiff_t)nc * nq * ne;
    auto            host = sfem::create_host_buffer<real_t>(n);

    const sfem::HistoryStorage formats[] = {sfem::HistoryStorage::FULL,
                                            sfem::HistoryStorage::FLOAT32,
                                            sfem::HistoryStorage::FLOAT16_SCALED,
                                            sfem::HistoryStorage::INT16_SCALED,
                                            sfem::HistoryStorage::INT8_SCALED,
                                            sfem::HistoryStorage::FLOAT16_MD,
                                            sfem::HistoryStorage::INT8_MD};

    printf("format,space,elements,n_qp,threads,elements_per_mean,stored,bytes_history,bytes_increment,bytes_frozen,"
           "begin_step_us,advance_us,advance_ns_per_value,frozen_rel_diff\n");
    std::vector<real_t> reference;
    for (const auto storage : formats) {
        sfem::PronyRelaxation::Options options;
        options.history_storage           = storage;
        options.execution_space           = es;
        options.history_elements_per_mean = per_mean;
        sfem::PronyRelaxation relaxation(epdm(), nc, nq, ne, options);

        double                     t_begin = 0, t_advance = 0;
        sfem::SharedBuffer<real_t> S_n, S_new;
        stress_at(nc, nq, ne, 0, host, S_n, es);
        for (int k = 1; k <= steps; k++) {
            stress_at(nc, nq, ne, k * dt, host, S_new, es);
            synchronize(es);
            auto t0 = clock_type::now();
            relaxation.begin_step(k * dt, dt, S_n->data());
            synchronize(es);
            auto t1 = clock_type::now();
            relaxation.advance(S_n->data(), S_new->data());
            synchronize(es);
            auto t2 = clock_type::now();
            // The first step allocates the store: not timed.
            if (k > 1) {
                t_begin += std::chrono::duration<double>(t1 - t0).count();
                t_advance += std::chrono::duration<double>(t2 - t1).count();
            }
            std::swap(S_n, S_new);
        }
        // The frozen history the kernels would read at the next step.
        relaxation.begin_step((steps + 1) * dt, dt, S_n->data());
        synchronize(es);
        auto frozen = to_host_vector(relaxation.history(), n, es);
        if (storage == sfem::HistoryStorage::FULL) reference = frozen;
        real_t diff = 0, max_ref = 0;
        for (ptrdiff_t i = 0; i < n; i++) {
            diff    = std::max(diff, std::abs(frozen[i] - reference[i]));
            max_ref = std::max(max_ref, std::abs(reference[i]));
        }

        const int    timed  = steps - 1;
        const double values = double(relaxation.n_stored_terms() + 1) * n;  // the stored terms and D
        printf("%s,%s,%td,%d,%d,%d,%d,%zu,%zu,%zu,%.3f,%.3f,%.4f,%.3e\n",
               sfem::to_string(storage),
               es == sfem::EXECUTION_SPACE_DEVICE ? "device" : "host",
               ne,
               nq,
               threads,
               per_mean,
               relaxation.n_stored_terms(),
               relaxation.history_bytes(),
               relaxation.increment_bytes(),
               relaxation.frozen_bytes(),
               1e6 * t_begin / timed,
               1e6 * t_advance / timed,
               1e9 * t_advance / timed / values,
               (double)(max_ref > 0 ? diff / max_ref : diff));
        fflush(stdout);
    }
    return EXIT_SUCCESS;
}
