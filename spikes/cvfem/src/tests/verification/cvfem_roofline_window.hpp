#ifndef CVFEM_ROOFLINE_WINDOW_HPP
#define CVFEM_ROOFLINE_WINDOW_HPP

#include "drivers/cvfem_perf_control.hpp"
#include <iostream>
#include <sstream>

template <typename Production, typename Reference>
static int run_roofline_window(MeshData &d, const std::string &window,
                               const std::vector<int> &fds, const std::string &layout,
                               const int limiter, const int pack_size,
                               const int repeat, const int warmup,
                               Production production, Reference reference) {
    try {
        CounterPhase counters(window == "normal" ? "" : window, fds);
        int team = 0;
#pragma omp parallel
        {
#pragma omp single
            team = cvfem_n_threads();
        }
        if (team != threads_active()) throw std::runtime_error("roofline OpenMP team mismatch");
        auto verify = [&](const char *stage) {
            std::vector<scalar_t> measured;
            pack_residual(d, measured);
            reference();
            std::vector<scalar_t> expected;
            pack_residual(d, expected);
            bool nonzero = false;
            for (int f = 0; f < 4; ++f) {
                double scale = 0, error = 0;
                for (size_t i = f; i < measured.size(); i += 4) {
                    const double a = measured[i], b = expected[i];
                    if (!std::isfinite(a) || !std::isfinite(b))
                        throw std::runtime_error("nonfinite roofline result");
                    scale = std::max(scale, std::abs(b));
                    error = std::max(error, std::abs(a-b));
                }
                nonzero = nonzero || scale > 0;
                const double relative = error / std::max(scale, 1e-30);
                if (!(relative <= 1e-10)) throw std::runtime_error("roofline result differs from atomic reference");
                std::printf("ROOFLINE_CHECK stage=%s field=%d relative=%.17g PASS\n", stage, f, relative);
            }
            if (!nonzero) throw std::runtime_error("zero roofline reference");
        };
        production();
        verify("before");
        // Verification buffers are released before warmup and do not enter the counter window.
        for (int i = 0; i < warmup; ++i) production();
        counters.begin("sweep-empty");
        counters.end("sweep-empty");
        counters.begin("sweep");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) production();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        counters.end("sweep");
        if (!std::isfinite(seconds) || seconds <= 0) throw std::runtime_error("invalid roofline duration");
        verify("after");
        counters.finish();
        std::printf("ROOFLINE_OPERATOR window=%s layout=%s limiter=%d elements=%td nodes=%td "
                    "pack=%d repeats=%d warmup=%d threads=%d seconds=%.17g "
                    "precision=fp64 geometry=affine rc=1 state_gradients=cached boundary=0 transient=0 PASS\n",
                    window.c_str(), layout.c_str(), limiter, d.nelements, d.nnodes,
                    pack_size, repeat, warmup, team, seconds);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ROOFLINE_OPERATOR FAILED: %s\n", e.what());
        return 1;
    }
}

template <typename Production, typename Reference>
static int run_roofline_session(MeshData &d, const std::vector<int> &fds,
                                const std::string &layout, const int limiter,
                                const int pack_size, Production production, Reference reference) {
    try {
        CounterPhase validate_channels("sweep", fds);
        auto identity = [&](const char *marker) {
            std::printf("%s pid=%ld a=%p b=%p c=%p\n", marker, long(getpid()),
                        static_cast<void *>(d.ux.data()), static_cast<void *>(d.p.data()),
                        static_cast<void *>(d.rx.data()));
        };
        identity("SESSION_READY");
        std::fflush(stdout);
        std::string line;
        int index = 0;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line);
            std::string window, repeats_text, extra;
            if (!(input >> window >> repeats_text) || (input >> extra) ||
                (window != "normal" && window != "sweep" && window != "sweep-empty"))
                throw std::runtime_error("invalid roofline session request");
            const int repeats = CounterPhase::descriptor(repeats_text);
            if (repeats < 1 || repeats > 1000) throw std::runtime_error("invalid roofline repeats");
            // Mesh, state, gradients and production scratch outlive every window in this block.
            const std::vector<int> channels = window == "normal" ? std::vector<int>{} : fds;
            if (run_roofline_window(d, window, channels, layout, limiter, pack_size,
                                    repeats, 30, production, reference)) return 1;
            identity("SESSION_BUFFERS");
            std::printf("SESSION_DONE index=%d\n", index++);
            std::fflush(stdout);
        }
        if (!std::cin.eof() || index == 0) throw std::runtime_error("incomplete roofline session input");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ROOFLINE_SESSION FAILED: %s\n", e.what());
        return 1;
    }
}
#endif
