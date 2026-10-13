#ifndef CVFEM_ROOFLINE_WINDOW_HPP
#define CVFEM_ROOFLINE_WINDOW_HPP

#include "drivers/cvfem_perf_control.hpp"
#include <iostream>
#include <sstream>
#include <sys/resource.h>

template <typename Production, typename Reference>
static int run_roofline_window(MeshData &d, const std::string &window,
                               const std::vector<int> &fds, const std::string &layout,
                               const int limiter, const int pack_size,
                               const int repeat, const int warmup,
                               Production production, Reference reference,
                               const std::vector<scalar_t> *action = nullptr,
                               const char *operation = "residual",
                               const bool trace_timing = false,
                               const bool trace_resources = false) {
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
            if (action) measured = *action;
            else pack_residual(d, measured);
            reference();
            std::vector<scalar_t> expected;
            if (action) expected = *action;
            else pack_residual(d, expected);
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
        // Allocate diagnostic storage before warmup; print only after counters stop.
        std::vector<double> warmup_times(trace_timing ? warmup : 0);
        std::vector<double> apply_times(trace_timing ? repeat : 0);
        std::vector<rusage> warmup_resources(trace_resources ? warmup+1 : 0);
        std::vector<rusage> apply_resources(trace_resources ? repeat+1 : 0);
        auto sample_resources = [](rusage &sample) {
            if (getrusage(RUSAGE_SELF, &sample))
                throw std::runtime_error("getrusage failed in timing diagnostic");
        };
        // Verification buffers are released before warmup and do not enter the counter window.
        if (trace_timing) {
            if (trace_resources) sample_resources(warmup_resources[0]);
            auto previous = std::chrono::steady_clock::now();
            for (int i = 0; i < warmup; ++i) {
                production();
                if (trace_resources) sample_resources(warmup_resources[i+1]);
                const auto now = std::chrono::steady_clock::now();
                warmup_times[i] = std::chrono::duration<double>(now-previous).count();
                previous = now;
            }
        } else {
            for (int i = 0; i < warmup; ++i) production();
        }
        counters.begin("sweep-empty");
        counters.end("sweep-empty");
        counters.begin("sweep");
        if (trace_resources) sample_resources(apply_resources[0]);
        const auto start = std::chrono::steady_clock::now();
        double seconds;
        if (trace_timing) {
            auto previous = start;
            for (int i = 0; i < repeat; ++i) {
                production();
                if (trace_resources) sample_resources(apply_resources[i+1]);
                const auto now = std::chrono::steady_clock::now();
                apply_times[i] = std::chrono::duration<double>(now-previous).count();
                previous = now;
            }
            seconds = std::chrono::duration<double>(previous-start).count();
        } else {
            for (int i = 0; i < repeat; ++i) production();
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        counters.end("sweep");
        if (!std::isfinite(seconds) || seconds <= 0) throw std::runtime_error("invalid roofline duration");
        verify("after");
        counters.finish();
        if (trace_timing) {
            for (int i = 0; i < warmup; ++i)
                std::printf("ROOFLINE_TIMING phase=warmup index=%d seconds=%.17g\n", i, warmup_times[i]);
            for (int i = 0; i < repeat; ++i)
                std::printf("ROOFLINE_TIMING phase=apply index=%d seconds=%.17g\n", i, apply_times[i]);
        }
        if (trace_resources) {
            auto cpu_seconds = [](const timeval &before, const timeval &after) {
                // Subtract integer timestamps before fast-math can reassociate them.
                const long long micros = 1000000LL*(after.tv_sec-before.tv_sec) + after.tv_usec-before.tv_usec;
                return 1e-6*double(micros);
            };
            auto print_resources = [&](const char *phase, const std::vector<rusage> &samples) {
                for (size_t i = 1; i < samples.size(); ++i) {
                    const auto &a = samples[i-1];
                    const auto &b = samples[i];
                    std::printf("ROOFLINE_RESOURCE phase=%s index=%zu user_seconds=%.17g system_seconds=%.17g "
                                "minor_faults=%ld major_faults=%ld voluntary_switches=%ld involuntary_switches=%ld\n",
                                phase, i-1, cpu_seconds(a.ru_utime, b.ru_utime),
                                cpu_seconds(a.ru_stime, b.ru_stime),
                                b.ru_minflt-a.ru_minflt, b.ru_majflt-a.ru_majflt,
                                b.ru_nvcsw-a.ru_nvcsw, b.ru_nivcsw-a.ru_nivcsw);
                }
            };
            print_resources("warmup", warmup_resources);
            print_resources("apply", apply_resources);
        }
        if (action)
            std::printf("ROOFLINE_ACTION operation=%s direction=coordinate_fixed reconstruction=%s PASS\n",
                        operation, std::string(operation) == "jacobian" ? "per_apply" : "none");
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
                                const int pack_size, Production production, Reference reference,
                                const std::vector<scalar_t> *action = nullptr,
                                const char *operation = "residual") {
    try {
        CounterPhase validate_channels("sweep", fds);
        auto identity = [&](const char *marker) {
            std::printf("%s pid=%ld a=%p b=%p c=%p\n", marker, long(getpid()),
                        static_cast<void *>(d.ux.data()), static_cast<void *>(d.p.data()),
                        action ? static_cast<const void *>(action->data())
                               : static_cast<const void *>(d.rx.data()));
        };
        identity("SESSION_READY");
        std::fflush(stdout);
        std::string line;
        int index = 0;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line);
            std::string window, repeats_text, option, extra;
            if (!(input >> window >> repeats_text) ||
                (window != "normal" && window != "sweep" && window != "sweep-empty"))
                throw std::runtime_error("invalid roofline session request");
            const bool trace_timing = bool(input >> option);
            const bool trace_resources = option == "resources";
            if (trace_timing && ((option != "trace" && !trace_resources) || (input >> extra)))
                throw std::runtime_error("invalid roofline timing request");
            const int repeats = CounterPhase::descriptor(repeats_text);
            if (repeats < 1 || repeats > 1000) throw std::runtime_error("invalid roofline repeats");
            // Mesh, state, gradients and production scratch outlive every window in this block.
            const std::vector<int> channels = window == "normal" ? std::vector<int>{} : fds;
            if (run_roofline_window(d, window, channels, layout, limiter, pack_size,
                                    repeats, 30, production, reference, action, operation,
                                    trace_timing, trace_resources)) return 1;
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
