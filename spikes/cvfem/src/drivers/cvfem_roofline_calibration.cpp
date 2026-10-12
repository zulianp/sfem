// Copy and triad controls for checking counter accounting before operator measurements.
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <poll.h>
#include <unistd.h>

#include "cvfem_perf_control.hpp"

__attribute__((noinline)) static void sweep(const bool triad, const size_t n,
                                           double *a, const double *b, const double *c) {
    if (triad) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    } else {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) a[i] = b[i];
    }
    // Keep every sweep observable, including repeated writes of the same values.
    asm volatile("" : : "r"(a) : "memory");
}

#include "cvfem_calibration_session.hpp"

int main(int argc, char **argv) try {
    if (argc == 8 && std::string(argv[1]) == "session") return calibration_session(argv);
    if (argc != 5 && argc != 8 && argc != 10)
        throw std::runtime_error("usage: calibration copy|triad elements repeats threads [phase control_fd ack_fd [control_fd ack_fd]]");
    std::vector<int> descriptors;
    for (int i = 6; i < argc; ++i) descriptors.push_back(CounterPhase::descriptor(argv[i]));
    CounterPhase counters(argc == 5 ? "" : argv[5], descriptors);
    const std::string mode = argv[1];
    const auto n = std::stoull(argv[2]);
    const int repeats = std::stoi(argv[3]), requested = std::stoi(argv[4]);
    if ((mode != "copy" && mode != "triad") || n == 0 || n > 1000000000ULL || repeats < 1 || requested < 1)
        throw std::runtime_error("invalid calibration arguments");
    const char *warmup_env = std::getenv("CVFEM_CALIBRATION_WARMUP");
    const int warmups = warmup_env ? CounterPhase::descriptor(warmup_env) : 3;
    if (warmups < 1 || warmups > 1000) throw std::runtime_error("invalid calibration warmup count");
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    if (team != requested) throw std::runtime_error("OpenMP team does not match request");
    counters.begin("empty");
    counters.end("empty");
    double *a = nullptr, *b = nullptr, *c = nullptr;
    counters.begin("setup");
    if (posix_memalign(reinterpret_cast<void **>(&a), 64, n*sizeof(double)) ||
        posix_memalign(reinterpret_cast<void **>(&b), 64, n*sizeof(double)) ||
        posix_memalign(reinterpret_cast<void **>(&c), 64, n*sizeof(double)))
        throw std::runtime_error("allocation failed");
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
    counters.end("setup");
#ifdef __linux__
    std::ifstream maps("/proc/self/numa_maps");
    std::cout << "NUMA_MAPS_BEGIN\n" << maps.rdbuf() << "NUMA_MAPS_END\n";
#endif
    const bool triad = mode == "triad";
    counters.begin("warmup");
    for (int i = 0; i < warmups; ++i) sweep(triad, n, a, b, c);
    counters.end("warmup");
    counters.begin("sweep-empty");
    counters.end("sweep-empty");
    counters.begin("sweep");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) sweep(triad, n, a, b, c);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    counters.end("sweep");
    size_t bad = 0;
    counters.begin("check");
#pragma omp parallel for reduction(+:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) if (a[i] != (triad ? 7.0 : 1.0)) ++bad;
    counters.end("check");
    if (bad || !std::isfinite(seconds) || seconds <= 0) {
        std::fprintf(stderr, "calibration result failed: bad=%zu seconds=%.12g first=%.17g\n", bad, seconds, a[0]);
        return 1;
    }
    std::printf("CALIBRATION mode=%s elements=%llu repeats=%d threads=%d seconds=%.12g useful_bytes=%.0f flops=%.0f PASS\n",
                mode.c_str(), n, repeats, team, seconds, double(n)*(triad ? 24 : 16), double(n)*(triad ? 2 : 0));
    counters.begin("teardown");
    std::free(a); std::free(b); std::free(c);
    counters.end("teardown");
    counters.finish();
    std::printf("CALIBRATION_WARMUP sweeps=%d\n", warmups);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
