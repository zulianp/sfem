#ifndef CVFEM_CALIBRATION_SESSION_HPP
#define CVFEM_CALIBRATION_SESSION_HPP
#include <sstream>

// Reuse the process and arrays while each measurement opens fresh perf counters.
static int calibration_session(char **argv) {
    const auto n = static_cast<unsigned long long>(CounterPhase::descriptor(argv[2]));
    const int requested = CounterPhase::descriptor(argv[3]);
    if (!n || n > 1000000000ULL || requested < 1)
        throw std::runtime_error("invalid session size or team");
    std::vector<int> descriptors;
    for (int i = 4; i < 8; ++i) descriptors.push_back(CounterPhase::descriptor(argv[i]));
    const char *warmup = std::getenv("CVFEM_CALIBRATION_WARMUP");
    if (!warmup || std::string(warmup) != "30")
        throw std::runtime_error("reuse diagnostic requires 30 warmups");
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    if (team != requested) throw std::runtime_error("OpenMP team does not match request");
    double *a = nullptr, *b = nullptr, *c = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&a), 64, n*sizeof(double)) ||
        posix_memalign(reinterpret_cast<void **>(&b), 64, n*sizeof(double)) ||
        posix_memalign(reinterpret_cast<void **>(&c), 64, n*sizeof(double)))
        throw std::runtime_error("session allocation failed");
    std::printf("SESSION_READY pid=%ld a=%p b=%p c=%p\n", long(getpid()),
                static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c));
    std::fflush(stdout);
    std::string line;
    int index = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string mode, repeat_text, phase, extra;
        if (!(input >> mode >> repeat_text >> phase) || (input >> extra) ||
            (mode != "copy" && mode != "triad") || (phase != "sweep" && phase != "sweep-empty"))
            throw std::runtime_error("invalid session request");
        const int repeats = CounterPhase::descriptor(repeat_text);
        if (repeats < 1 || repeats > 1000) throw std::runtime_error("invalid session repeats");
        CounterPhase counters(phase, descriptors);
        const bool triad = mode == "triad";
        // Both lifetimes initialize and warm up before every capture.
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
#ifdef __linux__
        std::ifstream maps("/proc/self/numa_maps");
        std::cout << "NUMA_MAPS_BEGIN\n" << maps.rdbuf() << "NUMA_MAPS_END\n";
#endif
        for (int i = 0; i < 30; ++i) sweep(triad, n, a, b, c);
        counters.begin("sweep-empty");
        counters.end("sweep-empty");
        counters.begin("sweep");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) sweep(triad, n, a, b, c);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        counters.end("sweep");
        size_t bad = 0;
#pragma omp parallel for reduction(+:bad) schedule(static)
        for (size_t i = 0; i < n; ++i) if (a[i] != (triad ? 7.0 : 1.0)) ++bad;
        if (bad || !std::isfinite(seconds) || seconds <= 0)
            throw std::runtime_error("session numerical check failed");
        std::printf("CALIBRATION mode=%s elements=%llu repeats=%d threads=%d seconds=%.12g useful_bytes=%.0f flops=%.0f PASS\n",
                    mode.c_str(), n, repeats, team, seconds, double(n)*(triad ? 24 : 16), double(n)*(triad ? 2 : 0));
        counters.finish();
        std::printf("CALIBRATION_WARMUP sweeps=30\nSESSION_BUFFERS pid=%ld a=%p b=%p c=%p\nSESSION_DONE index=%d\n",
                    long(getpid()), static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c), index++);
        std::fflush(stdout);
    }
    if (!std::cin.eof() || index == 0) throw std::runtime_error("incomplete session input");
    std::free(a); std::free(b); std::free(c);
    return 0;
}
#endif
