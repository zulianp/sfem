// A checked dense FP64 reference with the same acknowledged windows as the operators.
#include "cvfem_perf_control.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

extern "C" void dgemm_(const char *, const char *, const int *, const int *, const int *,
                       const double *, const double *, const int *, const double *, const int *,
                       const double *, double *, const int *);

static void library_identity(const int requested) {
    Dl_info info{};
    // The imported function address can point to the executable's PLT stub.
    const auto symbol = dlsym(RTLD_DEFAULT, "dgemm_");
    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname)
        throw std::runtime_error("cannot identify the loaded DGEMM library");
#ifdef __APPLE__
    const char *limit = std::getenv("VECLIB_MAXIMUM_THREADS");
    if (!limit || CounterPhase::descriptor(limit) != requested)
        throw std::runtime_error("Accelerate requested thread limit does not match");
    std::printf("GEMM_IDENTITY backend=accelerate requested=%d actual=unavailable thread_scope=requested_limit\n", requested);
#else
    using GetThreads = int (*)();
    using GetConfig = const char *(*)();
    const auto threads = reinterpret_cast<GetThreads>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    const auto config = reinterpret_cast<GetConfig>(dlsym(RTLD_DEFAULT, "openblas_get_config"));
    if (!threads || !config) throw std::runtime_error("OpenBLAS runtime identity functions unavailable");
    const int actual = threads();
    if (actual != requested) throw std::runtime_error("OpenBLAS runtime thread count does not match");
    const char *identity = config();
    if (!identity || !*identity || std::string(identity).find("USE64BITINT") != std::string::npos)
        throw std::runtime_error("DGEMM requires an identified LP64 OpenBLAS build");
    std::printf("GEMM_IDENTITY backend=openblas requested=%d actual=%d thread_scope=runtime\n", requested, actual);
    std::printf("GEMM_CONFIG %s\n", identity);
#endif
    std::printf("GEMM_LIBRARY %s\n", info.dli_fname);
}

struct Gemm {
    using Buffer = std::unique_ptr<double, decltype(&std::free)>;
    int n;
    size_t count;
    Buffer a{nullptr, &std::free}, b{nullptr, &std::free}, c{nullptr, &std::free};
    std::vector<std::pair<size_t, long double>> reference;
    explicit Gemm(int dimension) : n(dimension), count(size_t(n)*n) {
        for (Buffer *buffer : {&a, &b, &c}) {
            void *memory = nullptr;
            if (posix_memalign(&memory, 64, count*sizeof(double)))
                throw std::runtime_error("DGEMM allocation failed");
            buffer->reset(static_cast<double *>(memory));
        }
        // Nonseparable, dyadic inputs keep the independent reference well conditioned.
        for (size_t j = 0; j < size_t(n); ++j)
            for (size_t i = 0; i < size_t(n); ++i) {
                const size_t k = i+j*n;
                a.get()[k] = 1.0 + double((i*17+j*31+i*j*3)%97)/128.0;
                b.get()[k] = 1.0 + double((i*29+j*13+i*j*7)%89)/128.0;
                c.get()[k] = 0.0;
            }
        std::vector<size_t> samples{0, count-1, size_t(n)-1, count-size_t(n), count/2};
        for (size_t k = 0; k < 16; ++k) samples.push_back(((k*997+7)%n)+((k*619+3)%n)*n);
        std::sort(samples.begin(), samples.end());
        samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
        for (const auto index : samples) {
            const size_t i = index%n, j = index/n;
            long double sum = 0.0L;
            for (size_t k = 0; k < size_t(n); ++k)
                sum += static_cast<long double>(a.get()[i+k*n])*b.get()[k+j*n];
            reference.emplace_back(index, sum);
        }
    }
    void apply() {
        const char trans = 'N';
        const double alpha = 1.0, beta = 0.0;
        dgemm_(&trans, &trans, &n, &n, &n, &alpha, a.get(), &n, b.get(), &n, &beta, c.get(), &n);
        asm volatile("" : : "r"(c.get()) : "memory");
    }
    void check(const char *stage) const {
        for (size_t k = 0; k < count; ++k)
            if (!std::isfinite(c.get()[k])) throw std::runtime_error("nonfinite DGEMM result");
        long double largest = 0.0L;
        for (const auto &entry : reference) {
            const long double relative = std::abs(static_cast<long double>(c.get()[entry.first])-entry.second)/std::abs(entry.second);
            largest = std::max(largest, relative);
        }
        if (largest > 1e-12L) throw std::runtime_error("independent DGEMM dot check failed");
        std::printf("GEMM_CHECK stage=%s finite=%zu samples=%zu relative=%.12Lg tolerance=1e-12 PASS\n",
                    stage, count, reference.size(), largest);
    }
    void capture(const std::string &phase, const int repeats, const int threads, const std::vector<int> &fds) {
        CounterPhase counters(phase == "normal" ? "" : phase, phase == "normal" ? std::vector<int>{} : fds);
        apply();
        check("before");
        for (int i = 0; i < 3; ++i) apply();
        counters.begin("sweep-empty");
        counters.end("sweep-empty");
        counters.begin("sweep");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) apply();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        counters.end("sweep");
        check("after");
        if (!std::isfinite(seconds) || seconds <= 0) throw std::runtime_error("invalid DGEMM elapsed time");
        counters.finish();
        std::printf("ROOFLINE_GEMM window=%s n=%d repeats=%d warmup=3 threads=%d seconds=%.12g useful_flops=%.0f precision=fp64 layout=column_major transa=N transb=N alpha=1 beta=0 PASS\n",
                    phase.c_str(), n, repeats, threads, seconds, 2.0*n*n*n);
    }
    void buffers(const char *marker) const {
        std::printf("%s pid=%ld a=%p b=%p c=%p\n", marker, long(getpid()),
                    static_cast<void *>(a.get()), static_cast<void *>(b.get()), static_cast<void *>(c.get()));
    }
};

int main(int argc, char **argv) try {
    if (argc != 4 && argc != 5 && argc != 8)
        throw std::runtime_error("usage: gemm session N threads fd fd fd fd | gemm normal N repeats threads | gemm preflight N threads");
    const std::string mode = argv[1];
    const int n = CounterPhase::descriptor(argv[2]);
    if (n < 1 || n > 8192) throw std::runtime_error("DGEMM dimension outside 1..8192");
    const bool session = mode == "session", preflight = mode == "preflight";
    if ((session && argc != 8) || (preflight && argc != 4) || (!session && !preflight && (mode != "normal" || argc != 5)))
        throw std::runtime_error("invalid DGEMM command");
    const int threads = CounterPhase::descriptor(argv[session || preflight ? 3 : 4]);
    if (threads < 1 || threads > 72) throw std::runtime_error("invalid DGEMM thread count");
    library_identity(threads);
    Gemm gemm(n);
#ifdef __linux__
    std::ifstream maps("/proc/self/numa_maps");
    std::cout << "NUMA_MAPS_BEGIN\n" << maps.rdbuf() << "NUMA_MAPS_END\n";
#endif
    if (!session) {
        const int repeats = preflight ? 1 : CounterPhase::descriptor(argv[3]);
        if (repeats < 1 || repeats > 3) throw std::runtime_error("DGEMM repeats outside 1..3");
        gemm.capture("normal", repeats, threads, {});
        return 0;
    }
    std::vector<int> fds;
    for (int i = 4; i < 8; ++i) fds.push_back(CounterPhase::descriptor(argv[i]));
    CounterPhase validate_channels("sweep", fds);
    gemm.buffers("SESSION_READY");
    std::fflush(stdout);
    std::string line;
    int index = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream request(line);
        std::string phase, repeat_text, extra;
        if (!(request >> phase >> repeat_text) || (request >> extra) ||
            (phase != "normal" && phase != "sweep" && phase != "sweep-empty"))
            throw std::runtime_error("invalid DGEMM session request");
        const int repeats = CounterPhase::descriptor(repeat_text);
        if (repeats < 1 || repeats > 3) throw std::runtime_error("DGEMM repeats outside 1..3");
        gemm.capture(phase, repeats, threads, fds);
        gemm.buffers("SESSION_BUFFERS");
        std::printf("SESSION_DONE index=%d\n", index++);
        std::fflush(stdout);
    }
    if (!std::cin.eof() || !index) throw std::runtime_error("incomplete DGEMM session");
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
