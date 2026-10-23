#ifndef CVFEM_RETAINED_TIMING_HPP
#define CVFEM_RETAINED_TIMING_HPP
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <unistd.h>

// Checks and warmup are outside the same wall-time interval in every arm.
template <class Apply, class Reference, class T>
static void retained_action_timing(Apply apply, Reference reference, std::vector<T> &output,
        const T *direction, int windows, int warmup, const std::string &dump) {
    auto check = [&](bool run) {
        if (run) apply();
        const auto measured = output;
        reference();
        double worst = 0;
        for (int field = 0; field < 4; ++field) {
            double error = 0, scale = 0;
            for (size_t j = field; j < output.size(); j += 4) {
                if (!std::isfinite(double(output[j])) || !std::isfinite(double(measured[j])))
                    throw std::runtime_error("nonfinite retained action");
                error = std::max(error, std::abs(double(output[j])-double(measured[j])));
                scale = std::max(scale, std::abs(double(output[j])));
            }
            const double relative = error / std::max(scale, 1e-30);
            worst = std::max(worst, relative);
            if (relative > (sizeof(T) == 8 ? 1e-10 : 5e-5))
                throw std::runtime_error("retained action reference mismatch");
        }
        std::copy(measured.begin(), measured.end(), output.begin());
        return worst;
    };
    check(true);
    if (!dump.empty()) {
        std::ofstream file(dump, std::ios::binary);
        for (T value : output) { const double v=value; file.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
        if (!file) throw std::runtime_error("retained output dump failed");
    }
    const auto pointer = output.data();
    std::printf("RETAINED_READY pid=%ld direction=%p output=%p windows=%d warmup=%d\n",
        long(getpid()), static_cast<const void *>(direction), static_cast<void *>(pointer), windows, warmup);
    std::fflush(stdout);
    for (int index = 0; index < windows; ++index) {
        std::string line, extra;
        int repeats=0;
        if (!std::getline(std::cin, line)) throw std::runtime_error("incomplete retained session");
        std::istringstream input(line);
        if (!(input >> repeats) || (input >> extra) || repeats < 1 || repeats > 1000)
            throw std::runtime_error("invalid retained repeat count");
        double relative=check(true);
        for (int i=0; i<warmup; ++i) apply();
        const double start=wall_time();
        for (int i=0; i<repeats; ++i) apply();
        const double seconds=wall_time()-start;
        relative=std::max(relative,check(false));
        if (!(seconds>0) || !std::isfinite(seconds) || output.data()!=pointer)
            throw std::runtime_error("invalid retained measurement or changed storage");
        uint64_t fingerprint=14695981039346656037ull;
        for (T value:output) {
            const double v=value; uint64_t bits=0; std::memcpy(&bits,&v,sizeof(bits));
            fingerprint=(fingerprint^bits)*1099511628211ull;
        }
        std::printf("RETAINED_CAPTURE index=%d repeats=%d seconds=%.17g relative=%.9g fingerprint=%016llx pid=%ld direction=%p output=%p\n",
            index,repeats,seconds,relative,(unsigned long long)fingerprint,long(getpid()),
            static_cast<const void *>(direction),static_cast<void *>(pointer));
        std::fflush(stdout);
    }
    std::printf("RETAINED_DONE windows=%d\n",windows);
    std::fflush(stdout);
}
#endif
