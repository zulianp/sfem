// c++ -std=c++17 -O3 -ffast-math scripts/test_history_diagnostics.cpp -o /tmp/history-check-test
// /tmp/history-check-test
#include "../operators/hex8/hex8_history_scaling.hpp"
#include <cmath>
#include <sys/wait.h>
#include <unistd.h>

int main() {
    // Actual CSCS failing magnitude; exercise the production helper for both grouping sizes.
    for (const int n : {6, 48}) {
        for (const double magnitude : {0.0, std::numeric_limits<double>::min(),
                                       1.2218775401624972e-53, 1e-40, 1e-30, 1.0, 1e10}) {
            double H[48] = {magnitude, -magnitude / 2};
            const float scale = fp16_history_scale(H, n, true, 7, 2, 1);
            if (!sfem_history_finite(scale) || scale < std::numeric_limits<float>::min()) return 1;
            if (scale != fp16_history_scale(H, n, false, 7, 2, 1)) return 1;
            if (magnitude == 0 && scale != 1) return 1;
            if (magnitude == 1.2218775401624972e-53 && scale != std::numeric_limits<float>::min()) return 1;
            if (magnitude == 1 && scale != std::ldexp(1.0f, -14)) return 1;
            for (int c = 0; c < n; ++c) {
                const double normalized = H[c] * (1.0 / scale);
                sfem_check_history("normalized_regression", 7, 2, 1, c, normalized, scale, true);
                const _Float16 stored = static_cast<_Float16>(normalized);
                sfem_check_history("stored_regression", 7, 2, 1, c, double(stored), scale);
            }
        }
    }
    sfem_check_history("valid_zero", 7, 2, 1, 3, 0);
    sfem_check_history("valid_scaled", 7, 2, 1, 3, 32768, 0.00001, true);
    sfem_check_history("valid_negative", 7, 2, 1, 3, -65504, 0.5, true);
    sfem_check_history_scale(1, 0.5, 1, std::ldexp(1.0, -14), std::ldexp(1.0f, -14), 7, 2, 1);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double bad[][3] = {{nan, 1, 0}, {inf, 1, 0}, {1, 0, 0}, {1, -1, 0},
                            {1, nan, 0}, {1, inf, 0}, {1e308, 2, 0}, {65505, 1, 1}, {1e-50, 0, 0}};
    for (const auto &args : bad) {
        const pid_t pid = fork();
        if (pid < 0) { std::perror("fork"); return 1; }
        if (pid == 0) {
            if (args[0] == 1e-50) {
                int exponent;
                const double mantissa = std::frexp(args[0], &exponent);
                const double scale64 = std::ldexp(1.0, exponent - 15);
                sfem_check_history_scale(args[0], mantissa, exponent, scale64, float(scale64), 7, 2, 1);
                _exit(0);
            }
            sfem_check_history("expected_failure", 7, 2, 1, 3, args[0], args[1], args[2] != 0);
            _exit(0);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE) {
            std::fprintf(stderr, "Expected failure for value=%g scale=%g range=%g, status=%d\n",
                         args[0], args[1], args[2], status);
            return 1;
        }
    }
    std::puts("History diagnostics checks passed");
}
