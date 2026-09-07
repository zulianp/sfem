// c++ -std=c++17 -O3 -ffast-math scripts/test_history_diagnostics.cpp -o /tmp/history-check-test
// /tmp/history-check-test
#include "../operators/hex8/hex8_history_diagnostics.hpp"
#include <cmath>
#include <sys/wait.h>
#include <unistd.h>

int main() {
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
