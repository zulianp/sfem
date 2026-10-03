// Build-flag self-test for the CVFEM finiteness guards.
//
// cvfem_hex8_ns_steady.cpp guards against a singular block-Jacobi block and a
// diverged Newton residual with std::isfinite(). Those guards only exist if the
// build's floating-point flags let them exist: -ffast-math implies
// -ffinite-math-only, under which the compiler assumes no NaN and no infinity
// ever occurs and folds every such guard to a constant `true`. The guards then
// fail *open* -- the solver proceeds on a NaN state and reports success.
//
// This is invisible at the source level: the guard is still written, still
// reviewed, still passes a code search. Only the generated code shows it is
// gone. So the protection has to be a test that runs, not a comment.
//
// Built with the same flags as the solver (${_cvfem_perf_opts}). Returns
// non-zero if any guard has been folded away. Wire into CI; if this fails,
// -fno-finite-math-only has been dropped from CMakeLists.txt.

#include <cmath>
#include <cstdio>
#include <cstring>

// Volatile so the bit pattern is opaque at the point of use: this stands in for
// a NaN produced by real arithmetic deep in a solve, not a literal the compiler
// can reason about at the call site.
volatile unsigned long long g_qnan = 0x7ff8000000000000ULL;
volatile unsigned long long g_pinf = 0x7ff0000000000000ULL;
volatile unsigned long long g_ninf = 0xfff0000000000000ULL;
volatile unsigned long long g_one  = 0x3ff0000000000000ULL;

static double as_double(unsigned long long b) {
    double x;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

static int check(const char *what, const double x, const bool want_finite) {
    const bool got = std::isfinite(x);
    const bool ok  = (got == want_finite);
    std::printf("  %-24s isfinite=%-5s expected=%-5s  %s\n",
                what, got ? "true" : "false", want_finite ? "true" : "false",
                ok ? "ok" : "FOLDED -- guard is dead");
    return ok ? 0 : 1;
}

int main() {
    std::printf("CVFEM guard self-test: are the finiteness guards alive in this build?\n");
    int bad = 0;
    bad += check("quiet NaN", as_double(g_qnan), false);
    bad += check("+infinity", as_double(g_pinf), false);
    bad += check("-infinity", as_double(g_ninf), false);
    bad += check("1.0 (control)", as_double(g_one), true);

    if (bad) {
        std::printf(
                "\nFAIL: %d of 4 guards folded away.\n"
                "The build is compiling with -ffinite-math-only (implied by -ffast-math),\n"
                "so std::isfinite() in cvfem_hex8_ns_steady.cpp is a constant `true` and the\n"
                "solver will run on a NaN state instead of reporting failure.\n"
                "Fix: restore -fno-finite-math-only in spikes/cvfem/CMakeLists.txt.\n",
                bad);
        return 1;
    }
    std::printf("\nPASS: all guards are live.\n");
    return 0;
}
