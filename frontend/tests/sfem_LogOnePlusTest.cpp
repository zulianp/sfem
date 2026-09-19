// `sfem_log1p` is only correct while its floating-point fence survives.
//
// The generated kernels compute `log(det F)`, and a deformation gradient is
// the identity plus something small, so the argument lands near 1 and a plain
// logarithm discards everything below eps relative to 1.  `sfem_log1p` recovers
// it from the part of `x` that `1 + x` rounded away -- arithmetic that
// `-ffast-math`, which this project builds with, is entitled to fold to zero.
// A `float_control` / `GCC optimize` fence stops it.
//
// The failure mode this test exists for is silent.  A compiler that does not
// recognise the fence, or a build that inlines through it, leaves a function
// that still returns an answer -- just the one a plain `log(1 + x)` would give,
// four orders worse, with nothing to show for it.  The assertions below are
// chosen to sit in that gap: at x = 1e-8 the accurate answer is right to a few
// ulp and the unfenced one is wrong at 8e-5.

#include "sfem_test.hpp"

#include "kernel_math.hpp"

#include <cmath>

namespace {

    // Values spanning the range a deformation gradient produces, from a strain
    // far below eps-relative-to-1 up to a large deformation.
    const double kProbes[] = {1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, -0.1, -0.5, 1.0, 3.0};

    int test_log1p_matches_the_library() {
        for (const double x : kProbes) {
            const double ours = sfem::codegen::sfem_log1p(x);
            const double reference = std::log1p(x);
            SFEM_TEST_ASSERT(std::abs(ours - reference) <= 1e-15 * std::abs(reference));
        }
        return SFEM_TEST_SUCCESS;
    }

    /// The same, evaluated where the kernels evaluate it.
    ///
    /// A fence can hold for a direct call and be lost when the function is
    /// inlined into a vectorized loop compiled with the project's flags, which
    /// is the only place that matters, so it is checked there too.
    int test_log1p_survives_a_vectorized_loop() {
        constexpr int n = sizeof(kProbes) / sizeof(kProbes[0]);
        double        out[n];

#pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = sfem::codegen::sfem_log1p(kProbes[i]);
        }

        for (int i = 0; i < n; ++i) {
            const double reference = std::log1p(kProbes[i]);
            SFEM_TEST_ASSERT(std::abs(out[i] - reference) <= 1e-15 * std::abs(reference));
        }
        return SFEM_TEST_SUCCESS;
    }

    /// And that it is doing something a plain logarithm does not.
    ///
    /// Without this the two tests above would pass on a build where the fence
    /// was gone but the tolerance happened to be generous.
    int test_a_plain_logarithm_would_fail_these() {
        // `volatile` so the compiler evaluates this at run time.  Folded at
        // compile time it is computed exactly and the control passes for the
        // wrong reason, which would leave the two tests above unable to tell a
        // working fence from a dead one.
        volatile double opaque = 1e-8;
        const double    x = opaque;
        const double    plain = std::log(1.0 + x);
        const double reference = std::log1p(x);
        // The gap the tests above live in: an accurate log1p holds 1e-15 or
        // better here, a plain logarithm is out by about 6e-9.  Asserting the
        // middle of that keeps the control honest without pinning the exact
        // error of somebody else's libm.
        SFEM_TEST_ASSERT(std::abs(plain - reference) > 1e-12 * std::abs(reference));
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_log1p_matches_the_library);
    SFEM_RUN_TEST(test_log1p_survives_a_vectorized_loop);
    SFEM_RUN_TEST(test_a_plain_logarithm_would_fail_these);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
