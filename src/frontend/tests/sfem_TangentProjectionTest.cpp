// The projection of a quadrature point's tangent to positive semidefinite, |A| = Q |Lambda| Q^T,
// against what characterises it without an eigensolver: |A| is symmetric positive semidefinite,
// commutes with A and squares to A^2; a positive definite A is its own projection.

#include "sfem_test.hpp"

#include "sfem_TangentProjection.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

    constexpr int N = 9;

    void product(const double *const a, const double *const b, double *const c) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int k = 0; k < N; ++k) s += a[i * N + k] * b[k * N + j];
                c[i * N + j] = s;
            }
        }
    }

    double max_abs(const double *const a) {
        double m = 0;
        for (int i = 0; i < N * N; ++i) m = std::max(m, std::fabs(a[i]));
        return m;
    }

    // A symmetric matrix with eigenvalues `lambda` in a random orthogonal basis (Gram-Schmidt).
    void with_eigenvalues(const double *const lambda, const unsigned seed, double *const A) {
        std::mt19937                           gen(seed);
        std::uniform_real_distribution<double> u(-1, 1);
        double                                 Q[N * N];
        for (int k = 0; k < N; ++k) {
            double *const q = &Q[k * N];
            for (int i = 0; i < N; ++i) q[i] = u(gen);
            for (int l = 0; l < k; ++l) {
                double d = 0;
                for (int i = 0; i < N; ++i) d += q[i] * Q[l * N + i];
                for (int i = 0; i < N; ++i) q[i] -= d * Q[l * N + i];
            }
            double n = 0;
            for (int i = 0; i < N; ++i) n += q[i] * q[i];
            for (int i = 0; i < N; ++i) q[i] /= std::sqrt(n);
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int k = 0; k < N; ++k) s += Q[k * N + i] * lambda[k] * Q[k * N + j];
                A[i * N + j] = s;
            }
        }
        // Exactly symmetric, as the generated kernels mirror the tangent.
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) A[i * N + j] = A[j * N + i];
        }
    }

    int test_positive_definite_is_unchanged() {
        const double lambda[N] = {1e-3, 0.1, 1, 2, 5, 10, 100, 1e3, 1e4};
        double       A[N * N], P[N * N];
        with_eigenvalues(lambda, 1, A);
        std::copy(A, A + N * N, P);
        sfem::project_abs_symmetric<double, N>(P);
        double d[N * N];
        for (int i = 0; i < N * N; ++i) d[i] = P[i] - A[i];
        SFEM_TEST_ASSERT(max_abs(d) <= 1e-12 * max_abs(A));
        return SFEM_TEST_SUCCESS;
    }

    int test_indefinite_becomes_its_absolute_value() {
        // Eigenvalues of both signs and of the spread of a nearly incompressible rubber.
        const double lambda[N] = {-3e6, -2, -1e-3, 0, 0.5, 4, 1e2, 2e4, 1e7};
        double       A[N * N], P[N * N];
        with_eigenvalues(lambda, 2, A);
        std::copy(A, A + N * N, P);
        sfem::project_abs_symmetric<double, N>(P);
        const double scale = max_abs(A);

        double d[N * N];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) d[i * N + j] = P[i * N + j] - P[j * N + i];
        }
        SFEM_TEST_ASSERT(max_abs(d) == 0);

        // |A| |A| = A A, and |A| A = A |A|.
        double PP[N * N], AA[N * N], PA[N * N], AP[N * N];
        product(P, P, PP);
        product(A, A, AA);
        for (int i = 0; i < N * N; ++i) d[i] = PP[i] - AA[i];
        SFEM_TEST_ASSERT(max_abs(d) <= 1e-12 * scale * scale);
        product(P, A, PA);
        product(A, P, AP);
        for (int i = 0; i < N * N; ++i) d[i] = PA[i] - AP[i];
        SFEM_TEST_ASSERT(max_abs(d) <= 1e-12 * scale * scale);

        // x^T |A| x >= 0, and along the eigenvectors of A's negative eigenvalues it is |lambda|:
        // the trace of |A| is the sum of |lambda|.
        double trace = 0, abs_sum = 0;
        for (int i = 0; i < N; ++i) trace += P[i * N + i];
        for (int k = 0; k < N; ++k) abs_sum += std::fabs(lambda[k]);
        SFEM_TEST_APPROXEQ(trace, abs_sum, 1e-12 * abs_sum);
        std::mt19937                           gen(3);
        std::uniform_real_distribution<double> u(-1, 1);
        for (int t = 0; t < 1000; ++t) {
            double x[N], q = 0;
            for (int i = 0; i < N; ++i) x[i] = u(gen);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) q += x[i] * P[i * N + j] * x[j];
            }
            SFEM_TEST_ASSERT(q >= -1e-12 * scale);
        }
        return SFEM_TEST_SUCCESS;
    }

    int test_single_precision_follows_double() {
        const double lambda[N] = {-50, -1, 0.25, 1, 2, 3, 8, 20, 400};
        double       A[N * N];
        float        Af[N * N];
        with_eigenvalues(lambda, 4, A);
        for (int i = 0; i < N * N; ++i) Af[i] = (float)A[i];
        for (int i = 0; i < N * N; ++i) A[i] = Af[i];
        sfem::project_abs_symmetric<double, N>(A);
        sfem::project_abs_symmetric<float, N>(Af);
        double d[N * N];
        for (int i = 0; i < N * N; ++i) d[i] = A[i] - (double)Af[i];
        SFEM_TEST_ASSERT(max_abs(d) <= 1e-5 * max_abs(A));
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_positive_definite_is_unchanged);
    SFEM_RUN_TEST(test_indefinite_becomes_its_absolute_value);
    SFEM_RUN_TEST(test_single_precision_follows_double);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
