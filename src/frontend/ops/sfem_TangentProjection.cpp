#include "sfem_TangentProjection.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "sfem_config.h"

#ifdef SFEM_ENABLE_LAPACK
extern "C" void dsyev_(const char *jobz,
                       const char *uplo,
                       const int  *n,
                       double     *a,
                       const int  *lda,
                       double     *w,
                       double     *work,
                       const int  *lwork,
                       int        *info);
#endif

namespace sfem {

    static constexpr int MAX_N = 9;

    bool tangent_projection_available() {
#ifdef SFEM_ENABLE_LAPACK
        return true;
#else
        return false;
#endif
    }

    // Whether the symmetric A is positive definite: its Cholesky factorisation exists. A NaN
    // fails the test.
    static bool positive_definite(const int n, const double *const A) {
        double L[MAX_N * MAX_N];
        for (int j = 0; j < n; ++j) {
            double d = A[j * n + j];
            for (int k = 0; k < j; ++k) d -= L[j * n + k] * L[j * n + k];
            if (!(d > 0)) return false;
            L[j * n + j] = std::sqrt(d);
            for (int i = j + 1; i < n; ++i) {
                double s = A[i * n + j];
                for (int k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
                L[i * n + j] = s / L[j * n + j];
            }
        }
        return true;
    }

    void project_abs_symmetric(const int n, double *const A) {
        assert(n > 0 && n <= MAX_N);
        // Most points of a tangent are positive definite, and such an A is its own projection:
        // the factorisation, a sixth of the eigensolver's work, says so first.
        if (positive_definite(n, A)) return;
#ifdef SFEM_ENABLE_LAPACK
        // Symmetric, so row and column major coincide on the way in; on the way out dsyev leaves
        // eigenvector k in column k, Q[k * n + i].
        double    Q[MAX_N * MAX_N], w[MAX_N], work[64 * MAX_N];
        const int lwork = 64 * MAX_N;
        int       info  = 0;
        for (int i = 0; i < n * n; ++i) Q[i] = A[i];
        dsyev_("V", "U", &n, Q, &n, w, work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    double s = 0;
                    for (int k = 0; k < n; ++k) s += Q[k * n + i] * std::fabs(w[k]) * Q[k * n + j];
                    A[i * n + j] = s;
                    A[j * n + i] = s;
                }
            }
            return;
        }
#endif
        for (int i = 0; i < n * n; ++i) A[i] = std::numeric_limits<double>::quiet_NaN();
    }

    void project_abs_symmetric(const int n, float *const A) {
        assert(n > 0 && n <= MAX_N);
        double a[MAX_N * MAX_N];
        for (int i = 0; i < n * n; ++i) a[i] = A[i];
        project_abs_symmetric(n, a);
        for (int i = 0; i < n * n; ++i) A[i] = (float)a[i];
    }

}  // namespace sfem
