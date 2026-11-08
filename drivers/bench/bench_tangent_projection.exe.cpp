// The projection of a quadrature point's tangent to positive semidefinite (project_abs_symmetric,
// 9 x 9): the library's eigensolver (Householder tridiagonalisation and the implicit QL
// iteration), in double and in single precision, one matrix at a time and B in lockstep across SIMD
// lanes (project_abs_symmetric_lanes), against a cyclic Jacobi solver and, in a build with LAPACK,
// dsyev. Each arm projects the same matrices; rows report the arm, the family, the
// threads, nanoseconds per projection (median of 5 repetitions after a warm-up, per thread: the
// wall time times the threads over the projections) and the largest difference from the reference
// projection (dsyev where there is one, else Jacobi), relative to the matrix's largest entry.
//
//   bench_tangent_projection [n_matrices]
//
// Families: "uniform", entries uniform in [-1, 1]; "rubber", the spectrum of a stressed rubber's
// dP/dF: three rotation modes near zero, two of them negative, one negative stress mode, and five
// positive modes spread over three decades; "degenerate", eigenvalues of multiplicity two and three,
// a double negative one among them; "nearly_psd", one negative eigenvalue 1e-8 of the largest.
// All are indefinite, so every projection solves.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "sfem_TangentProjection.hpp"
#include "sfem_config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SFEM_ENABLE_LAPACK
extern "C" void dsyev_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, int *);
#endif

namespace {
    constexpr int N = 9;

#ifdef SFEM_ENABLE_LAPACK
    // The incumbent: dsyev, then Q |Lambda| Q^T.
    void project_lapack(double *const A) {
        double    Q[N * N], w[N], work[64 * N];
        const int n = N, lwork = 64 * N;
        int       info = 0;
        std::copy(A, A + N * N, Q);
        dsyev_("V", "U", &n, Q, &n, w, work, &lwork, &info);
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                double s = 0;
                for (int k = 0; k < N; ++k) s += Q[k * N + i] * std::fabs(w[k]) * Q[k * N + j];
                A[i * N + j] = A[j * N + i] = s;
            }
        }
    }
#endif

    // Cyclic Jacobi with Rutishauser's thresholds, then the same negative-part update.
    void project_jacobi(double *const A) {
        double a[N * N], V[N * N] = {0}, d[N], b[N], z[N];
        std::copy(A, A + N * N, a);
        for (int i = 0; i < N; ++i) {
            V[i * N + i] = 1;
            b[i] = d[i] = a[i * N + i];
            z[i]        = 0;
        }
        for (int sweep = 0; sweep < 50; ++sweep) {
            double off = 0;
            for (int p = 0; p < N - 1; ++p)
                for (int q = p + 1; q < N; ++q) off += std::fabs(a[p * N + q]);
            if (off == 0) break;
            const double threshold = sweep < 3 ? 0.2 * off / (N * N) : 0;
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) {
                    const double g = 100 * std::fabs(a[p * N + q]);
                    if (sweep > 3 && std::fabs(d[p]) + g == std::fabs(d[p]) && std::fabs(d[q]) + g == std::fabs(d[q])) {
                        a[p * N + q] = 0;
                    } else if (std::fabs(a[p * N + q]) > threshold) {
                        double       h = d[q] - d[p], t;
                        if (std::fabs(h) + g == std::fabs(h)) {
                            t = a[p * N + q] / h;
                        } else {
                            const double theta = 0.5 * h / a[p * N + q];
                            t                  = 1 / (std::fabs(theta) + std::sqrt(1 + theta * theta));
                            if (theta < 0) t = -t;
                        }
                        const double c = 1 / std::sqrt(1 + t * t), s = t * c, tau = s / (1 + c);
                        h = t * a[p * N + q];
                        z[p] -= h;
                        z[q] += h;
                        d[p] -= h;
                        d[q] += h;
                        a[p * N + q] = 0;
                        auto rotate  = [&](double *m, int i, int j, int k, int l) {
                            const double g_ = m[i * N + j], h_ = m[k * N + l];
                            m[i * N + j]    = g_ - s * (h_ + g_ * tau);
                            m[k * N + l]    = h_ + s * (g_ - h_ * tau);
                        };
                        for (int j = 0; j < p; ++j) rotate(a, j, p, j, q);
                        for (int j = p + 1; j < q; ++j) rotate(a, p, j, j, q);
                        for (int j = q + 1; j < N; ++j) rotate(a, p, j, q, j);
                        for (int j = 0; j < N; ++j) rotate(V, j, p, j, q);
                    }
                }
            }
            for (int p = 0; p < N; ++p) {
                b[p] += z[p];
                d[p] = b[p];
                z[p] = 0;
            }
        }
        for (int k = 0; k < N; ++k) {
            if (!(d[k] < 0)) continue;
            for (int i = 0; i < N; ++i)
                for (int j = i; j < N; ++j) A[i * N + j] -= 2 * d[k] * V[i * N + k] * V[j * N + k];
        }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < i; ++j) A[i * N + j] = A[j * N + i];
    }

    void project_ql(double *const A) { sfem::project_abs_symmetric<double, N>(A); }

    // B matrices in lockstep (project_abs_symmetric_lanes).
    template <int B>
    void project_lanes(double *const A) {
        sfem::project_abs_symmetric_lanes<double, N, B>(A);
    }

    template <int B>
    void project_lanes_f32(double *const A) {
        float a[B * N * N];
        for (int i = 0; i < B * N * N; ++i) a[i] = (float)A[i];
        sfem::project_abs_symmetric_lanes<float, N, B>(a);
        for (int i = 0; i < B * N * N; ++i) A[i] = a[i];
    }

    void project_ql_f32(double *const A) {
        float a[N * N];
        for (int i = 0; i < N * N; ++i) a[i] = (float)A[i];
        sfem::project_abs_symmetric<float, N>(a);
        for (int i = 0; i < N * N; ++i) A[i] = a[i];
    }

    void with_eigenvalues(const double *const lambda, std::mt19937 &gen, double *const A) {
        std::normal_distribution<double> u(0, 1);
        double                           Q[N * N];
        for (int k = 0; k < N; ++k) {
            double *const q = &Q[k * N];
            for (int i = 0; i < N; ++i) q[i] = u(gen);
            for (int l = 0; l < k; ++l) {
                double dot = 0;
                for (int i = 0; i < N; ++i) dot += q[i] * Q[l * N + i];
                for (int i = 0; i < N; ++i) q[i] -= dot * Q[l * N + i];
            }
            double n = 0;
            for (int i = 0; i < N; ++i) n += q[i] * q[i];
            for (int i = 0; i < N; ++i) q[i] /= std::sqrt(n);
        }
        for (int i = 0; i < N; ++i)
            for (int j = i; j < N; ++j) {
                double s = 0;
                for (int k = 0; k < N; ++k) s += Q[k * N + i] * lambda[k] * Q[k * N + j];
                A[i * N + j] = A[j * N + i] = s;
            }
    }

    std::vector<double> family(const std::string &name, const size_t count) {
        std::vector<double>                    out(count * N * N);
        std::mt19937                           gen(7);
        std::uniform_real_distribution<double> u(-1, 1);
        for (size_t m = 0; m < count; ++m) {
            double *const A = &out[m * N * N];
            if (name == "uniform") {
                for (int i = 0; i < N; ++i)
                    for (int j = i; j < N; ++j) A[i * N + j] = A[j * N + i] = u(gen);
            } else if (name == "degenerate") {
                const double lambda[N] = {-2, -2, -0.5, 0, 0, 3, 3, 3, 10};
                with_eigenvalues(lambda, gen, A);
            } else if (name == "nearly_psd") {
                const double lambda[N] = {-1e-8, 1e-3, 1e-2, 0.1, 1, 1, 2, 5, 1};
                with_eigenvalues(lambda, gen, A);
            } else {
                const double s         = 1e6 * (1 + 0.5 * u(gen));
                const double lambda[N] = {-1e-3 * s * (1 + u(gen)), -1e-4 * s * (1 + u(gen)), 1e-5 * s * u(gen),
                                          -0.05 * s * (1 + 0.5 * u(gen)), s * (1 + 0.1 * u(gen)), 3 * s,
                                          10 * s * (1 + 0.1 * u(gen)), 100 * s, 1e3 * s * (1 + 0.1 * u(gen))};
                with_eigenvalues(lambda, gen, A);
            }
        }
        return out;
    }

    int threads() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    // An arm projects `batch` consecutive matrices per call.
    struct Arm {
        const char                   *name;
        int                           batch;
        std::function<void(double *)> f;
    };

    // Nanoseconds per projection per thread, median of 5 after a warm-up, on `nt` threads.
    double time_arm(const Arm &arm, const std::vector<double> &input, const int nt) {
        const ptrdiff_t     count = (ptrdiff_t)(input.size() / (N * N)) / arm.batch * arm.batch;
        std::vector<double> work(input.size());
        std::vector<double> samples;
        for (int rep = 0; rep < 6; ++rep) {
            std::copy(input.begin(), input.end(), work.begin());
            const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(nt) schedule(static)
            for (ptrdiff_t m = 0; m < count; m += arm.batch) arm.f(&work[m * N * N]);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (rep > 0) samples.push_back(s);
        }
        std::sort(samples.begin(), samples.end());
        return 1e9 * samples[samples.size() / 2] * nt / count;
    }

    double max_difference(const Arm &arm, const std::function<void(double *)> &reference, const std::vector<double> &input) {
        const size_t        count = std::min<size_t>(input.size() / (N * N), 4096) / arm.batch * arm.batch;
        double              worst = 0;
        std::vector<double> a(input.begin(), input.begin() + count * N * N);
        for (size_t m = 0; m < count; m += arm.batch) arm.f(&a[m * N * N]);
        for (size_t m = 0; m < count; ++m) {
            double r[N * N], scale = 0;
            std::copy(&input[m * N * N], &input[(m + 1) * N * N], r);
            for (int i = 0; i < N * N; ++i) scale = std::max(scale, std::fabs(r[i]));
            reference(r);
            for (int i = 0; i < N * N; ++i) worst = std::max(worst, std::fabs(a[m * N * N + i] - r[i]) / scale);
        }
        return worst;
    }
}  // namespace

int main(int argc, char *argv[]) {
    const size_t count = argc > 1 ? (size_t)std::atol(argv[1]) : (size_t)1 << 16;
    std::vector<Arm> arms = {{"ql_f64", 1, project_ql},
                             {"ql_f32", 1, project_ql_f32},
                             {"lanes4_f64", 4, project_lanes<4>},
                             {"lanes8_f64", 8, project_lanes<8>},
                             {"lanes16_f64", 16, project_lanes<16>},
                             {"lanes8_f32", 8, project_lanes_f32<8>},
                             {"lanes16_f32", 16, project_lanes_f32<16>},
                             {"jacobi_f64", 1, project_jacobi}};
#ifdef SFEM_ENABLE_LAPACK
    arms.push_back({"dsyev_f64", 1, project_lapack});
    const auto reference = project_lapack;
#else
    const auto reference = project_jacobi;
#endif
    printf("arm,family,threads,ns_per_projection,max_rel_difference,matrices\n");
    for (const char *fam : {"uniform", "rubber", "degenerate", "nearly_psd"}) {
        const auto input = family(fam, count);
        for (auto &arm : arms) {
            const double difference = max_difference(arm, reference, input);
            for (const int nt : {1, threads()}) {
                printf("%s,%s,%d,%.1f,%.2e,%zu\n", arm.name, fam, nt, time_arm(arm, input, nt), difference, count);
                fflush(stdout);
            }
        }
    }
    return 0;
}
