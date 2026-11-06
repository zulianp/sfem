#ifndef SFEM_TANGENT_PROJECTION_HPP
#define SFEM_TANGENT_PROJECTION_HPP

// The projection of a quadrature point's tangent to positive semidefinite, |A| = Q |Lambda| Q^T.
//
// One header for the host and the device: the kernels call it per point, inside their element
// loops, so it is inline, allocation-free and templated on the scalar and the size (dP/dF is
// 9 x 9 in 3D, 4 x 4 in 2D). The eigensolver is Householder tridiagonalisation followed by the
// implicit QL iteration (tred2 and tql2 of EISPACK, as in JAMA), what LAPACK's dsyev does inside
// without the library call, whose locks serialise a threaded element loop. On Grace
// (bench_tangent_projection, 9 x 9) it takes 1.9-3.4 us per matrix on every thread at once, against
// 8 us for a cyclic Jacobi solver and 97 us for dsyev; computing the eigenpairs of the negative
// eigenvalues only (tridiagonal eigenvalues, inverse iteration, as dsyevx) was slower, 2.8-4.6 us.
//
// A host kernel projects all the points of an element (or of a slab) at once with
// project_abs_symmetric_lanes, the same algorithm in lockstep across SIMD lanes: 1.4-2.4 us per
// matrix at 8 lanes in double precision, 1.15-1.75 us at 16 in single. A device work item is a
// point, and projects it with project_abs_symmetric.

#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SFEM_TANGENT_PROJECTION_FUNCTION __host__ __device__ inline
#else
#define SFEM_TANGENT_PROJECTION_FUNCTION inline
#endif

// The lane loops of the batched projection, vectorised across its matrices.
#if defined(_OPENMP) && !defined(__CUDA_ARCH__)
#define SFEM_TANGENT_PROJECTION_LANES _Pragma("omp simd") for (int b = 0; b < B; ++b)
#else
#define SFEM_TANGENT_PROJECTION_LANES for (int b = 0; b < B; ++b)
#endif

namespace sfem {
    namespace tangent_projection {

        template <typename T>
        SFEM_TANGENT_PROJECTION_FUNCTION T abs_(const T x) {
            return x < T(0) ? -x : x;
        }

        /// Whether the symmetric A (row major) is positive definite: its Cholesky factorisation
        /// exists. A NaN fails the test.
        template <typename T, int N>
        SFEM_TANGENT_PROJECTION_FUNCTION bool positive_definite(const T *const A) {
            T L[N * N];
            for (int j = 0; j < N; ++j) {
                T d = A[j * N + j];
                for (int k = 0; k < j; ++k) d -= L[j * N + k] * L[j * N + k];
                if (!(d > T(0))) return false;
                const T l    = sqrt(d);
                L[j * N + j] = l;
                const T inv  = T(1) / l;
                for (int i = j + 1; i < N; ++i) {
                    T s = A[i * N + j];
                    for (int k = 0; k < j; ++k) s -= L[i * N + k] * L[j * N + k];
                    L[i * N + j] = s * inv;
                }
            }
            return true;
        }

        /// The eigensystem of the symmetric V (row major, N x N, entries of order one): on return
        /// d holds the eigenvalues and column k of V, V[i * N + k], the eigenvector of d[k].
        /// Returns false where the QL iteration takes more than 30 sweeps for an eigenvalue.
        template <typename T, int N>
        SFEM_TANGENT_PROJECTION_FUNCTION bool symmetric_eigensystem(T *const V, T *const d) {
            T e[N];
            // Householder reduction to tridiagonal form, the transformation accumulated in V.
            for (int j = 0; j < N; ++j) d[j] = V[(N - 1) * N + j];
            for (int i = N - 1; i > 0; --i) {
                T scale = 0, h = 0;
                for (int k = 0; k < i; ++k) scale += abs_(d[k]);
                if (scale == T(0)) {
                    e[i] = d[i - 1];
                    for (int j = 0; j < i; ++j) {
                        d[j]         = V[(i - 1) * N + j];
                        V[i * N + j] = 0;
                        V[j * N + i] = 0;
                    }
                } else {
                    const T inv_scale = T(1) / scale;
                    for (int k = 0; k < i; ++k) {
                        d[k] *= inv_scale;
                        h += d[k] * d[k];
                    }
                    T f = d[i - 1];
                    T g = sqrt(h);
                    if (f > T(0)) g = -g;
                    e[i]     = scale * g;
                    h        = h - f * g;
                    d[i - 1] = f - g;
                    for (int j = 0; j < i; ++j) e[j] = 0;
                    for (int j = 0; j < i; ++j) {
                        f            = d[j];
                        V[j * N + i] = f;
                        g            = e[j] + V[j * N + j] * f;
                        for (int k = j + 1; k <= i - 1; ++k) {
                            g += V[k * N + j] * d[k];
                            e[k] += V[k * N + j] * f;
                        }
                        e[j] = g;
                    }
                    f             = 0;
                    const T inv_h = T(1) / h;
                    for (int j = 0; j < i; ++j) {
                        e[j] *= inv_h;
                        f += e[j] * d[j];
                    }
                    const T hh = f / (h + h);
                    for (int j = 0; j < i; ++j) e[j] -= hh * d[j];
                    for (int j = 0; j < i; ++j) {
                        f = d[j];
                        g = e[j];
                        for (int k = j; k <= i - 1; ++k) V[k * N + j] -= (f * e[k] + g * d[k]);
                        d[j]         = V[(i - 1) * N + j];
                        V[i * N + j] = 0;
                    }
                }
                d[i] = h;
            }
            for (int i = 0; i < N - 1; ++i) {
                V[(N - 1) * N + i] = V[i * N + i];
                V[i * N + i]       = 1;
                const T h          = d[i + 1];
                if (h != T(0)) {
                    const T inv_h = T(1) / h;
                    for (int k = 0; k <= i; ++k) d[k] = V[k * N + i + 1] * inv_h;
                    for (int j = 0; j <= i; ++j) {
                        T g = 0;
                        for (int k = 0; k <= i; ++k) g += V[k * N + i + 1] * V[k * N + j];
                        for (int k = 0; k <= i; ++k) V[k * N + j] -= g * d[k];
                    }
                }
                for (int k = 0; k <= i; ++k) V[k * N + i + 1] = 0;
            }
            for (int j = 0; j < N; ++j) {
                d[j]               = V[(N - 1) * N + j];
                V[(N - 1) * N + j] = 0;
            }
            V[N * N - 1] = 1;
            e[0]         = 0;

            // The implicit QL iteration on the tridiagonal matrix, the rotations applied to V.
            for (int i = 1; i < N; ++i) e[i - 1] = e[i];
            e[N - 1]    = 0;
            T       f    = 0;
            T       tst1 = 0;
            const T eps  = sizeof(T) == sizeof(float) ? T(1.1920929e-7) : T(2.220446049250313e-16);
            for (int l = 0; l < N; ++l) {
                const T t = abs_(d[l]) + abs_(e[l]);
                if (t > tst1) tst1 = t;
                int m = l;
                while (m < N) {
                    if (abs_(e[m]) <= eps * tst1) break;
                    ++m;
                }
                if (m > l) {
                    int sweeps = 0;
                    do {
                        if (++sweeps > 30) return false;
                        T g = d[l];
                        T p = (d[l + 1] - g) / (T(2) * e[l]);
                        T r = sqrt(p * p + T(1));
                        if (p < T(0)) r = -r;
                        d[l]        = e[l] / (p + r);
                        d[l + 1]    = e[l] * (p + r);
                        const T dl1 = d[l + 1];
                        T       h   = g - d[l];
                        for (int i = l + 2; i < N; ++i) d[i] -= h;
                        f += h;
                        p           = d[m];
                        T       c = 1, c2 = 1, c3 = 1, s = 0, s2 = 0;
                        const T el1 = e[l + 1];
                        for (int i = m - 1; i >= l; --i) {
                            c3       = c2;
                            c2       = c;
                            s2       = s;
                            g        = c * e[i];
                            h        = c * p;
                            r        = sqrt(p * p + e[i] * e[i]);
                            e[i + 1] = s * r;
                            s        = e[i] / r;
                            c        = p / r;
                            p        = c * d[i] - s * g;
                            d[i + 1] = h + s * (c * g + s * d[i]);
                            for (int k = 0; k < N; ++k) {
                                h                = V[k * N + i + 1];
                                V[k * N + i + 1] = s * V[k * N + i] + c * h;
                                V[k * N + i]     = c * V[k * N + i] - s * h;
                            }
                        }
                        p    = -s * s2 * c3 * el1 * e[l] / dl1;
                        e[l] = s * p;
                        d[l] = c * p;
                    } while (abs_(e[l]) > eps * tst1);
                }
                d[l] += f;
                e[l] = 0;
            }
            return true;
        }

        /// symmetric_eigensystem for B matrices in lockstep, their entries interleaved: V[e][b] is
        /// entry e of matrix b. Every loop is a loop over the B lanes, which the compiler vectorises;
        /// nothing branches on a lane's values. The tridiagonalisation has no data-dependent control
        /// flow once its two special cases (a zero column, a zero norm) are folded into guarded
        /// reciprocals, under which the general formulas give what the special cases did. The QL
        /// iteration chases every sweep over the whole trailing matrix rather than to the first
        /// negligible subdiagonal, which costs rotations but keeps the lanes in step: at a zero
        /// subdiagonal the rotation is the identity, and the chase restarts there as a split one
        /// would. A lane whose eigenvalue has converged applies identity rotations and keeps its
        /// values while the others iterate. ok[b] is false where lane b took more than 30 sweeps.
        template <typename T, int N, int B>
        SFEM_TANGENT_PROJECTION_FUNCTION void symmetric_eigensystem_lanes(T (*const V)[B], T (*const d)[B], bool *const ok) {
            T e[N][B];
            for (int j = 0; j < N; ++j) SFEM_TANGENT_PROJECTION_LANES d[j][b] = V[(N - 1) * N + j][b];
            for (int i = N - 1; i > 0; --i) {
                T scale[B], h[B], f[B], g[B], inv[B];
                SFEM_TANGENT_PROJECTION_LANES {
                    scale[b] = 0;
                    h[b]     = 0;
                }
                for (int k = 0; k < i; ++k) SFEM_TANGENT_PROJECTION_LANES scale[b] += abs_(d[k][b]);
                SFEM_TANGENT_PROJECTION_LANES inv[b] = scale[b] == T(0) ? T(1) : T(1) / scale[b];
                for (int k = 0; k < i; ++k) SFEM_TANGENT_PROJECTION_LANES {
                        d[k][b] *= inv[b];
                        h[b] += d[k][b] * d[k][b];
                    }
                SFEM_TANGENT_PROJECTION_LANES {
                    f[b]        = d[i - 1][b];
                    const T s   = sqrt(h[b]);
                    g[b]        = f[b] > T(0) ? -s : s;
                    e[i][b]     = scale[b] * g[b];
                    h[b]        = h[b] - f[b] * g[b];
                    d[i - 1][b] = f[b] - g[b];
                }
                for (int j = 0; j < i; ++j) SFEM_TANGENT_PROJECTION_LANES e[j][b] = 0;
                for (int j = 0; j < i; ++j) {
                    T fj[B], gj[B];
                    SFEM_TANGENT_PROJECTION_LANES {
                        fj[b]           = d[j][b];
                        V[j * N + i][b] = fj[b];
                        gj[b]           = e[j][b] + V[j * N + j][b] * fj[b];
                    }
                    for (int k = j + 1; k <= i - 1; ++k) SFEM_TANGENT_PROJECTION_LANES {
                            gj[b] += V[k * N + j][b] * d[k][b];
                            e[k][b] += V[k * N + j][b] * fj[b];
                        }
                    SFEM_TANGENT_PROJECTION_LANES e[j][b] = gj[b];
                }
                SFEM_TANGENT_PROJECTION_LANES {
                    f[b]   = 0;
                    inv[b] = h[b] == T(0) ? T(0) : T(1) / h[b];
                }
                for (int j = 0; j < i; ++j) SFEM_TANGENT_PROJECTION_LANES {
                        e[j][b] *= inv[b];
                        f[b] += e[j][b] * d[j][b];
                    }
                SFEM_TANGENT_PROJECTION_LANES g[b] = T(0.5) * f[b] * inv[b];
                for (int j = 0; j < i; ++j) SFEM_TANGENT_PROJECTION_LANES e[j][b] -= g[b] * d[j][b];
                for (int j = 0; j < i; ++j) {
                    for (int k = j; k <= i - 1; ++k) SFEM_TANGENT_PROJECTION_LANES V[k * N + j][b] -= (d[j][b] * e[k][b] + e[j][b] * d[k][b]);
                    SFEM_TANGENT_PROJECTION_LANES {
                        d[j][b]         = V[(i - 1) * N + j][b];
                        V[i * N + j][b] = 0;
                    }
                }
                SFEM_TANGENT_PROJECTION_LANES d[i][b] = h[b];
            }
            for (int i = 0; i < N - 1; ++i) {
                T inv[B];
                SFEM_TANGENT_PROJECTION_LANES {
                    V[(N - 1) * N + i][b] = V[i * N + i][b];
                    V[i * N + i][b]       = 1;
                    inv[b]                = d[i + 1][b] == T(0) ? T(0) : T(1) / d[i + 1][b];
                }
                for (int k = 0; k <= i; ++k) SFEM_TANGENT_PROJECTION_LANES d[k][b] = V[k * N + i + 1][b] * inv[b];
                for (int j = 0; j <= i; ++j) {
                    T g[B];
                    SFEM_TANGENT_PROJECTION_LANES g[b] = 0;
                    for (int k = 0; k <= i; ++k) SFEM_TANGENT_PROJECTION_LANES g[b] += V[k * N + i + 1][b] * V[k * N + j][b];
                    for (int k = 0; k <= i; ++k) SFEM_TANGENT_PROJECTION_LANES V[k * N + j][b] -= g[b] * d[k][b];
                }
                for (int k = 0; k <= i; ++k) SFEM_TANGENT_PROJECTION_LANES V[k * N + i + 1][b] = 0;
            }
            for (int j = 0; j < N; ++j) SFEM_TANGENT_PROJECTION_LANES {
                    d[j][b]               = V[(N - 1) * N + j][b];
                    V[(N - 1) * N + j][b] = 0;
                }
            SFEM_TANGENT_PROJECTION_LANES {
                V[N * N - 1][b] = 1;
                e[0][b]         = 0;
            }

            for (int i = 1; i < N; ++i) SFEM_TANGENT_PROJECTION_LANES e[i - 1][b] = e[i][b];
            T       f[B], tst1[B];
            const T eps = sizeof(T) == sizeof(float) ? T(1.1920929e-7) : T(2.220446049250313e-16);
            SFEM_TANGENT_PROJECTION_LANES {
                e[N - 1][b] = 0;
                f[b]        = 0;
                tst1[b]     = 0;
                ok[b]       = true;
            }
            for (int l = 0; l < N - 1; ++l) {
                SFEM_TANGENT_PROJECTION_LANES {
                    const T t = abs_(d[l][b]) + abs_(e[l][b]);
                    tst1[b]   = t > tst1[b] ? t : tst1[b];
                }
                for (int sweep = 0;; ++sweep) {
                    bool active[B];
                    bool any = false;
                    for (int b = 0; b < B; ++b) {
                        active[b] = abs_(e[l][b]) > eps * tst1[b];
                        any       = any || active[b];
                    }
                    if (!any) break;
                    if (sweep == 30) {
                        for (int b = 0; b < B; ++b) ok[b] = ok[b] && !active[b];
                        break;
                    }
                    T p[B], c[B], c2[B], c3[B], s[B], s2[B], el1[B], dl1[B], shift[B];
                    SFEM_TANGENT_PROJECTION_LANES {
                        const T el = e[l][b];
                        const T g  = d[l][b];
                        T       q  = (d[l + 1][b] - g) / (active[b] ? T(2) * el : T(1));
                        T       r  = sqrt(q * q + T(1));
                        r          = q < T(0) ? -r : r;
                        const T dl = el / (q + r);
                        shift[b]   = active[b] ? g - dl : T(0);
                        d[l][b]    = active[b] ? dl : d[l][b];
                        d[l + 1][b] = active[b] ? el * (q + r) : d[l + 1][b];
                        dl1[b]     = active[b] ? d[l + 1][b] : T(1);
                    }
                    for (int i = l + 2; i < N; ++i) SFEM_TANGENT_PROJECTION_LANES d[i][b] -= shift[b];
                    SFEM_TANGENT_PROJECTION_LANES {
                        f[b] += shift[b];
                        p[b]   = d[N - 1][b];
                        c[b]   = 1;
                        c2[b]  = 1;
                        c3[b]  = 1;
                        s[b]   = 0;
                        s2[b]  = 0;
                        el1[b] = e[l + 1][b];
                    }
                    for (int i = N - 2; i >= l; --i) {
                        T cr[B], sr[B];
                        SFEM_TANGENT_PROJECTION_LANES {
                            const T ei   = e[i][b];
                            const T g    = c[b] * ei;
                            const T hh   = c[b] * p[b];
                            const T r    = sqrt(p[b] * p[b] + ei * ei);
                            const T rr   = r == T(0) ? T(1) : r;
                            const T sn   = r == T(0) ? T(0) : ei / rr;
                            const T cn   = r == T(0) ? T(1) : p[b] / rr;
                            const T di   = d[i][b];
                            const T e1   = s[b] * r;
                            const T d1   = hh + sn * (cn * g + sn * di);
                            const T pn   = cn * di - sn * g;
                            e[i + 1][b]  = active[b] ? e1 : e[i + 1][b];
                            d[i + 1][b]  = active[b] ? d1 : d[i + 1][b];
                            c3[b]        = c2[b];
                            c2[b]        = c[b];
                            s2[b]        = s[b];
                            s[b]         = sn;
                            c[b]         = cn;
                            p[b]         = pn;
                            sr[b]        = active[b] ? sn : T(0);
                            cr[b]        = active[b] ? cn : T(1);
                        }
                        for (int k = 0; k < N; ++k) SFEM_TANGENT_PROJECTION_LANES {
                                const T h            = V[k * N + i + 1][b];
                                V[k * N + i + 1][b]  = sr[b] * V[k * N + i][b] + cr[b] * h;
                                V[k * N + i][b]      = cr[b] * V[k * N + i][b] - sr[b] * h;
                            }
                    }
                    SFEM_TANGENT_PROJECTION_LANES {
                        const T pn = -s[b] * s2[b] * c3[b] * el1[b] * e[l][b] / dl1[b];
                        e[l][b]    = active[b] ? s[b] * pn : e[l][b];
                        d[l][b]    = active[b] ? c[b] * pn : d[l][b];
                    }
                }
                SFEM_TANGENT_PROJECTION_LANES {
                    d[l][b] += f[b];
                    e[l][b] = 0;
                }
            }
            SFEM_TANGENT_PROJECTION_LANES d[N - 1][b] += f[b];
        }

    }  // namespace tangent_projection

    /// project_abs_symmetric of B matrices at once, A[b * N * N + i * N + j], their eigensystems
    /// solved in lockstep (tangent_projection::symmetric_eigensystem_lanes).
    template <typename T, int N, int B>
    inline void project_abs_symmetric_lanes(T *const A) {
        namespace tp = tangent_projection;
        bool all_definite = true;
        for (int b = 0; b < B && all_definite; ++b) all_definite = tp::positive_definite<T, N>(A + b * N * N);
        if (all_definite) return;
        T    V[N * N][B], d[N][B], scale[B];
        bool finite[B], ok[B];
        for (int b = 0; b < B; ++b) {
            scale[b]  = 0;
            finite[b] = true;
            for (int i = 0; i < N * N; ++i) {
                const T a = tp::abs_(A[b * N * N + i]);
                finite[b] = finite[b] && a < T(INFINITY);
                scale[b]  = a > scale[b] ? a : scale[b];
            }
            const T inv = finite[b] && scale[b] > T(0) ? T(1) / scale[b] : T(0);
            for (int i = 0; i < N * N; ++i) V[i][b] = A[b * N * N + i] * inv;
        }
        tp::symmetric_eigensystem_lanes<T, N, B>(V, d, ok);
        T S[N * N][B];
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < N * N; ++i) S[i][b] = A[b * N * N + i];
        for (int k = 0; k < N; ++k) {
            T w[B];
            SFEM_TANGENT_PROJECTION_LANES w[b] = d[k][b] < T(0) ? T(-2) * d[k][b] * scale[b] : T(0);
            for (int i = 0; i < N; ++i)
                for (int j = i; j < N; ++j) SFEM_TANGENT_PROJECTION_LANES S[i * N + j][b] += w[b] * V[i * N + k][b] * V[j * N + k][b];
        }
        for (int b = 0; b < B; ++b) {
            const bool good = finite[b] && ok[b];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) A[b * N * N + i * N + j] = good ? S[(i <= j ? i * N + j : j * N + i)][b] : T(NAN);
        }
    }

    /// A <- Q |Lambda| Q^T for the symmetric N x N matrix A = Q Lambda Q^T (row major): positive
    /// semidefinite, and A itself where A already is. The absolute value rather than a clamp at
    /// zero keeps the curvature of a direction along which the energy is concave (Chen et al.,
    /// "Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton",
    /// SIGGRAPH 2024). The tangent dP/dF of a quadrature point is projected so; the element matrix,
    /// a weighted sum of B^T (dP/dF) B, is then positive semidefinite too.
    ///
    /// A positive definite A (its Cholesky factorisation exists) is returned as it is. Otherwise
    /// |A| = A - 2 sum_{lambda_k < 0} lambda_k q_k q_k^T, which leaves A's positive part as it was;
    /// the eigensystem is computed on A scaled to entries of order one. Where it fails, or A is not
    /// finite, A becomes NaN, which the solve that reads it reports.
    template <typename T, int N>
    SFEM_TANGENT_PROJECTION_FUNCTION void project_abs_symmetric(T *const A) {
        static_assert(N > 0, "N must be positive");
        if (tangent_projection::positive_definite<T, N>(A)) return;
        const T inf    = T(INFINITY);
        T       scale  = 0;
        bool    finite = true;
        for (int i = 0; i < N * N; ++i) {
            const T a = tangent_projection::abs_(A[i]);
            finite    = finite && a < inf;
            if (a > scale) scale = a;
        }
        if (finite && scale == T(0)) return;
        T V[N * N], d[N];
        if (finite) {
            const T inv_scale = T(1) / scale;
            for (int i = 0; i < N * N; ++i) V[i] = A[i] * inv_scale;
        }
        if (!finite || !tangent_projection::symmetric_eigensystem<T, N>(V, d)) {
            for (int i = 0; i < N * N; ++i) A[i] = T(NAN);
            return;
        }
        for (int k = 0; k < N; ++k) {
            if (!(d[k] < T(0))) continue;
            const T w = T(-2) * d[k] * scale;
            for (int i = 0; i < N; ++i) {
                const T wi = w * V[i * N + k];
                for (int j = i; j < N; ++j) A[i * N + j] += wi * V[j * N + k];
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) A[i * N + j] = A[j * N + i];
        }
    }

}  // namespace sfem

#endif  // SFEM_TANGENT_PROJECTION_HPP
