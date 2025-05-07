#ifndef CU_LAGRANGE_HEX_LAPLACIAN_INLINE_H
#define CU_LAGRANGE_HEX_LAPLACIAN_INLINE_H

#include "sfem_base.h"

template <int N, int Q, typename T>
inline __host__ __device__ void cu_lagrange_hex_triad_interpolate(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B0,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B1,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B2,
        // Coefficients S x S x S
        const T* const SFEM_RESTRICT u,
        // Evaluation Q x Q x Q
        T* const SFEM_RESTRICT out) {
    static const int N2 = N * N;
    static const int Q2 = Q * Q;

    T temp1[Q * N2];
    T temp2[Q2 * N];

    // Interpolate X
    for (int qi = 0; qi < Q; qi++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                const T* const A = &B0[qi * N];
                const T* const X = &u[k * N2 + j * N];

                T acc = A[0] * X[0];
                for (int i = 1; i < N; i++) {
                    acc += A[i] * X[i];
                }

                temp1[qi * N2 + k * N + j] = acc;
            }
        }
    }

    // Interpolate Y
    for (int qj = 0; qj < Q; qj++) {
        for (int qi = 0; qi < Q; qi++) {
            for (int k = 0; k < N; k++) {
                const T* const A = &B1[qj * N];
                const T* const X = &temp1[qi * N2 + k * N];

                T acc = A[0] * X[0];
                for (int j = 1; j < N; j++) {
                    acc += A[j] * X[j];
                }

                temp2[qj * (Q * N) + qi * N + k] = acc;
            }
        }
    }

    // Interpolate Z
    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                const T* const A = &B2[qk * N];
                const T* const X = &temp2[qj * Q * N + qi * N];

                T acc = A[0] * X[0];
                for (int k = 1; k < N; k++) {
                    acc += A[k] * X[k];
                }

                out[qk * Q2 + qj * Q + qi] = acc;
            }
        }
    }
}

template <int N, int Q, typename T>
static inline __device__ __host__ void cu_lagrange_hex_triad_integrate_add(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B0,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B1,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B2,
        // Coefficients Q x Q x Q
        const T* const SFEM_RESTRICT q,
        // Evaluation N x N x N
        T* const SFEM_RESTRICT out) {
    static const int N2 = N * N;
    static const int Q2 = Q * Q;

    T temp1[N * Q2];
    T temp2[N2 * Q];

    for (int i = 0; i < N; i++) {
        for (int qk = 0; qk < Q; qk++) {
            for (int qj = 0; qj < Q; qj++) {
                const T* const A = &B0[i];
                const T* const X = &q[qk * Q2 + qj * Q];

                T acc = A[0] * X[0];
                for (int qi = 1; qi < Q; qi++) {
                    acc += A[qi * N] * X[qi];
                }

                temp1[i * Q2 + qk * Q + qj] = acc;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int qk = 0; qk < Q; qk++) {
                const T* const A = &B1[j];
                const T* const X = &temp1[i * Q2 + qk * Q];

                T acc = A[0] * X[0];
                for (int qj = 1; qj < Q; qj++) {
                    acc += A[qj * N] * X[qj];
                }

                temp2[j * N * Q + i * Q + qk] = acc;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                const T* const A = &B2[k];
                const T* const X = &temp2[j * N * Q + i * Q];

                T acc = A[0] * X[0];
                for (int qk = 1; qk < Q; qk++) {
                    acc += A[qk * N] * X[qk];
                }

                out[k * N2 + j * N + i] += acc;
            }
        }
    }
}

template <int N, int Q, typename T>
inline __host__ __device__ void cu_lagrange_hex_laplacian_apply(const T* const SFEM_RESTRICT S,
                                                                // Shape functions per quad point Q x S
                                                                const T* const SFEM_RESTRICT D,
                                                                // Metric 6
                                                                const T* const SFEM_RESTRICT FFF,
                                                                // Quad-weights  Q
                                                                const T* const SFEM_RESTRICT qw,
                                                                // Coefficients S x S x S
                                                                const T* const SFEM_RESTRICT u,
                                                                // Evaluation Q x Q x Q
                                                                T* const SFEM_RESTRICT out) {
    static const int Q2 = Q * Q;
    static const int Q3 = Q2 * Q;

    T gx[Q3];
    cu_lagrange_hex_triad_interpolate<N, Q, T>(D, S, S, u, gx);

    T gy[Q3];
    cu_lagrange_hex_triad_interpolate<N, Q, T>(S, D, S, u, gy);

    T gz[Q3];
    cu_lagrange_hex_triad_interpolate<N, Q, T>(S, S, D, u, gz);

    for (int qz = 0; qz < Q; qz++) {
        for (int qy = 0; qy < Q; qy++) {
            for (int qx = 0; qx < Q; qx++) {
                const int q = qz * Q2 + qy * Q + qx;

                const T gxq = FFF[0] * gx[q] + FFF[1] * gy[q] + FFF[2] * gz[q];
                const T gyq = FFF[1] * gx[q] + FFF[3] * gy[q] + FFF[4] * gz[q];
                const T gzq = FFF[2] * gx[q] + FFF[4] * gy[q] + FFF[5] * gz[q];

                const T w = qw[qz] * qw[qy] * qw[qx];

                gx[q] = w * gxq;
                gy[q] = w * gyq;
                gz[q] = w * gzq;
            }
        }
    }

    cu_lagrange_hex_triad_integrate_add<N, Q, T>(D, S, S, gx, out);
    cu_lagrange_hex_triad_integrate_add<N, Q, T>(S, D, S, gy, out);
    cu_lagrange_hex_triad_integrate_add<N, Q, T>(S, S, D, gz, out);
}

#endif  // CU_LAGRANGE_HEX_LAPLACIAN_INLINE_H