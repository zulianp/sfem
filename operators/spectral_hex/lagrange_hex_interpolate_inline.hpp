#ifndef LAGRANGE_HEX_INTERPOLATE_INLINE_HPP
#define LAGRANGE_HEX_INTERPOLATE_INLINE_HPP
#include "sfem_base.h"

template <int N, int Q, typename T>
void lagrange_hex_interpolate(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT S,
        // Coefficients S x S x S
        const T* const SFEM_RESTRICT u,
        // Evaluation Q x Q x Q
        T* const SFEM_RESTRICT out) {
    static const int N2   = N * N;
    static const int N3   = N2 * N;
    static const int Q2   = Q * Q;
    static const int Q3   = Q2 * Q;
    static const int SIZE = MAX(Q3, N3);

    T temp1[SIZE];
    T temp2[SIZE];

    for (int qi = 0; qi < Q; qi++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                const T* const S0 = &S[qi * N];
                const T* const u0 = &u[k * N2 + j * N];

                T acc = S0[0] * u0[0];
                for (int i = 1; i < N; i++) {
                    // Sum over dimension 0
                    acc += S0[i] * u0[i];
                }

                temp1[qi * N2 + k * N + j] = acc;
            }
        }
    }

    for (int qj = 0; qj < Q; qj++) {
        for (int qi = 0; qi < Q; qi++) {
            for (int k = 0; k < N; k++) {
                const T* const S0 = &S[qj * N];
                const T* const u0 = &temp1[qi * N2 + k * N];

                T acc = S[0] * u0[0];
                for (int j = 1; j < N; j++) {
                    // Sum over dimension 1
                    acc += S0[j] * u0[j];
                }

                temp2[qj * Q * N + qi * N + k] = acc;
            }
        }
    }

    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                const T* const S0 = &S[qk * N];
                const T* const u0 = &temp2[qj * Q * N + qi * N];

                T acc = S[0] * u0[0];
                for (int k = 1; k < N; k++) {
                    // Sum over dimension 2
                    acc += S0[k] * u0[k];
                }

                out[qk * Q2 + qj * Q + qi] = acc;
            }
        }
    }
}

template <int N, int Q, typename T>
void lagrange_hex_integrate(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT S,
        // Weights Q
        const T* const SFEM_RESTRICT qw,
        // Coefficients S x S x S
        const T* const SFEM_RESTRICT q,
        // Evaluation Q x Q x Q
        T* const out) {
    static const int N2   = N * N;
    static const int N3   = N2 * N;
    static const int Q2   = Q * Q;
    static const int Q3   = Q2 * Q;
    static const int SIZE = MAX(Q3, N3);

    T temp1[N * Q2];
    T temp2[N2 * Q];

    for (int i = 0; i < N; i++) {
        for (int qk = 0; qk < Q; qk++) {
            for (int qj = 0; qj < Q; qj++) {
                const T* const S0  = &S[0 * N + i];
                const T* const q0  = &q[qk * Q2 + qj * Q + 0];
                T              acc = S0[0] * q0[0] * qw[0];
                for (int qi = 1; qi < Q; qi++) {
                    // Sum over dimension 0
                    acc += S0[qi * N] * q0[qi] * qw[qi];
                }

                temp1[i * Q2 + qk * Q + qj] = acc;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int qk = 0; qk < Q; qk++) {
                const T* const S0  = &S[j];
                const T* const t0  = &temp1[i * Q2 + qk * Q];
                T              acc = S0[0] * t0[0] * qw[0];

                for (int qj = 1; qj < Q; qj++) {
                    // Sum over dimension 1
                    acc += S0[qj * N] * t0[qj] * qw[qj];
                }

                temp2[j * N * Q + i * Q + qk] = acc;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                const T* const S0 = &S[k];
                const T* const t0 = &temp2[j * N * Q + i * Q];

                T acc = S0[0] * t0[0] * qw[0];
                for (int qk = 1; qk < Q; qk++) {
                    // Sum over dimension 2
                    acc += S0[qk * N] * t0[qk] * qw[qk];
                }

                out[k * N2 + j * N + i] = acc;
            }
        }
    }
}

#endif  // LAGRANGE_HEX_INTERPOLATE_INLINE_HPP