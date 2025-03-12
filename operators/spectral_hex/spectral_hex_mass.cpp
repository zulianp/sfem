#include "sfem_base.h"
#include "sfem_macros.h"

// 2*3*MAX(N, Q)^4

template <int N, int Q, typename T>
void spectral_hex_interpolate(
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
                T acc = S[qi * N + 0] * u[k * N2 + j * N + 0];
                for (int i = 1; i < N; i++) {
                    // Sum over dimension 0
                    acc += S[qi * N + i] * u[k * N2 + j * N + i];
                }

                temp1[qi * N2 + k * N + j] = acc;
            }
        }
    }

    for (int qj = 0; qj < Q; qj++) {
        for (int qi = 0; qi < Q; qi++) {
            for (int k = 0; k < N; k++) {
                T acc = S[qj * N + 0] * temp1[qi * N2 + k * N + 0];
                for (int j = 1; j < N; j++) {
                    // Sum over dimension 1
                    acc += S[qj * N + j] * temp1[qi * N2 + k * N + j];
                }

                temp2[qj * Q * N + qi * N + k] = acc;
            }
        }
    }

    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                T acc = S[qk * N + 0] * temp2[qj * Q * N + qi * N + 0];
                for (int k = 1; k < N; k++) {
                    // Sum over dimension 2
                    acc += S[qk * N + k] * temp2[qj * Q * N + qi * N + k];
                }

                out[qk * Q * Q + qj * Q + qi] = acc;
            }
        }
    }
}

template void spectral_hex_interpolate<2, 2, scalar_t>(const scalar_t* const SFEM_RESTRICT,
                                                       const scalar_t* const SFEM_RESTRICT,
                                                       scalar_t* const       SFEM_RESTRICT);

template <int N, int Q, typename T>
void spectral_hex_integrate(
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

    T temp1[SIZE];
    T temp2[SIZE];

    for (int i = 0; i < N; i++) {
        for (int qk = 0; qk < Q; qk++) {
            for (int qj = 0; qj < Q; qj++) {
                T acc = S[0 * N + i] * q[qk * Q2 + qj * Q + 0] * qw[0];
                for (int qi = 1; qi < Q; qi++) {
                    // Sum over dimension 0
                    acc += S[qi * N + i] * q[qk * Q2 + qj * Q + qi] * qw[qi];
                }

                temp1[i * Q2 + qk * Q + qj] = acc;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int qk = 0; qk < Q; qk++) {
                T acc = S[0 * N + j] * temp1[i * Q2 + qk * Q + 0] * qw[0];

                for (int qj = 1; qj < Q; qj++) {
                    // Sum over dimension 1
                    acc += S[qj * N + j] * temp1[i * Q2 + qk * Q + qj] * qw[qj];
                }

                temp2[j * N * Q + i * Q + qk] = acc;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                T acc = S[0 * N + k] * temp2[j * N * Q + i * Q + 0] * qw[0];
                for (int qk = 1; qk < Q; qk++) {
                    // Sum over dimension 2
                    acc += S[qk * N + k] * temp2[j * N * Q + i * Q + qk] * qw[qk];
                }

                out[k * N2 + j * N + i] = acc;
            }
        }
    }
}

template <int N, int Q, typename T>
void spectral_hex_mass_apply(const T jacobian_determinant,
                             // Shape functions per quad point Q x S
                             const T* const SFEM_RESTRICT S,
                             const T* const SFEM_RESTRICT qw,
                             // Coefficients S x S x S
                             const T* const SFEM_RESTRICT u,
                             // Evaluation Q x Q x Q
                             T* const SFEM_RESTRICT out) {
    spectral_hex_interpolate<N, Q, T>(S, u, out);

    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                out[qk * Q * Q + qj * Q + qi] *= jacobian_determinant;
            }
        }
    }

    spectral_hex_integrate<N, Q, T>(S, qw, out, out);
}

// template void spectral_hex_mass_apply<2, 2, scalar_t>(const scalar_t,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       scalar_t* const       SFEM_RESTRICT);

// template void spectral_hex_mass_apply<3, 3, scalar_t>(const scalar_t,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       scalar_t* const       SFEM_RESTRICT);

// template void spectral_hex_mass_apply<5, 5, scalar_t>(const scalar_t,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       const scalar_t* const SFEM_RESTRICT,
//                                                       scalar_t* const       SFEM_RESTRICT);

template void spectral_hex_mass_apply<9, 9, scalar_t>(const scalar_t,
                                                      const scalar_t* const SFEM_RESTRICT,
                                                      const scalar_t* const SFEM_RESTRICT,
                                                      const scalar_t* const SFEM_RESTRICT,
                                                      scalar_t* const       SFEM_RESTRICT);

// template void spectral_hex_mass_apply<9, 9, vec_t>(const vec_t,
//                                                    const vec_t* const SFEM_RESTRICT,
//                                                    const vec_t* const SFEM_RESTRICT,
//                                                    const vec_t* const SFEM_RESTRICT,
//                                                    vec_t* const       SFEM_RESTRICT);

// extern "C" void spectral_hex_mass_apply(const scalar_t jacobian_determinant,
//                              // Shape functions per quad point Q x S
//                              const scalar_t* const SFEM_RESTRICT S,
//                              const scalar_t* const SFEM_RESTRICT qw,
//                              // Coefficients S x S x S
//                              const scalar_t* const SFEM_RESTRICT u,
//                              // Evaluation Q x Q x Q
//                              scalar_t* const SFEM_RESTRICT out)
