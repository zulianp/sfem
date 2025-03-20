#include "spectral_hex_laplacian.h"

#include "sfem_defs.h"
#include "sfem_macros.h"

#include "hex8_laplacian_inline_cpu.h"
#include "hex8_quadrature.h"
#include "lagrange.hpp"
#include "lagrange_legendre_gauss_lobatto.hpp"
#include "line_quadrature.h"
#include "line_quadrature_gauss_lobatto.h"
#include "sfem_Tracer.hpp"
#include "sshex8.h"
#include "tet4_inline_cpu.h"

#include <cstdio>

// 2*3*MAX(N, Q)^4

template <int N, int Q, typename T>
SFEM_INLINE static void lagrange_hex_triad_interpolate(
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
    //
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
                T acc = B0[qi * N + 0] * u[k * N2 + j * N + 0];
                for (int i = 1; i < N; i++) {
                    // Sum over dimension 0
                    acc += B0[qi * N + i] * u[k * N2 + j * N + i];
                }

                temp1[qi * N2 + k * N + j] = acc;
            }
        }
    }

    for (int qj = 0; qj < Q; qj++) {
        for (int qi = 0; qi < Q; qi++) {
            for (int k = 0; k < N; k++) {
                T acc = B1[qj * N + 0] * temp1[qi * N2 + k * N + 0];
                for (int j = 1; j < N; j++) {
                    // Sum over dimension 1
                    acc += B1[qj * N + j] * temp1[qi * N2 + k * N + j];
                }

                temp2[qj * Q * N + qi * N + k] = acc;
            }
        }
    }

    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                T acc = B2[qk * N + 0] * temp2[qj * Q * N + qi * N + 0];
                for (int k = 1; k < N; k++) {
                    // Sum over dimension 2
                    acc += B2[qk * N + k] * temp2[qj * Q * N + qi * N + k];
                }

                out[qk * Q * Q + qj * Q + qi] = acc;
            }
        }
    }
}

template <int N, int Q, typename T>
SFEM_INLINE static void spectral_hex_triad_integrate_add(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B0,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B1,
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT B2,
        // Weights Q
        const T* const SFEM_RESTRICT qw,
        // Coefficients Q x Q x Q
        const T* const SFEM_RESTRICT q,
        // Evaluation N x N x N
        T* const SFEM_RESTRICT out) {
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
                T acc = B0[0 * N + i] * q[qk * Q2 + qj * Q + 0] * qw[0] * qw[qj] * qw[qk];
                for (int qi = 1; qi < Q; qi++) {
                    // Sum over dimension 0
                    acc += B0[qi * N + i] * q[qk * Q2 + qj * Q + qi] * qw[qi] * qw[qj] * qw[qk];
                }

                temp1[i * Q2 + qk * Q + qj] = acc;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int qk = 0; qk < Q; qk++) {
                T acc = B1[0 * N + j] * temp1[i * Q2 + qk * Q + 0];

                for (int qj = 1; qj < Q; qj++) {
                    // Sum over dimension 1
                    acc += B1[qj * N + j] * temp1[i * Q2 + qk * Q + qj];
                }

                temp2[j * N * Q + i * Q + qk] = acc;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                T acc = B2[0 * N + k] * temp2[j * N * Q + i * Q + 0];
                for (int qk = 1; qk < Q; qk++) {
                    // Sum over dimension 2
                    acc += B2[qk * N + k] * temp2[j * N * Q + i * Q + qk];
                }

                out[k * N2 + j * N + i] += acc;
            }
        }
    }
}

template <int N, int Q, typename T>
static void reference_lapl(const T* const SFEM_RESTRICT S,
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
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int test = k * N * N + j * N + i;

                T integr = 0;

                for (int qk = 0; qk < Q; qk++) {
                    for (int qj = 0; qj < Q; qj++) {
                        for (int qi = 0; qi < Q; qi++) {
                            T interp[3] = {0, 0, 0};

                            for (int l = 0; l < N; l++) {
                                for (int m = 0; m < N; m++) {
                                    for (int n = 0; n < N; n++) {
                                        int trial = l * N * N + m * N + n;

                                        T r0 = D[qi * N + n] * S[qj * N + m] * S[qk * N + l];
                                        T r1 = S[qi * N + n] * D[qj * N + m] * S[qk * N + l];
                                        T r2 = S[qi * N + n] * S[qj * N + m] * D[qk * N + l];

                                        interp[0] += r0 * u[trial];
                                        interp[1] += r1 * u[trial];
                                        interp[2] += r2 * u[trial];
                                    }
                                }
                            }

                            T FFFxr0 = FFF[0] * interp[0] + FFF[1] * interp[1] + FFF[2] * interp[2];
                            T FFFxr1 = FFF[1] * interp[0] + FFF[3] * interp[1] + FFF[4] * interp[2];
                            T FFFxr2 = FFF[2] * interp[0] + FFF[4] * interp[1] + FFF[5] * interp[2];

                            T l0 = D[qi * N + i] * S[qj * N + j] * S[qk * N + k];
                            T l1 = S[qi * N + i] * D[qj * N + j] * S[qk * N + k];
                            T l2 = S[qi * N + i] * S[qj * N + j] * D[qk * N + k];

                            integr += (FFFxr0 * l0 + FFFxr1 * l1 + FFFxr2 * l2) * qw[qi] * qw[qj] * qw[qk];
                        }
                    }
                }

                out[test] = integr;
            }
        }
    }
}

static void classical_lapl_qp(const scalar_t* const               fff,
                              const scalar_t                      qx,
                              const scalar_t                      qy,
                              const scalar_t                      qz,
                              const scalar_t                      qw,
                              const scalar_t* const SFEM_RESTRICT u,
                              scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t trial_operand[3];

    {
        const scalar_t x0  = 4 * qx;
        const scalar_t x1  = x0 - 3;
        const scalar_t x2  = qz - 1;
        const scalar_t x3  = qy - 1;
        const scalar_t x4  = x2 * x3;
        const scalar_t x5  = x1 * x4;
        const scalar_t x6  = qy * qz;
        const scalar_t x7  = 16 * x6;
        const scalar_t x8  = 2 * qx - 1;
        const scalar_t x9  = x4 * x8;
        const scalar_t x10 = 64 * u[13];
        const scalar_t x11 = x0 - 1;
        const scalar_t x12 = x11 * x4;
        const scalar_t x13 = 4 * qy;
        const scalar_t x14 = x1 * x13;
        const scalar_t x15 = 2 * qy - 1;
        const scalar_t x16 = qz * x2;
        const scalar_t x17 = x15 * x16;
        const scalar_t x18 = x15 * x8;
        const scalar_t x19 = x18 * x2;
        const scalar_t x20 = x11 * x13;
        const scalar_t x21 = 2 * qz - 1;
        const scalar_t x22 = qz * x3;
        const scalar_t x23 = x21 * x22;
        const scalar_t x24 = x21 * x8;
        const scalar_t x25 = x24 * x3;
        const scalar_t x26 = x15 * x21;
        const scalar_t x27 = x1 * x26;
        const scalar_t x28 = x11 * x26;
        const scalar_t x29 = x26 * x8;
        const scalar_t x30 = x13 * x29;
        const scalar_t x31 = x21 * x4;
        const scalar_t x32 = qy * x2;
        const scalar_t x33 = 16 * u[4];
        const scalar_t x34 = 16 * u[10];
        const scalar_t x35 = 4 * qz;
        const scalar_t x36 = x15 * x35;
        const scalar_t x37 = x29 * x35;
        const scalar_t x38 = -qz * u[25] * x30 + u[0] * x27 * x4 - u[11] * x12 * x36 + u[12] * x5 * x7 + u[14] * x12 * x7 -
                             u[15] * x14 * x17 + u[16] * x19 * x7 - u[17] * x17 * x20 + u[18] * x22 * x27 - u[19] * x3 * x37 -
                             4 * u[1] * x26 * x9 + u[20] * x22 * x28 - u[21] * x14 * x23 + u[22] * x25 * x7 - u[23] * x20 * x23 +
                             u[24] * x27 * x6 + u[26] * x28 * x6 + u[2] * x28 * x4 - u[3] * x14 * x31 - u[5] * x20 * x31 +
                             u[6] * x27 * x32 - u[7] * x2 * x30 + u[8] * x28 * x32 - u[9] * x36 * x5 - x10 * x6 * x9 +
                             x19 * x22 * x34 + x25 * x32 * x33;
        const scalar_t x39 = x13 - 3;
        const scalar_t x40 = qx * qz;
        const scalar_t x41 = qx - 1;
        const scalar_t x42 = x2 * x41;
        const scalar_t x43 = x40 * x42;
        const scalar_t x44 = x13 - 1;
        const scalar_t x45 = 16 * u[16];
        const scalar_t x46 = x0 * x39;
        const scalar_t x47 = x16 * x8;
        const scalar_t x48 = 16 * x40;
        const scalar_t x49 = x0 * x44;
        const scalar_t x50 = qz * x41;
        const scalar_t x51 = x21 * x50;
        const scalar_t x52 = x26 * x41;
        const scalar_t x53 = x24 * x39;
        const scalar_t x54 = x24 * x44;
        const scalar_t x55 = x0 * x29;
        const scalar_t x56 = x21 * x42;
        const scalar_t x57 = qx * x2;
        const scalar_t x58 = 16 * u[12];
        const scalar_t x59 = x35 * x42 * x8;
        const scalar_t x60 = 4 * x29;
        const scalar_t x61 = -qz * u[23] * x55 + u[0] * x42 * x53 - u[11] * x46 * x47 + u[14] * x19 * x48 - u[15] * x44 * x59 -
                             u[17] * x47 * x49 + u[18] * x50 * x53 - u[19] * x46 * x51 - u[1] * x46 * x56 + u[20] * x40 * x53 -
                             u[21] * x37 * x41 + u[22] * x48 * x52 + u[24] * x50 * x54 - u[25] * x49 * x51 + u[26] * x40 * x54 +
                             u[2] * x53 * x57 - u[3] * x42 * x60 - u[5] * x2 * x55 + u[6] * x42 * x54 - u[7] * x49 * x56 +
                             u[8] * x54 * x57 - u[9] * x39 * x59 - x10 * x15 * x43 + x19 * x50 * x58 + x33 * x52 * x57 +
                             x34 * x39 * x43 + x43 * x44 * x45;
        const scalar_t x62 = qx * qy;
        const scalar_t x63 = x3 * x41;
        const scalar_t x64 = x62 * x63;
        const scalar_t x65 = x35 - 1;
        const scalar_t x66 = x35 - 3;
        const scalar_t x67 = x0 * x65;
        const scalar_t x68 = qy * x3 * x8;
        const scalar_t x69 = qy * x41;
        const scalar_t x70 = x15 * x69;
        const scalar_t x71 = x0 * x66;
        const scalar_t x72 = x18 * x65;
        const scalar_t x73 = x18 * x66;
        const scalar_t x74 = qx * x3;
        const scalar_t x75 = x15 * x63;
        const scalar_t x76 = x13 * x63 * x8;
        const scalar_t x77 = -qy * u[17] * x55 + u[0] * x63 * x73 - u[11] * x3 * x55 + 16 * u[14] * x25 * x62 -
                             u[15] * x30 * x41 + u[18] * x63 * x72 - u[19] * x67 * x75 - u[1] * x71 * x75 + u[20] * x72 * x74 -
                             u[21] * x65 * x76 + 16 * u[22] * x64 * x65 - u[23] * x67 * x68 + u[24] * x69 * x72 -
                             u[25] * x67 * x70 + u[26] * x62 * x72 + u[2] * x73 * x74 - u[3] * x66 * x76 - u[5] * x68 * x71 +
                             u[6] * x69 * x73 - u[7] * x70 * x71 + u[8] * x62 * x73 - u[9] * x60 * x63 - x10 * x21 * x64 +
                             x25 * x58 * x69 + x33 * x64 * x66 + x34 * x52 * x74 + x45 * x52 * x62;
        trial_operand[0] = qw * (fff[0] * x38 + fff[1] * x61 + fff[2] * x77);
        trial_operand[1] = qw * (fff[1] * x38 + fff[3] * x61 + fff[4] * x77);
        trial_operand[2] = qw * (fff[2] * x38 + fff[4] * x61 + fff[5] * x77);
    }

    const scalar_t x0  = 4 * qx;
    const scalar_t x1  = x0 - 3;
    const scalar_t x2  = qy - 1;
    const scalar_t x3  = 2 * qy - 1;
    const scalar_t x4  = x2 * x3;
    const scalar_t x5  = qz - 1;
    const scalar_t x6  = 2 * qz - 1;
    const scalar_t x7  = x5 * x6;
    const scalar_t x8  = trial_operand[0] * x7;
    const scalar_t x9  = x4 * x8;
    const scalar_t x10 = qx - 1;
    const scalar_t x11 = 2 * qx - 1;
    const scalar_t x12 = x10 * x11;
    const scalar_t x13 = 4 * qy;
    const scalar_t x14 = x13 - 3;
    const scalar_t x15 = trial_operand[1] * x7;
    const scalar_t x16 = x14 * x15;
    const scalar_t x17 = 4 * qz;
    const scalar_t x18 = x17 - 3;
    const scalar_t x19 = trial_operand[2] * x18;
    const scalar_t x20 = x19 * x4;
    const scalar_t x21 = qx * x10;
    const scalar_t x22 = qx * x11;
    const scalar_t x23 = x0 - 1;
    const scalar_t x24 = qy * x2;
    const scalar_t x25 = x24 * x8;
    const scalar_t x26 = x19 * x24;
    const scalar_t x27 = x15 * x3;
    const scalar_t x28 = -qx * x10;
    const scalar_t x29 = -x2;
    const scalar_t x30 = qy * x8;
    const scalar_t x31 = x11 * x30;
    const scalar_t x32 = x3 * x30;
    const scalar_t x33 = qy * x12;
    const scalar_t x34 = x19 * x3;
    const scalar_t x35 = x13 - 1;
    const scalar_t x36 = x15 * x35;
    const scalar_t x37 = qy * x34;
    const scalar_t x38 = qz * x5;
    const scalar_t x39 = trial_operand[0] * x4;
    const scalar_t x40 = x38 * x39;
    const scalar_t x41 = trial_operand[1] * x12;
    const scalar_t x42 = x14 * x38;
    const scalar_t x43 = trial_operand[2] * x6;
    const scalar_t x44 = x4 * x43;
    const scalar_t x45 = -qz * x5;
    const scalar_t x46 = x11 * x45;
    const scalar_t x47 = x29 * x43;
    const scalar_t x48 = x3 * x45;
    const scalar_t x49 = qy * x47;
    const scalar_t x50 = trial_operand[1] * x48;
    const scalar_t x51 = qy * trial_operand[0];
    const scalar_t x52 = x46 * x51;
    const scalar_t x53 = qy * x3;
    const scalar_t x54 = trial_operand[0] * x38 * x53;
    const scalar_t x55 = x35 * x38;
    const scalar_t x56 = x3 * x33;
    const scalar_t x57 = x43 * x53;
    const scalar_t x58 = qz * x6;
    const scalar_t x59 = x39 * x58;
    const scalar_t x60 = x14 * x58;
    const scalar_t x61 = x17 - 1;
    const scalar_t x62 = trial_operand[2] * x61;
    const scalar_t x63 = x4 * x62;
    const scalar_t x64 = trial_operand[1] * x60;
    const scalar_t x65 = x1 * x58;
    const scalar_t x66 = trial_operand[0] * x24;
    const scalar_t x67 = x24 * x62;
    const scalar_t x68 = x3 * x58;
    const scalar_t x69 = trial_operand[1] * x68;
    const scalar_t x70 = x11 * x51;
    const scalar_t x71 = x35 * x58;
    const scalar_t x72 = x53 * x62;
    const scalar_t x73 = trial_operand[1] * x71;
    element_vector[0] += x1 * x9 + x12 * x16 + x12 * x20;
    element_vector[1] += -4 * x11 * x9 - 4 * x16 * x21 - 4 * x20 * x21;
    element_vector[2] += x16 * x22 + x20 * x22 + x23 * x9;
    element_vector[3] += -4 * x1 * x25 - 4 * x12 * x26 - 4 * x12 * x27;
    element_vector[4] += 16 * qx * qy * trial_operand[2] * x10 * x18 * x2 - 16 * x27 * x28 - 16 * x29 * x31;
    element_vector[5] += -4 * x22 * x26 - 4 * x22 * x27 - 4 * x23 * x25;
    element_vector[6] += x1 * x32 + x12 * x36 + x33 * x34;
    element_vector[7] += -4 * x21 * x36 - 4 * x21 * x37 - 4 * x3 * x31;
    element_vector[8] += x22 * x36 + x22 * x37 + x23 * x32;
    element_vector[9] += -4 * x1 * x40 - 4 * x12 * x44 - 4 * x41 * x42;
    element_vector[10] += 16 * qx * qz * trial_operand[1] * x10 * x14 * x5 - 16 * x28 * x44 - 16 * x39 * x46;
    element_vector[11] += -4 * trial_operand[1] * x22 * x42 - 4 * x22 * x44 - 4 * x23 * x40;
    element_vector[12] += 16 * qy * qz * trial_operand[0] * x1 * x2 * x5 - 16 * x33 * x47 - 16 * x41 * x48;
    element_vector[13] += -64 * x28 * x49 - 64 * x28 * x50 - 64 * x29 * x52;
    element_vector[14] += 16 * qy * qz * trial_operand[0] * x2 * x23 * x5 - 16 * x22 * x49 - 16 * x22 * x50;
    element_vector[15] += -4 * x1 * x54 - 4 * x41 * x55 - 4 * x43 * x56;
    element_vector[16] += 16 * qx * qz * trial_operand[1] * x10 * x35 * x5 - 16 * x28 * x57 - 16 * x3 * x52;
    element_vector[17] += -4 * trial_operand[1] * x22 * x55 - 4 * x22 * x57 - 4 * x23 * x54;
    element_vector[18] += x1 * x59 + x12 * x63 + x41 * x60;
    element_vector[19] += -4 * x11 * x59 - 4 * x21 * x63 - 4 * x21 * x64;
    element_vector[20] += x22 * x63 + x22 * x64 + x23 * x59;
    element_vector[21] += -4 * x12 * x67 - 4 * x41 * x68 - 4 * x65 * x66;
    element_vector[22] += 16 * qx * qy * trial_operand[2] * x10 * x2 * x61 - 16 * x28 * x69 - 16 * x29 * x58 * x70;
    element_vector[23] += -4 * x22 * x67 - 4 * x22 * x69 - 4 * x23 * x58 * x66;
    element_vector[24] += x3 * x51 * x65 + x41 * x71 + x56 * x62;
    element_vector[25] += -4 * x21 * x72 - 4 * x21 * x73 - 4 * x68 * x70;
    element_vector[26] += x22 * x72 + x22 * x73 + x23 * x51 * x68;
}

void classical_lapl(const scalar_t* const               fff,
                    const int                           n_qp,
                    const scalar_t* const               qx,
                    const scalar_t* const               qw,
                    const scalar_t* const SFEM_RESTRICT u,
                    scalar_t* const SFEM_RESTRICT       element_vector)

{
    for (int v = 0; v < 27; v++) {
        element_vector[v] = 0;
    }

    for (int k = 0; k < n_qp; k++) {
        for (int j = 0; j < n_qp; j++) {
            for (int i = 0; i < n_qp; i++) {
                classical_lapl_qp(fff, qx[i], qx[j], qx[k], qw[i] * qw[j] * qw[k], u, element_vector);
            }
        }
    }
}

// #define DEBUG_SUMFACT

template <int N, int Q, typename T>
void lagrange_hex_laplacian_apply(  // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT S,
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
    // Interpolate gradient

    T gx[Q * Q * Q];
    lagrange_hex_triad_interpolate<N, Q, T>(D, S, S, u, gx);

    T gy[Q * Q * Q];
    lagrange_hex_triad_interpolate<N, Q, T>(S, D, S, u, gy);

    T gz[Q * Q * Q];
    lagrange_hex_triad_interpolate<N, Q, T>(S, S, D, u, gz);

    for (int q = 0; q < Q * Q * Q; q++) {
        const T gxq = FFF[0] * gx[q] + FFF[1] * gy[q] + FFF[2] * gz[q];
        const T gyq = FFF[1] * gx[q] + FFF[3] * gy[q] + FFF[4] * gz[q];
        const T gzq = FFF[2] * gx[q] + FFF[4] * gy[q] + FFF[5] * gz[q];
        gx[q]       = gxq;
        gy[q]       = gyq;
        gz[q]       = gzq;
    }

#ifdef DEBUG_SUMFACT
    printf("--------------------------------\n");
    printf("gx) ");
    for (int i = 0; i < N * N * N; i++) {
        printf("%g ", gx[i]);
    }
    printf("\n");

    printf("gy) ");
    for (int i = 0; i < N * N * N; i++) {
        printf("%g ", gy[i]);
    }
    printf("\n");

    printf("gz) ");
    for (int i = 0; i < N * N * N; i++) {
        printf("%g ", gz[i]);
    }
    printf("\n");

    T expected[N * N * N] = {0};
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                for (int qk = 0; qk < Q; qk++) {
                    for (int qj = 0; qj < Q; qj++) {
                        for (int qi = 0; qi < Q; qi++) {
                            const int ii = qi * N + i;
                            const int jj = qj * N + j;
                            const int kk = qk * N + k;

                            const int q_kji = qk * Q * Q + qj * Q + qi;

                            T acc = 0;
                            // D * S * S
                            acc += D[ii] * S[jj] * S[kk] * gx[q_kji];

                            // S * D * S
                            acc += S[ii] * D[jj] * S[kk] * gy[q_kji];

                            // S * S * D
                            acc += S[ii] * S[jj] * D[kk] * gz[q_kji];

                            expected[k * N * N + j * N + i] += acc * qw[qk] * qw[qj] * qw[qi];
                        }
                    }
                }
            }
        }
    }
#endif

    spectral_hex_triad_integrate_add<N, Q, T>(D, S, S, qw, gx, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, gy, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, gz, out);

#ifdef DEBUG_SUMFACT

    printf("expected) ");
    for (int i = 0; i < N * N * N; i++) {
        printf("%g ", expected[i]);
    }
    printf("\n");

    printf("actual) ");
    for (int i = 0; i < N * N * N; i++) {
        printf("%g ", out[i]);
    }
    printf("\n");
    printf("--------------------------------\n");

#endif
}

template <int order>
int lagrange_hex_laplacian_apply_tpl(const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   nnodes,
                                     idx_t** const SFEM_RESTRICT       elements,
                                     geom_t** const SFEM_RESTRICT      points,
                                     const real_t* const SFEM_RESTRICT u,
                                     real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("lagrange_hex_laplacian_apply_tpl");

    SFEM_UNUSED(nnodes);

    static const int N  = order + 1;
    static const int N3 = N * N * N;
    static const int Q  = MAX(0, 3 * order);

    scalar_t S[Q * N] = {0};
    scalar_t D[Q * N] = {0};

    int             n_qp{0};
    const scalar_t* qx{nullptr};
    const scalar_t* qw{nullptr};

    switch (order) {
        case 1: {
            n_qp = line_q3_n;
            qx   = line_q3_x;
            qw   = line_q3_w;
            assert(n_qp == Q);
            break;
        }
        case 2: {
            n_qp = line_q6_n;
            qx   = line_q6_x;
            qw   = line_q6_w;
            assert(n_qp == Q);
            break;
        }
        case 4: {
            n_qp = line_q12_n;
            qx   = line_q12_x;
            qw   = line_q12_w;
            assert(n_qp == Q);
            break;
        }
        case 8: {
            n_qp = line_q24_n;
            qx   = line_q24_x;
            qw   = line_q24_w;
            assert(n_qp == Q);
            break;
        }
        case 16: {
            n_qp = line_q48_n;
            qx   = line_q48_x;
            qw   = line_q48_w;
            assert(n_qp == Q);
            break;
        }
        default: {
            SFEM_ERROR("lagrange_hex_laplacian_apply_tpl: Unsupported element order!\n");
            break;
        }
    }

    lagrange_eval<scalar_t>(order, Q, qx, S);
    lagrange_diff_eval<scalar_t>(order, Q, qx, D);

    const int nxe = sshex8_nxe(order);
    const int txe = sshex8_txe(order);

    const int hex8_corners[8] = {// Bottom
                                 sshex8_lidx(order, 0, 0, 0),
                                 sshex8_lidx(order, order, 0, 0),
                                 sshex8_lidx(order, order, order, 0),
                                 sshex8_lidx(order, 0, order, 0),

                                 // Top
                                 sshex8_lidx(order, 0, 0, order),
                                 sshex8_lidx(order, order, 0, order),
                                 sshex8_lidx(order, order, order, order),
                                 sshex8_lidx(order, 0, order, order)};

    const geom_t* const x = points[0];
    const geom_t* const y = points[1];
    const geom_t* const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[N3];
        accumulator_t element_vector[N3] = {0};
        scalar_t      element_u[N3];
        scalar_t      fff[6];
        scalar_t      lx[8], ly[8], lz[8];

        for (int v = 0; v < N3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < N3; ++v) {
            element_u[v] = u[ev[v]];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[hex8_corners[v]]];
            ly[v] = y[ev[hex8_corners[v]]];
            lz[v] = z[ev[hex8_corners[v]]];
        }

        // Assume affine here!
        hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, fff);
        lagrange_hex_laplacian_apply<N, Q, scalar_t>(S, D, fff, qw, element_u, element_vector);

        for (int v = 0; v < N3; v++) {
            assert(!isnan(element_vector[v]));
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }

    return SFEM_SUCCESS;
}

template <int order>
int spectral_hex_laplacian_apply_tpl(const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   nnodes,
                                     idx_t** const SFEM_RESTRICT       elements,
                                     geom_t** const SFEM_RESTRICT      points,
                                     const real_t* const SFEM_RESTRICT u,
                                     real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_laplacian_apply_tpl");

    SFEM_UNUSED(nnodes);

    static const int N  = order + 1;
    static const int N3 = N * N * N;
    static const int Q  = N;

    scalar_t S[Q * N] = {0};
    scalar_t D[Q * N] = {0};

    int             n_qp{0};
    const scalar_t* qx{nullptr};
    const scalar_t* qw{nullptr};

    switch (order) {
        case 1: {
            n_qp = line_GL_q2_n;
            qx   = line_GL_q2_x;
            qw   = line_GL_q2_w;
            assert(n_qp == Q);
            break;
        }
        case 2: {
            n_qp = line_GL_q3_n;
            qx   = line_GL_q3_x;
            qw   = line_GL_q3_w;
            assert(n_qp == Q);
            break;
        }
        case 4: {
            n_qp = line_GL_q5_n;
            qx   = line_GL_q5_x;
            qw   = line_GL_q5_w;
            assert(n_qp == Q);
            break;
        }
        case 8: {
            n_qp = line_GL_q9_n;
            qx   = line_GL_q9_x;
            qw   = line_GL_q9_w;
            assert(n_qp == Q);
            break;
        }
        case 16: {
            n_qp = line_GL_q17_n;
            qx   = line_GL_q17_x;
            qw   = line_GL_q17_w;
            assert(n_qp == Q);
            break;
        }
        default: {
            SFEM_ERROR("lagrange_hex_laplacian_apply_tpl: Unsupported element order!\n");
            break;
        }
    }

    lagrange_GLL_eval<scalar_t>(order, Q, qx, S);
    lagrange_GLL_diff_eval<scalar_t>(order, Q, qx, D);

    const int nxe = sshex8_nxe(order);
    const int txe = sshex8_txe(order);

    const int hex8_corners[8] = {// Bottom
                                 sshex8_lidx(order, 0, 0, 0),
                                 sshex8_lidx(order, order, 0, 0),
                                 sshex8_lidx(order, order, order, 0),
                                 sshex8_lidx(order, 0, order, 0),

                                 // Top
                                 sshex8_lidx(order, 0, 0, order),
                                 sshex8_lidx(order, order, 0, order),
                                 sshex8_lidx(order, order, order, order),
                                 sshex8_lidx(order, 0, order, order)};

    const geom_t* const x = points[0];
    const geom_t* const y = points[1];
    const geom_t* const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[N3];
        accumulator_t element_vector[N3] = {0};
        scalar_t      element_u[N3];
        scalar_t      fff[6];
        scalar_t      lx[8], ly[8], lz[8];

        for (int v = 0; v < N3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < N3; ++v) {
            element_u[v] = u[ev[v]];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[hex8_corners[v]]];
            ly[v] = y[ev[hex8_corners[v]]];
            lz[v] = z[ev[hex8_corners[v]]];
        }

        // Assume affine here!
        hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, fff);
        lagrange_hex_laplacian_apply<N, Q, scalar_t>(S, D, fff, qw, element_u, element_vector);

        for (int v = 0; v < N3; v++) {
            assert(!isnan(element_vector[v]));
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }

    return SFEM_SUCCESS;
}

extern "C" int spectral_hex_laplacian_apply(const int                         order,
                                            const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   nnodes,
                                            idx_t** const SFEM_RESTRICT       elements,
                                            geom_t** const SFEM_RESTRICT      points,
                                            const real_t* const SFEM_RESTRICT u,
                                            real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_laplacian_apply");

    int SFEM_USE_GLL = 0;
    SFEM_READ_ENV(SFEM_USE_GLL, atoi);

    if (SFEM_USE_GLL) {
        switch (order) {
            case 2: {
                return spectral_hex_laplacian_apply_tpl<2>(nelements, nnodes, elements, points, u, values);
            }
            case 4: {
                return spectral_hex_laplacian_apply_tpl<4>(nelements, nnodes, elements, points, u, values);
            }
            case 8: {
                return spectral_hex_laplacian_apply_tpl<8>(nelements, nnodes, elements, points, u, values);
            }
            case 16: {
                return spectral_hex_laplacian_apply_tpl<16>(nelements, nnodes, elements, points, u, values);
            }
            default: {
                SFEM_ERROR("spectral_hex_laplacian_apply Unsupported order!");
            }
        }

    } else {
        switch (order) {
            case 2: {
                return lagrange_hex_laplacian_apply_tpl<2>(nelements, nnodes, elements, points, u, values);
            }
            case 4: {
                return lagrange_hex_laplacian_apply_tpl<4>(nelements, nnodes, elements, points, u, values);
            }
            case 8: {
                return lagrange_hex_laplacian_apply_tpl<8>(nelements, nnodes, elements, points, u, values);
            }
            // case 16: {
            //     return lagrange_hex_laplacian_apply_tpl<16>(nelements, nnodes, elements, points, u, values);
            // }
            default: {
                SFEM_ERROR("spectral_hex_laplacian_apply Unsupported order!");
            }
        }
    }

    return SFEM_SUCCESS;
}
