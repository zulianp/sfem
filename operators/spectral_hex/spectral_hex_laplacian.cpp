#include "spectral_hex_laplacian.h"

#include "sfem_defs.h"
#include "sfem_macros.h"

#include "hex8_laplacian_inline_cpu.h"
#include "hex8_quadrature.h"
#include "line_quadrature.h"
#include "tet4_inline_cpu.h"

#include "sfem_Tracer.hpp"

// 2*3*MAX(N, Q)^4

template <int N, int Q, typename T>
SFEM_INLINE static void spectral_hex_triad_interpolate(
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
void spectral_hex_laplacian_apply(  // Shape functions per quad point Q x S
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
    spectral_hex_triad_interpolate<N, Q, T>(D, S, S, u, gx);

    T gy[Q * Q * Q];
    spectral_hex_triad_interpolate<N, Q, T>(S, D, S, u, gy);

    T gz[Q * Q * Q];
    spectral_hex_triad_interpolate<N, Q, T>(S, S, D, u, gz);

    for (int q = 0; q < Q * Q * Q; q++) {
        const T gxq = FFF[0] * gx[q] + FFF[1] * gy[q] + FFF[2] * gz[q];
        const T gyq = FFF[1] * gx[q] + FFF[3] * gy[q] + FFF[4] * gz[q];
        const T gzq = FFF[2] * gx[q] + FFF[4] * gy[q] + FFF[5] * gz[q];
        gx[q]       = gxq;
        gy[q]       = gyq;
        gz[q]       = gzq;
    }

#ifndef NDEBUG
    printf("gx) %g %g %g %g, %g %g %g %g\n", gx[0], gx[1], gx[2], gx[3], gx[4], gx[5], gx[6], gx[7]);
    printf("gy) %g %g %g %g, %g %g %g %g\n", gy[0], gy[1], gy[2], gy[3], gy[4], gy[5], gy[6], gy[7]);
    printf("gz) %g %g %g %g, %g %g %g %g\n", gz[0], gz[1], gz[2], gz[3], gz[4], gz[5], gz[6], gz[7]);

    T expected[8] = {0};
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
                            acc += D[ii] * S[jj] * S[kk] * gy[q_kji];
                            acc += D[ii] * S[jj] * S[kk] * gz[q_kji];

                            // S * D * S
                            acc += S[ii] * D[jj] * S[kk] * gx[q_kji];
                            acc += S[ii] * D[jj] * S[kk] * gy[q_kji];
                            acc += S[ii] * D[jj] * S[kk] * gz[q_kji];

                            // S * S * D
                            acc += S[ii] * S[jj] * D[kk] * gx[q_kji];
                            acc += S[ii] * S[jj] * D[kk] * gy[q_kji];
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
    spectral_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, gx, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, gx, out);

    spectral_hex_triad_integrate_add<N, Q, T>(D, S, S, qw, gy, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, gy, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, gy, out);

    spectral_hex_triad_integrate_add<N, Q, T>(D, S, S, qw, gz, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, gz, out);
    spectral_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, gz, out);

#ifndef NDEBUG
    printf("expected) %g %g %g %g, %g %g %g %g\n",
           expected[0],
           expected[1],
           expected[2],
           expected[3],
           expected[4],
           expected[5],
           expected[6],
           expected[7]);

    printf("actual) %g %g %g %g, %g %g %g %g\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
#endif
}

template void spectral_hex_laplacian_apply<2, 2, scalar_t>(const scalar_t* const SFEM_RESTRICT,
                                                           const scalar_t* const SFEM_RESTRICT,
                                                           const scalar_t* const SFEM_RESTRICT,
                                                           const scalar_t* const SFEM_RESTRICT,
                                                           const scalar_t* const SFEM_RESTRICT,
                                                           scalar_t* const       SFEM_RESTRICT);

// template void spectral_hex_laplacian_apply<9, 9, scalar_t>(const scalar_t* const SFEM_RESTRICT,
//                                                            const scalar_t* const SFEM_RESTRICT,
//                                                            const scalar_t* const SFEM_RESTRICT,
//                                                            const scalar_t* const SFEM_RESTRICT,
//                                                            const scalar_t* const SFEM_RESTRICT,
//                                                            scalar_t* const       SFEM_RESTRICT);

// template void spectral_hex_laplacian_apply<2, 2, vec_t>(const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         vec_t* const       SFEM_RESTRICT);

// template void spectral_hex_laplacian_apply<9, 9, vec_t>(const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         const vec_t* const SFEM_RESTRICT,
//                                                         vec_t* const       SFEM_RESTRICT);

extern "C" int spectral_hex_laplacian_apply(const int                         order,
                                            const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   nnodes,
                                            idx_t** const SFEM_RESTRICT       elements,
                                            geom_t** const SFEM_RESTRICT      points,
                                            const real_t* const SFEM_RESTRICT u,
                                            real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_laplacian_apply");

    SFEM_UNUSED(nnodes);

    assert(order == 1);

    scalar_t S[2 * 2];
    scalar_t D[2 * 2];

    // scalar_t qx[2] = {0, 1};
    // scalar_t qw[2]         = {0.5, 0.5};

    // scalar_t qx[2] = {0, 1};
    // scalar_t qw[2]         = {0.5, 0.5};

    static const scalar_t qx[line_q2_n] = {0.2113248654, 0.7886751346};
    static const scalar_t qw[line_q2_n] = {1. / 2, 1. / 2};

    for (int q = 0; q < 2; q++) {
        scalar_t x   = qx[q];
        S[q * 2 + 0] = (1 - x);
        S[q * 2 + 1] = x;

        D[q * 2 + 0] = -1;
        D[q * 2 + 1] = 1;
    }

    const geom_t* const x = points[0];
    const geom_t* const y = points[1];
    const geom_t* const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[8];
        accumulator_t element_vector[8] = {0};
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        element_u[0] = u[ev[0]];
        element_u[1] = u[ev[1]];
        element_u[3] = u[ev[2]];
        element_u[2] = u[ev[3]];
        element_u[4] = u[ev[4]];
        element_u[5] = u[ev[5]];
        element_u[7] = u[ev[6]];
        element_u[6] = u[ev[7]];

        const scalar_t lx[8] = {x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};
        const scalar_t ly[8] = {y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};
        const scalar_t lz[8] = {z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        // Assume affine here!
        hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, fff);

#if 1
        spectral_hex_laplacian_apply<2, 2, scalar_t>(S, D, fff, qw, element_u, element_vector);

#pragma omp atomic update
        values[ev[0]] += element_vector[0];
#pragma omp atomic update
        values[ev[1]] += element_vector[1];
#pragma omp atomic update
        values[ev[2]] += element_vector[3];
#pragma omp atomic update
        values[ev[3]] += element_vector[2];
#pragma omp atomic update
        values[ev[4]] += element_vector[4];
#pragma omp atomic update
        values[ev[5]] += element_vector[5];
#pragma omp atomic update
        values[ev[6]] += element_vector[7];
#pragma omp atomic update
        values[ev[7]] += element_vector[6];

#else
        hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
#endif

#ifndef NDEBUG
        // #if 0
        int perm[8] = {0, 1, 3, 2, 4, 5, 7, 6};

        for (int v = 0; v < 8; v++) {
            element_u[v] = v < 4;
        }

        accumulator_t actual[8] = {0};
        spectral_hex_laplacian_apply<2, 2, scalar_t>(S, D, fff, qw, element_u, actual);

        accumulator_t expected[8] = {0};
        hex8_laplacian_apply_fff_integral(fff, element_u, expected);

        int bugged = 0;
        for (int i = 0; i < 8; i++) {
            scalar_t diff = expected[i] - actual[perm[i]];
            if (fabs(diff) > 1e-8) {
                printf("%d) %g - %g = %g (%g)\n", i, expected[i], actual[perm[i]], diff, expected[i] / actual[perm[i]]);
                bugged++;
            }
        }

        if (bugged) {
            printf("Element %ld is bugged!\n", i);
        }

        assert(!bugged);
#endif
    }

    return SFEM_SUCCESS;
}