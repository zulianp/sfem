#ifndef LAGRANGE_HEX_LAPLACIAN_INLINE_HPP
#define LAGRANGE_HEX_LAPLACIAN_INLINE_HPP

#include "sfem_base.h"
#include <cstdio>


#include "sfem_Tracer.hpp"
// #define DEBUG_SUMFACT

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
SFEM_INLINE static void lagrange_hex_triad_integrate_add(
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

    lagrange_hex_triad_integrate_add<N, Q, T>(D, S, S, qw, gx, out);
    lagrange_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, gy, out);
    lagrange_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, gz, out);

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

    exit(0);
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

#endif  // LAGRANGE_HEX_LAPLACIAN_INLINE_HPP
