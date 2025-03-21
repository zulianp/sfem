#ifndef SPECTRAL_HEX_LAPLACIAN_INLINE_HPP
#define SPECTRAL_HEX_LAPLACIAN_INLINE_HPP

#include <cstdio>
#include "sfem_base.h"

template <int N, typename T>
void spectral_hex_laplacian_apply(
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
    static const int N2 = N * N;
    static const int N3 = N2 * N;

    T gx[N3];
    T gy[N3];
    T gz[N3];

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                T acc[3] = {0, 0, 0};

                for (int n = 0; n < N; n++) {
                    acc[0] += D[n + N * i] * u[k * N2 + j * N + n];
                    acc[1] += D[n + N * j] * u[k * N2 + n * N + i];
                    acc[2] += D[n + N * k] * u[n * N2 + j * N + i];
                }

                const T gxq = FFF[0] * acc[0] + FFF[1] * acc[1] + FFF[2] * acc[2];
                const T gyq = FFF[1] * acc[0] + FFF[3] * acc[1] + FFF[4] * acc[2];
                const T gzq = FFF[2] * acc[0] + FFF[4] * acc[1] + FFF[5] * acc[2];
                const T w   = qw[i] * qw[j] * qw[k];

                gx[k * N2 + j * N + i] = gxq * w;
                gy[k * N2 + j * N + i] = gyq * w;
                gz[k * N2 + j * N + i] = gzq * w;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                T acc = 0;
                for (int n = 0; n < N; n++) {
                    acc += D[n * N + i] * gx[k * N2 + j * N + n];
                    acc += D[n * N + j] * gy[k * N2 + n * N + i];
                    acc += D[n * N + k] * gz[n * N2 + j * N + i];
                }

                out[k * N2 + j * N + i] += acc;
            }
        }
    }
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
        spectral_hex_laplacian_apply<N, scalar_t>(D, fff, qw, element_u, element_vector);

        for (int v = 0; v < N3; v++) {
            assert(!isnan(element_vector[v]));
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }

    return SFEM_SUCCESS;
}

#endif  // SPECTRAL_HEX_LAPLACIAN_INLINE_HPP
