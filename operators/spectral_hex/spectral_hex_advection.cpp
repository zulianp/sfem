#include "spectral_hex_advection.h"

#include "hex8_laplacian_inline_cpu.h"
#include "lagrange.hpp"
#include "line_quadrature.h"
#include "sfem_base.h"
#include "sfem_macros.h"
#include "sshex8.h"

#include "lagrange_hex_interpolate_inline.hpp"
#include "lagrange_hex_laplacian_inline.hpp"
#include "sfem_Tracer.hpp"

#include <cassert>
#include <cstdio>

template <int N, int Q, typename T>
static SFEM_INLINE void lagrange_hex_advection_apply(
        // Shape functions per quad point Q x N
        const T* const SFEM_RESTRICT S,
        // Shape functions per quad point Q x N
        const T* const SFEM_RESTRICT D,
        // Metric 9
        const T* const SFEM_RESTRICT adjugate,
        // Quad-weights  Q
        const T* const SFEM_RESTRICT qw,
        // Evaluation Q x Q x Q
        const T* const SFEM_RESTRICT vx,
        // Evaluation Q x Q x Q
        const T* const SFEM_RESTRICT vy,
        // Evaluation Q x Q x Q
        const T* const SFEM_RESTRICT vz,
        // Coefficients N x N x N
        const T* const SFEM_RESTRICT c,
        // Evaluation Q x Q x Q
        T* const SFEM_RESTRICT out) {
    T cux[Q * Q * Q];
    T cuy[Q * Q * Q];
    T cuz[Q * Q * Q];

    // Use cux as temp for cq
    lagrange_hex_interpolate<N, Q, T>(S, c, cux);
    // lagrange_hex_triad_interpolate<N, Q, T>(S, S, S, c, cux);

    for (int q = 0; q < Q * Q * Q; q++) {
        const T cq = cux[q];

        // (Adjugate^T)^T
        const T cuxq = adjugate[0] * vx[q] + adjugate[1] * vy[q] + adjugate[2] * vz[q];
        const T cuyq = adjugate[3] * vx[q] + adjugate[4] * vy[q] + adjugate[5] * vz[q];
        const T cuzq = adjugate[6] * vx[q] + adjugate[7] * vy[q] + adjugate[8] * vz[q];

        cux[q] = cq * cuxq;
        cuy[q] = cq * cuyq;
        cuz[q] = cq * cuzq;
    }

    lagrange_hex_triad_integrate_add<N, Q, T>(D, S, S, qw, cux, out);
    lagrange_hex_triad_integrate_add<N, Q, T>(S, D, S, qw, cuy, out);
    lagrange_hex_triad_integrate_add<N, Q, T>(S, S, D, qw, cuz, out);
}

template <int order>
int lagrange_hex_advection_apply_tpl(const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   nnodes,
                                     idx_t** const SFEM_RESTRICT       elements,
                                     geom_t** const SFEM_RESTRICT      points,
                                     const real_t* const SFEM_RESTRICT vx,
                                     const real_t* const SFEM_RESTRICT vy,
                                     const real_t* const SFEM_RESTRICT vz,
                                     const real_t* const SFEM_RESTRICT c,
                                     real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("lagrange_hex_advection_apply_tpl");

    SFEM_UNUSED(nnodes);

    static const int N  = order + 1;
    static const int N3 = N * N * N;
    static const int Q  = MAX(0, 3 * order);
    static const int Q3 = Q * Q * Q;

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
        scalar_t element_c[N3];
        scalar_t temp[N3];

        accumulator_t element_vector[N3] = {0};

        scalar_t element_qvz[Q3];
        scalar_t element_qvy[Q3];
        scalar_t element_qvx[Q3];
        
        scalar_t adjugate[9];
        scalar_t lx[8], ly[8], lz[8];

        for (int v = 0; v < N3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < N3; ++v) {
            element_c[v] = c[ev[v]];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[hex8_corners[v]]];
            ly[v] = y[ev[hex8_corners[v]]];
            lz[v] = z[ev[hex8_corners[v]]];
        }

        // Assume affine here!
        hex8_adjugate(lx, ly, lz, 0.5, 0.5, 0.5, adjugate);

        // FIXME allow different order for vector field
        for (int v = 0; v < N3; ++v) {
            temp[v] = vx[ev[v]];
        }
        lagrange_hex_interpolate<N, Q, scalar_t>(S, temp, element_qvx);

        for (int v = 0; v < N3; ++v) {
            temp[v] = vy[ev[v]];
        }
        lagrange_hex_interpolate<N, Q, scalar_t>(S, temp, element_qvy);

        for (int v = 0; v < N3; ++v) {
            temp[v] = vz[ev[v]];
        }
        lagrange_hex_interpolate<N, Q, scalar_t>(S, temp, element_qvz);

        lagrange_hex_advection_apply<N, Q, scalar_t>(
                S, D, adjugate, qw, element_qvx, element_qvy, element_qvz, element_c, element_vector);

        for (int v = 0; v < N3; v++) {
            assert(!isnan(element_vector[v]));
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }

    return SFEM_SUCCESS;
}

template <int order>
int spectral_hex_advection_apply_tpl(const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   nnodes,
                                     idx_t** const SFEM_RESTRICT       elements,
                                     geom_t** const SFEM_RESTRICT      points,
                                     const real_t* const SFEM_RESTRICT vx,
                                     const real_t* const SFEM_RESTRICT vy,
                                     const real_t* const SFEM_RESTRICT vz,
                                     const real_t* const SFEM_RESTRICT c,
                                     real_t* const SFEM_RESTRICT       values) {
    SFEM_ERROR("IMPLEMENT ME!\n");
    return SFEM_SUCCESS;
}

extern "C" int spectral_hex_advection_apply(const int                         order,
                                            const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   nnodes,
                                            idx_t** const SFEM_RESTRICT       elements,
                                            geom_t** const SFEM_RESTRICT      points,
                                            const real_t* const SFEM_RESTRICT vx,
                                            const real_t* const SFEM_RESTRICT vy,
                                            const real_t* const SFEM_RESTRICT vz,
                                            const real_t* const SFEM_RESTRICT c,
                                            real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_advection_apply");

    int SFEM_USE_GLL = 0;
    SFEM_READ_ENV(SFEM_USE_GLL, atoi);

    if (SFEM_USE_GLL) {
        switch (order) {
            case 2: {
                return spectral_hex_advection_apply_tpl<2>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            case 4: {
                return spectral_hex_advection_apply_tpl<4>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            case 8: {
                return spectral_hex_advection_apply_tpl<8>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            case 16: {
                return spectral_hex_advection_apply_tpl<16>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            default: {
                SFEM_ERROR("spectral_hex_advection_apply: Unsupported order!");
            }
        }

    } else {
        switch (order) {
            case 2: {
                return lagrange_hex_advection_apply_tpl<2>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            case 4: {
                return lagrange_hex_advection_apply_tpl<4>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            case 8: {
                return lagrange_hex_advection_apply_tpl<8>(nelements, nnodes, elements, points, vx, vy, vz, c, values);
            }
            default: {
                SFEM_ERROR("lagrange_hex_advection_apply_tpl: Unsupported order!");
            }
        }
    }

    return SFEM_SUCCESS;
}
