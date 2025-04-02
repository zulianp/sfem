#include "spectral_hex_mass.h"

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
void lagrange_hex_mass_apply(const T jacobian_determinant,
                             // Shape functions per quad point Q x N
                             const T* const SFEM_RESTRICT S,
                             // Q
                             const T* const SFEM_RESTRICT qw,
                             // Coefficients N x N x N
                             const T* const SFEM_RESTRICT u,
                             // Evaluation N x N x N
                             T* const SFEM_RESTRICT out) {
    T temp[Q * Q * Q];
    lagrange_hex_interpolate<N, Q, T>(S, u, temp);
    // TODO Use this to test
    // lagrange_hex_triad_interpolate<N, Q, T>(S, S, S, u, temp);

    for (int qk = 0; qk < Q; qk++) {
        for (int qj = 0; qj < Q; qj++) {
            for (int qi = 0; qi < Q; qi++) {
                temp[qk * Q * Q + qj * Q + qi] *= jacobian_determinant;
            }
        }
    }

    lagrange_hex_integrate<N, Q, T>(S, qw, temp, out);
    // lagrange_hex_triad_integrate_add<N, Q, T>(S, S, S, qw, temp, out);
}

template <int order>
int lagrange_hex_mass_apply_tpl(const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t** const SFEM_RESTRICT       elements,
                                geom_t** const SFEM_RESTRICT      points,
                                const real_t* const SFEM_RESTRICT u,
                                real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("lagrange_hex_mass_apply_tpl");

    SFEM_UNUSED(nnodes);

    static const int N  = order + 1;
    static const int N3 = N * N * N;
    static const int Q  = MAX(0, 3 * order + 1);

    scalar_t S[Q * N] = {0};

    int             n_qp{0};
    const scalar_t* qx{nullptr};
    const scalar_t* qw{nullptr};

    switch (order) {
        case 1: {
            n_qp = line_q4_n;
            qx   = line_q4_x;
            qw   = line_q4_w;
            assert(n_qp == Q);
            break;
        }
        case 2: {
            n_qp = line_q7_n;
            qx   = line_q7_x;
            qw   = line_q7_w;
            assert(n_qp == Q);
            break;
        }
        case 4: {
            n_qp = line_q13_n;
            qx   = line_q13_x;
            qw   = line_q13_w;
            assert(n_qp == Q);
            break;
        }
        case 8: {
            n_qp = line_q25_n;
            qx   = line_q25_x;
            qw   = line_q25_w;
            assert(n_qp == Q);
            break;
        }
        default: {
            SFEM_ERROR("lagrange_hex_mass_apply_tpl: Unsupported element order!\n");
            break;
        }
    }

    lagrange_eval<scalar_t>(order, Q, qx, S);

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
        scalar_t      lx[8], ly[8], lz[8];
        scalar_t      adjugate[9];
        scalar_t      jacobian_determinant;

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
        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);
        lagrange_hex_mass_apply<N, Q, scalar_t>(jacobian_determinant, S, qw, element_u, element_vector);

        for (int v = 0; v < N3; v++) {
            assert(!isnan(element_vector[v]));
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }

    return SFEM_SUCCESS;
}

extern "C" int lagrange_hex_mass_apply(const int                         order,
                                       const ptrdiff_t                   nelements,
                                       const ptrdiff_t                   nnodes,
                                       idx_t** const SFEM_RESTRICT       elements,
                                       geom_t** const SFEM_RESTRICT      points,
                                       const real_t* const SFEM_RESTRICT u,
                                       real_t* const SFEM_RESTRICT       values) {
    switch (order) {
        case 2: {
            return lagrange_hex_mass_apply_tpl<2>(nelements, nnodes, elements, points, u, values);
        }
        case 4: {
            return lagrange_hex_mass_apply_tpl<4>(nelements, nnodes, elements, points, u, values);
        }
        case 8: {
            return lagrange_hex_mass_apply_tpl<8>(nelements, nnodes, elements, points, u, values);
        }
        default: {
            SFEM_ERROR("lagrange_hex_mass_apply Unsupported order!");
        }
    }

    return SFEM_FAILURE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int order>
int spectral_hex_mass_apply_tpl(const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t** const SFEM_RESTRICT       elements,
                                geom_t** const SFEM_RESTRICT      points,
                                const real_t* const SFEM_RESTRICT u,
                                real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_mass_apply_tpl");

    SFEM_UNUSED(nnodes);

    static const int N  = order + 1;
    static const int N3 = N * N * N;

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
        idx_t    ev[N3];
        scalar_t lx[8], ly[8], lz[8];
        scalar_t adjugate[9];
        scalar_t jacobian_determinant;

        for (int v = 0; v < N3; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[hex8_corners[v]]];
            ly[v] = y[ev[hex8_corners[v]]];
            lz[v] = z[ev[hex8_corners[v]]];
        }

        // Assume affine here!
        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);
        assert(!isnan(jacobian_determinant));

        for (int v = 0; v < N3; v++) {
#pragma omp atomic update
            values[ev[v]] += jacobian_determinant * u[ev[v]];
        }
    }

    return SFEM_SUCCESS;
}

extern "C" int spectral_hex_mass_apply(const int                         order,
                                       const ptrdiff_t                   nelements,
                                       const ptrdiff_t                   nnodes,
                                       idx_t** const SFEM_RESTRICT       elements,
                                       geom_t** const SFEM_RESTRICT      points,
                                       const real_t* const SFEM_RESTRICT u,
                                       real_t* const SFEM_RESTRICT       values) {
    SFEM_TRACE_SCOPE("spectral_hex_mass_apply");

    int SFEM_USE_GLL = 0;
    SFEM_READ_ENV(SFEM_USE_GLL, atoi);
    if (!SFEM_USE_GLL) {
        return lagrange_hex_mass_apply(order, nelements, nnodes, elements, points, u, values);
    }

    switch (order) {
        case 2: {
            return spectral_hex_mass_apply_tpl<2>(nelements, nnodes, elements, points, u, values);
        }
        case 4: {
            return spectral_hex_mass_apply_tpl<4>(nelements, nnodes, elements, points, u, values);
        }
        case 8: {
            return spectral_hex_mass_apply_tpl<8>(nelements, nnodes, elements, points, u, values);
        }
        case 16: {
            return spectral_hex_mass_apply_tpl<16>(nelements, nnodes, elements, points, u, values);
        }
        default: {
            SFEM_ERROR("lagrange_hex_mass_apply Unsupported order!");
        }
    }

    return SFEM_FAILURE;
}
