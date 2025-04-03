#include "spectral_hex_symmetric_interior_penalty.h"

#include "hex8_laplacian_inline_cpu.h"
#include "lagrange.hpp"
#include "line_quadrature.h"
#include "sfem_base.h"
#include "sfem_macros.h"
#include "sshex8.h"

#include "lagrange_hex_interpolate_inline.hpp"
#include "lagrange_hex_laplacian_inline.hpp"
#include "sfem_Tracer.hpp"
#include "sshex_side_code.h"

#include <cassert>
#include <cstdio>

template <int N, int Qx, int Qy, int Qz, typename T>
static SFEM_INLINE void lagrange_hex_interpolate_face(
        // Qx x N
        const scalar_t* const SFEM_RESTRICT Bx,
        // Qy x N
        const scalar_t* const SFEM_RESTRICT By,
        // Qz x N
        const scalar_t* const SFEM_RESTRICT Bz,
        // N x N x N
        const scalar_t* const SFEM_RESTRICT u,
        // Qx x Qy x Qz
        scalar_t* const SFEM_RESTRICT out) {
    static const int N2   = N * N;
    static const int N3   = N2 * N;
    static const int SIZE = MAX(Qx * Qy * Qz, N3);

    T temp1[Qx * N * N];
    T temp2[Qy * Qx * N];

    for (int qi = 0; qi < Qx; qi++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                T acc = Bx[qi * N + 0] * u[k * N2 + j * N + 0];
                for (int i = 1; i < N; i++) {
                    // Sum over dimension 0
                    acc += Bx[qi * N + i] * u[k * N2 + j * N + i];
                }

                temp1[qi * N2 + k * N + j] = acc;
            }
        }
    }

    for (int qj = 0; qj < Qy; qj++) {
        for (int qi = 0; qi < Qx; qi++) {
            for (int k = 0; k < N; k++) {
                T acc = By[qj * N + 0] * temp1[qi * N2 + k * N + 0];
                for (int j = 1; j < N; j++) {
                    // Sum over dimension 1
                    acc += By[qj * N + j] * temp1[qi * N2 + k * N + j];
                }

                temp2[qj * Qx * N + qi * N + k] = acc;
            }
        }
    }

    for (int qk = 0; qk < Qz; qk++) {
        for (int qj = 0; qj < Qy; qj++) {
            for (int qi = 0; qi < Qx; qi++) {
                T acc = Bz[qk * N + 0] * temp2[qj * Qx * N + qi * N + 0];
                for (int k = 1; k < N; k++) {
                    // Sum over dimension 2
                    acc += Bz[qk * N + k] * temp2[qj * Qx * N + qi * N + k];
                }

                out[qk * Qy * Qx + qj * Qx + qi] = acc;
            }
        }
    }
}

template <int N, int Qx, int Qy, int Qz, typename T>
void lagrange_hex_integrate_face_add(
        // Shape functions per quad point Qx x S
        const T* const SFEM_RESTRICT Bx,
        // Shape functions per quad point Qy x S
        const T* const SFEM_RESTRICT By,
        // Shape functions per quad point Qz x S
        const T* const SFEM_RESTRICT Bz,
        // Weights Q
        // Coefficients S x S x S
        const T* const SFEM_RESTRICT q,
        // Evaluation N x N x N
        T* const out) {
    static const int N2 = N * N;
    static const int N3 = N2 * N;

    T temp1[N * Qz * Qy];
    T temp2[N2 * Qz];

    for (int i = 0; i < N; i++) {
        for (int qk = 0; qk < Qz; qk++) {
            for (int qj = 0; qj < Qy; qj++) {
                const T* const S0  = &Bx[0 * N + i];
                const T* const q0  = &q[qk * Qy * Qx + qj * Qx + 0];
                T              acc = S0[0] * q0[0];
                for (int qi = 1; qi < Qx; qi++) {
                    // Sum over dimension 0
                    acc += S0[qi * N] * q0[qi];
                }

                temp1[i * Qz * Qy + qk * Qy + qj] = acc;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int qk = 0; qk < Qz; qk++) {
                const T* const S0  = &By[j];
                const T* const t0  = &temp1[i * Qz * Qy + qk * Qy];
                T              acc = S0[0] * t0[0];

                for (int qj = 1; qj < Qy; qj++) {
                    // Sum over dimension 1
                    acc += S0[qj * N] * t0[qj];
                }

                temp2[j * N * Qz + i * Qz + qk] = acc;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                const T* const S0 = &Bz[k];
                const T* const t0 = &temp2[j * N * Qz + i * Qz];

                T acc = S0[0] * t0[0];
                for (int qk = 1; qk < Qz; qk++) {
                    // Sum over dimension 2
                    acc += S0[qk * N] * t0[qk];
                }

                out[k * N2 + j * N + i] += acc;
            }
        }
    }
}

template <int N, int Nx, int Ny, int Nz, int Qx, int Qy, int Qz, typename T>
static SFEM_INLINE void lagrange_hex_integrate_adjoint_consistency_face_add(
        // Shape functions per quad point Qx x S
        const T* const SFEM_RESTRICT Sx,
        const T* const SFEM_RESTRICT Dx,
        // Shape functions per quad point Qy x S
        const T* const SFEM_RESTRICT Sy,
        const T* const SFEM_RESTRICT Dy,
        // Shape functions per quad point Qz x S
        const T* const SFEM_RESTRICT Sz,
        const T* const SFEM_RESTRICT Dz,
        // Weights Q
        const T* const SFEM_RESTRICT qwx,
        const T* const SFEM_RESTRICT qwy,
        const T* const SFEM_RESTRICT qwz,
        // 3
        const T* const SFEM_RESTRICT JinvXdS,
        // Coefficients Q x Q x Q
        const T* const SFEM_RESTRICT q,
        // Evaluation N x N x N
        T* const out) {

    scalar_t temp[Nx * Ny * Nz] = {0};
    lagrange_hex_integrate_face_add<N, Qx, Qy, Qz, T>(Dx, Sy, Sz, q, temp);

    for(int zi = 0; zi < Nz; zi++) {
        for(int yi = 0; yi < Ny; yi++) {
            for(int xi = 0; xi < Nx; xi++) {
                const int idx = zi * Ny * Nx + yi * Nx + zi;
                out[idx] += temp[idx] * JinvXdS[0];
            }
        }
    }

    for(int k = 0; k < Nx * Ny * Nz; k++) temp[k] = 0;

    lagrange_hex_integrate_face_add<N, Qx, Qy, Qz, T>(Sx, Dy, Sz, q, temp);

    for(int zi = 0; zi < Nz; zi++) {
        for(int yi = 0; yi < Ny; yi++) {
            for(int xi = 0; xi < Nx; xi++) {
                const int idx = zi * Ny * Nx + yi * Nx + zi;
                out[idx] += temp[idx] * JinvXdS[1];
            }
        }
    }

    for(int k = 0; k < Nx * Ny * Nz; k++) temp[k] = 0;

    lagrange_hex_integrate_face_add<N, Qx, Qy, Qz, T>(Sx, Sy, Dz, q, temp);

    for(int zi = 0; zi < Nz; zi++) {
        for(int yi = 0; yi < Ny; yi++) {
            for(int xi = 0; xi < Nx; xi++) {
                const int idx = zi * Qy * Qx + yi * Qx + zi;
                out[idx] += temp[idx] * JinvXdS[2];
            }
        }
    }
}

// template <int N, int Qx, int Qy, int Qz, typename T>
// static SFEM_INLINE void lagrange_hex_sip(const scalar_t u, const scalar_t gu, ) {
//     assert(false);
// }

template <int order>
static void dg_hex_add_to_neighbor(const uint8_t* const SFEM_RESTRICT  begin,
                                   const uint8_t* const SFEM_RESTRICT  end,
                                   const scalar_t* const SFEM_RESTRICT side_buffer,
                                   real_t* const SFEM_RESTRICT         neigh_buffer) {
    static const int N  = order + 1;
    static const int N2 = N * N;

    int offset = 0;
    for (int zi = begin[2] * N; zi < end[2] * N; zi += (end[2] - begin[2])) {
        for (int yi = begin[1] * N; yi < end[1] * N; yi += (end[1] - begin[1])) {
            for (int xi = begin[0] * N; xi < end[0] * N; xi += (end[0] - begin[0])) {
#pragma omp atomic update
                neigh_buffer[zi * N2 + yi * N + xi] += side_buffer[offset++];
            }
        }
    }
}

template <int order>
int lagrange_hex_symmetric_interior_penalty_apply_tpl(const ptrdiff_t                          nelements,
                                                      const ptrdiff_t                          nnodes,
                                                      idx_t** const SFEM_RESTRICT              elements,
                                                      const element_idx_t* const SFEM_RESTRICT adj_table,
                                                      const sshex_side_code_t* const           side_code,
                                                      geom_t** const SFEM_RESTRICT             points,
                                                      const real_t                             tau,
                                                      const real_t* const SFEM_RESTRICT        u,
                                                      real_t* const SFEM_RESTRICT              values) {
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

    static const int N  = order + 1;
    static const int N2 = N * N;
    static const int N3 = N * N * N;
    static const int Q  = MAX(0, 3 * order);
    static const int Q2 = Q * Q;
    static const int Q3 = Q2 * Q;

    scalar_t S[Q * N] = {0};
    scalar_t D[Q * N] = {0};

    const scalar_t* qw{nullptr};

    {
        int             n_qp{0};
        const scalar_t* qx{nullptr};

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
    }

    // Evaluate gradients at boundary limits {0, 1}
    scalar_t Sf[2 * N];
    scalar_t Df[2 * N];
    {
        scalar_t qxf[2] = {0, 1};

        lagrange_eval<scalar_t>(order, 2, qxf, Sf);
        lagrange_diff_eval<scalar_t>(order, 2, qxf, Df);
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t side_u[Q * Q];

        scalar_t side_gx[Q * Q];
        scalar_t side_gy[Q * Q];
        scalar_t side_gz[Q * Q];

        scalar_t      element_u[N3];
        accumulator_t element_vector[N3] = {0};

        scalar_t adjugate[9];
        scalar_t jacobian_determinant;
        scalar_t lx[8], ly[8], lz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        // Copy element block
        for (int v = 0; v < N3; ++v) {
            element_u[v] = u[i * N3 + v];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[hex8_corners[v]]];
            ly[v] = y[ev[hex8_corners[v]]];
            lz[v] = z[ev[hex8_corners[v]]];
        }

        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);

        // Loop over non-boundary faces
        // For each face
        // 1) interpolate u^- and grad u^- (transform using Adj(J)^T/det(J))
        // 2) Compute surface normal
        // 3) Compute v n
        // 4) Compute u n
        // 5) Compute grad v
        // 6) Compute composite terms
        // - <v n, {{ grad u }}>       (Primal consistency term, grad u and grad v also have the diffusivity K)
        // - <grad v, [[u]]/2>         (Adjoint consistency term)
        // + <v n, tau [[u]]>          (Penalty term tau=alpha*order^2/h, 2 <= alpha <= 10)
        // REARRANGED:
        // - <v, n^T {{ grad u }}>      (Primal consistency term, grad u and grad v also have the diffusivity K)
        // - <grad v, [[u]] / 2>        (Adjoint consistency term)
        // + <v, n^T tau [[u]]>         (Penalty term tau=alpha*order^2/h, 2 <= alpha <= 10)

        // 7)
        // We compute only the "-" part
        // - <v n^-, grad u^->   / 2
        // - <grad v, n^- * u^-> / 2
        // + <v n^-, tau n^- u^->
        // REARRANGED:
        // - <v     , n^- . grad u^->   / 2
        // - <grad v, n^- * u^->        / 2
        // + <v     , n^- . n^- u^->    tau
        // SIMPLIFIED: remove ^- for notation
        // - <v           , n . grad u>   / 2
        // - <n . grad v, u>              / 2
        // + <v           , u>              tau
        // CHANGE OF VARIABLES:
        // - <v                 , (J^-1 dS) . grad u^->   / 2
        // - <(J^-1 dS) . grad v, u>                      / 2
        // + <v                 , u> || dS ||             tau

        // 8) Accumulate owner
        // 9) Transpose (using side_code) and sign change
        // 10) Accumulate neighbor
        // Pros: less computation, less loads, Cons: atomics required

        // Alternative):
        /// Gather neighbor, interpolate u^+ and grad u^+ ..., transpose
        // Pros: no atomics required, Cons: large number of loads

        // Face 0
        // Normal = (0, -1, 0)
        // Grad_l0n :=
        // Dli * Sf0j * Snk
        // Sli * Df0j * Snk
        // Sli * Sf0j * Dnk

        // https://en.wikipedia.org/wiki/Surface_integral
        const element_idx_t neigh = adj_table[i * 6 + 0];
        if (neigh != SFEM_ELEMENT_IDX_INVALID) {
            lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, Sf, S, element_u, side_u);

            lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(D, Sf, S, element_u, side_gx);
            lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, Df, S, element_u, side_gy);
            lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, Sf, D, element_u, side_gz);

            // Sample the surface element at the midpoint (TODO use sum factorization here too and create full field)
            scalar_t dS[3];
            hex8_surface_element_0(0.5, 0.5, lx, ly, lz, dS);

            // Procedures
            // 1) - (v) ((J^-1 dS) w / 2 . grad u) [((S x S x S) x (Q x Q x Q)) : (Q x Q x Q)]
            // 2) - (u) ((J^-1 dS) w / 2 . grad v) [((Q x Q x Q)) : (Q x Q x Q) x (S x S x S) ]
            // 3) + v (u ||dS|| tau w)

            // Area element
            const scalar_t ds         = sqrt(dS[0] * dS[0] + dS[1] * dS[1] + dS[2] * dS[2]);
            scalar_t       JinvXdS[3] = {
                    (adjugate[0] * dS[0] + adjugate[1] * dS[1] + adjugate[2] * dS[2]) / (2 * jacobian_determinant),
                    (adjugate[3] * dS[0] + adjugate[4] * dS[1] + adjugate[5] * dS[2]) / (2 * jacobian_determinant),
                    (adjugate[6] * dS[0] + adjugate[7] * dS[1] + adjugate[8] * dS[2]) / (2 * jacobian_determinant)};

            scalar_t sip[N2] = {0}, temp[N2];
            for (int zi = 0; zi < Q; zi++) {
                for (int xi = 0; xi < Q; xi++) {
                    const int idx = zi * Q + xi;
                    temp[idx] =
                            (JinvXdS[0] * side_gx[idx] + JinvXdS[1] * side_gy[idx] + JinvXdS[2] * side_gz[idx]) * qw[zi] * qw[xi];
                }
            }

            // 1)
            lagrange_hex_integrate_face_add<N, Q, 1, Q, scalar_t>(S, Sf, S, temp, sip);

            {  // 2)
                scalar_t one = 1;
                lagrange_hex_integrate_adjoint_consistency_face_add<N, N, 1, N, Q, 1, Q, scalar_t>(
                        S, D, Sf, Df, S, D, qw, &one, qw, JinvXdS, u, sip);
            }

            {  // 3)
                for (int zi = 0; zi < Q; zi++) {
                    for (int xi = 0; xi < Q; xi++) {
                        const int idx = zi * Q + xi;
                        side_u[idx] *= (ds * tau * qw[zi] * qw[xi]);
                    }
                }
            }

            lagrange_hex_integrate_face_add<N, Q, 1, Q, scalar_t>(S, Sf, S, side_u, sip);

            // Accumulate on own buffer
            {
                for (int zi = 0; zi < N; zi++) {
                    for (int xi = 0; xi < N; xi++) {
                        int fidx = zi * N + xi;
                        int eidx = zi * N * N + 0 * N + xi;

#pragma omp atomic update
                        values[i * N3 + eidx] += sip[fidx];
                    }
                }
            }

            {  // Accumulate on neigh buffer
                uint8_t begin[3], end[3];
                sshex_coords_from_side_code(side_code[i], 0, begin, end);

                // Change sign
                for (int zi = 0; zi < N; zi++) {
                    for (int xi = 0; xi < N; xi++) {
                        const int idx = zi * Q + xi;
                        sip[idx]      = -sip[idx];
                    }
                }

                // Is the Transpose/Mirroring correct ?
                dg_hex_add_to_neighbor<order>(begin, end, sip, &values[neigh * N3]);
            }
        }

        else {
            // BC?
        }

        // TODO

        // Face 1
        // Normal = (1, 0, 0)
        // Grad_1mn :=
        // Df1i * Smj * Snk
        // Sf1i * Dmj * Snk
        // Sf1i * Smj * Dnk
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(&Sf[1], S, S, element_u, side_u);

        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(&Df[1], S, S, element_u, side_gx);
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(&Sf[1], D, S, element_u, side_gy);
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(&Sf[1], S, D, element_u, side_gz);

        // TODO

        // Face 2
        // Normal = (0, 1, 0)
        // Grad_l1n :=
        // Dli * Sf1j * Snk
        // Sli * Df1j * Snk
        // Sli * Sf1j * Dnk

        lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, &Sf[1], S, element_u, side_u);

        lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(D, &Sf[1], S, element_u, side_gx);
        lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, &Df[1], S, element_u, side_gy);
        lagrange_hex_interpolate_face<N, Q, 1, Q, scalar_t>(S, &Sf[1], D, element_u, side_gz);

        // TODO

        // Face 3
        // Normal = (-1, 0, 0)
        // Grad_0mn :=
        // Df0i * Smj * Snk
        // Sf0i * Dmj * Snk
        // Sf0i * Smj * Dnk
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(Sf, S, S, element_u, side_u);

        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(Df, S, S, element_u, side_gx);
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(Sf, D, S, element_u, side_gy);
        lagrange_hex_interpolate_face<N, 1, Q, Q, scalar_t>(Sf, S, D, element_u, side_gz);

        // TODO

        // Face 4
        // Normal = (0, 0, -1)
        // Grad_lm0 :=
        // Dli * Smj * Sf0k
        // Sli * Dmj * Sf0k
        // Sli * Smj * Df0k

        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, Sf, element_u, side_u);

        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, Sf, element_u, side_gx);
        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, D, Sf, element_u, side_gy);
        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, Df, element_u, side_gz);

        // TODO

        // Face 5
        // Normal = (0, 0, 1)
        // Grad_lm1 :=
        // Dli * Smj * Sf1k
        // Sli * Dmj * Sf1k
        // Sli * Smj * Df1k

        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, &Sf[1], element_u, side_u);

        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, &Sf[1], element_u, side_gx);
        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, D, &Sf[1], element_u, side_gy);
        lagrange_hex_interpolate_face<N, Q, Q, 1, scalar_t>(S, S, &Df[1], element_u, side_gz);

        // TODO
    }

    return SFEM_SUCCESS;
}

template <int order>
int spectral_hex_symmetric_interior_penalty_apply_tpl(const ptrdiff_t                          nelements,
                                                      const ptrdiff_t                          nnodes,
                                                      idx_t** const SFEM_RESTRICT              elements,
                                                      const element_idx_t* const SFEM_RESTRICT adj_table,
                                                      const sshex_side_code_t* const           side_code,
                                                      geom_t** const SFEM_RESTRICT             points,
                                                      const real_t                             tau,
                                                      const real_t* const SFEM_RESTRICT        u,
                                                      real_t* const SFEM_RESTRICT              values) {
    return SFEM_SUCCESS;
}

extern "C" int spectral_hex_symmetric_interior_penalty_apply(const int                                order,
                                                             const ptrdiff_t                          nelements,
                                                             const ptrdiff_t                          nnodes,
                                                             idx_t** const SFEM_RESTRICT              elements,
                                                             const element_idx_t* const SFEM_RESTRICT adj_table,
                                                             const sshex_side_code_t* const           side_code,
                                                             geom_t** const SFEM_RESTRICT             points,
                                                             const real_t                             tau,
                                                             const real_t* const SFEM_RESTRICT        u,
                                                             real_t* const SFEM_RESTRICT              values) {
    SFEM_TRACE_SCOPE("spectral_hex_advection_apply");

    int SFEM_USE_GLL = 0;
    SFEM_READ_ENV(SFEM_USE_GLL, atoi);

    if (SFEM_USE_GLL) {
        switch (order) {
            case 2: {
                return spectral_hex_symmetric_interior_penalty_apply_tpl<2>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            case 4: {
                return spectral_hex_symmetric_interior_penalty_apply_tpl<4>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            case 8: {
                return spectral_hex_symmetric_interior_penalty_apply_tpl<8>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            case 16: {
                return spectral_hex_symmetric_interior_penalty_apply_tpl<16>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            default: {
                SFEM_ERROR("spectral_hex_advection_apply: Unsupported order!");
            }
        }

    } else {
        switch (order) {
            case 2: {
                return lagrange_hex_symmetric_interior_penalty_apply_tpl<2>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            case 4: {
                return lagrange_hex_symmetric_interior_penalty_apply_tpl<4>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            case 8: {
                return lagrange_hex_symmetric_interior_penalty_apply_tpl<8>(
                        nelements, nnodes, elements, adj_table, side_code, points, tau, u, values);
            }
            default: {
                SFEM_ERROR("lagrange_hex_symmetric_interior_penalty_apply_tpl: Unsupported order!");
            }
        }
    }

    return SFEM_FAILURE;
}
