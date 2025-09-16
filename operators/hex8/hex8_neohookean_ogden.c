#include "hex8_neohookean_ogden.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sfem_macros.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "hex8_inline_cpu.h"
#include "line_quadrature.h"

#include "hex8_partial_assembly_neohookean_inline.h"

static SFEM_INLINE void hex8_neohookean_grad(const scalar_t *const SFEM_RESTRICT adjugate,
                                             const scalar_t                      jacobian_determinant,
                                             const scalar_t                      qx,
                                             const scalar_t                      qy,
                                             const scalar_t                      qz,
                                             const scalar_t                      qw,
                                             const scalar_t                      mu,
                                             const scalar_t                      lmbda,
                                             const scalar_t *const SFEM_RESTRICT dispx,
                                             const scalar_t *const SFEM_RESTRICT dispy,
                                             const scalar_t *const SFEM_RESTRICT dispz,
                                             scalar_t *const SFEM_RESTRICT       gx,
                                             scalar_t *const SFEM_RESTRICT       gy,
                                             scalar_t *const SFEM_RESTRICT       gz) {
    scalar_t F[9];
    {
        // mundane ops: 267 divs: 1 sqrts: 0
        // total ops: 275
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = qy * qz;
        const scalar_t x2 = 1 - qz;
        const scalar_t x3 = qy * x2;
        const scalar_t x4 = 1 - qy;
        const scalar_t x5 = qz * x4;
        const scalar_t x6 = x2 * x4;
        const scalar_t x7 = dispx[0] * x6 - dispx[1] * x6 - dispx[2] * x3 + dispx[3] * x3 + dispx[4] * x5 - dispx[5] * x5 -
                            dispx[6] * x1 + dispx[7] * x1;
        const scalar_t x8  = qx * qz;
        const scalar_t x9  = qx * x2;
        const scalar_t x10 = 1 - qx;
        const scalar_t x11 = qz * x10;
        const scalar_t x12 = x10 * x2;
        const scalar_t x13 = dispx[0] * x12 + dispx[1] * x9 - dispx[2] * x9 - dispx[3] * x12 + dispx[4] * x11 + dispx[5] * x8 -
                             dispx[6] * x8 - dispx[7] * x11;
        const scalar_t x14 = qx * qy;
        const scalar_t x15 = qx * x4;
        const scalar_t x16 = qy * x10;
        const scalar_t x17 = x10 * x4;
        const scalar_t x18 = dispx[0] * x17 + dispx[1] * x15 + dispx[2] * x14 + dispx[3] * x16 - dispx[4] * x17 - dispx[5] * x15 -
                             dispx[6] * x14 - dispx[7] * x16;
        const scalar_t x19 = dispy[0] * x6 - dispy[1] * x6 - dispy[2] * x3 + dispy[3] * x3 + dispy[4] * x5 - dispy[5] * x5 -
                             dispy[6] * x1 + dispy[7] * x1;
        const scalar_t x20 = dispy[0] * x12 + dispy[1] * x9 - dispy[2] * x9 - dispy[3] * x12 + dispy[4] * x11 + dispy[5] * x8 -
                             dispy[6] * x8 - dispy[7] * x11;
        const scalar_t x21 = dispy[0] * x17 + dispy[1] * x15 + dispy[2] * x14 + dispy[3] * x16 - dispy[4] * x17 - dispy[5] * x15 -
                             dispy[6] * x14 - dispy[7] * x16;
        const scalar_t x22 = dispz[0] * x6 - dispz[1] * x6 - dispz[2] * x3 + dispz[3] * x3 + dispz[4] * x5 - dispz[5] * x5 -
                             dispz[6] * x1 + dispz[7] * x1;
        const scalar_t x23 = dispz[0] * x12 + dispz[1] * x9 - dispz[2] * x9 - dispz[3] * x12 + dispz[4] * x11 + dispz[5] * x8 -
                             dispz[6] * x8 - dispz[7] * x11;
        const scalar_t x24 = dispz[0] * x17 + dispz[1] * x15 + dispz[2] * x14 + dispz[3] * x16 - dispz[4] * x17 - dispz[5] * x15 -
                             dispz[6] * x14 - dispz[7] * x16;
        F[0] = -adjugate[0] * x0 * x7 - adjugate[3] * x0 * x13 - adjugate[6] * x0 * x18 + 1;
        F[1] = -x0 * (adjugate[1] * x7 + adjugate[4] * x13 + adjugate[7] * x18);
        F[2] = -x0 * (adjugate[2] * x7 + adjugate[5] * x13 + adjugate[8] * x18);
        F[3] = -x0 * (adjugate[0] * x19 + adjugate[3] * x20 + adjugate[6] * x21);
        F[4] = -adjugate[1] * x0 * x19 - adjugate[4] * x0 * x20 - adjugate[7] * x0 * x21 + 1;
        F[5] = -x0 * (adjugate[2] * x19 + adjugate[5] * x20 + adjugate[8] * x21);
        F[6] = -x0 * (adjugate[0] * x22 + adjugate[3] * x23 + adjugate[6] * x24);
        F[7] = -x0 * (adjugate[1] * x22 + adjugate[4] * x23 + adjugate[7] * x24);
        F[8] = -adjugate[2] * x0 * x22 - adjugate[5] * x0 * x23 - adjugate[8] * x0 * x24 + 1;
    }

    // mundane ops: 111 divs: 1 sqrts: 0
    // total ops: 119
    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qy - 1;
    const scalar_t x2  = F[4] * F[8];
    const scalar_t x3  = F[5] * F[7];
    const scalar_t x4  = x2 - x3;
    const scalar_t x5  = F[1] * F[5];
    const scalar_t x6  = F[2] * F[7];
    const scalar_t x7  = F[1] * F[8];
    const scalar_t x8  = F[2] * F[4];
    const scalar_t x9  = F[0] * x2 - F[0] * x3 + F[3] * x6 - F[3] * x7 + F[6] * x5 - F[6] * x8;
    const scalar_t x10 = 1.0 / x9;
    const scalar_t x11 = mu * x10;
    const scalar_t x12 = lmbda * x10 * log(x9);
    const scalar_t x13 = F[0] * mu - x11 * x4 + x12 * x4;
    const scalar_t x14 = -x6 + x7;
    const scalar_t x15 = F[3] * mu + x11 * x14 - x12 * x14;
    const scalar_t x16 = x5 - x8;
    const scalar_t x17 = F[6] * mu - x11 * x16 + x12 * x16;
    const scalar_t x18 = adjugate[2] * x13 + adjugate[5] * x15 + adjugate[8] * x17;
    const scalar_t x19 = x1 * x18;
    const scalar_t x20 = x0 * x19;
    const scalar_t x21 = qz - 1;
    const scalar_t x22 = adjugate[1] * x13 + adjugate[4] * x15 + adjugate[7] * x17;
    const scalar_t x23 = x21 * x22;
    const scalar_t x24 = x0 * x23;
    const scalar_t x25 = adjugate[0] * x13 + adjugate[3] * x15 + adjugate[6] * x17;
    const scalar_t x26 = x21 * x25;
    const scalar_t x27 = x1 * x26;
    const scalar_t x28 = -qw * (x20 + x24 + x27);
    const scalar_t x29 = qx * x19;
    const scalar_t x30 = qx * x23;
    const scalar_t x31 = qw * (x27 + x29 + x30);
    const scalar_t x32 = qy * x18;
    const scalar_t x33 = qx * x32;
    const scalar_t x34 = qy * x26;
    const scalar_t x35 = -qw * (x30 + x33 + x34);
    const scalar_t x36 = x0 * x32;
    const scalar_t x37 = qw * (x24 + x34 + x36);
    const scalar_t x38 = qz * x22;
    const scalar_t x39 = x0 * x38;
    const scalar_t x40 = qz * x25;
    const scalar_t x41 = x1 * x40;
    const scalar_t x42 = qw * (x20 + x39 + x41);
    const scalar_t x43 = qx * x38;
    const scalar_t x44 = -qw * (x29 + x41 + x43);
    const scalar_t x45 = qy * x40;
    const scalar_t x46 = qw * (x33 + x43 + x45);
    const scalar_t x47 = -qw * (x36 + x39 + x45);
    gx[0] += x28;
    gx[1] += x31;
    gx[2] += x35;
    gx[3] += x37;
    gx[4] += x42;
    gx[5] += x44;
    gx[6] += x46;
    gx[7] += x47;
    gy[0] += x28;
    gy[1] += x31;
    gy[2] += x35;
    gy[3] += x37;
    gy[4] += x42;
    gy[5] += x44;
    gy[6] += x46;
    gy[7] += x47;
    gz[0] += x28;
    gz[1] += x31;
    gz[2] += x35;
    gz[3] += x37;
    gz[4] += x42;
    gz[5] += x44;
    gz[6] += x46;
    gz[7] += x47;
}

int hex8_neohookean_ogden_gradient(const ptrdiff_t              nelements,
                                   const ptrdiff_t              stride,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t                 mu,
                                   const real_t                 lambda,
                                   const ptrdiff_t              u_stride,
                                   const real_t *const          ux,
                                   const real_t *const          uy,
                                   const real_t *const          uz,
                                   const ptrdiff_t              out_stride,
                                   real_t *const                outx,
                                   real_t *const                outy,
                                   real_t *const                outz) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t element_dispx[8];
        scalar_t element_dispy[8];
        scalar_t element_dispz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_dispx[v]    = ux[idx];
            element_dispy[v]    = uy[idx];
            element_dispz[v]    = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_neohookean_grad(jacobian_adjugate,
                                         jacobian_determinant,
                                         qx[kx],
                                         qx[ky],
                                         qx[kz],
                                         qw[kx],
                                         mu,
                                         lambda,
                                         element_dispx,
                                         element_dispy,
                                         element_dispz,
                                         element_outx,
                                         element_outy,
                                         element_outz);
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(element_outx[edof_i] == element_outx[edof_i]);
            assert(element_outy[edof_i] == element_outy[edof_i]);
            assert(element_outz[edof_i] == element_outz[edof_i]);

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                      nelements,
                                                   const ptrdiff_t                      stride,
                                                   idx_t **const SFEM_RESTRICT          elements,
                                                   geom_t **const SFEM_RESTRICT         points,
                                                   const real_t                         mu,
                                                   const real_t                         lambda,
                                                   const ptrdiff_t                      u_stride,
                                                   const real_t *const SFEM_RESTRICT    ux,
                                                   const real_t *const SFEM_RESTRICT    uy,
                                                   const real_t *const SFEM_RESTRICT    uz,
                                                   metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        static const scalar_t samplex = 0.5, sampley = 0.5, samplez = 0.5;
        hex8_adjugate_and_det(lx, ly, lz, samplex, sampley, samplez, jacobian_adjugate, &jacobian_determinant);

        // Sample at the centroid
        scalar_t F[9] = {0};
        hex8_F(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, element_ux, element_uy, element_uz, F);
        scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
        hex8_S_ikmn_neohookean(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, F, mu, lambda, 1, S_ikmn);

        metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            assert(S_ikmn[k] == S_ikmn[k]);
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                            nelements,
                                                 const ptrdiff_t                            stride,
                                                 idx_t **const SFEM_RESTRICT                elements,
                                                 const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                 const ptrdiff_t                            h_stride,
                                                 const real_t *const                        hx,
                                                 const real_t *const                        hy,
                                                 const real_t *const                        hz,
                                                 const ptrdiff_t                            out_stride,
                                                 real_t *const                              outx,
                                                 real_t *const                              outy,
                                                 real_t *const                              outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        const metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                     S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = pai[k];
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, element_outx, element_outy, element_outz);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(element_outx[edof_i] == element_outx[edof_i]);
            assert(element_outy[edof_i] == element_outy[edof_i]);
            assert(element_outz[edof_i] == element_outz[edof_i]);

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

// Apply partially assembled operator
int hex8_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                         nelements,
                                                            const ptrdiff_t                         stride,
                                                            idx_t **const SFEM_RESTRICT             elements,
                                                            const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                            const scaling_t *const SFEM_RESTRICT    scaling,
                                                            const ptrdiff_t                         h_stride,
                                                            const real_t *const                     hx,
                                                            const real_t *const                     hy,
                                                            const real_t *const                     hz,
                                                            const ptrdiff_t                         out_stride,
                                                            real_t *const                           outx,
                                                            real_t *const                           outy,
                                                            real_t *const                           outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t element_outx[8] = {0};
        accumulator_t element_outy[8] = {0};
        accumulator_t element_outz[8] = {0};

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        // Load and decompress low precision tensor
        const scalar_t            s   = scaling[i];
        const compressed_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                  S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = s * (scalar_t)(pai[k]);
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, element_outx, element_outy, element_outz);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(element_outx[edof_i] == element_outx[edof_i]);
            assert(element_outy[edof_i] == element_outy[edof_i]);
            assert(element_outz[edof_i] == element_outz[edof_i]);

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}
