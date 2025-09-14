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

#include "hex8_partial_assembly_neohookean_inline.h"

int hex8_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                       nelements,
                                                   idx_t **const SFEM_RESTRICT           elements,
                                                   geom_t **const SFEM_RESTRICT          points,
                                                   const real_t                          mu,
                                                   const real_t                          lambda,
                                                   const ptrdiff_t                       u_stride,
                                                   const real_t *const SFEM_RESTRICT     ux,
                                                   const real_t *const SFEM_RESTRICT     uy,
                                                   const real_t *const SFEM_RESTRICT     uz,
                                                   metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[4];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
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
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

// Apply partially assembled operator
int         hex8_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                      nelements,
                                                                    idx_t **const SFEM_RESTRICT          elements,
                                                                    const compressed_t *const SFEM_RESTRICT   partial_assembly,
                                                                    const scaling_t *const SFEM_RESTRICT scaling,
                                                                    const ptrdiff_t                      h_stride,
                                                                    const real_t *const                  hx,
                                                                    const real_t *const                  hy,
                                                                    const real_t *const                  hz,
                                                                    const ptrdiff_t                      out_stride,
                                                                    real_t *const                        outx,
                                                                    real_t *const                        outy,
                                                                    real_t *const                        outz) {
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
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        // Load and decompress low precision tensor
        const scalar_t s = scaling[i];
        const compressed_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t       S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = s * (scalar_t)(pai[k]);
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        // scalar_t inc_grad[9];
        // hex8_ref_inc_grad(element_hx, element_hy, element_hz, inc_grad);
        // hex8_apply_S_ikmn(S_ikmn, inc_grad, element_outx, element_outy, element_outz);

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
