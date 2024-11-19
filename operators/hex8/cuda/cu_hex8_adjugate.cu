#include "cu_hex8_adjugate.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "cu_hex8_inline.hpp"

int cu_hex8_adjugate_allocate(const ptrdiff_t nelements,
                              void **const SFEM_RESTRICT jacobian_adjugate,
                              void **const SFEM_RESTRICT jacobian_determinant) {
    SFEM_CUDA_CHECK(cudaMalloc(jacobian_adjugate, 9 * nelements * sizeof(cu_jacobian_t)));
    SFEM_CUDA_CHECK(cudaMalloc(jacobian_determinant, nelements * sizeof(cu_jacobian_t)));

    return SFEM_SUCCESS;
}

int cu_hex8_adjugate_fill(const ptrdiff_t nelements,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          void *const SFEM_RESTRICT jacobian_adjugate,
                          void *const SFEM_RESTRICT jacobian_determinant) {
    cu_jacobian_t *h_jacobian_adjugate =
            (cu_jacobian_t *)calloc(9 * nelements, sizeof(cu_jacobian_t));
    cu_jacobian_t *h_jacobian_determinant =
            (cu_jacobian_t *)calloc(nelements, sizeof(cu_jacobian_t));

    {
        const geom_t *const x = points[0];
        const geom_t *const y = points[1];
        const geom_t *const z = points[2];

#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            idx_t ev[8];
            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            for (int v = 0; v < 8; v++) {
                ev[v] = elements[v][e];
            }

            for (int v = 0; v < 8; v++) {
                lx[v] = x[ev[v]];
                ly[v] = y[ev[v]];
                lz[v] = z[ev[v]];
            }

            const scalar_t half = 0.5;
            cu_hex8_adjugate_and_det(
                    lx, ly, lz, half, half, half, nelements, &h_jacobian_adjugate[e], &h_jacobian_determinant[e]);
        }
    }

    SFEM_CUDA_CHECK(cudaMemcpy(jacobian_adjugate,
                               h_jacobian_adjugate,
                               9 * nelements * sizeof(cu_jacobian_t),
                               cudaMemcpyHostToDevice));

    SFEM_CUDA_CHECK(cudaMemcpy(jacobian_determinant,
                               h_jacobian_determinant,
                               nelements * sizeof(cu_jacobian_t),
                               cudaMemcpyHostToDevice));

    // clean-up
    free(h_jacobian_adjugate);
    free(h_jacobian_determinant);
    return SFEM_SUCCESS;
}
