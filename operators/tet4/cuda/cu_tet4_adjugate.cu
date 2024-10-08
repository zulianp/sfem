#include "cu_tet4_adjugate.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "cu_tet4_inline.hpp"

static inline __device__ __host__ void adjugate_and_det_micro_kernel(
        const geom_t px0,
        const geom_t px1,
        const geom_t px2,
        const geom_t px3,
        const geom_t py0,
        const geom_t py1,
        const geom_t py2,
        const geom_t py3,
        const geom_t pz0,
        const geom_t pz1,
        const geom_t pz2,
        const geom_t pz3,
        const ptrdiff_t stride,
        cu_jacobian_t *adjugate,
        cu_jacobian_t *jacobian_determinant) {
    // Compute Jacobian in high precision
    real_t jacobian[9];
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;

    const real_t x0 = jacobian[4] * jacobian[8];
    const real_t x1 = jacobian[5] * jacobian[7];
    const real_t x2 = jacobian[1] * jacobian[8];
    const real_t x3 = jacobian[1] * jacobian[5];
    const real_t x4 = jacobian[2] * jacobian[4];

    // Store adjugate in lower precision
    adjugate[0 * stride] = x0 - x1;
    adjugate[1 * stride] = jacobian[2] * jacobian[7] - x2;
    adjugate[2 * stride] = x3 - x4;
    adjugate[3 * stride] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4 * stride] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5 * stride] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6 * stride] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7 * stride] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8 * stride] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];

    // Store determinant in lower precision
    jacobian_determinant[0 * stride] = jacobian[0] * x0 - jacobian[0] * x1 +
                                       jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                                       jacobian[6] * x3 - jacobian[6] * x4;
}

int cu_tet4_adjugate_allocate(const ptrdiff_t nelements,
                              void **const SFEM_RESTRICT jacobian_adjugate,
                              void **const SFEM_RESTRICT jacobian_determinant) {
    SFEM_CUDA_CHECK(cudaMalloc(jacobian_adjugate, 9 * nelements * sizeof(cu_jacobian_t)));
    SFEM_CUDA_CHECK(cudaMalloc(jacobian_determinant, nelements * sizeof(cu_jacobian_t)));

    return SFEM_SUCCESS;
}

int cu_tet4_adjugate_fill(const ptrdiff_t nelements,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          void *const SFEM_RESTRICT jacobian_adjugate,
                          void *const SFEM_RESTRICT jacobian_determinant) {
    cu_jacobian_t *h_jacobian_adjugate =
            (cu_jacobian_t *)calloc(9 * nelements, sizeof(cu_jacobian_t));
    cu_jacobian_t *h_jacobian_determinant = (cu_jacobian_t *)calloc(nelements, sizeof(cu_jacobian_t));

#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        adjugate_and_det_micro_kernel(points[0][elements[0][e]],
                                      points[0][elements[1][e]],
                                      points[0][elements[2][e]],
                                      points[0][elements[3][e]],
                                      points[1][elements[0][e]],
                                      points[1][elements[1][e]],
                                      points[1][elements[2][e]],
                                      points[1][elements[3][e]],
                                      points[2][elements[0][e]],
                                      points[2][elements[1][e]],
                                      points[2][elements[2][e]],
                                      points[2][elements[3][e]],
                                      nelements,
                                      &h_jacobian_adjugate[e],
                                      &h_jacobian_determinant[e]);
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
