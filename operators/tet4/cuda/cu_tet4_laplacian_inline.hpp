#ifndef CU_TET4_LAPLACIAN_INLINE_HPP
#define CU_TET4_LAPLACIAN_INLINE_HPP

#include "sfem_base.h"

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_tet4_laplacian_apply_fff(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        const scalar_t *const SFEM_RESTRICT u,
        scalar_t *const SFEM_RESTRICT element_vector) {
    const scalar_t x0 = fff[0 * stride] + fff[1 * stride] + fff[2 * stride];
    const scalar_t x1 = fff[1 * stride] + fff[3 * stride] + fff[4 * stride];
    const scalar_t x2 = fff[2 * stride] + fff[4 * stride] + fff[5 * stride];
    const scalar_t x3 = fff[1 * stride] * u[0];
    const scalar_t x4 = fff[2 * stride] * u[0];
    const scalar_t x5 = fff[4 * stride] * u[0];
    element_vector[0] = u[0] * x0 + u[0] * x1 + u[0] * x2 - u[1] * x0 - u[2] * x1 - u[3] * x2;
    element_vector[1] = -fff[0 * stride] * u[0] + fff[0 * stride] * u[1] + fff[1 * stride] * u[2] +
                        fff[2 * stride] * u[3] - x3 - x4;
    element_vector[2] = fff[1 * stride] * u[1] - fff[3 * stride] * u[0] + fff[3 * stride] * u[2] +
                        fff[4 * stride] * u[3] - x3 - x5;
    element_vector[3] = fff[2 * stride] * u[1] + fff[4 * stride] * u[2] - fff[5 * stride] * u[0] +
                        fff[5 * stride] * u[3] - x4 - x5;
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_tet4_laplacian_diag_fff(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        scalar_t *const SFEM_RESTRICT element_vector) {
    element_vector[0] = fff[0 * stride] + 2 * fff[1 * stride] + 2 * fff[2 * stride] +
                        fff[3 * stride] + 2 * fff[4 * stride] + fff[5 * stride];
    element_vector[1] = fff[0 * stride];
    element_vector[2] = fff[3 * stride];
    element_vector[3] = fff[5 * stride];
}

template <typename fff_t, typename accumulator_t>
static inline __device__ __host__ void cu_tet4_laplacian_matrix_fff(
        const fff_t *const SFEM_RESTRICT fff,
        accumulator_t *const SFEM_RESTRICT element_matrix) {
    const accumulator_t x0 = -fff[0] - fff[1] - fff[2];
    const accumulator_t x1 = -fff[1] - fff[3] - fff[4];
    const accumulator_t x2 = -fff[2] - fff[4] - fff[5];
    element_matrix[0] = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    element_matrix[1] = x0;
    element_matrix[2] = x1;
    element_matrix[3] = x2;
    element_matrix[4] = x0;
    element_matrix[5] = fff[0];
    element_matrix[6] = fff[1];
    element_matrix[7] = fff[2];
    element_matrix[8] = x1;
    element_matrix[9] = fff[1];
    element_matrix[10] = fff[3];
    element_matrix[11] = fff[4];
    element_matrix[12] = x2;
    element_matrix[13] = fff[2];
    element_matrix[14] = fff[4];
    element_matrix[15] = fff[5];
}

#endif  // CU_TET4_LAPLACIAN_INLINE_HPP
