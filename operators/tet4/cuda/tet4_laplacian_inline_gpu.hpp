#ifndef TET4_LAPLACIAN_INLINE_GPU_HPP
#define TET4_LAPLACIAN_INLINE_GPU_HPP

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void tet4_laplacian_apply_fff(
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
static inline __device__ __host__ void tet4_laplacian_diag_fff(const fff_t *const SFEM_RESTRICT fff,
                                                              const ptrdiff_t stride,
                                                              scalar_t *const SFEM_RESTRICT
                                                                  element_vector) {
    element_vector[0] = fff[0 * stride] + 2 * fff[1 * stride] + 2 * fff[2 * stride] +
                        fff[3 * stride] + 2 * fff[4 * stride] + fff[5 * stride];
    element_vector[1] = fff[0 * stride];
    element_vector[2] = fff[3 * stride];
    element_vector[3] = fff[5 * stride];
}

// template <typename fff_t, typename scalar_t>
// static inline __device__ __host__ void tet4_laplacian_diag_fff(const fff_t *const SFEM_RESTRICT fff,
//                                                                scalar_t *const SFEM_RESTRICT e0,
//                                                                scalar_t *const SFEM_RESTRICT e1,
//                                                                scalar_t *const SFEM_RESTRICT e2,
//                                                                scalar_t *const SFEM_RESTRICT e3) {
//     *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
//     *e1 += fff[0];
//     *e2 += fff[3];
//     *e3 += fff[5];
// }



// template <typename fff_t, typename scalar_t>
// static /*inline*/ __device__ __host__ void tet4_laplacian_apply_fff(
//         const fff_t *const SFEM_RESTRICT fff,
//         const scalar_t u0,
//         const scalar_t u1,
//         const scalar_t u2,
//         const scalar_t u3,
//         scalar_t *const SFEM_RESTRICT e0,
//         scalar_t *const SFEM_RESTRICT e1,
//         scalar_t *const SFEM_RESTRICT e2,
//         scalar_t *const SFEM_RESTRICT e3) {
//     const scalar_t x0 = fff[0] + fff[1] + fff[2];
//     const scalar_t x1 = fff[1] + fff[3] + fff[4];
//     const scalar_t x2 = fff[2] + fff[4] + fff[5];
//     const scalar_t x3 = fff[1] * u0;
//     const scalar_t x4 = fff[2] * u0;
//     const scalar_t x5 = fff[4] * u0;
//     *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
//     *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
//     *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
//     *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
// }

#endif  // TET4_LAPLACIAN_INLINE_GPU_HPP
