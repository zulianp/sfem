#ifndef CU_MACRO_TET4_INLINE_GPU_HPP
#define CU_MACRO_TET4_INLINE_GPU_HPP

#include "cu_tet4_inline.hpp"

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_sub_fff_0(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        sub_fff_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
    sub_fff[1] = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[2] = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[3] = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    sub_fff[4] = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride];
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_sub_fff_4(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = fff[1 * stride] + (1.0 / 2.0) * fff[3 * stride] + x0;
    sub_fff[1] = (sub_fff_t)(-1.0 / 2.0) * fff[1 * stride] - x0;
    sub_fff[2] = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_sub_fff_5(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    const fff_t x1 = fff[4 * stride] + (sub_fff_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    const fff_t x2 = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride] + x0;
    const fff_t x3 = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = (sub_fff_t)(-1.0 / 2.0) * fff[2 * stride] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + fff[2 * stride] + x1;
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_sub_fff_6(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride];
    const fff_t x2 = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride] + x0;
    sub_fff[0] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + x0;
    sub_fff[1] = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4 * stride] + (sub_fff_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_sub_fff_7(
        const fff_t *const SFEM_RESTRICT fff,
        const ptrdiff_t stride,
        sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = x0;
    sub_fff[1] = (sub_fff_t)(-1.0 / 2.0) * fff[4 * stride] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride] + fff[4 * stride] + x0;
    sub_fff[4] = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride] + x1;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
}

template <typename scalar_t>
static inline __device__ void subtet_gather(const int i0,
                                            const int i1,
                                            const int i2,
                                            const int i3,
                                            const scalar_t *const SFEM_RESTRICT in,
                                            scalar_t *const SFEM_RESTRICT out) {
    out[0] = in[i0];
    out[1] = in[i1];
    out[2] = in[i2];
    out[3] = in[i3];
}

template <typename accumulator_t>
static inline __device__ void subtet_scatter_add(const int i0,
                                                 const int i1,
                                                 const int i2,
                                                 const int i3,
                                                 const accumulator_t *const SFEM_RESTRICT in,
                                                 accumulator_t *const SFEM_RESTRICT out) {
    out[i0] += in[0];
    out[i1] += in[1];
    out[i2] += in[2];
    out[i3] += in[3];
}

#define cu_tet4_gather_idx(from, i0, i1, i2, i3, to) \
    do {                                             \
        to[0] = from[i0];                            \
        to[1] = from[i1];                            \
        to[2] = from[i2];                            \
        to[3] = from[i3];                            \
    } while (0);

template <typename fff_t, typename scalar_t, typename accumulator_t>
static /*inline*/ __device__ __host__ void cu_macro_tet4_laplacian_apply_fff(
        const fff_t *const SFEM_RESTRICT fff,
        const scalar_t u0,
        const scalar_t u1,
        const scalar_t u2,
        const scalar_t u3,
        accumulator_t *const SFEM_RESTRICT e0,
        accumulator_t *const SFEM_RESTRICT e1,
        accumulator_t *const SFEM_RESTRICT e2,
        accumulator_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u0;
    const scalar_t x4 = fff[2] * u0;
    const scalar_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

template <typename fff_t, typename accumulator_t>
static inline __device__ __host__ void cu_macro_tet4_laplacian_diag_fff(
        const fff_t *const SFEM_RESTRICT fff,
        accumulator_t *const SFEM_RESTRICT e0,
        accumulator_t *const SFEM_RESTRICT e1,
        accumulator_t *const SFEM_RESTRICT e2,
        accumulator_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

#endif  // CU_MACRO_TET4_INLINE_GPU_HPP
