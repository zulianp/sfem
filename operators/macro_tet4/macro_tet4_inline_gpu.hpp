#ifndef MACRO_TET4_INLINE_GPU_HPP
#define MACRO_TET4_INLINE_GPU_HPP

#include "tet4_inline_gpu.hpp"

template <typename geom_t>
static /*inline*/ __device__ __host__ void tet4_sub_fff_0(const geom_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t stride,
                                                          geom_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (geom_t)(1.0 / 2.0) * fff[0 * stride];
    sub_fff[1] = (geom_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[2] = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[3] = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    sub_fff[4] = (geom_t)(1.0 / 2.0) * fff[4 * stride];
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename geom_t>
static /*inline*/ __device__ __host__ void tet4_sub_fff_4(const geom_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t stride,
                                                          geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[0 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = fff[1 * stride] + (1.0 / 2.0) * fff[3 * stride] + x0;
    sub_fff[1] = (geom_t)(-1.0 / 2.0) * fff[1 * stride] - x0;
    sub_fff[2] = (geom_t)(1.0 / 2.0) * fff[4 * stride] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename geom_t>
static /*inline*/ __device__ __host__ void tet4_sub_fff_5(const geom_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t stride,
                                                          geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    const geom_t x1 = fff[4 * stride] + (geom_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    const geom_t x2 = (geom_t)(1.0 / 2.0) * fff[4 * stride] + x0;
    const geom_t x3 = (geom_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = (geom_t)(-1.0 / 2.0) * fff[2 * stride] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + fff[2 * stride] + x1;
}

template <typename geom_t>
static /*inline*/ __device__ __host__ void tet4_sub_fff_6(const geom_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t stride,
                                                          geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[4 * stride];
    const geom_t x2 = (geom_t)(1.0 / 2.0) * fff[1 * stride] + x0;
    sub_fff[0] = (geom_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + x0;
    sub_fff[1] = (geom_t)(1.0 / 2.0) * fff[2 * stride] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4 * stride] + (geom_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

template <typename geom_t>
static /*inline*/ __device__ __host__ void tet4_sub_fff_7(const geom_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t stride,
                                                          geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[5 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = x0;
    sub_fff[1] = (geom_t)(-1.0 / 2.0) * fff[4 * stride] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (geom_t)(1.0 / 2.0) * fff[3 * stride] + fff[4 * stride] + x0;
    sub_fff[4] = (geom_t)(1.0 / 2.0) * fff[1 * stride] + x1;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[0 * stride];
}

// Adjugate

template <typename adjugate_t>
static inline __device__ void sub_adj_0(const adjugate_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t stride,
                                        adjugate_t *const SFEM_RESTRICT sub_adjugate) {
    sub_adjugate[0] = 2 * adjugate[0 * stride];
    sub_adjugate[1] = 2 * adjugate[1 * stride];
    sub_adjugate[2] = 2 * adjugate[2 * stride];
    sub_adjugate[3] = 2 * adjugate[3 * stride];
    sub_adjugate[4] = 2 * adjugate[4 * stride];
    sub_adjugate[5] = 2 * adjugate[5 * stride];
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}

template <typename adjugate_t>
static inline __device__ void sub_adj_4(const adjugate_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t stride,
                                        adjugate_t *const SFEM_RESTRICT sub_adjugate) {
    const adjugate_t x0 = 2 * adjugate[0 * stride];
    const adjugate_t x1 = 2 * adjugate[1 * stride];
    const adjugate_t x2 = 2 * adjugate[2 * stride];
    sub_adjugate[0] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[3] = -x0;
    sub_adjugate[4] = -x1;
    sub_adjugate[5] = -x2;
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}

template <typename adjugate_t>
static inline __device__ void sub_adj_5(const adjugate_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t stride,
                                        adjugate_t *const SFEM_RESTRICT sub_adjugate) {
    const adjugate_t x0 = 2 * adjugate[3 * stride];
    const adjugate_t x1 = 2 * adjugate[6 * stride] + x0;
    const adjugate_t x2 = 2 * adjugate[4 * stride];
    const adjugate_t x3 = 2 * adjugate[7 * stride] + x2;
    const adjugate_t x4 = 2 * adjugate[5 * stride];
    const adjugate_t x5 = 2 * adjugate[8 * stride] + x4;
    sub_adjugate[0] = -x1;
    sub_adjugate[1] = -x3;
    sub_adjugate[2] = -x5;
    sub_adjugate[3] = x0;
    sub_adjugate[4] = x2;
    sub_adjugate[5] = x4;
    sub_adjugate[6] = 2 * adjugate[0 * stride] + x1;
    sub_adjugate[7] = 2 * adjugate[1 * stride] + x3;
    sub_adjugate[8] = 2 * adjugate[2 * stride] + x5;
}

template <typename adjugate_t>
static inline __device__ void sub_adj_6(const adjugate_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t stride,
                                        adjugate_t *const SFEM_RESTRICT sub_adjugate) {
    const adjugate_t x0 = 2 * adjugate[3 * stride];
    const adjugate_t x1 = 2 * adjugate[4 * stride];
    const adjugate_t x2 = 2 * adjugate[5 * stride];
    sub_adjugate[0] = 2 * adjugate[0 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[1 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[2 * stride] + x2;
    sub_adjugate[3] = 2 * adjugate[6 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[7 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[8 * stride] + x2;
    sub_adjugate[6] = -x0;
    sub_adjugate[7] = -x1;
    sub_adjugate[8] = -x2;
}

template <typename adjugate_t>
static inline __device__ void sub_adj_7(const adjugate_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t stride,
                                        adjugate_t *const SFEM_RESTRICT sub_adjugate) {
    const adjugate_t x0 = 2 * adjugate[6 * stride];
    const adjugate_t x1 = 2 * adjugate[7 * stride];
    const adjugate_t x2 = 2 * adjugate[8 * stride];
    sub_adjugate[0] = -x0;
    sub_adjugate[1] = -x1;
    sub_adjugate[2] = -x2;
    sub_adjugate[3] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[6] = 2 * adjugate[0 * stride];
    sub_adjugate[7] = 2 * adjugate[1 * stride];
    sub_adjugate[8] = 2 * adjugate[2 * stride];
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

#endif  // MACRO_TET4_INLINE_GPU_HPP
