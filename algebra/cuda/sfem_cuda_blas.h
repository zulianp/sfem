#ifndef SFEM_CUDA_BLAS_H
#define SFEM_CUDA_BLAS_H

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

real_t *d_allocate(const std::size_t n);
void d_memset(void *ptr, int value, const std::size_t n);

void d_destroy(real_t *a);

void d_copy(const ptrdiff_t n, const real_t *const src, real_t *const dest);

real_t d_dot(const ptrdiff_t n, const real_t *const l, const real_t *const r);

void d_axpby(const ptrdiff_t n,
             const real_t alpha,
             const real_t *const x,
             const real_t beta,
             real_t *const y);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_CUDA_BLAS_H
