#ifndef SFEM_CUDA_BLAS_H
#define SFEM_CUDA_BLAS_H

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif
// void sfem_blas_init();
real_t *d_allocate(const size_t n);
void    d_destroy(void *a);

void *d_buffer_alloc(const size_t n);
void  d_buffer_destroy(void *a);

void d_memset(void *ptr, int value, const size_t n);
void buffer_device_to_host(const size_t n, const void *const d, void *h);
void buffer_host_to_device(const size_t n, const void *const h, void *d);

void device_to_host(const size_t n, const real_t *const d, real_t *h);
void host_to_device(const size_t n, const real_t *const h, real_t *d);

void d_copy(const ptrdiff_t n, const real_t *const src, real_t *const dest);

real_t d_dot(const ptrdiff_t n, const real_t *const l, const real_t *const r);

void d_ediv(const ptrdiff_t n, const real_t *const l, const real_t *const r, real_t *const result);

void d_axpby(const ptrdiff_t n, const real_t alpha, const real_t *const x, const real_t beta, real_t *const y);

void d_axpy(const ptrdiff_t n, const real_t alpha, const real_t *const x, real_t *const y);

void d_scal(const ptrdiff_t n, const real_t alpha, real_t *const x);

real_t d_nrm2(const ptrdiff_t n, const real_t *const x);

void d_zaxpby(const ptrdiff_t,
              const real_t        alpha,
              const real_t *const x,
              const real_t        beta,
              const real_t *const y,
              real_t *const       z);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_CUDA_BLAS_H
