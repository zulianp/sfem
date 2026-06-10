#ifndef SFEM_CUDA_BLAS_HPP
#define SFEM_CUDA_BLAS_HPP

#include "sfem_base.hpp"
#include "sfem_tpl_blas.hpp"

#ifdef __cplusplus
extern "C" {
#endif

real_t *d_allocate(const size_t n);
void    d_destroy(void *a);

void *d_buffer_alloc(const size_t n);
void  d_buffer_destroy(void *a);

void d_memset(void *ptr, int value, const size_t n);
void buffer_device_to_host(const size_t n, const void *const d, void *h);
void buffer_host_to_device(const size_t n, const void *const h, void *d);
void d_memcpy(const ptrdiff_t n, const void *const src, void *const dest);

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

namespace sfem {

    template <typename T>
    struct CUDA_BLAS {
        static void build_blas(struct BLAS_Tpl<T>& tpl);
    };

    void device_synchronize();
    bool is_ptr_device(const void* ptr);

    template <typename T>
    int sbv_mult3(const ptrdiff_t    n_blocks,
                  const idx_t* const idx,
                  const T* const     dd,
                  const T* const     s,
                  const T* const     x,
                  T* const           y);

}  // namespace sfem

#endif  // SFEM_CUDA_BLAS_HPP
