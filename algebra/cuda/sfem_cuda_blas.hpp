#ifndef SFEM_CUDA_BLAS_HPP
#define SFEM_CUDA_BLAS_HPP

#include "sfem_base.hpp"
#include "sfem_tpl_blas.hpp"

#include <memory>

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
    class CUDA_BLAS final : public BLAS<T> {
    public:
        T* allocate(const std::size_t n) override;
        void destroy(void* a) override;

        void zeros(const std::size_t size, T* const x) override;
        void values(const std::size_t size, const T value, T* const x) override;

        void copy(const ptrdiff_t n, const T* const src, T* const dest) override;

        T dot(const ptrdiff_t n, const T* const l, const T* const r) override;
        void axpy(const ptrdiff_t n, const T alpha, const T* const x, T* const y) override;
        void axpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) override;
        void scal(const std::ptrdiff_t n, const T alpha, T* const x) override;
        void reciprocal(const std::ptrdiff_t n, const T alpha, T* const x) override;
        T norm2(const ptrdiff_t n, const T* const x) override;
        void zaxpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, const T* const y, T* const z) override;
        void xypaz(const ptrdiff_t n, const T* const x, const T* const y, const T alpha, T* const z) override;
    };

    template <typename T>
    std::shared_ptr<BLAS<T>> make_cuda_blas() {
        return std::make_shared<CUDA_BLAS<T>>();
    }

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
