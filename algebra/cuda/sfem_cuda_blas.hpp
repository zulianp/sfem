#ifndef SFEM_CUDA_BLAS_HPP
#define SFEM_CUDA_BLAS_HPP

#include "sfem_tpl_blas.hpp"

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
