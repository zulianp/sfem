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

}  // namespace sfem

#endif  // SFEM_CUDA_BLAS_HPP
