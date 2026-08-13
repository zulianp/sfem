#ifndef SFEM_CUDA_CG_IMPL_HPP
#define SFEM_CUDA_CG_IMPL_HPP

#include "sfem_openmp_cg_impl.hpp"

namespace sfem {
    template <typename T>
    struct CUDA_CG {
        static void build(struct CG_Tpl<T>& tpl);
    };
}  // namespace sfem

#endif  // SFEM_CUDA_CG_IMPL_HPP
