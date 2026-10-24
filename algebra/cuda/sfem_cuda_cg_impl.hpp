#ifndef SFEM_CUDA_CG_IMPL_HPP
#define SFEM_CUDA_CG_IMPL_HPP

#include "sfem_execution_context.hpp"
#include "sfem_openmp_cg_impl.hpp"

namespace sfem {
    template <typename T>
    struct CUDA_CG {
        /// The passes on the device, ordered on `context`'s stream.
        static void build(struct CG_Tpl<T>& tpl, const ExecutionContext& context = ExecutionContext::device());
    };
}  // namespace sfem

#endif  // SFEM_CUDA_CG_IMPL_HPP
