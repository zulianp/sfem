#ifndef SFEM_CUDA_CG_IMPL_HPP
#define SFEM_CUDA_CG_IMPL_HPP

#include "sfem_execution_context.hpp"
#include "sfem_ConjugateGradientKernels_openmp.hpp"

namespace sfem {
    template <typename T>
    struct GPUConjugateGradientKernels {
        /// The passes on the device, ordered on `context`'s stream.
        static void build(struct ConjugateGradientKernels<T>& tpl, const ExecutionContext& context = ExecutionContext::device());
    };
}  // namespace sfem

#endif  // SFEM_CUDA_CG_IMPL_HPP
