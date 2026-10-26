#ifndef SFEM_CUDA_MPRGP_IMPL_HPP
#define SFEM_CUDA_MPRGP_IMPL_HPP

#include "sfem_execution_context.hpp"
#include "sfem_openmp_mprgp_impl.hpp"

namespace sfem {
	template <typename T>
	struct GPUMPRGPKernels {
            /// The passes on the device, ordered on `context`'s stream.
            static void build_mprgp(struct MPRGPKernels<T>& tpl, const ExecutionContext& context = ExecutionContext::device());
        };
}

#endif //SFEM_CUDA_MPRGP_IMPL_HPP
