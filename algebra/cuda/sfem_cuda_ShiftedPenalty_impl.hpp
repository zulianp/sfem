#ifndef SFEM_CUDA_ShiftedPenalty_IMPL_HPP
#define SFEM_CUDA_ShiftedPenalty_IMPL_HPP

#include "sfem_ShiftedPenalty_impl.hpp"
#include "sfem_execution_context.hpp"

namespace sfem {
	template <typename T>
	struct GPUShiftedPenaltyKernels {
            /// The passes on the device, ordered on `context`'s stream.
            static void build(struct ShiftedPenaltyKernels<T>& tpl, const ExecutionContext& context = ExecutionContext::device());
        };
}

#endif //SFEM_CUDA_ShiftedPenalty_IMPL_HPP
