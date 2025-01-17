#ifndef SFEM_CUDA_ShiftedPenalty_IMPL_HPP
#define SFEM_CUDA_ShiftedPenalty_IMPL_HPP

#include "sfem_ShiftedPenalty_impl.hpp"

namespace sfem {
	template <typename T>
	struct CUDA_ShiftedPenalty {
		static void build(struct ShiftedPenalty_Tpl<T>& tpl);
	};
}

#endif //SFEM_CUDA_ShiftedPenalty_IMPL_HPP
