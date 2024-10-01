#ifndef SFEM_CUDA_MPRGP_IMPL_HPP
#define SFEM_CUDA_MPRGP_IMPL_HPP

#include "sfem_openmp_mprgp_impl.hpp"

namespace sfem {
	template <typename T>
	struct CUDA_MPRGP {
		static void build_mprgp(struct MPRGP_Tpl<T>& tpl);
	};
}

#endif //SFEM_CUDA_MPRGP_IMPL_HPP
