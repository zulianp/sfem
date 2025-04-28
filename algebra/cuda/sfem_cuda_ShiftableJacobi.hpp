#ifndef SFEM_CUDA_SHIFTABLE_JACOBI_HPP
#define SFEM_CUDA_SHIFTABLE_JACOBI_HPP

#include "sfem_openmp_ShiftableJacobi.hpp"

namespace sfem {
	template <typename T>
	struct ShiftableBlockSymJacobi_CUDA {
		static int build(const int dim, struct ShiftableBlockSymJacobi_Tpl<T>& tpl);
	};
}

#endif  // SFEM_CUDA_SHIFTABLE_JACOBI_HPP
