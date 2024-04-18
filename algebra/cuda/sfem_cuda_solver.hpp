#ifndef SFEM_CUDA_SOLVER_HPP
#define SFEM_CUDA_SOLVER_HPP

#include <memory>
#include "sfem_cg.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cuda_blas.h"

namespace sfem {
	template <typename T>
	std::shared_ptr<ConjugateGradient<T>> d_cg()  {
		auto cg = std::make_shared<ConjugateGradient<T>>();
	    cg->allocate = d_allocate;
	    cg->destroy = d_destroy;
	    cg->copy = d_copy;
	    cg->dot = d_dot;
	    cg->axpby = d_axpby;
	    return cg;
	}

	template <typename T>
	std::shared_ptr<BiCGStab<T>> d_bcgstab() {
		auto cg = std::make_shared<BiCGStab<T>>();
	    cg->allocate = d_allocate;
	    cg->destroy = d_destroy;
	    cg->copy = d_copy;
	    cg->dot = d_dot;
	    cg->axpby = d_axpby;
	    cg->zaxpby = d_zaxpby;
	    return cg;
	}
}

#endif
