#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_cuda_blas.h"

namespace sfem {
	void register_device_ops();
	std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);

	template <typename T>
	std::shared_ptr<Buffer<T>> d_buffer(const std::ptrdiff_t n) {
	    auto ret = std::make_shared<Buffer<T>>(
	        (T *)d_buffer_alloc(n * sizeof(T)), 
	        &d_buffer_destroy, 
	        MEMORY_SPACE_DEVICE
	    );
	    return ret;
	}

	std::string d_op_str(const std::string &name);

}

#endif //SFEM_FUNCTION_INCORE_CUDA_HPP
