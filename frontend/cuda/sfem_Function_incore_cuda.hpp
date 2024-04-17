#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

namespace sfem {
	void register_device_ops();
	std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);
}

#endif //SFEM_FUNCTION_INCORE_CUDA_HPP
