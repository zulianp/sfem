#ifndef SFEM_CONTACT_CONDITIONS_CUDA_HPP
#define SFEM_CONTACT_CONDITIONS_CUDA_HPP

#include "sfem_ContactConditions.hpp"

namespace sfem {
	std::shared_ptr<Constraint> to_device(const std::shared_ptr<ContactConditions> &dc);
}

#endif //SFEM_CONTACT_CONDITIONS_CUDA_HPP
