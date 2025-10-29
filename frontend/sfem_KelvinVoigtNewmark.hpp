#ifndef SFEM_KELVIN_VOIGT_NEWMARK_HPP
#define SFEM_KELVIN_VOIGT_NEWMARK_HPP

#include "sfem_Function.hpp"

namespace sfem {
    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space);
}  // namespace sfem

#endif  // SFEM_KELVIN_VOIGT_NEWMARK_HPP
