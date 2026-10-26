#include "sfem_GeneratedLaplace.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLaplace_generated_op() {
    OpFactory::register_op("GeneratedLaplace", GeneratedLaplace::create);
    OpFactory::register_op("ss:GeneratedLaplace", GeneratedLaplace::create);
  }
}  // namespace sfem
