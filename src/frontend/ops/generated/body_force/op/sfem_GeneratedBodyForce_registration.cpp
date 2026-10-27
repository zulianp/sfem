#include "sfem_GeneratedBodyForce.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedBodyForce_generated_op() {
    OpFactory::register_op("GeneratedBodyForce", GeneratedBodyForce::create);
    OpFactory::register_op("ss:GeneratedBodyForce", GeneratedBodyForce::create);
  }
}  // namespace sfem
