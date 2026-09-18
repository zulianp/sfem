#include "sfem_GeneratedBodyForce.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedBodyForce_generated_op() {
    Factory::register_op("GeneratedBodyForce", GeneratedBodyForce::create);
    Factory::register_op("ss:GeneratedBodyForce", GeneratedBodyForce::create);
  }
}  // namespace sfem
