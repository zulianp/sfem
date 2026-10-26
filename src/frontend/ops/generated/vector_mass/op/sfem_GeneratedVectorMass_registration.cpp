#include "sfem_GeneratedVectorMass.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedVectorMass_generated_op() {
    OpFactory::register_op("GeneratedVectorMass", GeneratedVectorMass::create);
    OpFactory::register_op("ss:GeneratedVectorMass", GeneratedVectorMass::create);
  }
}  // namespace sfem
