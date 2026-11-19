#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLinearElasticity_generated_op() {
    OpFactory::register_op("GeneratedLinearElasticity", GeneratedLinearElasticity::create);
    OpFactory::register_op("ss:GeneratedLinearElasticity", GeneratedLinearElasticity::create);
  }
}  // namespace sfem
