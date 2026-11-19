#include "sfem_GeneratedNeumannGeneral.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumannGeneral_generated_op() {
    OpFactory::register_op("GeneratedNeumannGeneral", GeneratedNeumannGeneral::create);
    OpFactory::register_op("ss:GeneratedNeumannGeneral", GeneratedNeumannGeneral::create);
  }
}  // namespace sfem
