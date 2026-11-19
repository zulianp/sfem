#include "sfem_GeneratedModifiedMooneyRivlin.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedModifiedMooneyRivlin_generated_op() {
    OpFactory::register_op("GeneratedModifiedMooneyRivlin", GeneratedModifiedMooneyRivlin::create);
    OpFactory::register_op("ss:GeneratedModifiedMooneyRivlin", GeneratedModifiedMooneyRivlin::create);
  }
}  // namespace sfem
