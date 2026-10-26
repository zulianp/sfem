#include "sfem_GeneratedMooneyRivlinProny.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinProny_generated_op() {
    OpFactory::register_op("GeneratedMooneyRivlinProny", GeneratedMooneyRivlinProny::create);
    OpFactory::register_op("ss:GeneratedMooneyRivlinProny", GeneratedMooneyRivlinProny::create);
  }
}  // namespace sfem
