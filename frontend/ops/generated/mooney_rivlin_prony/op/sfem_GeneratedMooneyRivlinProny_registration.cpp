#include "sfem_GeneratedMooneyRivlinProny.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinProny_generated_op() {
    Factory::register_op("GeneratedMooneyRivlinProny", GeneratedMooneyRivlinProny::create);
    Factory::register_op("ss:GeneratedMooneyRivlinProny", GeneratedMooneyRivlinProny::create);
  }
}  // namespace sfem
