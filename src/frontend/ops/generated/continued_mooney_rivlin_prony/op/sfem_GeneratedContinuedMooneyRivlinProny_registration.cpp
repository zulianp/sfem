#include "sfem_GeneratedContinuedMooneyRivlinProny.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedContinuedMooneyRivlinProny_generated_op() {
    OpFactory::register_op("GeneratedContinuedMooneyRivlinProny", GeneratedContinuedMooneyRivlinProny::create);
    OpFactory::register_op("ss:GeneratedContinuedMooneyRivlinProny", GeneratedContinuedMooneyRivlinProny::create);
  }
}  // namespace sfem
