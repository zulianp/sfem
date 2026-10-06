#include "sfem_GeneratedMooneyRivlinKelvinVoigt.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinKelvinVoigt_generated_op() {
    Factory::register_op("GeneratedMooneyRivlinKelvinVoigt", GeneratedMooneyRivlinKelvinVoigt::create);
    Factory::register_op("ss:GeneratedMooneyRivlinKelvinVoigt", GeneratedMooneyRivlinKelvinVoigt::create);
  }
}  // namespace sfem
