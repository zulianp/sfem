#include "sfem_GeneratedLinearKelvinVoigt.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLinearKelvinVoigt_generated_op() {
    Factory::register_op("GeneratedLinearKelvinVoigt", GeneratedLinearKelvinVoigt::create);
    Factory::register_op("ss:GeneratedLinearKelvinVoigt", GeneratedLinearKelvinVoigt::create);
  }
}  // namespace sfem
