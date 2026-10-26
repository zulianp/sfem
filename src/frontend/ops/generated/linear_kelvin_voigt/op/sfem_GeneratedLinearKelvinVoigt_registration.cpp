#include "sfem_GeneratedLinearKelvinVoigt.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLinearKelvinVoigt_generated_op() {
    OpFactory::register_op("GeneratedLinearKelvinVoigt", GeneratedLinearKelvinVoigt::create);
    OpFactory::register_op("ss:GeneratedLinearKelvinVoigt", GeneratedLinearKelvinVoigt::create);
  }
}  // namespace sfem
