#include "sfem_GeneratedNeumann.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumann_generated_op() {
    OpFactory::register_op("GeneratedNeumann", GeneratedNeumann::create);
    OpFactory::register_op("ss:GeneratedNeumann", GeneratedNeumann::create);
  }
}  // namespace sfem
