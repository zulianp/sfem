#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedTwoPhaseFlow_generated_op() {
    OpFactory::register_op("GeneratedTwoPhaseFlow", GeneratedTwoPhaseFlow::create);
    OpFactory::register_op("ss:GeneratedTwoPhaseFlow", GeneratedTwoPhaseFlow::create);
  }
}  // namespace sfem
