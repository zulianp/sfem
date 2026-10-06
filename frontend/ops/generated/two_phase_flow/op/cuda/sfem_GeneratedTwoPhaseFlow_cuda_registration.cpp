#include "sfem_GeneratedTwoPhaseFlow_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedTwoPhaseFlow_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedTwoPhaseFlow", GPUGeneratedTwoPhaseFlow::create);
    Factory::register_op("gpu:ss:GeneratedTwoPhaseFlow", GPUGeneratedTwoPhaseFlow::create);
  }
}  // namespace sfem
