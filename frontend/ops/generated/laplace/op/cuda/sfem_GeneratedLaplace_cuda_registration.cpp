#include "sfem_GeneratedLaplace_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLaplace_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedLaplace", GPUGeneratedLaplace::create);
    Factory::register_op("gpu:ss:GeneratedLaplace", GPUGeneratedLaplace::create);
  }
}  // namespace sfem
