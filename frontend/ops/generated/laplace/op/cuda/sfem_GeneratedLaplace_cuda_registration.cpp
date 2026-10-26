#include "sfem_GeneratedLaplace_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLaplace_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedLaplace", GPUGeneratedLaplace::create);
    OpFactory::register_op("gpu:ss:GeneratedLaplace", GPUGeneratedLaplace::create);
  }
}  // namespace sfem
