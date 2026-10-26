#include "sfem_GeneratedBodyForce_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedBodyForce_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedBodyForce", GPUGeneratedBodyForce::create);
    OpFactory::register_op("gpu:ss:GeneratedBodyForce", GPUGeneratedBodyForce::create);
  }
}  // namespace sfem
