#include "sfem_GeneratedVectorMass_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedVectorMass_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedVectorMass", GPUGeneratedVectorMass::create);
    Factory::register_op("gpu:ss:GeneratedVectorMass", GPUGeneratedVectorMass::create);
  }
}  // namespace sfem
