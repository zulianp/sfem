#include "sfem_GeneratedNeumannGeneral_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumannGeneral_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedNeumannGeneral", GPUGeneratedNeumannGeneral::create);
    Factory::register_op("gpu:ss:GeneratedNeumannGeneral", GPUGeneratedNeumannGeneral::create);
  }
}  // namespace sfem
