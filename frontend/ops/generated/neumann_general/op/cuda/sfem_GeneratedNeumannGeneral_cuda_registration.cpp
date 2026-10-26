#include "sfem_GeneratedNeumannGeneral_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumannGeneral_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedNeumannGeneral", GPUGeneratedNeumannGeneral::create);
    OpFactory::register_op("gpu:ss:GeneratedNeumannGeneral", GPUGeneratedNeumannGeneral::create);
  }
}  // namespace sfem
