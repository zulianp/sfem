#include "sfem_GeneratedNeumann_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumann_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedNeumann", GPUGeneratedNeumann::create);
    OpFactory::register_op("gpu:ss:GeneratedNeumann", GPUGeneratedNeumann::create);
  }
}  // namespace sfem
