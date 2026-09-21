#include "sfem_GeneratedNeumann_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeumann_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedNeumann", GPUGeneratedNeumann::create);
    Factory::register_op("gpu:ss:GeneratedNeumann", GPUGeneratedNeumann::create);
  }
}  // namespace sfem
