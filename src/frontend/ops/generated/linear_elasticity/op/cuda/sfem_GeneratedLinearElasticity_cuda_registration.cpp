#include "sfem_GeneratedLinearElasticity_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLinearElasticity_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedLinearElasticity", GPUGeneratedLinearElasticity::create);
    OpFactory::register_op("gpu:ss:GeneratedLinearElasticity", GPUGeneratedLinearElasticity::create);
  }
}  // namespace sfem
