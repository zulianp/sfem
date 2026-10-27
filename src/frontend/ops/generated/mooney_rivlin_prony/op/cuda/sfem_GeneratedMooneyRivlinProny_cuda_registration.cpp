#include "sfem_GeneratedMooneyRivlinProny_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinProny_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedMooneyRivlinProny", GPUGeneratedMooneyRivlinProny::create);
    OpFactory::register_op("gpu:ss:GeneratedMooneyRivlinProny", GPUGeneratedMooneyRivlinProny::create);
  }
}  // namespace sfem
