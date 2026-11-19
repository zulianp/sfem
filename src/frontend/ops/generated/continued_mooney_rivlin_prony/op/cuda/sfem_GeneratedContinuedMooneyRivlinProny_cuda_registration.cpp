#include "sfem_GeneratedContinuedMooneyRivlinProny_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedContinuedMooneyRivlinProny_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedContinuedMooneyRivlinProny", GPUGeneratedContinuedMooneyRivlinProny::create);
    OpFactory::register_op("gpu:ss:GeneratedContinuedMooneyRivlinProny", GPUGeneratedContinuedMooneyRivlinProny::create);
  }
}  // namespace sfem
