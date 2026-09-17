#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinKelvinVoigtNewmark_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedMooneyRivlinKelvinVoigtNewmark", GPUGeneratedMooneyRivlinKelvinVoigtNewmark::create);
    Factory::register_op("gpu:ss:GeneratedMooneyRivlinKelvinVoigtNewmark", GPUGeneratedMooneyRivlinKelvinVoigtNewmark::create);
  }
}  // namespace sfem
