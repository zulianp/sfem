#include "sfem_GeneratedMooneyRivlinKelvinVoigt_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedMooneyRivlinKelvinVoigt_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedMooneyRivlinKelvinVoigt", GPUGeneratedMooneyRivlinKelvinVoigt::create);
    OpFactory::register_op("gpu:ss:GeneratedMooneyRivlinKelvinVoigt", GPUGeneratedMooneyRivlinKelvinVoigt::create);
  }
}  // namespace sfem
