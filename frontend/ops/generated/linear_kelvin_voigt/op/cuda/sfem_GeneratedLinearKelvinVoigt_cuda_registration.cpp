#include "sfem_GeneratedLinearKelvinVoigt_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedLinearKelvinVoigt_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedLinearKelvinVoigt", GPUGeneratedLinearKelvinVoigt::create);
    Factory::register_op("gpu:ss:GeneratedLinearKelvinVoigt", GPUGeneratedLinearKelvinVoigt::create);
  }
}  // namespace sfem
