#include "sfem_GeneratedNavierStokes_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNavierStokes_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedNavierStokes", GPUGeneratedNavierStokes::create);
    Factory::register_op("gpu:ss:GeneratedNavierStokes", GPUGeneratedNavierStokes::create);
  }
}  // namespace sfem
