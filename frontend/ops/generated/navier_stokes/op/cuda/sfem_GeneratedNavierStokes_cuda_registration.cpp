#include "sfem_GeneratedNavierStokes_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNavierStokes_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedNavierStokes", GPUGeneratedNavierStokes::create);
    OpFactory::register_op("gpu:ss:GeneratedNavierStokes", GPUGeneratedNavierStokes::create);
  }
}  // namespace sfem
