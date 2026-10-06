#include "sfem_GeneratedSaintVenantKirchhoff_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedSaintVenantKirchhoff_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedSaintVenantKirchhoff", GPUGeneratedSaintVenantKirchhoff::create);
    Factory::register_op("gpu:ss:GeneratedSaintVenantKirchhoff", GPUGeneratedSaintVenantKirchhoff::create);
  }
}  // namespace sfem
