#include "sfem_GeneratedSaintVenantKirchhoff_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedSaintVenantKirchhoff_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedSaintVenantKirchhoff", GPUGeneratedSaintVenantKirchhoff::create);
    OpFactory::register_op("gpu:ss:GeneratedSaintVenantKirchhoff", GPUGeneratedSaintVenantKirchhoff::create);
  }
}  // namespace sfem
