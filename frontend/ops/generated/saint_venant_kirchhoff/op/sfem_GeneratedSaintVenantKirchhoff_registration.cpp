#include "sfem_GeneratedSaintVenantKirchhoff.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedSaintVenantKirchhoff_generated_op() {
    OpFactory::register_op("GeneratedSaintVenantKirchhoff", GeneratedSaintVenantKirchhoff::create);
    OpFactory::register_op("ss:GeneratedSaintVenantKirchhoff", GeneratedSaintVenantKirchhoff::create);
  }
}  // namespace sfem
