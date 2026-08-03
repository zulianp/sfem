#include "sfem_GeneratedSaintVenantKirchhoff.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedSaintVenantKirchhoff_generated_op() {
        Factory::register_op("GeneratedSaintVenantKirchhoff", GeneratedSaintVenantKirchhoff::create);
        Factory::register_op("ss:GeneratedSaintVenantKirchhoff", GeneratedSaintVenantKirchhoff::create);
    }
}  // namespace sfem
