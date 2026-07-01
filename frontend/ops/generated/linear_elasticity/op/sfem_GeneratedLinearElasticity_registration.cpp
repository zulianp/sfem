#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedLinearElasticity_generated_op() {
        Factory::register_op("GeneratedLinearElasticity", GeneratedLinearElasticity::create);
    }
}  // namespace sfem
