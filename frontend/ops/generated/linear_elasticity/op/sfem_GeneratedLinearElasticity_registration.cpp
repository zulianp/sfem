#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedLinearElasticity_generated_op() {
        Factory::register_op("GeneratedLinearElasticity", GeneratedLinearElasticity::create);
        Factory::register_op("ss:GeneratedLinearElasticity", GeneratedLinearElasticity::create);
    }
}  // namespace sfem
