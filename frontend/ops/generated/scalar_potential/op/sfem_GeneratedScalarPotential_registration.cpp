#include "sfem_GeneratedScalarPotential.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedScalarPotential_generated_op() {
        Factory::register_op("GeneratedScalarPotential", GeneratedScalarPotential::create);
        Factory::register_op("ss:GeneratedScalarPotential", GeneratedScalarPotential::create);
    }
}  // namespace sfem
