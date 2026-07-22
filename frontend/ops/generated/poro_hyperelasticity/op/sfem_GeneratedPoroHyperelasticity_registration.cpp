#include "sfem_GeneratedPoroHyperelasticity.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedPoroHyperelasticity_generated_op() {
        Factory::register_op("GeneratedPoroHyperelasticity", GeneratedPoroHyperelasticity::create);
        Factory::register_op("ss:GeneratedPoroHyperelasticity", GeneratedPoroHyperelasticity::create);
    }
}  // namespace sfem
