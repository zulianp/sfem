#include "sfem_GeneratedPoroHyperelasticity.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedPoroHyperelasticity_generated_op() {
        Factory::register_op("GeneratedPoroHyperelasticity", GeneratedPoroHyperelasticity::create);
    }
}  // namespace sfem
