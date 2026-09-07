#include "sfem_GeneratedStokes.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedStokes_generated_op() {
        Factory::register_op("GeneratedStokes", GeneratedStokes::create);
        Factory::register_op("ss:GeneratedStokes", GeneratedStokes::create);
    }
}  // namespace sfem
