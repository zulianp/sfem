#include "sfem_GeneratedNavierStokes.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedNavierStokes_generated_op() {
        Factory::register_op("GeneratedNavierStokes", GeneratedNavierStokes::create);
    }
}  // namespace sfem
