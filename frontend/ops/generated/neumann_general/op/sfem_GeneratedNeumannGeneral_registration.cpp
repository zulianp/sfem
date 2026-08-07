#include "sfem_GeneratedNeumannGeneral.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedNeumannGeneral_generated_op() {
        Factory::register_op("GeneratedNeumannGeneral", GeneratedNeumannGeneral::create);
        Factory::register_op("ss:GeneratedNeumannGeneral", GeneratedNeumannGeneral::create);
    }
}  // namespace sfem
