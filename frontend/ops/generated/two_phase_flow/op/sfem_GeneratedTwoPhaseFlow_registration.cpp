#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedTwoPhaseFlow_generated_op() {
        Factory::register_op("GeneratedTwoPhaseFlow", GeneratedTwoPhaseFlow::create);
    }
}  // namespace sfem
