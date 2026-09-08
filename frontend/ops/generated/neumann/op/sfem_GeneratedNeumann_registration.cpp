#include "sfem_GeneratedNeumann.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedNeumann_generated_op() {
        Factory::register_op("GeneratedNeumann", GeneratedNeumann::create);
    }
}  // namespace sfem
