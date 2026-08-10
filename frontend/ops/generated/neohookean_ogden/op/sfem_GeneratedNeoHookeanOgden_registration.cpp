#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedNeoHookeanOgden_generated_op() {
        Factory::register_op("GeneratedNeoHookeanOgden", GeneratedNeoHookeanOgden::create);
    }
}  // namespace sfem
