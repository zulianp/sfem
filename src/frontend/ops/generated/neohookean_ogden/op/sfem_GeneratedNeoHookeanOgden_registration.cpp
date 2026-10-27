#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeoHookeanOgden_generated_op() {
    OpFactory::register_op("GeneratedNeoHookeanOgden", GeneratedNeoHookeanOgden::create);
    OpFactory::register_op("ss:GeneratedNeoHookeanOgden", GeneratedNeoHookeanOgden::create);
  }
}  // namespace sfem
