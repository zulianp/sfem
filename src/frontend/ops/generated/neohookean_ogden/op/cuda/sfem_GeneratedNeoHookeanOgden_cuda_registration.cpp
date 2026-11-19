#include "sfem_GeneratedNeoHookeanOgden_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedNeoHookeanOgden_cuda_generated_op() {
    OpFactory::register_op("gpu:GeneratedNeoHookeanOgden", GPUGeneratedNeoHookeanOgden::create);
    OpFactory::register_op("gpu:ss:GeneratedNeoHookeanOgden", GPUGeneratedNeoHookeanOgden::create);
  }
}  // namespace sfem
