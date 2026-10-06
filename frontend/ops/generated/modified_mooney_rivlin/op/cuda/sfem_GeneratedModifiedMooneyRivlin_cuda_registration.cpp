#include "sfem_GeneratedModifiedMooneyRivlin_cuda.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
  void register_GeneratedModifiedMooneyRivlin_cuda_generated_op() {
    Factory::register_op("gpu:GeneratedModifiedMooneyRivlin", GPUGeneratedModifiedMooneyRivlin::create);
    Factory::register_op("gpu:ss:GeneratedModifiedMooneyRivlin", GPUGeneratedModifiedMooneyRivlin::create);
  }
}  // namespace sfem
