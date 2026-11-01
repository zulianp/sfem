#include "linear_kelvin_voigt_total_proteus_hex27_inexact.cuh"
#include "sfem_cuda_InexactStore.cuh"

namespace sfem {
namespace codegen {

std::unique_ptr<InexactStore> linear_kelvin_voigt_total_proteus_hex27_inexact_store(const ptrdiff_t nelements, const InexactOptions &options, void *const stream) {
  return inexact::cuda::create_store<linear_kelvin_voigt_total_proteus_hex27_inexact>(nelements, options, stream);
}

}  // namespace codegen
}  // namespace sfem
