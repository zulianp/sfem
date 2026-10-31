#include "mooney_rivlin_kelvin_voigt_total_tet4_inexact.cuh"
#include "sfem_cuda_InexactStore.cuh"

namespace sfem {
namespace codegen {

std::unique_ptr<InexactStore> mooney_rivlin_kelvin_voigt_total_tet4_inexact_store(const ptrdiff_t nelements, const InexactOptions &options, void *const stream) {
  return inexact::cuda::create_store<mooney_rivlin_kelvin_voigt_total_tet4_inexact>(nelements, options, stream);
}

}  // namespace codegen
}  // namespace sfem
