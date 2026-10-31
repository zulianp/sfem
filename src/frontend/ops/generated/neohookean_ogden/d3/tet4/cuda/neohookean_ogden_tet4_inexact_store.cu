#include "neohookean_ogden_tet4_inexact.cuh"
#include "sfem_cuda_InexactStore.cuh"

namespace sfem {
namespace codegen {

std::unique_ptr<InexactStore> neohookean_ogden_tet4_inexact_store(const ptrdiff_t nelements, const InexactOptions &options, void *const stream) {
  return inexact::cuda::create_store<neohookean_ogden_tet4_inexact>(nelements, options, stream);
}

}  // namespace codegen
}  // namespace sfem
