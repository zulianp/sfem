#ifndef SFEM_CUDA_MAX_HPP
#define SFEM_CUDA_MAX_HPP

#include "sfem_base.hpp"

namespace sfem {
    /// max(init, max_i x_i) over n device values: blocks reduce in shared memory, the host the
    /// blocks' maxima.
    real_t d_max(const ptrdiff_t n, const real_t *const x, const real_t init);
}  // namespace sfem

#endif  // SFEM_CUDA_MAX_HPP
