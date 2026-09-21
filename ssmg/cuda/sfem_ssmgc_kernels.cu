#include "sfem_ssmgc_kernels.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"

#include <algorithm>

namespace sfem {

    __global__ void pack_nodal_diag_to_block_sym6_kernel(const ptrdiff_t                   n_nodes,
                                                         const real_t *const SFEM_RESTRICT d3,
                                                         real_t *const SFEM_RESTRICT       d6) {
        for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_nodes;
             node += blockDim.x * gridDim.x) {
            // The off-diagonals are zero because the source is a diagonal: this
            // widens a nodal (xx, yy, zz) into the symmetric-6 layout the block
            // Jacobi smoother reads, it does not invent coupling.
            d6[node * 6 + 0] = d3[node * 3 + 0];
            d6[node * 6 + 1] = real_t(0);
            d6[node * 6 + 2] = real_t(0);
            d6[node * 6 + 3] = d3[node * 3 + 1];
            d6[node * 6 + 4] = real_t(0);
            d6[node * 6 + 5] = d3[node * 3 + 2];
        }
    }

    void pack_nodal_diag_to_block_sym6_device(const ptrdiff_t     n_nodes,
                                              const real_t *const d3,
                                              real_t *const       d6) {
        if (n_nodes <= 0 || !d3 || !d6) {
            return;
        }
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (n_nodes + kernel_block_size - 1) / kernel_block_size);
        pack_nodal_diag_to_block_sym6_kernel<<<n_blocks, kernel_block_size, 0>>>(n_nodes, d3, d6);
        SFEM_DEBUG_SYNCHRONIZE();
    }

}  // namespace sfem
