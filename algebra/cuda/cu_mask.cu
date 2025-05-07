#include "cu_mask.h"
#include "cu_mask.cuh"

#include "sfem_base.h"
#include "sfem_cuda_base.h"

__global__ void cu_mask_nodes_kernel(const ptrdiff_t                  nnodes,
                                     const idx_t *const SFEM_RESTRICT nodes,
                                     const int                        block_size,
                                     const int                        component,
                                     mask_t *const SFEM_RESTRICT      inout) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < nnodes; node += blockDim.x * gridDim.x) {
        idx_t idx = nodes[node] * block_size + component;
        cu_mask_atomic_set(idx, inout);
    }
}

extern int cu_mask_nodes(const ptrdiff_t                  nnodes,
                         const idx_t *const SFEM_RESTRICT nodes,
                         const int                        block_size,
                         const int                        component,
                         mask_t *const SFEM_RESTRICT      inout) {
    SFEM_DEBUG_SYNCHRONIZE();

    int       kernel_block_size = 128;
    ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (nnodes + kernel_block_size - 1) / kernel_block_size);
    cu_mask_nodes_kernel<<<n_blocks, kernel_block_size, 0>>>(nnodes, nodes, block_size, component, inout);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
