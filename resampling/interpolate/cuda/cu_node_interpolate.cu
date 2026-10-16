#include "cu_node_interpolate.hpp"

#include "sfem_cuda_base.hpp"

namespace {
    template <int ORDER>
    __global__ void cu_sample_nodes_kernel(const ptrdiff_t                   nnodes,
                                           const idx_t *const SFEM_RESTRICT  idx,
                                           const NodeSamplerPoints           points,
                                           const real_t *const SFEM_RESTRICT disp,
                                           const NodeSamplerGrid             grid,
                                           const NodeSamplerOutput           out) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nnodes; i += blockDim.x * gridDim.x) {
            sample_node<ORDER>(i, idx, points, disp, grid, out);
        }
    }
}  // namespace

int cu_sample_nodes(const int                         order,
                    const ptrdiff_t                   nnodes,
                    const idx_t *const SFEM_RESTRICT  idx,
                    const NodeSamplerPoints          &points,
                    const real_t *const SFEM_RESTRICT disp,
                    const NodeSamplerGrid            &grid,
                    const NodeSamplerOutput          &out) {
    if (!nnodes) return SFEM_SUCCESS;

    SFEM_DEBUG_SYNCHRONIZE();
    const int       block_size = 128;
    const ptrdiff_t n_blocks   = (nnodes + block_size - 1) / block_size;
    if (order == 2) {
        cu_sample_nodes_kernel<2><<<n_blocks, block_size, 0>>>(nnodes, idx, points, disp, grid, out);
    } else {
        cu_sample_nodes_kernel<1><<<n_blocks, block_size, 0>>>(nnodes, idx, points, disp, grid, out);
    }
    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
