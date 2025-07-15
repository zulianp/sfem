#include "cu_contact_surface.h"

#include "sfem_cuda_base.h"
#include "sfem_macros.h"

__global__ void cu_displace_points_kernel(const int                         dim,
                                          const ptrdiff_t                   n_nodes,
                                          const idx_t *const SFEM_RESTRICT  idx,
                                          geom_t **const SFEM_RESTRICT      bulk_points,
                                          const real_t *const SFEM_RESTRICT disp,
                                          geom_t **const SFEM_RESTRICT      surface_points) {
    ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);

    for (int d = 0; d < dim; d++) {
        surface_points[d][i] = bulk_points[d][b] + disp[b * dim + d];
    }
}

int cu_displace_points(const int          dim,
                       const ptrdiff_t    n_nodes,
                       const idx_t *const idx,
                       geom_t **const     bulk_points,
                       const real_t      *disp,
                       geom_t **const     surface_points) {
    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (n_nodes + block_size - 1) / block_size);

    cu_displace_points_kernel<<<n_blocks, block_size>>>(dim, n_nodes, idx, bulk_points, disp, surface_points);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

__global__ void cu_displace_surface_points_kernel(const int                         dim,
                                                  const ptrdiff_t                   n_nodes,
                                                  const idx_t *const SFEM_RESTRICT  idx,
                                                  geom_t **const SFEM_RESTRICT      surface_points_rest,
                                                  const real_t *const SFEM_RESTRICT disp,
                                                  geom_t **const SFEM_RESTRICT      surface_points) {
    ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);

    for (int d = 0; d < dim; d++) {
        surface_points[d][i] = surface_points_rest[d][i] + disp[b * dim + d];
    }
}

int cu_displace_surface_points(const int          dim,
                               const ptrdiff_t    n_nodes,
                               const idx_t *const idx,
                               geom_t **const     surface_points_rest,
                               const real_t      *disp,
                               geom_t **const     surface_points) {
    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (n_nodes + block_size - 1) / block_size);

    cu_displace_surface_points_kernel<<<n_blocks, block_size>>>(dim, n_nodes, idx, surface_points_rest, disp, surface_points);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
