#include "cu_obstacle.h"

#include "sfem_macros.h"
#include "sfem_cuda_base.h"

__global__ void obstacle_normal_project_kernel(const int                         dim,
                                               const ptrdiff_t                   n,
                                               const idx_t *const SFEM_RESTRICT  idx,
                                               real_t **const SFEM_RESTRICT      normals,
                                               const real_t *const SFEM_RESTRICT h,
                                               real_t *const SFEM_RESTRICT       out) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockIdx.x) {
        const ptrdiff_t     ii  = idx[i] * dim;
        const real_t *const hii = &h[ii];
        for (int d = 0; d < dim; d++) {
            out[i] += hii[d] * normals[d][i];
        }
    }
}

__global__ void obstacle_distribute_contact_forces_kernel(const int                         dim,
                                                          const ptrdiff_t                   n,
                                                          const idx_t *const SFEM_RESTRICT  idx,
                                                          real_t **const SFEM_RESTRICT      normals,
                                                          const real_t *const SFEM_RESTRICT m,
                                                          const real_t *const               f,
                                                          real_t *const                     out) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockIdx.x) {
        const ptrdiff_t ii  = idx[i] * dim;
        real_t *const   oii = &out[ii];
        const real_t    fi  = f[i] * m[i];
        for (int d = 0; d < dim; d++) {
            oii[d] += normals[d][i] * fi;
        }
    }
}

extern int cu_obstacle_normal_project(const int                         dim,
                                      const ptrdiff_t                   n,
                                      const idx_t *const SFEM_RESTRICT  idx,
                                      real_t **const SFEM_RESTRICT      normals,
                                      const real_t *const SFEM_RESTRICT h,
                                      real_t *const SFEM_RESTRICT       out) {
    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (n + block_size - 1) / block_size);
    obstacle_normal_project_kernel<<<n_blocks, block_size, 0>>>(dim, n, idx, normals, h, out);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_obstacle_distribute_contact_forces(const int                         dim,
                                                 const ptrdiff_t                   n,
                                                 const idx_t *const SFEM_RESTRICT  idx,
                                                 real_t **const SFEM_RESTRICT      normals,
                                                 const real_t *const SFEM_RESTRICT m,
                                                 const real_t *const               f,
                                                 real_t *const                     out) {
    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (n + block_size - 1) / block_size);
    obstacle_distribute_contact_forces_kernel<<<n_blocks, block_size, 0>>>(dim, n, idx, normals, m, f, out);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
