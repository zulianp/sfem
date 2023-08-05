// Micro kernels
#include "Tri6_kernels.cu"

#include <stddef.h>

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_KERNEL static __global__
#else
#define SFEM_DEVICE_KERNEL static
#endif

SFEM_DEVICE_KERNEL void Tri6_jacobian_kernel(const ptrdiff_t nelements,
                                             const geom_t *const SFEM_RESTRICT xyz,
                                             real_t *const SFEM_RESTRICT jacobian) {
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {
        // The element coordinates and jacobian
        const geom_t *const this_xyz = &xyz[e];
        real_t *const this_jacobian = &jacobian[e];

        // x-coordinates
        const real_t px0 = this_xyz[(0 * fe_subparam_n_nodes + 0) * nelements];
        const real_t px1 = this_xyz[(0 * fe_subparam_n_nodes + 1) * nelements];
        const real_t px2 = this_xyz[(0 * fe_subparam_n_nodes + 2) * nelements];
        // y-coordinates
        const real_t py0 = this_xyz[(1 * fe_subparam_n_nodes + 0) * nelements];
        const real_t py1 = this_xyz[(1 * fe_subparam_n_nodes + 1) * nelements];
        const real_t py2 = this_xyz[(1 * fe_subparam_n_nodes + 2) * nelements];

        Tri6_mk_jacobian(
            // x-coordinates
            px0,
            px1,
            px2,
            // y-coordinates
            py0,
            py1,
            py2,
            // arrays
            nelements,
            this_jacobian);
    }
}

SFEM_DEVICE_KERNEL void Tri6_jacobian_inverse_kernel(const ptrdiff_t nelements,
                                                     const geom_t *const SFEM_RESTRICT xyz,
                                                     real_t *const SFEM_RESTRICT jacobian) {
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {
        // The element coordinates and jacobian
        const geom_t *const this_xyz = &xyz[e];
        real_t *const this_jacobian = &jacobian[e];

        // x-coordinates
        const real_t px0 = this_xyz[(0 * fe_subparam_n_nodes + 0) * nelements];
        const real_t px1 = this_xyz[(0 * fe_subparam_n_nodes + 1) * nelements];
        const real_t px2 = this_xyz[(0 * fe_subparam_n_nodes + 2) * nelements];
        // y-coordinates
        const real_t py0 = this_xyz[(1 * fe_subparam_n_nodes + 0) * nelements];
        const real_t py1 = this_xyz[(1 * fe_subparam_n_nodes + 1) * nelements];
        const real_t py2 = this_xyz[(1 * fe_subparam_n_nodes + 2) * nelements];

        Tri6_mk_jacobian_inverse(
            // x-coordinates
            px0,
            px1,
            px2,
            // y-coordinates
            py0,
            py1,
            py2,
            // arrays
            nelements,
            this_jacobian);
    }
}

static void Tri6_pack_elements(const ptrdiff_t n,
                               const ptrdiff_t element_offset,
                               idx_t **const SFEM_RESTRICT elems,
                               geom_t **const SFEM_RESTRICT xyz,
                               geom_t *const SFEM_RESTRICT he_xyz) {
    SFEM_RANGE_PUSH("Tri6_pack_elements");
    {
        for (int d = 0; d < fe_spatial_dim; ++d) {
            for (int e_node = 0; e_node < fe_subparam_n_nodes; e_node++) {
                const geom_t *const x = xyz[d];
                ptrdiff_t offset = (d * fe_subparam_n_nodes + e_node) * n;
                const idx_t *const nodes = &elems[e_node][element_offset];

                geom_t *buff = &he_xyz[offset];

#pragma omp parallel
                {
#pragma omp for nowait
                    for (ptrdiff_t k = 0; k < n; k++) {
                        buff[k] = x[nodes[k]];
                    }
                }
            }
        }
    }

    SFEM_RANGE_POP();
}
