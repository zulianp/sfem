#include <algorithm>

#include "cu_boundary_condition.h"
#include "sfem_cuda_base.h"

extern "C" {

void boundary_conditions_host_to_device(const boundary_condition_t *const host,
                                        boundary_condition_t *const device) {
    device->local_size = host->local_size;
    device->global_size = host->global_size;
    device->component = host->component;
    device->value = host->value;

    // Copy indices
    SFEM_CUDA_CHECK(cudaMalloc(&device->idx, device->local_size * sizeof(idx_t)));
    SFEM_CUDA_CHECK(cudaMemcpy(
            device->idx, host->idx, device->local_size * sizeof(idx_t), cudaMemcpyHostToDevice));

    if (host->values) {
        SFEM_CUDA_CHECK(cudaMalloc(&device->values, device->local_size * sizeof(real_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(device->values,
                                   host->values,
                                   device->local_size * sizeof(real_t),
                                   cudaMemcpyHostToDevice));
    } else {
        device->values = nullptr;
    }
}

__global__ void constraint_nodes_copy_vec_kernel(const ptrdiff_t n_dirichlet_nodes,
                                                 const idx_t *dirichlet_nodes,
                                                 const int block_size,
                                                 const int component,
                                                 const real_t *source,
                                                 real_t *dest) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;
        dest[i] = source[i];
    }
}

void d_constraint_nodes_copy_vec(const ptrdiff_t n_dirichlet_nodes,
                                 const idx_t *dirichlet_nodes,
                                 const int block_size,
                                 const int component,
                                 const real_t *source,
                                 real_t *dest) {
    // TODO Hand tuned
    int kernel_block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &kernel_block_size, constraint_nodes_copy_vec_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
    constraint_nodes_copy_vec_kernel<<<n_blocks, kernel_block_size, 0>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, source, dest);

    SFEM_DEBUG_SYNCHRONIZE();
}

void d_copy_at_dirichlet_nodes_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int block_size,
                                   const real_t *const in,
                                   real_t *const out) {
    for (int i = 0; i < n_conditions; i++) {
        d_constraint_nodes_copy_vec(
                cond[i].local_size, cond[i].idx, block_size, cond[i].component, in, out);
    }
}

__global__ void constraint_nodes_to_value_vec_kernel(const ptrdiff_t n_dirichlet_nodes,
                                                     const idx_t *dirichlet_nodes,
                                                     const int block_size,
                                                     const int component,
                                                     const real_t value,
                                                     real_t *values) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;
        values[i] = value;
    }
}

void d_constraint_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                     const idx_t *dirichlet_nodes,
                                     const int block_size,
                                     const int component,
                                     const real_t value,
                                     real_t *values) {
    // TODO Hand tuned
    int kernel_block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &kernel_block_size, constraint_nodes_to_value_vec_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
    constraint_nodes_to_value_vec_kernel<<<n_blocks, kernel_block_size, 0>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, value, values);

    SFEM_DEBUG_SYNCHRONIZE();
}

__global__ void constraint_nodes_to_values_vec_kernel(const ptrdiff_t n_dirichlet_nodes,
                                                      const idx_t *dirichlet_nodes,
                                                      const int block_size,
                                                      const int component,
                                                      const real_t *dirichlet_values,
                                                      real_t *values) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;
        values[i] = dirichlet_values[node];
    }
}

void d_constraint_nodes_to_values_vec(const ptrdiff_t n_dirichlet_nodes,
                                      const idx_t *dirichlet_nodes,
                                      const int block_size,
                                      const int component,
                                      const real_t *dirichlet_values,
                                      real_t *values) {
    // TODO Hand tuned
    int kernel_block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &kernel_block_size, constraint_nodes_to_values_vec_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
    constraint_nodes_to_values_vec_kernel<<<n_blocks, kernel_block_size, 0>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, dirichlet_values, values);

    SFEM_DEBUG_SYNCHRONIZE();
}

void d_apply_dirichlet_condition_vec(const int n_conditions,
                                     const boundary_condition_t *const cond,
                                     const int block_size,
                                     real_t *const x) {
    for (int i = 0; i < n_conditions; i++) {
        if (cond[i].values) {
            d_constraint_nodes_to_values_vec(cond[i].local_size,
                                             cond[i].idx,
                                             block_size,
                                             cond[i].component,
                                             cond[i].values,
                                             x);
        } else {
            d_constraint_nodes_to_value_vec(cond[i].local_size,
                                            cond[i].idx,
                                            block_size,
                                            cond[i].component,
                                            cond[i].value,
                                            x);
        }
    }
}

__global__ void constraint_gradient_nodes_to_value_vec_kernel(const ptrdiff_t n_dirichlet_nodes,
                                                              const idx_t *dirichlet_nodes,
                                                              const int block_size,
                                                              const int component,
                                                              const real_t value,
                                                              const real_t *const SFEM_RESTRICT x,
                                                              real_t *const SFEM_RESTRICT g) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;
        g[i] = x[i] - value;
    }
}

void d_constraint_gradient_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                              const idx_t *dirichlet_nodes,
                                              const int block_size,
                                              const int component,
                                              const real_t value,
                                              const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT g) {
    int kernel_block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &kernel_block_size,
                                           constraint_gradient_nodes_to_value_vec_kernel,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
    constraint_gradient_nodes_to_value_vec_kernel<<<n_blocks, kernel_block_size, 0>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, value, x, g);

    SFEM_DEBUG_SYNCHRONIZE();
}

__global__ void cu_crs_constraint_nodes_to_identity_vec_kernel(
        const ptrdiff_t n_dirichlet_nodes,
        const idx_t *const SFEM_RESTRICT dirichlet_nodes,
        const int block_size,
        const int component,
        const real_t diag_value,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;

        idx_t begin = rowptr[i];
        idx_t end = rowptr[i + 1];
        idx_t lenrow = end - begin;
        const idx_t *cols = &colidx[begin];
        real_t *row = &values[begin];

        for (int k = 0; k < lenrow; k++) {
            row[k] = (cols[k] == i) ? diag_value : 0;
        }
    }
}

void cu_crs_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                             const idx_t *const SFEM_RESTRICT dirichlet_nodes,
                                             const int block_size,
                                             const int component,
                                             const real_t diag_value,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT colidx,
                                             real_t *const SFEM_RESTRICT values) {
    int kernel_block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &kernel_block_size,
                                           cu_crs_constraint_nodes_to_identity_vec_kernel,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);

    cu_crs_constraint_nodes_to_identity_vec_kernel<<<n_blocks, kernel_block_size, 0>>>(
            n_dirichlet_nodes,
            dirichlet_nodes,
            block_size,
            component,
            diag_value,
            rowptr,
            colidx,
            values);

    SFEM_DEBUG_SYNCHRONIZE();
}

void d_destroy_conditions(const int n_conditions, boundary_condition_t *cond) {
    for (int i = 0; i < n_conditions; i++) {
        cond[i].local_size = 0;
        cond[i].global_size = 0;
        cond[i].component = 0;

        cudaFree(cond[i].idx);

        if (cond[i].values) {
            cudaFree(cond[i].values);
        }
    }
}
}
