#include <algorithm>
#include <vector>

#include "boundary_condition_cuda.hpp"
#include "sfem_cuda_base.hpp"

namespace sfem {
    boundary_condition_t to_device(const boundary_condition_t &host) {
        boundary_condition_t device;
        device.local_size  = host.local_size;
        device.global_size = host.global_size;
        device.component   = host.component;
        device.value       = host.value;

        // Copy indices
        SFEM_CUDA_CHECK(cudaMalloc(&device.idx, device.local_size * sizeof(idx_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(device.idx, host.idx, device.local_size * sizeof(idx_t), cudaMemcpyHostToDevice));

        if (host.values) {
            SFEM_CUDA_CHECK(cudaMalloc(&device.values, device.local_size * sizeof(real_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(device.values, host.values, device.local_size * sizeof(real_t), cudaMemcpyHostToDevice));
        } else {
            device.values = nullptr;
        }
        return device;
    }
}  // namespace sfem

extern "C" {

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
                                 const idx_t    *dirichlet_nodes,
                                 const int       block_size,
                                 const int       component,
                                 const real_t   *source,
                                 real_t         *dest,
                                 void *const     stream) {
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
    constraint_nodes_copy_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, source, dest);

    SFEM_DEBUG_SYNCHRONIZE(stream);
}

void d_copy_at_dirichlet_nodes_vec(const int                         n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int                         block_size,
                                   const real_t *const               in,
                                   real_t *const                     out,
                                   void *const                       stream) {
    for (int i = 0; i < n_conditions; i++) {
        d_constraint_nodes_copy_vec(cond[i].local_size, cond[i].idx, block_size, cond[i].component, in, out, stream);
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
                                     const idx_t    *dirichlet_nodes,
                                     const int       block_size,
                                     const int       component,
                                     const real_t    value,
                                     real_t         *values,
                                     void *const     stream) {
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
    constraint_nodes_to_value_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, value, values);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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
                                      const idx_t    *dirichlet_nodes,
                                      const int       block_size,
                                      const int       component,
                                      const real_t   *dirichlet_values,
                                      real_t         *values,
                                      void *const     stream) {
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
    constraint_nodes_to_values_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, dirichlet_values, values);

    SFEM_DEBUG_SYNCHRONIZE(stream);
}

void d_apply_dirichlet_condition_vec(const int                         n_conditions,
                                     const boundary_condition_t *const cond,
                                     const int                         block_size,
                                     real_t *const                     x,
                                     void *const                       stream) {
    for (int i = 0; i < n_conditions; i++) {
        if (cond[i].values) {
            d_constraint_nodes_to_values_vec(
                    cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].values, x, stream);
        } else {
            d_constraint_nodes_to_value_vec(
                    cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].value, x, stream);
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

void d_constraint_gradient_nodes_to_value_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                              const idx_t                      *dirichlet_nodes,
                                              const int                         block_size,
                                              const int                         component,
                                              const real_t                      value,
                                              const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT       g,
                                              void *const                       stream) {
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
    constraint_gradient_nodes_to_value_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, value, x, g);

    SFEM_DEBUG_SYNCHRONIZE(stream);
}

// The same with one prescribed value per constrained node, as
// `d_constraint_nodes_to_values_vec` is to `d_constraint_nodes_to_value_vec`.
__global__ void constraint_gradient_nodes_to_values_vec_kernel(const ptrdiff_t n_dirichlet_nodes,
                                                               const idx_t *dirichlet_nodes,
                                                               const int block_size,
                                                               const int component,
                                                               const real_t *const SFEM_RESTRICT dirichlet_values,
                                                               const real_t *const SFEM_RESTRICT x,
                                                               real_t *const SFEM_RESTRICT g) {
    for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
         node += blockDim.x * gridDim.x) {
        idx_t i = dirichlet_nodes[node] * block_size + component;
        g[i]    = x[i] - dirichlet_values[node];
    }
}

void d_constraint_gradient_nodes_to_values_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                               const idx_t                      *dirichlet_nodes,
                                               const int                         block_size,
                                               const int                         component,
                                               const real_t *const SFEM_RESTRICT dirichlet_values,
                                               const real_t *const SFEM_RESTRICT x,
                                               real_t *const SFEM_RESTRICT       g,
                                               void *const                       stream) {
    const int kernel_block_size = 128;
    ptrdiff_t n_blocks =
            std::max(ptrdiff_t(1), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
    constraint_gradient_nodes_to_values_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, dirichlet_values, x, g);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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

void cu_crs_constraint_nodes_to_identity_vec(const ptrdiff_t                    n_dirichlet_nodes,
                                             const idx_t *const SFEM_RESTRICT   dirichlet_nodes,
                                             const int                          block_size,
                                             const int                          component,
                                             const real_t                       diag_value,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT   colidx,
                                             real_t *const SFEM_RESTRICT        values,
                                             void *const                        stream) {
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

    cu_crs_constraint_nodes_to_identity_vec_kernel<<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
            n_dirichlet_nodes, dirichlet_nodes, block_size, component, diag_value, rowptr, colidx, values);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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

namespace sfem {
    namespace {
        // The rows of `component` at the nodes: diag_value on the diagonal, zero elsewhere (the
        // device bsr_constraint_nodes_to_identity_vec).
        template <typename V>
        __global__ void bsr_constraint_nodes_to_identity_vec_kernel(const ptrdiff_t                    n_dirichlet_nodes,
                                                                    const idx_t *const SFEM_RESTRICT   dirichlet_nodes,
                                                                    const int                          block_size,
                                                                    const int                          component,
                                                                    const real_t                       diag_value,
                                                                    const count_t *const SFEM_RESTRICT rowptr,
                                                                    const idx_t *const SFEM_RESTRICT   colidx,
                                                                    V *const SFEM_RESTRICT             values) {
            const int mat_block_size = block_size * block_size;
            const int b_offset       = component * block_size;
            for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
                 node += blockDim.x * gridDim.x) {
                const idx_t i = dirichlet_nodes[node];
                for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                    V *const block = &values[(ptrdiff_t)k * mat_block_size];
                    for (int d = 0; d < block_size; d++) {
                        block[b_offset + d] = colidx[k] == i ? V((d == component) * diag_value) : V(0);
                    }
                }
            }
        }

        // Per block of threads and step s: sum over its nodes of (x_i + steps[s] h_i - v_i)^2 (h
        // null: zero), into partial[block * nsteps + s].
        __global__ void constraint_objective_vec_steps_kernel(const ptrdiff_t                   n_dirichlet_nodes,
                                                              const idx_t *const SFEM_RESTRICT  dirichlet_nodes,
                                                              const int                         block_size,
                                                              const int                         component,
                                                              const real_t                      value,
                                                              const real_t *const SFEM_RESTRICT values,
                                                              const real_t *const SFEM_RESTRICT x,
                                                              const real_t *const SFEM_RESTRICT h,
                                                              const int                         nsteps,
                                                              const real_t *const SFEM_RESTRICT steps,
                                                              real_t *const SFEM_RESTRICT       partial) {
            extern __shared__ real_t cache[];
            for (int s = 0; s < nsteps; ++s) {
                real_t acc = 0;
                for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_dirichlet_nodes;
                     node += blockDim.x * gridDim.x) {
                    const ptrdiff_t i    = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
                    const real_t    diff = (x[i] + (h ? steps[s] * h[i] : real_t(0))) - (values ? values[node] : value);
                    acc += diff * diff;
                }
                cache[threadIdx.x] = acc;
                __syncthreads();
                for (int w = blockDim.x / 2; w > 0; w /= 2) {
                    if (threadIdx.x < w) cache[threadIdx.x] += cache[threadIdx.x + w];
                    __syncthreads();
                }
                if (threadIdx.x == 0) partial[blockIdx.x * nsteps + s] = cache[0];
                __syncthreads();
            }
        }
    }  // namespace

    template <typename V>
    int d_bsr_constraint_nodes_to_identity_vec(const ptrdiff_t                    n_dirichlet_nodes,
                                               const idx_t *const SFEM_RESTRICT   dirichlet_nodes,
                                               const int                          block_size,
                                               const int                          component,
                                               const real_t                       diag_value,
                                               const count_t *const SFEM_RESTRICT rowptr,
                                               const idx_t *const SFEM_RESTRICT   colidx,
                                               V *const SFEM_RESTRICT             values,
                                               void *const                        stream) {
        if (n_dirichlet_nodes <= 0) return SFEM_SUCCESS;
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size;
        bsr_constraint_nodes_to_identity_vec_kernel<V><<<n_blocks, kernel_block_size, 0, (cudaStream_t)stream>>>(
                n_dirichlet_nodes, dirichlet_nodes, block_size, component, diag_value, rowptr, colidx, values);
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return SFEM_SUCCESS;
    }

    template int d_bsr_constraint_nodes_to_identity_vec<double>(const ptrdiff_t, const idx_t *const, const int, const int,
                                                                const real_t, const count_t *const, const idx_t *const,
                                                                double *const, void *const);
    template int d_bsr_constraint_nodes_to_identity_vec<float>(const ptrdiff_t, const idx_t *const, const int, const int,
                                                               const real_t, const count_t *const, const idx_t *const,
                                                               float *const, void *const);

    int d_constraint_objective_vec_steps(const ptrdiff_t                   n_dirichlet_nodes,
                                         const idx_t *const SFEM_RESTRICT  dirichlet_nodes,
                                         const int                         block_size,
                                         const int                         component,
                                         const real_t                      value,
                                         const real_t *const SFEM_RESTRICT values,
                                         const real_t *const SFEM_RESTRICT x,
                                         const real_t *const SFEM_RESTRICT h,
                                         const int                         nsteps,
                                         const real_t *const               steps,
                                         real_t *const                     out,
                                         void *const                       stream) {
        if (n_dirichlet_nodes <= 0 || nsteps <= 0) return SFEM_SUCCESS;
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks =
                std::min(ptrdiff_t(1024), (n_dirichlet_nodes + kernel_block_size - 1) / kernel_block_size);
        const cudaStream_t s = (cudaStream_t)stream;
        real_t *d_steps = nullptr, *d_partial = nullptr;
        if (cudaMallocAsync((void **)&d_steps, nsteps * sizeof(real_t), s) != cudaSuccess) return SFEM_FAILURE;
        if (cudaMallocAsync((void **)&d_partial, n_blocks * nsteps * sizeof(real_t), s) != cudaSuccess) return SFEM_FAILURE;
        cudaMemcpyAsync(d_steps, steps, nsteps * sizeof(real_t), cudaMemcpyHostToDevice, s);
        constraint_objective_vec_steps_kernel<<<n_blocks, kernel_block_size, kernel_block_size * sizeof(real_t), s>>>(
                n_dirichlet_nodes, dirichlet_nodes, block_size, component, value, values, x, h, nsteps, d_steps, d_partial);
        std::vector<real_t> partial(n_blocks * nsteps);
        cudaMemcpyAsync(partial.data(), d_partial, partial.size() * sizeof(real_t), cudaMemcpyDeviceToHost, s);
        cudaFreeAsync(d_steps, s);
        cudaFreeAsync(d_partial, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) return SFEM_FAILURE;
        // The blocks' sums in block order: the same total on every call.
        for (int k = 0; k < nsteps; ++k) {
            real_t acc = 0;
            for (ptrdiff_t b = 0; b < n_blocks; ++b) acc += partial[b * nsteps + k];
            out[k] += acc;
        }
        return SFEM_SUCCESS;
    }
}  // namespace sfem
