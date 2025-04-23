#include "cu_hex8_laplacian.h"

#include "cu_hex8_laplacian_inline.hpp"
#include "sfem_cuda_base.h"

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

template <typename T>
__global__ void cu_affine_hex8_laplacian_apply_tiled_kernel(const ptrdiff_t nelements,
                                                            const ptrdiff_t stride,  // Stride for elements and fff
                                                            const idx_t *const SFEM_RESTRICT         elements,
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                            const T *const SFEM_RESTRICT             u,
                                                            T *const SFEM_RESTRICT                   values) {
    // Tile number in group
    const int       tile    = threadIdx.x >> 3;   // same as threadIdx.x / 8
    const int       n_tiles = blockDim.x >> 3;    // same as blockDim.x / 8
    const int       sub_idx = threadIdx.x & 0x7;  // same as threadIdx.x % 8
    const ptrdiff_t e       = blockIdx.x * n_tiles + tile;

    // Boundary check
    if (e >= nelements) return;

    extern __shared__ unsigned char shared_mem[];

    const ptrdiff_t offset       = tile * 8 * sizeof(T);
    const ptrdiff_t block_offset = sizeof(T) * blockDim.x;

    T *element_u   = (T *)&shared_mem[offset];
    T *gx          = (T *)&shared_mem[offset + block_offset];
    T *gy          = (T *)&shared_mem[offset + block_offset * 2];
    T *gz          = (T *)&shared_mem[offset + block_offset * 3];
    T *element_out = (T *)&shared_mem[offset + block_offset * 4];

    // Reorder HEX8 to cartesian
    const ptrdiff_t vidx =
            e + stride * ((sub_idx == 3 || sub_idx == 7) ? (sub_idx - 1)
                                                         : ((sub_idx == 2 || sub_idx == 6) ? (sub_idx + 1) : sub_idx));
    const ptrdiff_t gidx = elements[vidx];

    T fff[6];
    for (int d = 0; d < 6; d++) {
        fff[d] = g_fff[d * stride + e];
    }
    static const T D[2 * 2] = {-1, 1, -1, 1};
    static const T qw[2]    = {0.5, 0.5};

    // Gather
    element_u[sub_idx]   = (e < nelements) ? u[gidx] : 0;
    element_out[sub_idx] = 0;
    gx[sub_idx]          = 0;
    gy[sub_idx]          = 0;
    gz[sub_idx]          = 0;

    // hex8 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2
    const int zi = (sub_idx >> 2);        // equivalent to sub_idx / 4

    cu_spectral_hex_laplacian_apply_tiled<2, T>(xi, yi, zi, D, fff, qw, element_u, gx, gy, gz, element_out);
    atomicAdd(&values[gidx], element_out[sub_idx]);
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_tiled_tpl(const ptrdiff_t                  nelements,
                                                    const ptrdiff_t                  stride,  // Stride for elements and fff
                                                    const idx_t *const SFEM_RESTRICT elements,
                                                    const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                    const T *const                           x,
                                                    T *const                                 y,
                                                    void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
    // int block_size = 256;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    size_t    shmem_size = block_size * sizeof(T) * 5;
    ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size / 8 - 1) / (block_size / 8));

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_tiled_kernel<<<n_blocks, block_size, shmem_size, s>>>(
                nelements, stride, elements, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_tiled_kernel<<<n_blocks, block_size, shmem_size>>>(nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_kernel(const ptrdiff_t                  nelements,
                                                      const ptrdiff_t                  stride,  // Stride for elements and fff
                                                      const idx_t *const SFEM_RESTRICT elements,
                                                      const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                      const real_t *const SFEM_RESTRICT        u,
                                                      real_t *const SFEM_RESTRICT              values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * stride + e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * stride + e];
        }

#if 0
        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v]];
        }

        cu_hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            atomicAdd(&values[dof_i], element_vector[edof_i]);
        }
#else

        element_u[0] = u[ev[0]];
        element_u[1] = u[ev[1]];
        element_u[2] = u[ev[3]];
        element_u[3] = u[ev[2]];

        element_u[4] = u[ev[4]];
        element_u[5] = u[ev[5]];
        element_u[6] = u[ev[7]];
        element_u[7] = u[ev[6]];

        for (int v = 0; v < 8; ++v) {
            element_vector[v] = 0;
        }

        cu_hex8_laplacian_apply_add_fff_sum_factorization(fff, element_u, element_vector);

        atomicAdd(&values[ev[0]], element_vector[0]);
        atomicAdd(&values[ev[1]], element_vector[1]);
        atomicAdd(&values[ev[2]], element_vector[3]);
        atomicAdd(&values[ev[3]], element_vector[2]);

        atomicAdd(&values[ev[4]], element_vector[4]);
        atomicAdd(&values[ev[5]], element_vector[5]);
        atomicAdd(&values[ev[6]], element_vector[7]);
        atomicAdd(&values[ev[7]], element_vector[6]);
#endif
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_tpl(const ptrdiff_t                          nelements,
                                              const ptrdiff_t                          stride,  // Stride for elements and fff
                                              const idx_t *const SFEM_RESTRICT         elements,
                                              const cu_jacobian_t *const SFEM_RESTRICT fff,
                                              const T *const                           x,
                                              T *const                                 y,
                                              void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_apply(const ptrdiff_t                  nelements,
                                          const ptrdiff_t                  stride,  // Stride for elements and fff
                                          const idx_t *const SFEM_RESTRICT elements,
                                          const void *const SFEM_RESTRICT  fff,
                                          const enum RealType              real_type_xy,
                                          const void *const                x,
                                          void *const                      y,
                                          void                            *stream) {
    int SFEM_ENABLE_TAYLOR_EXPANSION = 0;
    SFEM_READ_ENV(SFEM_ENABLE_TAYLOR_EXPANSION, atoi);

    if (SFEM_ENABLE_TAYLOR_EXPANSION) {
        return cu_affine_hex8_laplacian_taylor_apply(nelements, stride, elements, fff, real_type_xy, x, y, stream);
    }

    int SFEM_AFFINE_HEX8_TILED = 0;
    SFEM_READ_ENV(SFEM_AFFINE_HEX8_TILED, atoi);

    if (SFEM_AFFINE_HEX8_TILED) {
        switch (real_type_xy) {
            case SFEM_REAL_DEFAULT: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, stride, elements, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
            }
            case SFEM_FLOAT32: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, stride, elements, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
            }
            case SFEM_FLOAT64: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, stride, elements, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
            }
            default: {
                SFEM_ERROR("[Error] cu_hex8_laplacian_apply: not implemented for type %s (code %d)\n",
                           real_type_to_string(real_type_xy),
                           real_type_xy);
                return SFEM_FAILURE;
            }
        }
    }

    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_apply_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_apply_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_apply_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_hex8_laplacian_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type_xy),
                       real_type_xy);
            return SFEM_FAILURE;
        }
    }
}

#if 1

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_taylor_kernel(const ptrdiff_t nelements,
                                                             const ptrdiff_t stride,  // Stride for elements and fff
                                                             const idx_t *const SFEM_RESTRICT         elements,
                                                             const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                             const real_t *const SFEM_RESTRICT        u,
                                                             real_t *const SFEM_RESTRICT              values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * stride + e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * stride];
        }

        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v]];
        }

        cu_hex8_laplacian_apply_fff_taylor(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            atomicAdd(&values[dof_i], element_vector[edof_i]);
        }
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_taylor_tpl(const ptrdiff_t                  nelements,
                                                     const ptrdiff_t                  stride,  // Stride for elements and fff
                                                     const idx_t *const SFEM_RESTRICT elements,
                                                     const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                     const T *const                           x,
                                                     T *const                                 y,
                                                     void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    cu_hex8_taylor_expansion_init();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_laplacian_apply_taylor_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0>>>(nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

#else

#define HEX8_LAPLACE_TAYLOR_BLOCK_SIZE 512

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_taylor_kernel(const ptrdiff_t nelements,
                                                             const ptrdiff_t stride,  // Stride for elements and fff
                                                             const idx_t *const SFEM_RESTRICT elements,
                                                             const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                             const real_t *const SFEM_RESTRICT u,
                                                             real_t *const SFEM_RESTRICT values) {
    const int elements_per_block = blockDim.x / 8;
    const int node = threadIdx.x % 8;
    const int e_block = threadIdx.x / 8;
    const int offset = e_block * 8;

    __shared__ scalar_t element_u[HEX8_LAPLACE_TAYLOR_BLOCK_SIZE];
    scalar_t fff[6];

    for (ptrdiff_t e = blockIdx.x * elements_per_block + e_block; e < nelements; e += elements_per_block * gridDim.x) {
        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * stride];
        }

        const idx_t idx = elements[node * stride + e];

        scalar_t row[8];
        for (int j = 0; j < 8; j++) {
            cu_hex8_laplacian_matrix_ij_taylor(fff,
                                               // Trial
                                               hex8_g_0_x[j],
                                               hex8_g_0_y[j],
                                               hex8_g_0_z[j],
                                               hex8_H_0_x[j],
                                               hex8_H_0_y[j],
                                               hex8_H_0_z[j],
                                               hex8_diff3_0[j],
                                               // Test
                                               hex8_g_0_x[node],
                                               hex8_g_0_y[node],
                                               hex8_g_0_z[node],
                                               hex8_H_0_x[node],
                                               hex8_H_0_y[node],
                                               hex8_H_0_z[node],
                                               hex8_diff3_0[node],
                                               &row[j]);
        }

        element_u[threadIdx.x] = u[idx];

        __syncwarp();

        scalar_t thread_u[8];
        for (int j = 0; j < 8; j++) {
            // const int rot_idx = (j + node) % 8; // Avoid bank conflicts? This makes perf worse
            const int rot_idx = j;
            thread_u[rot_idx] = element_u[offset + rot_idx];
        }

        scalar_t result = 0;
        for (int j = 0; j < 8; j++) {
            result += row[j] * thread_u[j];
        }

        atomicAdd(&values[idx], result);

        __syncwarp();
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_taylor_tpl(const ptrdiff_t nelements,
                                                     const ptrdiff_t stride,  // Stride for elements and fff
                                                     const idx_t *const SFEM_RESTRICT elements,
                                                     const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                     const T *const x,
                                                     T *const y,
                                                     void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    cu_hex8_taylor_expansion_init();

    // Hand tuned
    int block_size = HEX8_LAPLACE_TAYLOR_BLOCK_SIZE;
    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements * 8 + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0>>>(nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

#endif

extern int cu_affine_hex8_laplacian_taylor_apply(const ptrdiff_t                  nelements,
                                                 const ptrdiff_t                  stride,  // Stride for elements and fff
                                                 const idx_t *const SFEM_RESTRICT elements,
                                                 const void *const SFEM_RESTRICT  fff,
                                                 const enum RealType              real_type_xy,
                                                 const void *const                x,
                                                 void *const                      y,
                                                 void                            *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_hex8_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_hex8_laplacian_crs_sym_kernel(const ptrdiff_t                  nelements,
                                                        const ptrdiff_t                  stride,  // Stride for elements and fff
                                                        const idx_t *const SFEM_RESTRICT elements,
                                                        const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                        const count_t *const SFEM_RESTRICT       rowptr,
                                                        const idx_t *const SFEM_RESTRICT         colidx,
                                                        T *const SFEM_RESTRICT                   diag,
                                                        T *const SFEM_RESTRICT                   offdiag) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t    ev[8];
        scalar_t fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * stride + e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * stride];
        }

        T element_matrix[8 * 8];
        cu_hex8_laplacian_matrix_fff_integral(fff, element_matrix);

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            atomicAdd(&diag[ev[edof_i]], element_matrix[edof_i * 8 + edof_i]);
        }

        // Assemble the upper-triangular part of the matrix
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            // For each row we find the corresponding entries in the off-diag
            // We select the entries associated with ev[row] < ev[col]
            const int    lenrow = rowptr[ev[edof_i] + 1] - rowptr[ev[edof_i]];
            const idx_t *cols   = &colidx[rowptr[ev[edof_i]]];
            // Find the columns associated with the current row and mask what is not found with
            // -1
            idx_t ks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            for (int i = 0; i < lenrow; i++) {
                for (int k = 0; k < 8; k++) {
                    if (cols[i] == ev[k]) {
                        ks[k] = i;
                        break;
                    }
                }
            }

            for (int edof_j = 0; edof_j < 8; edof_j++) {
                if (ev[edof_j] > ev[edof_i]) {
                    assert(ks[edof_j] != SFEM_IDX_INVALID);
                    atomicAdd(&offdiag[rowptr[ev[edof_i]] + ks[edof_j]], element_matrix[edof_i * 8 + edof_j]);
                }
            }
        }
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_crs_sym_tpl(const ptrdiff_t                          nelements,
                                                const ptrdiff_t                          stride,  // Stride for elements and fff
                                                const idx_t *const SFEM_RESTRICT         elements,
                                                const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                const count_t *const SFEM_RESTRICT       rowptr,
                                                const idx_t *const SFEM_RESTRICT         colidx,
                                                T *const SFEM_RESTRICT                   diag,
                                                T *const SFEM_RESTRICT                   offdiag,
                                                void                                    *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_laplacian_crs_sym_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_crs_sym_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, rowptr, colidx, diag, offdiag);
    } else {
        cu_affine_hex8_laplacian_crs_sym_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, rowptr, colidx, diag, offdiag);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_crs_sym(const ptrdiff_t                    nelements,
                                            const ptrdiff_t                    stride,  // Stride for elements and fff
                                            const idx_t *const SFEM_RESTRICT   elements,
                                            const void *const SFEM_RESTRICT    fff,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT   colidx,
                                            const enum RealType                real_type,
                                            void *const SFEM_RESTRICT          diag,
                                            void *const SFEM_RESTRICT          offdiag,
                                            void                              *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, rowptr, colidx, (real_t *)diag, (real_t *)offdiag, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, rowptr, colidx, (float *)diag, (float *)offdiag, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, rowptr, colidx, (double *)diag, (double *)offdiag, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_hex8_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}