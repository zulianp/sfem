#include "cu_hex8_laplacian.h"

#include "cu_hex8_laplacian_inline.hpp"
#include "sfem_cuda_base.h"

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

template <typename T, int block_size>
__global__ void cu_affine_hex8_laplacian_apply_tiled_kernel(const ptrdiff_t             nelements,
                                                            idx_t **const SFEM_RESTRICT elements,
                                                            const ptrdiff_t fff_stride,  // Stride for elements and fff
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                            const T *const SFEM_RESTRICT             u,
                                                            T *const SFEM_RESTRICT                   values) {
    // Tile number in group
    const int tile    = threadIdx.x >> 3;   // same as threadIdx.x / 8
    const int n_tiles = block_size >> 3;    // same as block_size / 8
    const int sub_idx = threadIdx.x & 0x7;  // same as threadIdx.x % 8

    extern __shared__ unsigned char shared_mem[];

    const ptrdiff_t offset       = tile * 8 * sizeof(T);
    const ptrdiff_t block_offset = sizeof(T) * block_size;

    // !! Bank conflicts if T=double
    T *element_u = (T *)&shared_mem[offset];
    T *gx        = (T *)&shared_mem[offset + block_offset];
    T *gy        = (T *)&shared_mem[offset + block_offset * 2];
    T *gz        = (T *)&shared_mem[offset + block_offset * 3];

    idx_t         *ev    = (idx_t *)&shared_mem[block_offset * 4];
    cu_jacobian_t *s_fff = (cu_jacobian_t *)&shared_mem[block_offset * 4 + block_size * 8 * sizeof(idx_t)];

    {  // Coalesced read from global memory
        ptrdiff_t e_coalesced = blockIdx.x * block_size + threadIdx.x;
        if (e_coalesced < nelements) {
            // TODO transpose to correct shmem layout
            for (int v = 0; v < 8; v++) {
                ev[v * block_size + threadIdx.x] = elements[v][e_coalesced];
            }

            // ???
            for (int v = 0; v < 6; v++) {
                s_fff[v * block_size + threadIdx.x] = g_fff[v * fff_stride + e_coalesced];
            }
        }
#ifndef NDEBUG
        else {
            for (int v = 0; v < 8; v++) {
                ev[v * block_size + threadIdx.x] = -1;
            }

            for (int v = 0; v < 6; v++) {
                s_fff[v * block_size + threadIdx.x] = 0;
            }
        }
#endif
    }

    const int hex8_vidx =
            block_size *
            ((sub_idx == 3 || sub_idx == 7) ? (sub_idx - 1) : ((sub_idx == 2 || sub_idx == 6) ? (sub_idx + 1) : sub_idx));

    const T D[2 * 2] = {-1, 1, -1, 1};
    const T qw[2]    = {0.5, 0.5};

    // Gather from global mem to shared mem

    // hex8 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2
    const int zi = (sub_idx >> 2);        // equivalent to sub_idx / 4

    __syncthreads();

    for (int pack = 0; pack < 8; pack++) {
        const ptrdiff_t e = pack * n_tiles + tile;

        // Reorder HEX8 to cartesian
        const int vidx = e + hex8_vidx;

        bool not_esists_tile = blockIdx.x * block_size + e >= nelements;
        // bool not_esists_warp = __ballot_sync(SFEM_WARP_FULL_MASK, (blockIdx.x * block_size + e >= nelements));
        // if (not_esists_warp || not_esists_tile) continue;

        if (not_esists_tile) continue;

        // read from shared mem
        const ptrdiff_t gidx = ev[vidx];

        // FIX using shuffles?
        T fff[6];
        for (int d = 0; d < 6; d++) {
            fff[d] = s_fff[d * block_size + e];
        }

        element_u[sub_idx] = u[gidx];
        const T acc        = cu_spectral_hex_laplacian_apply_tiled<2, T>(xi, yi, zi, D, fff, qw, element_u, gx, gy, gz);

        // Scatter from shared mem to global mem
        atomicAdd(&values[gidx], acc);
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_tiled_tpl(const ptrdiff_t                          nelements,
                                                    idx_t **const SFEM_RESTRICT              elements,
                                                    const ptrdiff_t                          fff_stride,
                                                    const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                    const T *const                           x,
                                                    T *const                                 y,
                                                    void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    static const int block_size = 128;
    size_t           shmem_size = block_size * (sizeof(T) * 4 + sizeof(idx_t) * 8 + 6 * sizeof(cu_jacobian_t));
    ptrdiff_t        n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size / 8 - 1) / (block_size / 8));

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_tiled_kernel<T, block_size>
                <<<n_blocks, block_size, shmem_size, s>>>(nelements, elements, fff_stride, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_tiled_kernel<T, block_size>
                <<<n_blocks, block_size, shmem_size>>>(nelements, elements, fff_stride, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_kernel(const ptrdiff_t                          nelements,
                                                      idx_t **const SFEM_RESTRICT              elements,
                                                      const ptrdiff_t                          fff_stride,
                                                      const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                      const real_t *const SFEM_RESTRICT        u,
                                                      real_t *const SFEM_RESTRICT              values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * fff_stride + e];
        }

        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v]];
        }

#if 1
        cu_hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);
#else
        // Higher numerical error in MG galerkin test
        cu_hex8_laplacian_apply_add_fff_trick(fff, element_u, element_vector);
#endif

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            atomicAdd(&values[dof_i], element_vector[edof_i]);
        }
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_tpl(const ptrdiff_t                          nelements,
                                              idx_t **const SFEM_RESTRICT              elements,
                                              const ptrdiff_t                          fff_stride,
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
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(nelements, elements, fff_stride, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(nelements, elements, fff_stride, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_apply(const ptrdiff_t                 nelements,
                                          idx_t **const SFEM_RESTRICT     elements,
                                          const ptrdiff_t                 fff_stride,
                                          const void *const SFEM_RESTRICT fff,
                                          const enum RealType             real_type_xy,
                                          const void *const               x,
                                          void *const                     y,
                                          void                           *stream) {
    int SFEM_ENABLE_TAYLOR_EXPANSION = 0;
    SFEM_READ_ENV(SFEM_ENABLE_TAYLOR_EXPANSION, atoi);

    if (SFEM_ENABLE_TAYLOR_EXPANSION) {
        return cu_affine_hex8_laplacian_taylor_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
    }

    int SFEM_AFFINE_HEX8_TILED = 0;
    SFEM_READ_ENV(SFEM_AFFINE_HEX8_TILED, atoi);

    if (SFEM_AFFINE_HEX8_TILED) {
        // This is slower than the other variant
        switch (real_type_xy) {
            case SFEM_REAL_DEFAULT: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
            }
            case SFEM_FLOAT32: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
            }
            case SFEM_FLOAT64: {
                return cu_affine_hex8_laplacian_apply_tiled_tpl(
                        nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
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
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_apply_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_apply_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_hex8_laplacian_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type_xy),
                       real_type_xy);
            return SFEM_FAILURE;
        }
    }
}

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_taylor_kernel(const ptrdiff_t                          nelements,
                                                             idx_t **const SFEM_RESTRICT              elements,
                                                             const ptrdiff_t                          fff_stride,
                                                             const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                             const real_t *const SFEM_RESTRICT        u,
                                                             real_t *const SFEM_RESTRICT              values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * fff_stride];
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
static int cu_affine_hex8_laplacian_apply_taylor_tpl(const ptrdiff_t                          nelements,
                                                     idx_t **const SFEM_RESTRICT              elements,
                                                     const ptrdiff_t                          fff_stride,
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
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0, s>>>(nelements, elements, fff_stride, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_taylor_kernel<<<n_blocks, block_size, 0>>>(nelements, elements, fff_stride, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_taylor_apply(const ptrdiff_t                 nelements,
                                                 idx_t **const SFEM_RESTRICT     elements,
                                                 const ptrdiff_t                 fff_stride,
                                                 const void *const SFEM_RESTRICT fff,
                                                 const enum RealType             real_type_xy,
                                                 const void *const               x,
                                                 void *const                     y,
                                                 void                           *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_apply_taylor_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_hex8_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_hex8_laplacian_crs_sym_kernel(const ptrdiff_t                          nelements,
                                                        idx_t **const SFEM_RESTRICT              elements,
                                                        const ptrdiff_t                          fff_stride,
                                                        const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                        const count_t *const SFEM_RESTRICT       rowptr,
                                                        const idx_t *const SFEM_RESTRICT              colidx,
                                                        T *const SFEM_RESTRICT                   diag,
                                                        T *const SFEM_RESTRICT                   offdiag) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t    ev[8];
        scalar_t fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * fff_stride];
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
                                                idx_t **const SFEM_RESTRICT              elements,
                                                const ptrdiff_t                          fff_stride,
                                                const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                const count_t *const SFEM_RESTRICT       rowptr,
                                                const idx_t *const SFEM_RESTRICT              colidx,
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
                nelements, elements, fff_stride, fff, rowptr, colidx, diag, offdiag);
    } else {
        cu_affine_hex8_laplacian_crs_sym_kernel<<<n_blocks, block_size, 0>>>(
                nelements, elements, fff_stride, fff, rowptr, colidx, diag, offdiag);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_crs_sym(const ptrdiff_t                    nelements,
                                            idx_t **const SFEM_RESTRICT        elements,
                                            const ptrdiff_t                    fff_stride,
                                            const void *const SFEM_RESTRICT    fff,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT        colidx,
                                            const enum RealType                real_type,
                                            void *const SFEM_RESTRICT          diag,
                                            void *const SFEM_RESTRICT          offdiag,
                                            void                              *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(nelements,
                                                        elements,
                                                        fff_stride,
                                                        (cu_jacobian_t *)fff,
                                                        rowptr,
                                                        colidx,
                                                        (real_t *)diag,
                                                        (real_t *)offdiag,
                                                        stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(nelements,
                                                        elements,
                                                        fff_stride,
                                                        (cu_jacobian_t *)fff,
                                                        rowptr,
                                                        colidx,
                                                        (float *)diag,
                                                        (float *)offdiag,
                                                        stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_crs_sym_tpl(nelements,
                                                        elements,
                                                        fff_stride,
                                                        (cu_jacobian_t *)fff,
                                                        rowptr,
                                                        colidx,
                                                        (double *)diag,
                                                        (double *)offdiag,
                                                        stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_hex8_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}