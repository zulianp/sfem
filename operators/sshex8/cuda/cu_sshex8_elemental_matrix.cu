#include "cu_sshex8_elemental_matrix.h"

#include "cu_sshex8_inline.hpp"
#include "sfem_cuda_base.h"

template <typename T>
__global__ void cu_affine_hex8_elemental_matrix_apply_kernel(const ptrdiff_t              nelements,
                                                             idx_t **const SFEM_RESTRICT  elements,
                                                             T **const SFEM_RESTRICT      elemental_matrix,
                                                             const T *const SFEM_RESTRICT x,
                                                             T *const SFEM_RESTRICT       y) {
    idx_t ev[8];
    T     mat[8 * 8];
    T     in[8], out[8];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int v = 0; v < 8; v++) {
            ev[v] = elements[v][e];
        }

        for (int v = 0; v < 64; v++) {
            mat[v] = elemental_matrix[v][e];
        }

        for (int v = 0; v < 8; v++) {
            in[v] = x[ev[v]];
        }

        for (int v = 0; v < 8; v++) {
            out[v] = 0;
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                out[i] += mat[i * 8 + j] * in[j];
            }
        }

        for (int v = 0; v < 8; v++) {
            atomicAdd(&y[ev[v]], out[v]);
        }
    }
}

template <int BLOCK_SIZE, typename T>
__global__ void cu_affine_hex8_elemental_matrix_apply_kernel_warp(const ptrdiff_t              nelements,
                                                                  idx_t **const SFEM_RESTRICT  elements,
                                                                  T **const SFEM_RESTRICT      elemental_matrix,
                                                                  const T *const SFEM_RESTRICT x,
                                                                  T *const SFEM_RESTRICT       y) {
    // Shared memory
    __shared__ T in[BLOCK_SIZE];

    // Registers
    T row[8];
    T coeffs[8];

    // 8 threads per element
    const ptrdiff_t start    = blockIdx.x * blockDim.x / 8;
    const int       tile     = threadIdx.x / 8;
    const int       tile_idx = threadIdx.x % 8;

    const ptrdiff_t nelements_padded = (nelements / 8) * 8;
    const ptrdiff_t stride           = blockDim.x * gridDim.x / 8;

    for (ptrdiff_t e = start + tile; e < nelements_padded; e += stride) {
        const bool exists = e < nelements;

        __syncwarp();

        idx_t lidx = SFEM_IDX_INVALID;
        if (exists) {
            // Copy element indices from global to shared mem
            lidx = elements[tile_idx][e];

            // Copy row of elemental matrix from global to possibly "registers"
            // Assume symmetry
            for (int v = 0; v < 8; v++) {
                assert(tile_idx + v * 8 < 64);
                row[v] = elemental_matrix[tile_idx + v * 8][e];
            }

            in[threadIdx.x] = x[lidx];
        }

        __syncwarp();

        if (exists) {
            T out = 0;

            // Round-robin to reduce bank conflicts
            for (int v = 0; v < 8; v++) {
                int circular     = (tile_idx + v) % 8;
                assert(tile * 8 + circular < BLOCK_SIZE);
                coeffs[circular] = in[tile * 8 + circular];
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                out += row[j] * coeffs[j];
            }

            assert(SFEM_IDX_INVALID != lidx);
            atomicAdd(&y[lidx], out);
        }
    }
}

template <typename T>
int cu_affine_hex8_elemental_matrix_apply_tpl(const ptrdiff_t              nelements,
                                              idx_t **const SFEM_RESTRICT  elements,
                                              T **const SFEM_RESTRICT      elemental_matrix,
                                              const T *const SFEM_RESTRICT x,
                                              T *const SFEM_RESTRICT       y,
                                              void                        *stream) {
    int SFEM_AFFINE_HEX8_ELEMENTAL_MATRIX_WARP = 0;
    SFEM_READ_ENV(SFEM_AFFINE_HEX8_ELEMENTAL_MATRIX_WARP, atoi);

    if (SFEM_AFFINE_HEX8_ELEMENTAL_MATRIX_WARP) {
        static const int block_size = 128;
        const ptrdiff_t  n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size * 8 - 1) / (block_size * 8));

        if (stream) {
            cudaStream_t s = *static_cast<cudaStream_t *>(stream);
            cu_affine_hex8_elemental_matrix_apply_kernel_warp<block_size, T>
                    <<<n_blocks, block_size, 0, s>>>(nelements, elements, elemental_matrix, x, y);
        } else {
            cu_affine_hex8_elemental_matrix_apply_kernel_warp<block_size, T>
                    <<<n_blocks, block_size, 0>>>(nelements, elements, elemental_matrix, x, y);
        }

    } else {
        static const int block_size = 128;
        const ptrdiff_t  n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

        if (stream) {
            cudaStream_t s = *static_cast<cudaStream_t *>(stream);
            cu_affine_hex8_elemental_matrix_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                    nelements, elements, elemental_matrix, x, y);
        } else {
            cu_affine_hex8_elemental_matrix_apply_kernel<<<n_blocks, block_size, 0>>>(
                    nelements, elements, elemental_matrix, x, y);
        }
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_elemental_matrix_apply(const ptrdiff_t                 nelements,
                                                 idx_t **const SFEM_RESTRICT     elements,
                                                 const enum RealType             real_type,
                                                 void **const SFEM_RESTRICT      elemental_matrix,
                                                 const void *const SFEM_RESTRICT x,
                                                 void *const SFEM_RESTRICT       y,
                                                 void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_elemental_matrix_apply_tpl<real_t>(
                    nelements, elements, (real_t **)elemental_matrix, (const real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_elemental_matrix_apply_tpl<float>(
                    nelements, elements, (float **)elemental_matrix, (const float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_elemental_matrix_apply_tpl<double>(
                    nelements, elements, (double **)elemental_matrix, (const double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_affine_hex8_elemental_matrix_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type),
                       real_type);
            return SFEM_FAILURE;
        }
    }
}

////////////////////////

template <typename T>
__global__ void cu_affine_sshex8_elemental_matrix_apply_kernel(const int                    level,
                                                               const ptrdiff_t              nelements,
                                                               idx_t **const SFEM_RESTRICT  elements,
                                                               T **const SFEM_RESTRICT      elemental_matrix,
                                                               const T *const SFEM_RESTRICT x,
                                                               T *const SFEM_RESTRICT       y) {
    T mat[8 * 8];
    T in[8], out[8];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int v = 0; v < 64; v++) {
            mat[v] = elemental_matrix[v][e];
        }

        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {
                    int ev[8] = {// Bottom
                                 elements[cu_sshex8_lidx(level, xi, yi, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi, yi + 1, zi)][e],
                                 // Top
                                 elements[cu_sshex8_lidx(level, xi, yi, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi, yi + 1, zi + 1)][e]};

                    for (int v = 0; v < 8; v++) {
                        in[v] = x[ev[v]];
                    }

                    for (int v = 0; v < 8; v++) {
                        out[v] = 0;
                    }

                    for (int i = 0; i < 8; i++) {
                        const auto SFEM_RESTRICT mi = &mat[i * 8];
                        for (int j = 0; j < 8; j++) {
                            out[i] += mi[j] * in[j];
                        }
                    }

                    for (int v = 0; v < 8; v++) {
                        atomicAdd(&y[ev[v]], out[v]);
                    }
                }
            }
        }
    }
}

///////////////////////////////

// template <int BLOCK_SIZE, typename T>
// __global__ void cu_affine_sshex8_elemental_matrix_apply_kernel_warp(const int                    level,
//                                                                     const ptrdiff_t              nelements,
//                                                                     idx_t **const SFEM_RESTRICT  elements,
//                                                                     T **const SFEM_RESTRICT      elemental_matrix,
//                                                                     const T *const SFEM_RESTRICT x,
//                                                                     T *const SFEM_RESTRICT       y) {
//     extern __shared__ unsigned char shared_mem[];

//     static const int n_matrix_elements    = 64;
//     static const int n_warps_per_element  = n_matrix_elements / SFEM_WARP_SIZE;
//     static const int n_elements_per_block = BLOCK_SIZE / n_matrix_elements;

//     const int warp_id = blockDim.x / SFEM_WARP_SIZE;

//     const int       cu_id         = threadIdx.x % (n_warps_per_element * SFEM_WARP_SIZE);
//     const int       block_element = warp_id / n_warps_per_element;
//     const ptrdiff_t e             = blockIdx.x * n_elements_per_block + block_element;

//     const bool exists   = e < nelements;
//     const int  mat_lidx = threadIdx.x % n_matrix_elements;
//     const int  nxe      = cu_sshex8_nxe(level);

//     // Grab shared mem
//     T   *emat = &((T *)shared_mem)[block_element * n_matrix_elements];
//     T   *ex = &((T *)shared_mem)[[n_elements_per_block * n_matrix_elements + block_element * nxe];
//     T   *ey = &ex[nxe];

//     idx_t *idx = ((T *)shared_mem)[n_elements_per_block * (n_matrix_elements + 2 * nxe)];

//     // Global to shared mem
//     if (exists)
//     {
//         // Matrix coefficients
//         emat[mat_lidx] = elemental_matrix[mat_lidx][e];

//         // Elemental indices
//         for (int offset = cu_id; offset < nxe; offset += n_warps_per_element * SFEM_WARP_SIZE) {
//             idx[offset] = elements[offset][e];
//         }
//     }

//     __syncthreads();

//     for (int zi = 0; zi < level; zi++) {
//         for (int yi = 0; yi < level; yi++) {
//             for (int xi = 0; xi < level; xi++) {
//                 int ev[8] = {// Bottom
//                              elements[cu_sshex8_lidx(level, xi, yi, zi)][e],
//                              elements[cu_sshex8_lidx(level, xi + 1, yi, zi)][e],
//                              elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi)][e],
//                              elements[cu_sshex8_lidx(level, xi, yi + 1, zi)][e],
//                              // Top
//                              elements[cu_sshex8_lidx(level, xi, yi, zi + 1)][e],
//                              elements[cu_sshex8_lidx(level, xi + 1, yi, zi + 1)][e],
//                              elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi + 1)][e],
//                              elements[cu_sshex8_lidx(level, xi, yi + 1, zi + 1)][e]};

//                 for (int v = 0; v < 8; v++) {
//                     in[v] = x[ev[v]];
//                 }

//                 for (int v = 0; v < 8; v++) {
//                     out[v] = 0;
//                 }

//                 for (int i = 0; i < 8; i++) {
//                     const auto SFEM_RESTRICT mi = &mat[i * 8];
//                     for (int j = 0; j < 8; j++) {
//                         out[i] += mi[j] * in[j];
//                     }
//                 }

//                 for (int v = 0; v < 8; v++) {
//                     atomicAdd(&y[ev[v]], out[v]);
//                 }
//             }
//         }
//     }
// }

/////////////////////////////////

template <typename T>
int cu_affine_sshex8_elemental_matrix_apply_tpl(const int                    level,
                                                const ptrdiff_t              nelements,
                                                idx_t **const SFEM_RESTRICT  elements,
                                                T **const SFEM_RESTRICT      elemental_matrix,
                                                const T *const SFEM_RESTRICT x,
                                                T *const SFEM_RESTRICT       y,
                                                void                        *stream) {
    // Hand tuned
    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_elemental_matrix_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                level, nelements, elements, elemental_matrix, x, y);
    } else {
        cu_affine_sshex8_elemental_matrix_apply_kernel<<<n_blocks, block_size, 0>>>(
                level, nelements, elements, elemental_matrix, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_sshex8_elemental_matrix_apply(const int                       level,
                                                   const ptrdiff_t                 nelements,
                                                   idx_t **const SFEM_RESTRICT     elements,
                                                   const enum RealType             real_type,
                                                   void **const SFEM_RESTRICT      elemental_matrix,
                                                   const void *const SFEM_RESTRICT x,
                                                   void *const SFEM_RESTRICT       y,
                                                   void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_elemental_matrix_apply_tpl<real_t>(
                    level, nelements, elements, (real_t **)elemental_matrix, (const real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_elemental_matrix_apply_tpl<float>(
                    level, nelements, elements, (float **)elemental_matrix, (const float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_elemental_matrix_apply_tpl<double>(
                    level, nelements, elements, (double **)elemental_matrix, (const double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_affine_sshex8_elemental_matrix_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type),
                       real_type);
            return SFEM_FAILURE;
        }
    }
}
