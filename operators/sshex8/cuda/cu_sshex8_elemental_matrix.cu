#include "cu_sshex8_elemental_matrix.h"

#include "cu_sshex8_inline.hpp"
#include "sfem_cuda_base.h"

#if 0

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

#else

template <typename T>
__global__ void cu_affine_hex8_elemental_matrix_apply_kernel(const ptrdiff_t              nelements,
                                                             idx_t **const SFEM_RESTRICT  elements,
                                                             T **const SFEM_RESTRICT      elemental_matrix,
                                                             const T *const SFEM_RESTRICT x,
                                                             T *const SFEM_RESTRICT       y) {
    const ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nelements) return;

    idx_t ev[8];
    for (int v = 0; v < 8; v++) {
        ev[v] = elements[v][e];
    }

    for (int i = 0; i < 8; i++) {
        T out = 0;
        for (int j = 0; j < 8; j++) {
            out += elemental_matrix[i * 8 + j][e] * x[ev[j]];
        }

        atomicAdd(&y[ev[i]], out);
    }
}

#endif

// template <int BLOCK_SIZE, typename T>
// __global__ void cu_affine_hex8_elemental_matrix_apply_kernel_warp(const ptrdiff_t              nelements,
//                                                                   idx_t **const SFEM_RESTRICT  elements,
//                                                                   T **const SFEM_RESTRICT      elemental_matrix,
//                                                                   const T *const SFEM_RESTRICT x,
//                                                                   T *const SFEM_RESTRICT       y) {
//     // Shared memory
//     __shared__ T in[BLOCK_SIZE];
//     __shared__ idx_t ev[BLOCK_SIZE];

//     // Registers
//     T row[8];
//     T coeffs[8];

//     // 8 threads per element
//     const ptrdiff_t start    = blockIdx.x * blockDim.x / 8;
//     const int       tile     = threadIdx.x / 8;
//     const int       tile_idx = threadIdx.x % 8;
//     const int       tilex8   = tile * 8;

//     const ptrdiff_t nelements_padded = (nelements / 8) * 8;
//     const ptrdiff_t stride           = blockDim.x * gridDim.x / 8;

//     for (ptrdiff_t e = start + tile; e < nelements_padded; e += stride) {
//         const bool exists = e < nelements;

//         __syncwarp();

//         idx_t lidx = SFEM_IDX_INVALID;
//         if (exists) {
//             // Copy element indices from global to shared mem
//             lidx = elements[tile_idx][e];

//             // Copy row of elemental matrix from global to possibly "registers"
//             // Assume symmetry
//             for (int v = 0; v < 8; v++) {
//                 assert(tile_idx + v * 8 < 64);
//                 row[v] = elemental_matrix[tile_idx + v * 8][e];
//             }

//             in[threadIdx.x] = x[lidx];
//         }

//         __syncwarp();

//         if (exists) {
//             T out = 0;

//             // Round-robin to reduce bank conflicts
//             for (int v = 0; v < 8; v++) {
//                 const int circular = (tile_idx + v) % 8;
//                 assert(tilex8 + circular < BLOCK_SIZE);
//                 coeffs[circular] = in[tilex8 + circular];
//             }

// #pragma unroll
//             for (int j = 0; j < 8; j++) {
//                 out += row[j] * coeffs[j];
//             }

//             assert(SFEM_IDX_INVALID != lidx);
//             atomicAdd(&y[lidx], out);
//         }
//     }
// }

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
    const ptrdiff_t start    = (blockIdx.x * blockDim.x) >> 3;
    const int       tile     = threadIdx.x >> 3;
    const int       tile_idx = threadIdx.x & 0x7;
    const int       tilex8   = tile << 3;

    const ptrdiff_t nelements_padded = (nelements / 8) * 8;
    const ptrdiff_t stride           = blockDim.x * gridDim.x / 8;

    const idx_t *const SFEM_RESTRICT ii = elements[tile_idx];

    for (ptrdiff_t e = start + tile; e < nelements_padded; e += stride) {
        const bool exists = e < nelements;

        __syncwarp();

        idx_t lidx = SFEM_IDX_INVALID;
        if (exists) {
            // Copy element indices from global to shared mem
            lidx = ii[e];

            // Copy row of elemental matrix from global to possibly "registers"
            // Assume symmetry
            for (int v = 0; v < 8; v++) {
                assert(tile_idx + v * 8 < 64);
                // row[v] = elemental_matrix[tile_idx + v * 8][e];
                row[v] = elemental_matrix[tile_idx * 8 + v][e];
            }

            in[threadIdx.x] = x[lidx];
        }

        __syncwarp();

        if (exists) {
            // Round-robin to reduce bank conflicts (only works well for fp32)
            for (int v = 0; v < 8; v++) {
                const int circular = (tile_idx + v) % 8;
                assert(tilex8 + circular < BLOCK_SIZE);
                coeffs[circular] = in[tilex8 + circular];
            }

            T out = 0;
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

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_elemental_matrix_apply_kernel_warp(const ptrdiff_t              nelements,
                                                                    idx_t **const SFEM_RESTRICT  elements,
                                                                    T **const SFEM_RESTRICT      elemental_matrix,
                                                                    const T *const SFEM_RESTRICT x,
                                                                    T *const SFEM_RESTRICT       y) {
    static const int BLOCK_SIZE   = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    assert(blockDim.x == BLOCK_SIZE);
    assert(blockDim.y == BLOCK_SIZE);
    assert(blockDim.z == BLOCK_SIZE);

    __shared__ T x_block[BLOCK_SIZE_3];
    __shared__ T y_block[BLOCK_SIZE_3];
    __shared__ T emat[64];

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const int lidx = threadIdx.z * BLOCK_SIZE_2 + threadIdx.y * BLOCK_SIZE + threadIdx.x;

        if (lidx < 64) {
            emat[lidx] = elemental_matrix[lidx][e];
        }

        const ptrdiff_t idx = elements[lidx][e];

        x_block[lidx] = x[idx];  // Copy coeffs to shared mem
        y_block[lidx] = 0;       // Reset

        const bool is_element = threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        T element_vector[8] = {0};
        if (is_element) {
            // gather
            T element_u[8] = {x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)]};

            for (int i = 0; i < 8; i++) {
                // Shared mem Broadcast (should be ok)
                const T *const row = &emat[i * 8];
                const T        ui  = element_u[i];
                assert(ui == ui);
                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }

            // TODO With stencil version atomics can be avoided
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)], element_vector[0]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)], element_vector[1]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)], element_vector[2]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)], element_vector[3]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)], element_vector[4]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)], element_vector[5]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)], element_vector[6]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)], element_vector[7]);
        }

        const int interior = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0 && threadIdx.x < LEVEL &&
                             threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        if (interior)
            y[idx] += y_block[lidx];
        else
            atomicAdd(&y[idx], y_block[lidx]);
    }
}

template <typename T, int LEVEL>
int cu_affine_sshex8_elemental_matrix_apply_warp_tpl(const ptrdiff_t              nelements,
                                                     idx_t **const SFEM_RESTRICT  elements,
                                                     T **const SFEM_RESTRICT      elemental_matrix,
                                                     const T *const SFEM_RESTRICT x,
                                                     T *const SFEM_RESTRICT       y,
                                                     void                        *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    static const int BLOCK_SIZE = LEVEL + 1;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_elemental_matrix_apply_kernel_warp<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements, elements, elemental_matrix, x, y);
    } else {
        cu_affine_sshex8_elemental_matrix_apply_kernel_warp<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements, elements, elemental_matrix, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

/////////////////////////////////

template <typename T>
int cu_affine_sshex8_elemental_matrix_apply_tpl(const int                    level,
                                                const ptrdiff_t              nelements,
                                                idx_t **const SFEM_RESTRICT  elements,
                                                T **const SFEM_RESTRICT      elemental_matrix,
                                                const T *const SFEM_RESTRICT x,
                                                T *const SFEM_RESTRICT       y,
                                                void                        *stream) {
    int SFEM_HEX8_SHARED_MEM_KERNEL = 1;
    SFEM_READ_ENV(SFEM_HEX8_SHARED_MEM_KERNEL, atoi);

    if (SFEM_HEX8_SHARED_MEM_KERNEL) {
        switch (level) {
            case 4: {
                return cu_affine_sshex8_elemental_matrix_apply_warp_tpl<T, 4>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            case 6: {
                return cu_affine_sshex8_elemental_matrix_apply_warp_tpl<T, 6>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            case 8: {
                return cu_affine_sshex8_elemental_matrix_apply_warp_tpl<T, 8>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            default:
                break;
        }
    }

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

#define USE_CARTESIAN_ORDERING

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_warp(const ptrdiff_t                  nelements,
                                                                        const idx_t *const SFEM_RESTRICT elements,
                                                                        const T *const SFEM_RESTRICT     elemental_matrix,
                                                                        const T *const SFEM_RESTRICT     x,
                                                                        T *const SFEM_RESTRICT           y) {
    static const int BLOCK_SIZE   = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    assert(blockDim.x == BLOCK_SIZE);
    assert(blockDim.y == BLOCK_SIZE);
    assert(blockDim.z == BLOCK_SIZE);

    __shared__ T x_block[BLOCK_SIZE_3];
    __shared__ T y_block[BLOCK_SIZE_3];
    __shared__ T emat[64];

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const int lidx = threadIdx.z * BLOCK_SIZE_2 + threadIdx.y * BLOCK_SIZE + threadIdx.x;

        if (lidx < 64) {
            emat[lidx] = elemental_matrix[e * 64 + lidx];
        }

        const ptrdiff_t idx = elements[e * BLOCK_SIZE_3 + lidx];

        x_block[lidx] = x[idx];  // Copy coeffs to shared mem
        y_block[lidx] = 0;       // Reset

        const bool is_element = threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        T element_vector[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (is_element) {
            // gather

#ifndef USE_CARTESIAN_ORDERING
            // T element_u[8] = {x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
            //                   x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)]};
#else
            T element_u[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int x_idx = (i & 1) ? threadIdx.x + 1 : threadIdx.x;
                const int y_idx = (i & 2) ? threadIdx.y + 1 : threadIdx.y;
                const int z_idx = (i & 4) ? threadIdx.z + 1 : threadIdx.z;
                element_u[i]    = x_block[cu_sshex8_lidx(LEVEL, x_idx, y_idx, z_idx)];
            }
#endif

            for (int i = 0; i < 8; i++) {
                // Shared mem Broadcast (should be ok)
                const T *const row = &emat[i * 8];
                const T        ui  = element_u[i];
                assert(ui == ui);
                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }

#ifndef USE_CARTESIAN_ORDERING
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)], element_vector[0]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)], element_vector[1]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)], element_vector[2]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)], element_vector[3]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)], element_vector[4]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)], element_vector[5]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)], element_vector[6]);
            // atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)], element_vector[7]);
#else
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int x_idx = (i & 1) ? threadIdx.x + 1 : threadIdx.x;
                const int y_idx = (i & 2) ? threadIdx.y + 1 : threadIdx.y;
                const int z_idx = (i & 4) ? threadIdx.z + 1 : threadIdx.z;
                atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, x_idx, y_idx, z_idx)], element_vector[i]);
            }
#endif
        }

        const int interior = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0 && threadIdx.x < LEVEL &&
                             threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();

        if (interior)
            y[idx] += y_block[lidx];
        else
            atomicAdd(&y[idx], y_block[lidx]);
    }
}

template <typename T, int LEVEL>
int cu_affine_sshex8_elemental_matrix_apply_AoS_warp_tpl(const ptrdiff_t                  nelements,
                                                         const idx_t *const SFEM_RESTRICT elements,
                                                         const T *const SFEM_RESTRICT     elemental_matrix,
                                                         const T *const SFEM_RESTRICT     x,
                                                         T *const SFEM_RESTRICT           y,
                                                         void                            *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    static const int BLOCK_SIZE = LEVEL + 1;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_warp<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements, elements, elemental_matrix, x, y);
    } else {
        cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_warp<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements, elements, elemental_matrix, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_TC(const ptrdiff_t                  nelements,
                                                                      const idx_t *const SFEM_RESTRICT elements,
                                                                      const T *const SFEM_RESTRICT     elemental_matrix,
                                                                      const T *const SFEM_RESTRICT     x,
                                                                      T *const SFEM_RESTRICT           y) {
    assert(blockDim.x == 4);  // 4 shape function in the x direction
    assert(blockDim.y == 8);  // 8 elements in the y directions
    assert(blockDim.x * blockDim.y * blockDim.z >= 64);

    static const int BLOCK_SIZE       = LEVEL + 1;
    static const int BLOCK_SIZE_2     = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3     = BLOCK_SIZE_2 * BLOCK_SIZE;
    static const int N_MICRO_ELEMENTS = LEVEL * LEVEL * LEVEL;

    __shared__ T x_block[BLOCK_SIZE_3];
    __shared__ T y_block[BLOCK_SIZE_3];
    __shared__ T emat[64];

    const int lane_id = threadIdx.x + blockDim.x * threadIdx.y;
    const int lidx = lane_id + (blockDim.x * blockDim.y) * threadIdx.z;

    // Elemental matrix entries for this thread
    const int offset_0 = threadIdx.y * 8 + threadIdx.x;
    const int offset_1 = threadIdx.y * 8 + 4 + threadIdx.x;

    assert(offset_0 < 64);
    assert(offset_1 < 64);

    // Nodes on quad input
    const int in_x_offset = !!(threadIdx.x & 1);
    const int in_y_offset = !!(threadIdx.x & 2);

    assert(in_x_offset <= 1);
    assert(in_y_offset <= 1);

    const int out_x_offset = !!(threadIdx.y & 1);
    const int out_y_offset = !!(threadIdx.y & 2);
    const int out_z_offset = !!(threadIdx.y & 4);

    assert(out_x_offset <= 1);
    assert(out_y_offset <= 1);
    assert(out_z_offset <= 1);

    // Max number of elements per batch
    const int batch_size = blockDim.y * blockDim.z;

    // Number of rounds/batches
    const int n_rounds     = (N_MICRO_ELEMENTS + batch_size - 1) / batch_size;
    const int batch_offset = threadIdx.z * (N_MICRO_ELEMENTS / 8);

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        // Copy elemental matrix from global memory to shared memory (Colaesced)
        if (lidx < 64) {
            emat[lidx] = elemental_matrix[e * 64 + lidx];
        }

        // Copy coefficients from global memory to shared memory (Random Access)
        if (lidx < BLOCK_SIZE_3) {
            const ptrdiff_t idx = elements[e * BLOCK_SIZE_3 + lidx];
            x_block[lidx]       = x[idx];
            y_block[lidx]       = 0;  // Reset
        }

        __syncthreads();

        const double A0 = emat[offset_0];
        const double A1 = emat[offset_1];

        for (int r = 0; r < n_rounds; r++) {
            const int in_micro_e  = threadIdx.y + batch_offset + r * batch_size;
            const int out_micro_e = 2 * threadIdx.x + batch_offset + r * batch_size;

            double u0 = 0;
            double u1 = 0;

            if (in_micro_e < N_MICRO_ELEMENTS) {
                // construct micro-element tensorial index
                const int in_xe = in_micro_e % LEVEL;
                const int in_ye = (in_micro_e / LEVEL) % LEVEL;
                const int in_ze = in_micro_e / (LEVEL * LEVEL);
                const int x0    = in_xe + in_x_offset;
                const int y0    = in_ye + in_y_offset;

                assert(x0 < LEVEL + 1);
                assert(y0 < LEVEL + 1);
                assert(in_ze < LEVEL);

                // Bottom node
                int idx0 = cu_sshex8_lidx(LEVEL, x0, y0, in_ze);
                assert(idx0 < BLOCK_SIZE_3);

                // Top node
                int idx1 = cu_sshex8_lidx(LEVEL, x0, y0, in_ze + 1);
                assert(idx1 < BLOCK_SIZE_3);

                u0 = x_block[idx0];
                u1 = x_block[idx1];
            }

            double C[2] = {0, 0};

            // C += A0 * u0
            asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1},{%2},{%3},{%4,%5};\n"
                    : "=d"(C[0]), "=d"(C[1])
                    : "d"(A0), "d"(u0), "d"(C[0]), "d"(C[1]));

            // C += A1 * u1
            asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1},{%2},{%3},{%4,%5};\n"
                    : "=d"(C[0]), "=d"(C[1])
                    : "d"(A1), "d"(u1), "d"(C[0]), "d"(C[1]));

            if (out_micro_e < N_MICRO_ELEMENTS) {
                const int out_xe = out_micro_e % LEVEL;
                const int out_ye = (out_micro_e / LEVEL) % LEVEL;
                const int out_ze = out_micro_e / (LEVEL * LEVEL);

                const int x0 = out_xe + out_x_offset;
                const int y0 = out_ye + out_y_offset;
                const int z0 = out_ze + out_z_offset;

                assert(x0 < LEVEL + 1);
                assert(y0 < LEVEL + 1);
                assert(z0 < LEVEL + 1);

                int idx0 = cu_sshex8_lidx(LEVEL, x0, y0, z0);
                assert(idx0 < BLOCK_SIZE_3);

                atomicAdd(&y_block[idx0], C[0]);
            }

            if (out_micro_e + 1 < N_MICRO_ELEMENTS) {
                const int out_xe = (out_micro_e + 1) % LEVEL;
                const int out_ye = ((out_micro_e + 1) / LEVEL) % LEVEL;
                const int out_ze = (out_micro_e + 1) / (LEVEL * LEVEL);

                const int x1 = out_xe + out_x_offset;
                const int y1 = out_ye + out_y_offset;
                const int z1 = out_ze + out_z_offset;

                assert(x1 < LEVEL + 1);
                assert(y1 < LEVEL + 1);
                assert(z1 < LEVEL + 1);

                int idx1 = cu_sshex8_lidx(LEVEL, x1, y1, z1);
                assert(idx1 < BLOCK_SIZE_3);

                atomicAdd(&y_block[idx1], C[1]);
            }
        }

        __syncthreads();

        if (lidx < BLOCK_SIZE_3) {
            const ptrdiff_t idx = elements[e * BLOCK_SIZE_3 + lidx];
            atomicAdd(&y[idx], y_block[lidx]);
        }
    }
}

template <typename T, int LEVEL>
int cu_affine_sshex8_elemental_matrix_apply_AoS_TC_tpl(const ptrdiff_t                  nelements,
                                                       const idx_t *const SFEM_RESTRICT elements,
                                                       const T *const SFEM_RESTRICT     elemental_matrix,
                                                       const T *const SFEM_RESTRICT     x,
                                                       T *const SFEM_RESTRICT           y,
                                                       void                            *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    static const int Z_SIZE = (LEVEL * LEVEL * LEVEL + 8 - 1) / 8;

    dim3 block_size(4, 8, Z_SIZE);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_TC<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements, elements, elemental_matrix, x, y);
    } else {
        cu_affine_sshex8_elemental_matrix_apply_kernel_AoS_TC<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements, elements, elemental_matrix, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

template <typename T>
int cu_affine_sshex8_elemental_matrix_apply_AoS_tpl(const int                        level,
                                                    const ptrdiff_t                  nelements,
                                                    const idx_t *const SFEM_RESTRICT elements,
                                                    const T *const SFEM_RESTRICT     elemental_matrix,
                                                    const T *const SFEM_RESTRICT     x,
                                                    T *const SFEM_RESTRICT           y,
                                                    void                            *stream) {
    int SFEM_ENABLE_TC = 0;
    SFEM_READ_ENV(SFEM_ENABLE_TC, atoi);

    if (SFEM_ENABLE_TC) {
        switch (level) {
            case 4: {
                return cu_affine_sshex8_elemental_matrix_apply_AoS_TC_tpl<T, 4>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            case 6: {
                return cu_affine_sshex8_elemental_matrix_apply_AoS_TC_tpl<T, 6>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            case 8: {
                return cu_affine_sshex8_elemental_matrix_apply_AoS_TC_tpl<T, 8>(
                        nelements, elements, elemental_matrix, x, y, stream);
            }
            default:
                break;
        }
    }

    switch (level) {
        case 4: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_warp_tpl<T, 4>(
                    nelements, elements, elemental_matrix, x, y, stream);
        }
        case 6: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_warp_tpl<T, 6>(
                    nelements, elements, elemental_matrix, x, y, stream);
        }
        case 8: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_warp_tpl<T, 8>(
                    nelements, elements, elemental_matrix, x, y, stream);
        }
        default:
            break;
    }

    SFEM_ERROR("NOT implemented!\n");

    return SFEM_FAILURE;
}

int cu_affine_sshex8_elemental_matrix_apply_AoS(const int                        level,
                                                const ptrdiff_t                  nelements,
                                                const idx_t *const SFEM_RESTRICT elements,
                                                const enum RealType              real_type,
                                                const void *const SFEM_RESTRICT  elemental_matrix,
                                                const void *const SFEM_RESTRICT  x,
                                                void *const SFEM_RESTRICT        y,
                                                void                            *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_tpl<real_t>(
                    level, nelements, elements, (const real_t *)elemental_matrix, (const real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_tpl<float>(
                    level, nelements, elements, (const float *)elemental_matrix, (const float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_elemental_matrix_apply_AoS_tpl<double>(
                    level, nelements, elements, (const double *)elemental_matrix, (const double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_affine_sshex8_elemental_matrix_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type),
                       real_type);
            return SFEM_FAILURE;
        }
    }
}
