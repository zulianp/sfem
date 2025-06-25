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

#if 1

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

#else

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

    const int lidx = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

    assert(blockDim.x == 4); // 4 shape function in the x direction
    assert(blockDim.y == 8); // 8 elements in the y directions
    assert(blockDim.z == BLOCK_SIZE / (blockDim.x * blockDim.y)); // ceil(n_micro_elements / 8)

    const int warp_id = lidx / SFEM_WARP_SIZE;
    const int lane_id = lidx % SFEM_WARP_SIZE;
    
    const int mat_i = lane_id % 4;
    const int mat_j = lane_id / 4;

    const int offset_0 = mat_i * 4 + mat_j;
    const int offset_1 = mat_i * 4 + 4 + mat_j;
    
    const int x_offset = (threadIdx.x & 1);
    const int y_offset = (threadIdx.x & 2);
    const int micro_element_offset = threadIdx.z / 8;

    // y_elements * z_elements
    const bool is_element = threadIdx.y + blockDim.y * threadIdx.z  < (LEVEL * LEVEL * LEVEL);

    // Bottom or top
    const int vertex_id = lidx % 4;
    const int shape_id  = vertex_id == 3 ? 2 : (vertex_id == 2 ? 3 : vertex_id);

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        if (lidx < 64) {
            emat[lidx] = elemental_matrix[e * 64 + lidx];
        }

        const ptrdiff_t idx = elements[e * BLOCK_SIZE_3 + lidx];

        x_block[lidx] = x[idx];  // Copy coeffs to shared mem
        y_block[lidx] = 0;       // Reset

        __syncthreads();  //

        // 8 elements at time
        // for (int pack = warp_id * 8; pack < (LEVEL * LEVEL * LEVEL); pack += n_warps * 8) {
        // C += [A0, A1] * [u0,u1]^T

        // Assume Symmetric
        const double A0 = emat[offset_0];
        const double A1 = emat[offset_1];

        double u0 = 0;
        double u1 = 0;

        double C[2] = {0, 0};

        // A0 * u0
        asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0,%1},{%2},{%3},{%4,%5};\n"
                : "=d"(C[0]), "=d"(C[1])
                : "d"(A0), "d"(u0), "d"(C[0]), "d"(C[1]));

        // A1 * u1
        asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0,%1},{%2},{%3},{%4,%5};\n"
                : "=d"(C[0]), "=d"(C[1])
                : "d"(A1), "d"(u1), "d"(C[0]), "d"(C[1]));

        // atomicAdd(y_block[lidx], C[0]);
        // atomicAdd(y_block[lidx], C[1]);
        // }

        __syncthreads();  //
        atomicAdd(&y[idx], y_block[lidx]);
    }

#endif

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

template <typename T>
int cu_affine_sshex8_elemental_matrix_apply_AoS_tpl(const int                        level,
                                                    const ptrdiff_t                  nelements,
                                                    const idx_t *const SFEM_RESTRICT elements,
                                                    const T *const SFEM_RESTRICT     elemental_matrix,
                                                    const T *const SFEM_RESTRICT     x,
                                                    T *const SFEM_RESTRICT           y,
                                                    void                            *stream) {
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
            SFEM_ERROR("NOT implemented!\n");
            break;
    }

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
