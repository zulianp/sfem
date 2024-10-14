#include "cu_hex8_laplacian_inline.hpp"

template <typename real_t, int LEVEL>
static inline __host__ __device__ void cu_proteus_hex8_gather(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const ptrdiff_t e,
        const idx_t *const SFEM_RESTRICT elements,
        const real_t *const SFEM_RESTRICT x,
        scalar_t *const SFEM_RESTRICT x_block) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
#ifndef NDEBUG
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;
#endif

    // z=0
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, yi, 0);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(xi, yi, 0)] = x[idx];
            }
        }
    }

    // z=LEVEL
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, yi, LEVEL);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(xi, yi, LEVEL)] = x[idx];
            }
        }
    }

    // x=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, 0, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(0, yi, zi)] = x[idx];
            }
        }
    }

    // x=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, LEVEL, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(LEVEL, yi, zi)] = x[idx];
            }
        }
    }

    // y=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, 0, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(xi, 0, zi)] = x[idx];
            }
        }
    }

    // y=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, LEVEL, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[B_(xi, LEVEL, zi)] = x[idx];
            }
        }
    }

    // Interior
    {
        for (int zi = 1; zi < LEVEL; zi++) {
            for (int yi = 1; yi < LEVEL; yi++) {
                for (int xi = 1; xi < LEVEL; xi++) {
                    const int lidx_vol = cu_proteus_hex8_lidx(LEVEL, xi, yi, zi);
                    const int Lm1 = LEVEL - 1;
                    const int en = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
                    const ptrdiff_t idx = interior_start + e * (Lm1 * Lm1 * Lm1) + en;
                    x_block[B_(xi, yi, zi)] = x[idx];
                }
            }
        }
    }
}

template <typename real_t, int LEVEL>
static inline __host__ __device__ void cu_proteus_hex8_scatter(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const ptrdiff_t e,
        const idx_t *const SFEM_RESTRICT elements,
        scalar_t *const SFEM_RESTRICT y_block,
        real_t *const SFEM_RESTRICT y) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;

#ifndef NDEBUG
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;
#endif

    // z=0
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, yi, 0);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(xi, yi, 0)]);
            }
        }
    }

    // z=LEVEL
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, yi, LEVEL);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(xi, yi, LEVEL)]);
            }
        }
    }

    // x=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, 0, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(0, yi, zi)]);
            }
        }
    }

    // x=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, LEVEL, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(LEVEL, yi, zi)]);
            }
        }
    }

    // y=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, 0, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(xi, 0, zi)]);
            }
        }
    }

    // y=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, LEVEL, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx], y_block[B_(xi, LEVEL, zi)]);
            }
        }
    }

    // Interior
    {
        for (int zi = 1; zi < LEVEL; zi++) {
            for (int yi = 1; yi < LEVEL; yi++) {
                for (int xi = 1; xi < LEVEL; xi++) {
                    const int lidx_vol = cu_proteus_hex8_lidx(LEVEL, xi, yi, zi);
                    const int Lm1 = LEVEL - 1;
                    const int en = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
                    const ptrdiff_t idx = interior_start + e * (Lm1 * Lm1 * Lm1) + en;
                    y[idx] += y_block[B_(xi, yi, zi)];
                }
            }
        }
    }
}

template <typename real_t, int LEVEL>
__global__ void cu_proteus_affine_hex8_laplacian_apply_kernel_volgen(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const real_t *const SFEM_RESTRICT x,
        real_t *const SFEM_RESTRICT y) {
#ifndef NDEBUG
    const int nxe = cu_proteus_hex8_nxe(LEVEL);
#endif

    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    // Uses "local" memory
    scalar_t x_block[BLOCK_SIZE_3];
    scalar_t y_block[BLOCK_SIZE_3];
#define CU_PROTEUS_HEX8_USE_ELEMENTAL_MATRIX
#ifdef CU_PROTEUS_HEX8_USE_ELEMENTAL_MATRIX
    scalar_t laplacian_matrix[8 * 8];
#endif

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
#ifdef CU_PROTEUS_HEX8_USE_ELEMENTAL_MATRIX
        // Build operator
        {
            scalar_t sub_fff[6];
            const scalar_t h = 1. / LEVEL;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }
#else
        const scalar_t h = 1. / LEVEL;
        scalar_t sub_fff[6];
        cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
#endif

        // Gather
        cu_proteus_hex8_gather<real_t, LEVEL>(
                nelements, stride, interior_start, e, elements, x, x_block);

        // Reset
        for (int i = 0; i < BLOCK_SIZE_3; i++) {
            y_block[i] = 0;
        }

        // Compute
        for (int zi = 0; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE - 1; yi++) {
                for (int xi = 0; xi < BLOCK_SIZE - 1; xi++) {
                    assert(B_(xi + 1, yi + 1, zi + 1) < BLOCK_SIZE_3);

                    scalar_t element_u[8] = {x_block[B_(xi, yi, zi)],
                                             x_block[B_(xi + 1, yi, zi)],
                                             x_block[B_(xi + 1, yi + 1, zi)],
                                             x_block[B_(xi, yi + 1, zi)],
                                             x_block[B_(xi, yi, zi + 1)],
                                             x_block[B_(xi + 1, yi, zi + 1)],
                                             x_block[B_(xi + 1, yi + 1, zi + 1)],
                                             x_block[B_(xi, yi + 1, zi + 1)]};

#ifdef CU_PROTEUS_HEX8_USE_ELEMENTAL_MATRIX
                    scalar_t element_vector[8] = {0};
                    for (int i = 0; i < 8; i++) {
                        const scalar_t *const row = &laplacian_matrix[i * 8];
                        const scalar_t ui = element_u[i];
                        assert(ui == ui);
                        for (int j = 0; j < 8; j++) {
                            assert(row[j] == row[j]);
                            element_vector[j] += ui * row[j];
                        }
                    }

#else
                    scalar_t element_vector[8];
                    cu_hex8_laplacian_apply_fff_integral(sub_fff, element_u, element_vector);
#endif

                    y_block[B_(xi, yi, zi)] += element_vector[0];
                    y_block[B_(xi + 1, yi, zi)] += element_vector[1];
                    y_block[B_(xi + 1, yi + 1, zi)] += element_vector[2];
                    y_block[B_(xi, yi + 1, zi)] += element_vector[3];
                    y_block[B_(xi, yi, zi + 1)] += element_vector[4];
                    y_block[B_(xi + 1, yi, zi + 1)] += element_vector[5];
                    y_block[B_(xi + 1, yi + 1, zi + 1)] += element_vector[6];
                    y_block[B_(xi, yi + 1, zi + 1)] += element_vector[7];
                }
            }
        }

        // Scatter
        cu_proteus_hex8_scatter<real_t, LEVEL>(
                nelements, stride, interior_start, e, elements, y_block, y);
    }
}



template <typename T, int LEVEL>
static int cu_proteus_affine_hex8_laplacian_apply_volgen_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const T *const x,
        T *const y,
        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_proteus_affine_hex8_laplacian_apply_kernel_volgen<T, LEVEL>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_laplacian_apply_kernel_volgen<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements, stride, interior_start, elements, fff, x, y);
    } else {
        cu_proteus_affine_hex8_laplacian_apply_kernel_volgen<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements, stride, interior_start, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
