#ifndef SFEM_PROTEUS_HEX8_INLINE_HPP
#define SFEM_PROTEUS_HEX8_INLINE_HPP

#include "sfem_base.h"

#define SFEM_HEX8_BLOCK_IDX(x, y, z) ((z)*BLOCK_SIZE_2 + (y)*BLOCK_SIZE + (x))

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
                x_block[SFEM_HEX8_BLOCK_IDX(xi, yi, 0)] = x[idx];
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
                x_block[SFEM_HEX8_BLOCK_IDX(xi, yi, LEVEL)] = x[idx];
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
                x_block[SFEM_HEX8_BLOCK_IDX(0, yi, zi)] = x[idx];
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
                x_block[SFEM_HEX8_BLOCK_IDX(LEVEL, yi, zi)] = x[idx];
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
                x_block[SFEM_HEX8_BLOCK_IDX(xi, 0, zi)] = x[idx];
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
                x_block[SFEM_HEX8_BLOCK_IDX(xi, LEVEL, zi)] = x[idx];
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
                    x_block[SFEM_HEX8_BLOCK_IDX(xi, yi, zi)] = x[idx];
                }
            }
        }
    }
}

template <typename real_t, int LEVEL>
static inline __host__ __device__ void cu_proteus_hex8_scatter_add(
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(xi, yi, 0)]);
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(xi, yi, LEVEL)]);
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(0, yi, zi)]);
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(LEVEL, yi, zi)]);
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(xi, 0, zi)]);
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
                atomicAdd(&y[idx], y_block[SFEM_HEX8_BLOCK_IDX(xi, LEVEL, zi)]);
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
                    y[idx] += y_block[SFEM_HEX8_BLOCK_IDX(xi, yi, zi)];
                }
            }
        }
    }
}

#endif //SFEM_PROTEUS_HEX8_INLINE_HPP
