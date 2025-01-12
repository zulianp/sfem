#ifndef SFEM_SSHEX8_INLINE_HPP
#define SFEM_SSHEX8_INLINE_HPP

#include "sfem_base.h"

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#endif

template<typename scalar_t>
static inline __device__ __host__ void cu_hex8_sub_fff_0(const ptrdiff_t stride,
                                                         const cu_jacobian_t *const SFEM_RESTRICT
                                                                 fff,
                                                         const scalar_t h,
                                                         scalar_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (scalar_t)fff[0 * stride] * h;
    sub_fff[1] = (scalar_t)fff[1 * stride] * h;
    sub_fff[2] = (scalar_t)fff[2 * stride] * h;
    sub_fff[3] = (scalar_t)fff[3 * stride] * h;
    sub_fff[4] = (scalar_t)fff[4 * stride] * h;
    sub_fff[5] = (scalar_t)fff[5 * stride] * h;
}

static inline __device__ __host__ int cu_sshex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

static inline __device__ __host__ int cu_sshex8_txe(int level) {
    return level * level * level;
}

static inline __device__ __host__ int cu_sshex8_lidx(const int L,
                                                           const int x,
                                                           const int y,
                                                           const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < cu_sshex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

#define SFEM_HEX8_BLOCK_IDX(x, y, z) ((z)*BLOCK_SIZE_2 + (y)*BLOCK_SIZE + (x))

template <typename real_t, int LEVEL, typename scalar_t>
static inline __host__ __device__ void cu_sshex8_gather(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const ptrdiff_t e,
        const idx_t *const SFEM_RESTRICT elements,
        const ptrdiff_t x_stride,
        const real_t *const SFEM_RESTRICT x,
        scalar_t *const SFEM_RESTRICT x_block) {
    static const int BLOCK_SIZE = LEVEL + 1;

#ifndef NDEBUG
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;
#endif

    // z=0
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, 0);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // z=LEVEL
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, LEVEL);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // x=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, 0, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // x=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, LEVEL, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // y=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, 0, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // y=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, LEVEL, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                x_block[lidx] = x[idx * x_stride];
            }
        }
    }

    // Interior
    {
        for (int zi = 1; zi < LEVEL; zi++) {
            for (int yi = 1; yi < LEVEL; yi++) {
                for (int xi = 1; xi < LEVEL; xi++) {
                    const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, zi);
                    const int Lm1 = LEVEL - 1;
                    const int en = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
                    const ptrdiff_t idx = interior_start + e * (Lm1 * Lm1 * Lm1) + en;
                    x_block[lidx] = x[idx * x_stride];
                }
            }
        }
    }
}

template <typename real_t, int LEVEL, typename scalar_t>
static inline __host__ __device__ void cu_sshex8_scatter_add(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const ptrdiff_t e,
        const idx_t *const SFEM_RESTRICT elements,
        scalar_t *const SFEM_RESTRICT y_block,
        const ptrdiff_t y_stride,
        real_t *const SFEM_RESTRICT y) {
    static const int BLOCK_SIZE = LEVEL + 1;

#ifndef NDEBUG
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;
#endif

    // z=0
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, 0);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // z=LEVEL
    {
        for (int yi = 0; yi < BLOCK_SIZE; yi++) {
            for (int xi = 0; xi < BLOCK_SIZE; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, LEVEL);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // x=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, 0, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // x=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int yi = 0; yi < BLOCK_SIZE; yi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, LEVEL, yi, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // y=0
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, 0, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // y=LEVEL
    {
        for (int zi = 1; zi < BLOCK_SIZE - 1; zi++) {
            for (int xi = 1; xi < BLOCK_SIZE - 1; xi++) {
                const int lidx = cu_sshex8_lidx(LEVEL, xi, LEVEL, zi);
                assert(lidx < BLOCK_SIZE_3);
                const idx_t idx = elements[lidx * stride + e];
                atomicAdd(&y[idx * y_stride], y_block[lidx]);
            }
        }
    }

    // Interior
    {
        for (int zi = 1; zi < LEVEL; zi++) {
            for (int yi = 1; yi < LEVEL; yi++) {
                for (int xi = 1; xi < LEVEL; xi++) {
                    const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, zi);
                    const int Lm1 = LEVEL - 1;
                    const int en = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
                    const ptrdiff_t idx = interior_start + e * (Lm1 * Lm1 * Lm1) + en;
                    y[idx * y_stride] += y_block[lidx];
                }
            }
        }
    }
}

#endif  // SFEM_SSHEX8_INLINE_HPP
