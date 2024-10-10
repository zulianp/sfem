#include "cu_proteus_hex8_interpolate.h"

#include "sfem_cuda_base.h"

#include <cassert>
#include <cstdio>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef NDEBUG
static inline __device__ __host__ int cu_proteus_hex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}
#endif

static inline __device__ __host__ int cu_proteus_hex8_lidx(const int L,
                                                           const int x,
                                                           const int y,
                                                           const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < cu_proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

// PROLONGATION

template <typename From, typename To>
__global__ void cu_proteus_hex8_hierarchical_prolongation_kernel(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const int vec_size,
        const ptrdiff_t from_stride,
        const From *const SFEM_RESTRICT from,
        const ptrdiff_t to_stride,
        To *const SFEM_RESTRICT to) {
    const int corners[8] = {// Bottom
                            cu_proteus_hex8_lidx(level, 0, 0, 0),
                            cu_proteus_hex8_lidx(level, level, 0, 0),
                            cu_proteus_hex8_lidx(level, level, level, 0),
                            cu_proteus_hex8_lidx(level, 0, level, 0),
                            // Top
                            cu_proteus_hex8_lidx(level, 0, 0, level),
                            cu_proteus_hex8_lidx(level, level, 0, level),
                            cu_proteus_hex8_lidx(level, level, level, level),
                            cu_proteus_hex8_lidx(level, 0, level, level)};

    const scalar_t h = 1. / level;

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        for (int zi = 0; zi < level + 1; zi++) {
            for (int yi = 0; yi < level + 1; yi++) {
                for (int xi = 0; xi < level + 1; xi++) {
                    idx_t idx = elements[cu_proteus_hex8_lidx(level, xi, yi, zi) * stride + e];

                    const scalar_t x = xi * h;
                    const scalar_t y = yi * h;
                    const scalar_t z = zi * h;

                    // Evaluate Hex8 basis functions at x, y, z
                    const scalar_t xm = (1 - x);
                    const scalar_t ym = (1 - y);
                    const scalar_t zm = (1 - z);

                    scalar_t f[8];
                    f[0] = xm * ym * zm;  // (0, 0, 0)
                    f[1] = x * ym * zm;   // (1, 0, 0)
                    f[2] = x * y * zm;    // (1, 1, 0)
                    f[3] = xm * y * zm;   // (0, 1, 0)
                    f[4] = xm * ym * z;   // (0, 0, 1)
                    f[5] = x * ym * z;    // (1, 0, 1)
                    f[6] = x * y * z;     // (1, 1, 1)
                    f[7] = xm * y * z;    // (0, 1, 1)

                    for (int d = 0; d < vec_size; d++) {
                        scalar_t val = 0;

                        for (int v = 0; v < 8; v++) {
                            const ptrdiff_t global_from_idx =
                                    (elements[corners[v] * stride + e] * vec_size + d) * from_stride;
                            val += f[v] * from[global_from_idx];
                        }

                        

                        const ptrdiff_t global_to_idx = (idx * vec_size + d) * to_stride;
                        to[global_to_idx] = val;
                    }
                }
            }
        }
    }
}

template <typename From, typename To>
static int cu_proteus_hex8_hierarchical_prolongation_tpl(const int level,
                                                         const ptrdiff_t nelements,
                                                         const ptrdiff_t stride,
                                                         const idx_t *const SFEM_RESTRICT elements,
                                                         const int vec_size,
                                                         const ptrdiff_t from_stride,
                                                         const From *const SFEM_RESTRICT from,
                                                         const ptrdiff_t to_stride,
                                                         To *const SFEM_RESTRICT to,
                                                         void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_proteus_hex8_hierarchical_prolongation_kernel<From, To>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_proteus_hex8_hierarchical_prolongation_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_proteus_hex8_hierarchical_prolongation_kernel<From, To><<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_proteus_hex8_hierarchical_prolongation(const int level,
                                              const ptrdiff_t nelements,
                                              const ptrdiff_t stride,
                                              const idx_t *const SFEM_RESTRICT elements,
                                              const int vec_size,
                                              const enum RealType from_type,
                                              const ptrdiff_t from_stride,
                                              const void *const SFEM_RESTRICT from,
                                              const enum RealType to_type,
                                              const ptrdiff_t to_stride,
                                              void *const SFEM_RESTRICT to,
                                              void *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_proteus_hex8_hierarchical_prolongation_tpl(level,
                                                                 nelements,
                                                                 stride,
                                                                 elements,
                                                                 vec_size,
                                                                 from_stride,
                                                                 (real_t *)from,
                                                                 to_stride,
                                                                 (real_t *)to,
                                                                 stream);
        }
        case SFEM_FLOAT32: {
            return cu_proteus_hex8_hierarchical_prolongation_tpl(level,
                                                                 nelements,
                                                                 stride,
                                                                 elements,
                                                                 vec_size,
                                                                 from_stride,
                                                                 (float *)from,
                                                                 to_stride,
                                                                 (float *)to,
                                                                 stream);
        }
        case SFEM_FLOAT64: {
            return cu_proteus_hex8_hierarchical_prolongation_tpl(level,
                                                                 nelements,
                                                                 stride,
                                                                 elements,
                                                                 vec_size,
                                                                 from_stride,
                                                                 (double *)from,
                                                                 to_stride,
                                                                 (double *)to,
                                                                 stream);
        }
        default: {
            fprintf(stderr,
                    "[Error]  cu_proteus_hex8_prolongation_tpl: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

// RESTRICTION

template <typename From, typename To>
__global__ void cu_proteus_hex8_hierarchical_restriction_kernel(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const uint16_t *const SFEM_RESTRICT e2n_count,
        const int vec_size,
        const ptrdiff_t from_stride,
        const From *const SFEM_RESTRICT from,
        const ptrdiff_t to_stride,
        To *const SFEM_RESTRICT to) {
    const int corners[8] = {// Bottom
                            cu_proteus_hex8_lidx(level, 0, 0, 0),
                            cu_proteus_hex8_lidx(level, level, 0, 0),
                            cu_proteus_hex8_lidx(level, level, level, 0),
                            cu_proteus_hex8_lidx(level, 0, level, 0),
                            // Top
                            cu_proteus_hex8_lidx(level, 0, 0, level),
                            cu_proteus_hex8_lidx(level, level, 0, level),
                            cu_proteus_hex8_lidx(level, level, level, level),
                            cu_proteus_hex8_lidx(level, 0, level, level)};

    const scalar_t h = 1. / level;
    scalar_t acc[8];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        for (int d = 0; d < vec_size; d++) {
            for (int i = 0; i < 8; i++) {
                acc[i] = 0;
            }

            for (int zi = 0; zi < level + 1; zi++) {
                for (int yi = 0; yi < level + 1; yi++) {
                    for (int xi = 0; xi < level + 1; xi++) {
                        const int lidx = cu_proteus_hex8_lidx(level, xi, yi, zi);
                        const ptrdiff_t idx = elements[lidx * stride + e];

                        const scalar_t x = xi * h;
                        const scalar_t y = yi * h;
                        const scalar_t z = zi * h;

                        // Evaluate Hex8 basis functions at x, y, z
                        const scalar_t xm = (1 - x);
                        const scalar_t ym = (1 - y);
                        const scalar_t zm = (1 - z);

                        scalar_t f[8];
                        f[0] = xm * ym * zm;  // (0, 0, 0)
                        f[1] = x * ym * zm;   // (1, 0, 0)
                        f[2] = x * y * zm;    // (1, 1, 0)
                        f[3] = xm * y * zm;   // (0, 1, 0)
                        f[4] = xm * ym * z;   // (0, 0, 1)
                        f[5] = x * ym * z;    // (1, 0, 1)
                        f[6] = x * y * z;     // (1, 1, 1)
                        f[7] = xm * y * z;    // (0, 1, 1)

                        const ptrdiff_t global_from_idx = (idx * vec_size + d) * from_stride;
                        const scalar_t val = from[global_from_idx] / e2n_count[global_from_idx];

                        for (int i = 0; i < 8; i++) {
                            acc[i] += f[i] * val;
                        }
                    }
                }
            }

            for (int v = 0; v < 8; v++) {
                const ptrdiff_t global_to_idx = (elements[corners[v] * stride + e] * vec_size + d) * to_stride;
                atomicAdd(&to[global_to_idx], acc[v]);
            }
        }
    }
}

template <typename From, typename To>
static int cu_proteus_hex8_hierarchical_restriction_tpl(const int level,
                                                        const ptrdiff_t nelements,
                                                        const ptrdiff_t stride,
                                                        const idx_t *const SFEM_RESTRICT elements,
                                                        const uint16_t *const SFEM_RESTRICT
                                                                element_to_node_incidence_count,
                                                        const int vec_size,
                                                        const ptrdiff_t from_stride,
                                                        const From *const SFEM_RESTRICT from,
                                                        const ptrdiff_t to_stride,
                                                        To *const SFEM_RESTRICT to,
                                                        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();
    
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_proteus_hex8_hierarchical_restriction_kernel<From, To>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_proteus_hex8_hierarchical_restriction_kernel<From, To>
                <<<n_blocks, block_size, 0, s>>>(level,
                                                 nelements,
                                                 stride,
                                                 elements,
                                                 element_to_node_incidence_count,
                                                 vec_size,
                                                 from_stride,
                                                 from,
                                                 to_stride,
                                                 to);
    } else {
        cu_proteus_hex8_hierarchical_restriction_kernel<From, To>
                <<<n_blocks, block_size, 0>>>(level,
                                              nelements,
                                              stride,
                                              elements,
                                              element_to_node_incidence_count,
                                              vec_size,
                                              from_stride,
                                              from,
                                              to_stride,
                                              to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_proteus_hex8_hierarchical_restriction(const int level,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t stride,
                                             const idx_t *const SFEM_RESTRICT elements,
                                             const uint16_t *const SFEM_RESTRICT
                                                     element_to_node_incidence_count,
                                             const int vec_size,
                                             const enum RealType from_type,
                                             const ptrdiff_t from_stride,
                                             const void *const SFEM_RESTRICT from,
                                             const enum RealType to_type,
                                             const ptrdiff_t to_stride,
                                             void *const SFEM_RESTRICT to,
                                             void *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_proteus_hex8_hierarchical_restriction_tpl(level,
                                                                nelements,
                                                                stride,
                                                                elements,
                                                                element_to_node_incidence_count,
                                                                vec_size,
                                                                from_stride,
                                                                (real_t *)from,
                                                                to_stride,
                                                                (real_t *)to,
                                                                stream);
        }
        case SFEM_FLOAT32: {
            return cu_proteus_hex8_hierarchical_restriction_tpl(level,
                                                                nelements,
                                                                stride,
                                                                elements,
                                                                element_to_node_incidence_count,
                                                                vec_size,
                                                                from_stride,
                                                                (float *)from,
                                                                to_stride,
                                                                (float *)to,
                                                                stream);
        }
        case SFEM_FLOAT64: {
            return cu_proteus_hex8_hierarchical_restriction_tpl(level,
                                                                nelements,
                                                                stride,
                                                                elements,
                                                                element_to_node_incidence_count,
                                                                vec_size,
                                                                from_stride,
                                                                (double *)from,
                                                                to_stride,
                                                                (double *)to,
                                                                stream);
        }
        default: {
            fprintf(stderr,
                    "[Error]  cu_proteus_hex8_prolongation_tpl: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
