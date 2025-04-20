#include "cu_sshex8_interpolate.h"

#include "sfem_cuda_base.h"

#include "cu_sshex8_inline.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

static const int TILE_SIZE = 8;
#define ROUND_ROBIN(val, shift) ((val + shift) & (TILE_SIZE - 1))
#define ROUND_ROBIN_2(val, shift) ((val + shift) & (2 - 1))

static inline __device__ int is_even(const int v) { return !(v & 1); }
static inline __device__ int is_odd(const int v) { return (v & 1); }

template <typename T>
class ShapeInterpolation {
public:
    T     *data{nullptr};
    size_t nodes{0};

    ShapeInterpolation(const int steps, const int padding = 0) {
        nodes = (steps + 1);
        std::vector<T> S_host(2 * (nodes + padding), 0);
        double         h = 1. / steps;
        for (int i = 0; i < nodes; i++) {
            S_host[0 * (nodes + padding) + i] = h * i;
            S_host[1 * (nodes + padding) + i] = (1 - h * i);
        }

        auto nbytes = S_host.size() * sizeof(T);

        SFEM_CUDA_CHECK(cudaMalloc((void **)&data, nbytes));
        SFEM_CUDA_CHECK(cudaMemcpy(data, S_host.data(), nbytes, cudaMemcpyHostToDevice));
    }

    ~ShapeInterpolation() { cudaFree(data); }
};

// PROLONGATION

template <typename From, typename To>
__global__ void cu_sshex8_hierarchical_prolongation_kernel(const int                        level,
                                                           const ptrdiff_t                  nelements,
                                                           const ptrdiff_t                  stride,
                                                           const idx_t *const SFEM_RESTRICT elements,
                                                           const int                        vec_size,
                                                           const ptrdiff_t                  from_stride,
                                                           const From *const SFEM_RESTRICT  from,
                                                           const ptrdiff_t                  to_stride,
                                                           To *const SFEM_RESTRICT          to) {
    const int corners[8] = {// Bottom
                            cu_sshex8_lidx(level, 0, 0, 0),
                            cu_sshex8_lidx(level, level, 0, 0),
                            cu_sshex8_lidx(level, level, level, 0),
                            cu_sshex8_lidx(level, 0, level, 0),
                            // Top
                            cu_sshex8_lidx(level, 0, 0, level),
                            cu_sshex8_lidx(level, level, 0, level),
                            cu_sshex8_lidx(level, level, level, level),
                            cu_sshex8_lidx(level, 0, level, level)};

    const scalar_t h = 1. / level;

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int zi = 0; zi < level + 1; zi++) {
            for (int yi = 0; yi < level + 1; yi++) {
                for (int xi = 0; xi < level + 1; xi++) {
                    idx_t idx = elements[cu_sshex8_lidx(level, xi, yi, zi) * stride + e];

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
                            const ptrdiff_t global_from_idx = (elements[corners[v] * stride + e] * vec_size + d) * from_stride;

                            assert(from[global_from_idx] == from[global_from_idx]);
                            assert(f[v] == f[v]);

                            val += f[v] * from[global_from_idx];
                        }

                        assert(val == val);

                        const ptrdiff_t global_to_idx = (idx * vec_size + d) * to_stride;
                        to[global_to_idx]             = val;
                    }
                }
            }
        }
    }
}

template <typename From, typename To>
static int cu_sshex8_hierarchical_prolongation_tpl(const int                        level,
                                                   const ptrdiff_t                  nelements,
                                                   const ptrdiff_t                  stride,
                                                   const idx_t *const SFEM_RESTRICT elements,
                                                   const int                        vec_size,
                                                   const ptrdiff_t                  from_stride,
                                                   const From *const SFEM_RESTRICT  from,
                                                   const ptrdiff_t                  to_stride,
                                                   To *const SFEM_RESTRICT          to,
                                                   void                            *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_sshex8_hierarchical_prolongation_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_sshex8_hierarchical_prolongation_kernel<From, To>
                <<<n_blocks, block_size, 0, s>>>(level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_sshex8_hierarchical_prolongation_kernel<From, To>
                <<<n_blocks, block_size, 0>>>(level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_sshex8_hierarchical_prolongation(const int                        level,
                                               const ptrdiff_t                  nelements,
                                               const ptrdiff_t                  stride,
                                               const idx_t *const SFEM_RESTRICT elements,
                                               const int                        vec_size,
                                               const enum RealType              from_type,
                                               const ptrdiff_t                  from_stride,
                                               const void *const SFEM_RESTRICT  from,
                                               const enum RealType              to_type,
                                               const ptrdiff_t                  to_stride,
                                               void *const SFEM_RESTRICT        to,
                                               void                            *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_sshex8_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (real_t *)from, to_stride, (real_t *)to, stream);
        }
        case SFEM_FLOAT32: {
            return cu_sshex8_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (float *)from, to_stride, (float *)to, stream);
        }
        case SFEM_FLOAT64: {
            return cu_sshex8_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (double *)from, to_stride, (double *)to, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error]  cu_sshex8_prolongation_tpl: not implemented for type %s "
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
__global__ void cu_sshex8_hierarchical_restriction_kernel(const int                           level,
                                                          const ptrdiff_t                     nelements,
                                                          const ptrdiff_t                     stride,
                                                          const idx_t *const SFEM_RESTRICT    elements,
                                                          const uint16_t *const SFEM_RESTRICT e2n_count,
                                                          const int                           vec_size,
                                                          const ptrdiff_t                     from_stride,
                                                          const From *const SFEM_RESTRICT     from,
                                                          const ptrdiff_t                     to_stride,
                                                          To *const SFEM_RESTRICT             to) {
    const int corners[8] = {// Bottom
                            cu_sshex8_lidx(level, 0, 0, 0),
                            cu_sshex8_lidx(level, level, 0, 0),
                            cu_sshex8_lidx(level, level, level, 0),
                            cu_sshex8_lidx(level, 0, level, 0),
                            // Top
                            cu_sshex8_lidx(level, 0, 0, level),
                            cu_sshex8_lidx(level, level, 0, level),
                            cu_sshex8_lidx(level, level, level, level),
                            cu_sshex8_lidx(level, 0, level, level)};

    const scalar_t h = 1. / level;
    scalar_t       acc[8];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int d = 0; d < vec_size; d++) {
            for (int i = 0; i < 8; i++) {
                acc[i] = 0;
            }

            for (int zi = 0; zi < level + 1; zi++) {
                for (int yi = 0; yi < level + 1; yi++) {
                    for (int xi = 0; xi < level + 1; xi++) {
                        const int       lidx = cu_sshex8_lidx(level, xi, yi, zi);
                        const ptrdiff_t idx  = elements[lidx * stride + e];

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
                        const scalar_t  val             = from[global_from_idx] / e2n_count[idx];

                        assert(from[global_from_idx] == from[global_from_idx]);
                        assert(e2n_count[idx] > 0);
                        assert(val == val);

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
static int cu_sshex8_hierarchical_restriction_tpl(const int                           level,
                                                  const ptrdiff_t                     nelements,
                                                  const ptrdiff_t                     stride,
                                                  const idx_t *const SFEM_RESTRICT    elements,
                                                  const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                                  const int                           vec_size,
                                                  const ptrdiff_t                     from_stride,
                                                  const From *const SFEM_RESTRICT     from,
                                                  const ptrdiff_t                     to_stride,
                                                  To *const SFEM_RESTRICT             to,
                                                  void                               *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_sshex8_hierarchical_restriction_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_sshex8_hierarchical_restriction_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_sshex8_hierarchical_restriction_kernel<From, To><<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_sshex8_hierarchical_restriction(const int                           level,
                                              const ptrdiff_t                     nelements,
                                              const ptrdiff_t                     stride,
                                              const idx_t *const SFEM_RESTRICT    elements,
                                              const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                              const int                           vec_size,
                                              const enum RealType                 from_type,
                                              const ptrdiff_t                     from_stride,
                                              const void *const SFEM_RESTRICT     from,
                                              const enum RealType                 to_type,
                                              const ptrdiff_t                     to_stride,
                                              void *const SFEM_RESTRICT           to,
                                              void                               *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_sshex8_hierarchical_restriction_tpl(level,
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
            return cu_sshex8_hierarchical_restriction_tpl(level,
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
            return cu_sshex8_hierarchical_restriction_tpl(level,
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
                    "[Error]  cu_sshex8_prolongation_tpl: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

template <typename From, typename To>
__global__ void cu_sshex8_restrict_kernel(const ptrdiff_t                     nelements,
                                          const ptrdiff_t                     stride,
                                          const int                           from_level,
                                          const int                           from_level_stride,
                                          idx_t *const SFEM_RESTRICT          from_elements,
                                          const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                                          const int                           to_level,
                                          const int                           to_level_stride,
                                          idx_t *const SFEM_RESTRICT          to_elements,
                                          const To *const SFEM_RESTRICT       S,
                                          const int                           vec_size,
                                          const enum RealType                 from_type,
                                          const ptrdiff_t                     from_stride,
                                          const From *const SFEM_RESTRICT     from,
                                          const enum RealType                 to_type,
                                          const ptrdiff_t                     to_stride,
                                          To *const SFEM_RESTRICT             to) {
    static_assert(TILE_SIZE == 8,
                  "This only works with tile size 8 because the implementation assumes a fixed tile size for shared memory "
                  "layout and indexing.");

    // Unsigned char necessary for multiple template instantiations of this kernel
    extern __shared__ unsigned char cu_buff[];

    const int step_factor = from_level / to_level;

    // Tile number in group
    const int tile    = threadIdx.x >> 3;   // same as threadIdx.x / 8
    const int n_tiles = blockDim.x >> 3;    // same as blockDim.x / 8
    const int sub_idx = threadIdx.x & 0x7;  // same as threadIdx.x % 8

    From *in = (From *)&cu_buff[tile * TILE_SIZE * sizeof(From)];

    // hex8 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2
    const int zi = (sub_idx >> 2);        // equivalent to sub_idx / 4
    assert(n_tiles * TILE_SIZE == blockDim.x);

    // // 1 macro element per tile
    const ptrdiff_t e         = blockIdx.x * n_tiles + tile;
    const int       from_even = is_even(from_level);
    const int       to_nloops = to_level + from_even;

    // Add padding (Make sure that S has the correct padding)
    const int S_stride = from_level + 1 + from_even;

    // // Vector loop
    for (int d = 0; d < vec_size; d++) {
        // loop on all TO micro elements
        for (int to_zi = 0; to_zi < to_nloops; to_zi++) {
            for (int to_yi = 0; to_yi < to_nloops; to_yi++) {
                for (int to_xi = 0; to_xi < to_nloops; to_xi++) {
                    // Attention: parallelism in tile using xi, yi, zi
                    To acc = 0;

                    const int z_start = to_zi * step_factor;
                    const int y_start = to_yi * step_factor;
                    const int x_start = to_xi * step_factor;

                    const int z_odd = is_odd(z_start);
                    const int y_odd = is_odd(y_start);
                    const int x_odd = is_odd(x_start);

                    const int z_end = (zi == to_level ? 1 : step_factor) + z_odd;
                    const int y_end = (yi == to_level ? 1 : step_factor) + y_odd;
                    const int x_end = (xi == to_level ? 1 : step_factor) + x_odd;

                    for (int from_zi = z_odd; from_zi < z_end; from_zi += 2) {
                        for (int from_yi = y_odd; from_yi < y_end; from_yi += 2) {
                            for (int from_xi = x_odd; from_xi < x_end; from_xi += 2) {
                                // read from global mem
                                {
                                    const int zz = z_start + from_zi;
                                    const int yy = y_start + from_yi;
                                    const int xx = x_start + from_xi;

                                    const int off_from_zi = (zz + zi);
                                    const int off_from_yi = (yy + yi);
                                    const int off_from_xi = (xx + xi);

                                    const bool from_exists = e < nelements && off_from_zi <= from_level &&
                                                             off_from_yi <= from_level && off_from_xi <= from_level;

                                    __syncwarp();
                                    if (from_exists) {
                                        const int idx_from = cu_sshex8_lidx(from_level * from_level_stride,
                                                                            off_from_xi * from_level_stride,
                                                                            off_from_yi * from_level_stride,
                                                                            off_from_zi * from_level_stride);

                                        const idx_t     gidx = from_elements[idx_from * stride + e];
                                        const ptrdiff_t idx  = (gidx * vec_size + d) * from_stride;
                                        in[sub_idx]          = from[idx] / from_element_to_node_incidence_count[gidx];
                                    } else {
                                        in[sub_idx] = 0;
                                    }
                                    __syncwarp();
                                }

                                const To *const Sz = &S[zi * S_stride + from_zi];
                                const To *const Sy = &S[yi * S_stride + from_yi];
                                const To *const Sx = &S[xi * S_stride + from_xi];

                                for (int dz = 0; dz < 2; dz++) {
                                    const int rrdz = ROUND_ROBIN_2(dz, zi);
                                    for (int dy = 0; dy < 2; dy++) {
                                        const int rrdy = ROUND_ROBIN_2(dy, yi);
                                        for (int dx = 0; dx < 2; dx++) {
                                            const int rrdx = ROUND_ROBIN_2(dx, xi);
                                            // No bank conflicts due to round robin for single precision
                                            const To c = in[rrdz * 4 + rrdy * 2 + rrdx];
                                            acc += c * Sx[rrdx] * Sy[rrdy] * Sz[rrdz];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    const bool exists = e < nelements && to_zi <= to_level && to_yi <= to_level && to_xi <= to_level;
                    if (exists) {
                        const int idx_to = cu_sshex8_lidx(to_level * to_level_stride,
                                                          to_xi * to_level_stride,
                                                          to_yi * to_level_stride,
                                                          to_zi * to_level_stride);

                        atomicAdd(&to[(to_elements[idx_to * stride + e] * vec_size + d) * to_stride], acc);
                    }
                }
            }
        }
    }
}

template <typename From, typename To>
int cu_sshex8_restrict_tpl(const ptrdiff_t                     nelements,
                           const ptrdiff_t                     stride,
                           const int                           from_level,
                           const int                           from_level_stride,
                           idx_t *const SFEM_RESTRICT          from_elements,
                           const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                           const int                           to_level,
                           const int                           to_level_stride,
                           idx_t *const SFEM_RESTRICT          to_elements,
                           const int                           vec_size,
                           const enum RealType                 from_type,
                           const ptrdiff_t                     from_stride,
                           const From *const SFEM_RESTRICT     from,
                           const enum RealType                 to_type,
                           const ptrdiff_t                     to_stride,
                           To *const SFEM_RESTRICT             to,
                           void                               *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_sshex8_restrict_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size / TILE_SIZE - 1) / (block_size / TILE_SIZE));

    ShapeInterpolation<To> S(from_level / to_level, from_level % 2 == 0);

    size_t shared_mem_size = block_size * sizeof(From);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_sshex8_restrict_kernel<From, To><<<n_blocks, block_size, shared_mem_size, s>>>(nelements,
                                                                                          stride,
                                                                                          from_level,
                                                                                          from_level_stride,
                                                                                          from_elements,
                                                                                          from_element_to_node_incidence_count,
                                                                                          to_level,
                                                                                          to_level_stride,
                                                                                          to_elements,
                                                                                          S.data,
                                                                                          vec_size,
                                                                                          from_type,
                                                                                          from_stride,
                                                                                          from,
                                                                                          to_type,
                                                                                          to_stride,
                                                                                          to);
    } else {
        cu_sshex8_restrict_kernel<From, To><<<n_blocks, block_size, shared_mem_size>>>(nelements,
                                                                                       stride,
                                                                                       from_level,
                                                                                       from_level_stride,
                                                                                       from_elements,
                                                                                       from_element_to_node_incidence_count,
                                                                                       to_level,
                                                                                       to_level_stride,
                                                                                       to_elements,
                                                                                       S.data,
                                                                                       vec_size,
                                                                                       from_type,
                                                                                       from_stride,
                                                                                       from,
                                                                                       to_type,
                                                                                       to_stride,
                                                                                       to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_sshex8_restrict(const ptrdiff_t                     nelements,
                              const ptrdiff_t                     stride,
                              const int                           from_level,
                              const int                           from_level_stride,
                              idx_t *const SFEM_RESTRICT          from_elements,
                              const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                              const int                           to_level,
                              const int                           to_level_stride,
                              idx_t *const SFEM_RESTRICT          to_elements,
                              const int                           vec_size,
                              const enum RealType                 from_type,
                              const ptrdiff_t                     from_stride,
                              const void *const SFEM_RESTRICT     from,
                              const enum RealType                 to_type,
                              const ptrdiff_t                     to_stride,
                              void *const SFEM_RESTRICT           to,
                              void                               *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_sshex8_restrict_tpl<real_t, real_t>(nelements,
                                                          stride,
                                                          from_level,
                                                          from_level_stride,
                                                          from_elements,
                                                          from_element_to_node_incidence_count,
                                                          to_level,
                                                          to_level_stride,
                                                          to_elements,
                                                          vec_size,
                                                          from_type,
                                                          from_stride,
                                                          (real_t *)from,
                                                          to_type,
                                                          to_stride,
                                                          (real_t *)to,
                                                          stream);
        }
        case SFEM_FLOAT32: {
            return cu_sshex8_restrict_tpl<float, float>(nelements,
                                                        stride,
                                                        from_level,
                                                        from_level_stride,
                                                        from_elements,
                                                        from_element_to_node_incidence_count,
                                                        to_level,
                                                        to_level_stride,
                                                        to_elements,
                                                        vec_size,
                                                        from_type,
                                                        from_stride,
                                                        (float *)from,
                                                        to_type,
                                                        to_stride,
                                                        (float *)to,
                                                        stream);
        }
        case SFEM_FLOAT64: {
            return cu_sshex8_restrict_tpl<double, double>(nelements,
                                                          stride,
                                                          from_level,
                                                          from_level_stride,
                                                          from_elements,
                                                          from_element_to_node_incidence_count,
                                                          to_level,
                                                          to_level_stride,
                                                          to_elements,
                                                          vec_size,
                                                          from_type,
                                                          from_stride,
                                                          (double *)from,
                                                          to_type,
                                                          to_stride,
                                                          (double *)to,
                                                          stream);
        }

        default: {
            fprintf(stderr,
                    "[Error]  cu_sshex8_prolongate: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

#define PROLONGATE_IN_KERNEL_BASIS_VERSION  // The other version needs debugging
#ifdef PROLONGATE_IN_KERNEL_BASIS_VERSION

// Even TO sub-elements are used to interpolate from FROM sub-elements
template <typename From, typename To>
__global__ void cu_sshex8_prolongate_kernel(const ptrdiff_t                 nelements,
                                            const ptrdiff_t                 stride,
                                            const int                       from_level,
                                            const int                       from_level_stride,
                                            idx_t *const SFEM_RESTRICT      from_elements,
                                            const int                       to_level,
                                            const int                       to_level_stride,
                                            idx_t *const SFEM_RESTRICT      to_elements,
                                            const int                       vec_size,
                                            const enum RealType             from_type,
                                            const ptrdiff_t                 from_stride,
                                            const From *const SFEM_RESTRICT from,
                                            const enum RealType             to_type,
                                            const ptrdiff_t                 to_stride,
                                            To *const SFEM_RESTRICT         to) {
    static_assert(TILE_SIZE == 8, "This only works with tile size 8!");

    // Uunsigned char necessary for multiple instantiations
    extern __shared__ unsigned char cu_buff[];

    const int step_factor = to_level / from_level;

    // Tile number in group
    const int tile    = threadIdx.x >> 3;   // same as threadIdx.x / 8
    const int n_tiles = blockDim.x >> 3;    // same as blockDim.x / 8
    const int sub_idx = threadIdx.x & 0x7;  // same as threadIdx.x % 8

    From *in = (From *)&cu_buff[tile * TILE_SIZE * sizeof(From)];

    // hex8 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2
    const int zi = (sub_idx >> 2);        // equivalent to sub_idx / 4
    assert(n_tiles * TILE_SIZE == blockDim.x);

    // 1 macro element per tile
    const ptrdiff_t e = blockIdx.x * n_tiles + tile;

    const To between_h = (From)from_level / (To)to_level;

    const int to_even     = is_even(to_level);
    const int from_nloops = from_level + to_even;
    const int to_nloops   = to_level + to_even;

    // Vector loop
    for (int d = 0; d < vec_size; d++) {
        // loop on all FROM micro elements
        for (int from_zi = 0; from_zi < from_nloops; from_zi++) {
            for (int from_yi = 0; from_yi < from_nloops; from_yi++) {
                for (int from_xi = 0; from_xi < from_nloops; from_xi++) {
                    const int off_from_zi = (from_zi + zi);
                    const int off_from_yi = (from_yi + yi);
                    const int off_from_xi = (from_xi + xi);

                    const bool from_exists =
                            e < nelements && off_from_zi <= from_level && off_from_yi <= from_level && off_from_xi <= from_level;

                    // Wait for shared memory transactions to be finished
                    __syncwarp();

                    // Gather
                    if (from_exists) {
                        const int idx_from = cu_sshex8_lidx(from_level * from_level_stride,
                                                            off_from_xi * from_level_stride,
                                                            off_from_yi * from_level_stride,
                                                            off_from_zi * from_level_stride);

                        const idx_t     gidx = from_elements[idx_from * stride + e];
                        const ptrdiff_t idx  = (gidx * vec_size + d) * from_stride;
                        in[sub_idx]          = from[idx];
                    } else {
                        in[sub_idx] = 0;
                    }

                    // Wait for in to be filled
                    __syncwarp();

                    int start_zi = from_zi * step_factor;
                    start_zi += is_odd(start_zi);  // Skip odd numbers

                    int start_yi = from_yi * step_factor;
                    start_yi += is_odd(start_yi);  // Skip odd numbers

                    int start_xi = from_xi * step_factor;
                    start_xi += is_odd(start_xi);  // Skip odd numbers

                    const int end_zi = MIN(to_nloops, start_zi + step_factor);
                    const int end_yi = MIN(to_nloops, start_yi + step_factor);
                    const int end_xi = MIN(to_nloops, start_xi + step_factor);

                    // sub-loop on even TO micro-elements
                    for (int to_zi = start_zi; to_zi < end_zi; to_zi += 2) {
                        for (int to_yi = start_yi; to_yi < end_yi; to_yi += 2) {
                            for (int to_xi = start_xi; to_xi < end_xi; to_xi += 2) {
                                const int off_to_zi = (to_zi + zi);
                                const int off_to_yi = (to_yi + yi);
                                const int off_to_xi = (to_xi + xi);

                                const To x = (off_to_xi - from_xi * step_factor) * between_h;
                                const To y = (off_to_yi - from_yi * step_factor) * between_h;
                                const To z = (off_to_zi - from_zi * step_factor) * between_h;

                                assert(x >= 0);
                                assert(x <= 1);
                                assert(y >= 0);
                                assert(y <= 1);
                                assert(z >= 0);
                                assert(z <= 1);

                                // This requires 64 bytes on the stack frame
                                // Cartesian order
                                To f[8] = {// Bottom
                                           (1 - x) * (1 - y) * (1 - z),
                                           x * (1 - y) * (1 - z),
                                           (1 - x) * y * (1 - z),
                                           x * y * (1 - z),
                                           // Top
                                           (1 - x) * (1 - y) * z,
                                           x * (1 - y) * z,
                                           (1 - x) * y * z,
                                           x * y * z};

#ifndef NDEBUG
                                To pou = 0;
                                for (int i = 0; i < 8; i++) {
                                    pou += f[i];
                                }

                                assert(fabs(1 - pou) < 1e-8);
#endif

                                To out = 0;
                                for (int v = 0; v < 8; v++) {
                                    const int round_robin = ROUND_ROBIN(v, sub_idx);
                                    // There should be no bank conflicts due to round robin
                                    out += f[round_robin] * in[round_robin];
                                }

                                // Check if not ghost nodes for scatter assign
                                const bool to_exists =
                                        e < nelements && off_to_zi <= to_level && off_to_yi <= to_level && off_to_xi <= to_level;

                                if (to_exists) {
                                    // set the value to the output
                                    const int idx_to = cu_sshex8_lidx(to_level * to_level_stride,
                                                                      off_to_xi * to_level_stride,
                                                                      off_to_yi * to_level_stride,
                                                                      off_to_zi * to_level_stride);

                                    to[(to_elements[idx_to * stride + e] * vec_size + d) * to_stride] = out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#else  // PROLONGATE_IN_KERNEL_BASIS_VERSION

template <typename From, typename To>
__global__ void cu_sshex8_prolongate_kernel(const ptrdiff_t                 nelements,
                                            const ptrdiff_t                 stride,
                                            const int                       from_level,
                                            const int                       from_level_stride,
                                            idx_t *const SFEM_RESTRICT      from_elements,
                                            const int                       to_level,
                                            const int                       to_level_stride,
                                            idx_t *const SFEM_RESTRICT      to_elements,
                                            const To *const SFEM_RESTRICT   S,
                                            const int                       vec_size,
                                            const enum RealType             from_type,
                                            const ptrdiff_t                 from_stride,
                                            const From *const SFEM_RESTRICT from,
                                            const enum RealType             to_type,
                                            const ptrdiff_t                 to_stride,
                                            To *const SFEM_RESTRICT         to) {
    static_assert(TILE_SIZE == 8, "This only works with tile size 8!");

    // Uunsigned char necessary for multiple instantiations
    extern __shared__ unsigned char cu_buff[];

    const int step_factor = to_level / from_level;
    const int to_npoints  = to_level + 1;

    // Tile number in group
    const int tile    = threadIdx.x >> 3;   // same as threadIdx.x / 8
    const int n_tiles = blockDim.x >> 3;    // same as blockDim.x / 8
    const int sub_idx = threadIdx.x & 0x7;  // same as threadIdx.x % 8

    // Potential bug ??
    From *in = (From *)&cu_buff[tile * TILE_SIZE * sizeof(From)];

    // hex8 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2
    const int zi = (sub_idx >> 2);        // equivalent to sub_idx / 4
    assert(n_tiles * TILE_SIZE == blockDim.x);

    // 1 macro element per tile
    const ptrdiff_t e = blockIdx.x * n_tiles + tile;

    const int to_even     = is_even(to_level);
    const int from_nloops = from_level + to_even;
    const int to_nloops   = to_level + to_even;

    // if (e == 0) {
    //     printf("%d) %d %d %d\n", sub_idx, xi, yi, zi);
    //     if (!sub_idx) {
    //         printf("from_level %d, to_level %d, step_factor %d, to_even %d, from_nloops %d, to_nloops %d, tile %d, n_tiles
    //         %d\n",
    //                from_level,
    //                to_level,
    //                step_factor,
    //                to_even,
    //                from_nloops,
    //                to_nloops,
    //                tile,
    //                n_tiles);
    //     }
    // }

    // Vector loop
    for (int d = 0; d < vec_size; d++) {
        // loop on all FROM micro elements
        for (int from_zi = 0; from_zi < from_nloops; from_zi++) {
            for (int from_yi = 0; from_yi < from_nloops; from_yi++) {
                for (int from_xi = 0; from_xi < from_nloops; from_xi++) {
                    const int off_from_zi = (from_zi + zi);
                    const int off_from_yi = (from_yi + yi);
                    const int off_from_xi = (from_xi + xi);

                    const bool from_exists =
                            e < nelements && off_from_zi <= from_level && off_from_yi <= from_level && off_from_xi <= from_level;

                    // Wait for shared memory transactions to be finished
                    __syncwarp();

                    // Gather
                    if (from_exists) {
                        const int idx_from = cu_sshex8_lidx(from_level * from_level_stride,
                                                            off_from_zi * from_level_stride,
                                                            off_from_yi * from_level_stride,
                                                            off_from_xi * from_level_stride);

                        const idx_t     gidx = from_elements[idx_from * stride + e];
                        const ptrdiff_t idx  = (gidx * vec_size + d) * from_stride;
                        in[sub_idx]          = from[idx];
                    } else {
                        in[sub_idx] = 0;
                    }

                    // Wait for in to be filled
                    __syncwarp();

                    int start_zi = from_zi * step_factor;
                    start_zi += is_odd(start_zi);  // Skip odd numbers

                    int start_yi = from_yi * step_factor;
                    start_yi += is_odd(start_yi);  // Skip odd numbers

                    int start_xi = from_xi * step_factor;
                    start_xi += is_odd(start_xi);  // Skip odd numbers

                    const int end_zi = MIN(to_nloops, start_zi + step_factor);
                    const int end_yi = MIN(to_nloops, start_yi + step_factor);
                    const int end_xi = MIN(to_nloops, start_xi + step_factor);

                    // sub-loop on even TO micro-elements
                    for (int to_zi = start_zi; to_zi < end_zi; to_zi += 2) {
                        for (int to_yi = start_yi; to_yi < end_yi; to_yi += 2) {
                            for (int to_xi = start_xi; to_xi < end_xi; to_xi += 2) {
                                // Tile-level parallelism due to xi, yi, zi
                                const int off_to_zi = (to_zi + zi);
                                const int off_to_yi = (to_yi + yi);
                                const int off_to_xi = (to_xi + xi);

                                const From *const Sx = &S[(off_to_xi - from_xi)];
                                const From *const Sy = &S[(off_to_yi - from_yi)];
                                const From *const Sz = &S[(off_to_zi - from_zi)];

                                To out = 0;
                                // for (int vz = 0; vz < 2; vz++) {
                                //     const int rrvz = ROUND_ROBIN_2(vz, zi);
                                //     for (int vy = 0; vy < 2; vy++) {
                                //         const int rrvy = ROUND_ROBIN_2(vy, yi);
                                //         for (int vx = 0; vx < 2; vx++) {
                                //             const int rrvx = ROUND_ROBIN_2(vx, xi);
                                //             const int idx  = rrvz * 4 + rrvy * 2 + rrvx;
                                //             out += Sx[rrvx * to_npoints] * Sy[rrvy * to_npoints] * Sz[rrvz * to_npoints] *
                                //                    in[idx];
                                //         }
                                //     }
                                // }

                                for (int vz = 0; vz < 2; vz++) {
                                    for (int vy = 0; vy < 2; vy++) {
                                        for (int vx = 0; vx < 2; vx++) {
                                            const int idx = vz * 4 + vy * 2 + vx;
                                            out += Sx[vx * to_npoints] * Sy[vy * to_npoints] * Sz[vz * to_npoints] * in[idx];
                                        }
                                    }
                                }

                                // Check if not ghost nodes for scatter assign
                                const bool to_exists =
                                        e < nelements && off_to_zi <= to_level && off_to_yi <= to_level && off_to_xi <= to_level;

                                if (to_exists) {
                                    // set the value to the output
                                    const int idx_to = cu_sshex8_lidx(to_level * to_level_stride,
                                                                      off_to_xi * to_level_stride,
                                                                      off_to_yi * to_level_stride,
                                                                      off_to_zi * to_level_stride);

                                    to[(to_elements[idx_to * stride + e] * vec_size + d) * to_stride] = out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif  // PROLONGATE_IN_KERNEL_BASIS_VERSION

template <typename From, typename To>
int cu_sshex8_prolongate_tpl(const ptrdiff_t                 nelements,
                             const ptrdiff_t                 stride,
                             const int                       from_level,
                             const int                       from_level_stride,
                             idx_t *const SFEM_RESTRICT      from_elements,
                             const int                       to_level,
                             const int                       to_level_stride,
                             idx_t *const SFEM_RESTRICT      to_elements,
                             const int                       vec_size,
                             const enum RealType             from_type,
                             const ptrdiff_t                 from_stride,
                             const From *const SFEM_RESTRICT from,
                             const enum RealType             to_type,
                             const ptrdiff_t                 to_stride,
                             To *const SFEM_RESTRICT         to,
                             void                           *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    const int block_size      = 128;
    ptrdiff_t n_blocks        = MAX(ptrdiff_t(1), (nelements + block_size / TILE_SIZE - 1) / (block_size / TILE_SIZE));
    size_t    shared_mem_size = block_size * sizeof(From);

#ifndef PROLONGATE_IN_KERNEL_BASIS_VERSION
    ShapeInterpolation<To> S(to_level / from_level);
#endif

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_sshex8_prolongate_kernel<From, To><<<n_blocks, block_size, shared_mem_size, s>>>(nelements,
                                                                                            stride,
                                                                                            from_level,
                                                                                            from_level_stride,
                                                                                            from_elements,
                                                                                            to_level,
                                                                                            to_level_stride,
                                                                                            to_elements,
#ifndef PROLONGATE_IN_KERNEL_BASIS_VERSION
                                                                                            S.data,
#endif
                                                                                            vec_size,
                                                                                            from_type,
                                                                                            from_stride,
                                                                                            from,
                                                                                            to_type,
                                                                                            to_stride,
                                                                                            to);
    } else {
        cu_sshex8_prolongate_kernel<From, To><<<n_blocks, block_size, shared_mem_size>>>(nelements,
                                                                                         stride,
                                                                                         from_level,
                                                                                         from_level_stride,
                                                                                         from_elements,
                                                                                         to_level,
                                                                                         to_level_stride,
                                                                                         to_elements,
#ifndef PROLONGATE_IN_KERNEL_BASIS_VERSION
                                                                                         S.data,
#endif
                                                                                         vec_size,
                                                                                         from_type,
                                                                                         from_stride,
                                                                                         from,
                                                                                         to_type,
                                                                                         to_stride,
                                                                                         to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_sshex8_prolongate(const ptrdiff_t                 nelements,
                                const ptrdiff_t                 stride,
                                const int                       from_level,
                                const int                       from_level_stride,
                                idx_t *const SFEM_RESTRICT      from_elements,
                                const int                       to_level,
                                const int                       to_level_stride,
                                idx_t *const SFEM_RESTRICT      to_elements,
                                const int                       vec_size,
                                const enum RealType             from_type,
                                const ptrdiff_t                 from_stride,
                                const void *const SFEM_RESTRICT from,
                                const enum RealType             to_type,
                                const ptrdiff_t                 to_stride,
                                void *const SFEM_RESTRICT       to,
                                void                           *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_sshex8_prolongate_tpl<real_t, real_t>(nelements,
                                                            stride,
                                                            from_level,
                                                            from_level_stride,
                                                            from_elements,
                                                            to_level,
                                                            to_level_stride,
                                                            to_elements,
                                                            vec_size,
                                                            from_type,
                                                            from_stride,
                                                            (real_t *)from,
                                                            to_type,
                                                            to_stride,
                                                            (real_t *)to,
                                                            stream);
        }
        case SFEM_FLOAT32: {
            return cu_sshex8_prolongate_tpl<float, float>(nelements,
                                                          stride,
                                                          from_level,
                                                          from_level_stride,
                                                          from_elements,
                                                          to_level,
                                                          to_level_stride,
                                                          to_elements,
                                                          vec_size,
                                                          from_type,
                                                          from_stride,
                                                          (float *)from,
                                                          to_type,
                                                          to_stride,
                                                          (float *)to,
                                                          stream);
        }
        case SFEM_FLOAT64: {
            return cu_sshex8_prolongate_tpl<double, double>(nelements,
                                                            stride,
                                                            from_level,
                                                            from_level_stride,
                                                            from_elements,
                                                            to_level,
                                                            to_level_stride,
                                                            to_elements,
                                                            vec_size,
                                                            from_type,
                                                            from_stride,
                                                            (double *)from,
                                                            to_type,
                                                            to_stride,
                                                            (double *)to,
                                                            stream);
        }

        default: {
            SFEM_ERROR(
                    "[Error]  cu_sshex8_prolongate: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            return SFEM_FAILURE;
        }
    }
}
