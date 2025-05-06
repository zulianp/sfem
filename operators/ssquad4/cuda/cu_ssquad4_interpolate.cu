#include "cu_ssquad4_interpolate.h"

#include "sfem_cuda_base.h"
#include "sfem_macros.h"

#include "cu_ssquad4_inline.cuh"

#include <cassert>
#include <cstdio>
#include <vector>

// PROLONGATION

template <typename From, typename To>
__global__ void cu_ssquad4_hierarchical_prolongation_kernel(const int                        level,
                                                            const ptrdiff_t                  nelements,
                                                            const ptrdiff_t                  stride,
                                                            const idx_t *const SFEM_RESTRICT elements,
                                                            const int                        vec_size,
                                                            const ptrdiff_t                  from_stride,
                                                            const From *const SFEM_RESTRICT  from,
                                                            const ptrdiff_t                  to_stride,
                                                            To *const SFEM_RESTRICT          to) {
    const int corners[4] = {// Bottom
                            cu_ssquad4_lidx(level, 0, 0),
                            cu_ssquad4_lidx(level, level, 0),
                            cu_ssquad4_lidx(level, level, level),
                            cu_ssquad4_lidx(level, 0, level)};

    const scalar_t h = 1. / level;

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int yi = 0; yi < level + 1; yi++) {
            for (int xi = 0; xi < level + 1; xi++) {
                idx_t idx = elements[cu_ssquad4_lidx(level, xi, yi) * stride + e];

                const scalar_t x = xi * h;
                const scalar_t y = yi * h;

                // Evaluate Hex8 basis functions at x, y, z
                const scalar_t xm = (1 - x);
                const scalar_t ym = (1 - y);

                scalar_t f[4];
                f[0] = xm * ym;  // (0, 0)
                f[1] = x * ym;   // (1, 0)
                f[2] = x * y;    // (1, 1)
                f[3] = xm * y;   // (0, 1)

                for (int d = 0; d < vec_size; d++) {
                    scalar_t val = 0;

                    for (int v = 0; v < 4; v++) {
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

template <typename From, typename To>
static int cu_ssquad4_hierarchical_prolongation_tpl(const int                        level,
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
                &min_grid_size, &block_size, cu_ssquad4_hierarchical_prolongation_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_ssquad4_hierarchical_prolongation_kernel<From, To>
                <<<n_blocks, block_size, 0, s>>>(level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_ssquad4_hierarchical_prolongation_kernel<From, To>
                <<<n_blocks, block_size, 0>>>(level, nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_ssquad4_hierarchical_prolongation(const int                        level,
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
            return cu_ssquad4_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (real_t *)from, to_stride, (real_t *)to, stream);
        }
        case SFEM_FLOAT32: {
            return cu_ssquad4_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (float *)from, to_stride, (float *)to, stream);
        }
        case SFEM_FLOAT64: {
            return cu_ssquad4_hierarchical_prolongation_tpl(
                    level, nelements, stride, elements, vec_size, from_stride, (double *)from, to_stride, (double *)to, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error]  cu_ssquad4_prolongation_tpl: not implemented for type %s "
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
__global__ void cu_ssquad4_hierarchical_restriction_kernel(const int                           level,
                                                           const ptrdiff_t                     nelements,
                                                           const ptrdiff_t                     stride,
                                                           const idx_t *const SFEM_RESTRICT    elements,
                                                           const uint16_t *const SFEM_RESTRICT e2n_count,
                                                           const int                           vec_size,
                                                           const ptrdiff_t                     from_stride,
                                                           const From *const SFEM_RESTRICT     from,
                                                           const ptrdiff_t                     to_stride,
                                                           To *const SFEM_RESTRICT             to) {
    const int corners[4] = {// Bottom
                            cu_ssquad4_lidx(level, 0, 0),
                            cu_ssquad4_lidx(level, level, 0),
                            cu_ssquad4_lidx(level, level, level),
                            cu_ssquad4_lidx(level, 0, level)};

    const scalar_t h = 1. / level;
    scalar_t       acc[4];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int d = 0; d < vec_size; d++) {
            for (int i = 0; i < 4; i++) {
                acc[i] = 0;
            }

            for (int yi = 0; yi < level + 1; yi++) {
                for (int xi = 0; xi < level + 1; xi++) {
                    const int       lidx = cu_ssquad4_lidx(level, xi, yi);
                    const ptrdiff_t idx  = elements[lidx * stride + e];

                    const scalar_t x = xi * h;
                    const scalar_t y = yi * h;

                    // Evaluate Quad4 basis functions at x, y, z
                    const scalar_t xm = (1 - x);
                    const scalar_t ym = (1 - y);

                    scalar_t f[4];
                    f[0] = xm * ym;  // (0, 0, 0)
                    f[1] = x * ym;   // (1, 0, 0)
                    f[2] = x * y;    // (1, 1, 0)
                    f[3] = xm * y;   // (0, 1, 0)

                    const ptrdiff_t global_from_idx = (idx * vec_size + d) * from_stride;
                    const scalar_t  val             = from[global_from_idx] / e2n_count[idx];

                    assert(from[global_from_idx] == from[global_from_idx]);
                    assert(e2n_count[idx] > 0);
                    assert(val == val);

                    for (int i = 0; i < 4; i++) {
                        acc[i] += f[i] * val;
                    }
                }
            }

            for (int v = 0; v < 4; v++) {
                const ptrdiff_t global_to_idx = (elements[corners[v] * stride + e] * vec_size + d) * to_stride;
                atomicAdd(&to[global_to_idx], acc[v]);
            }
        }
    }
}

template <typename From, typename To>
static int cu_ssquad4_hierarchical_restriction_tpl(const int                           level,
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
                &min_grid_size, &block_size, cu_ssquad4_hierarchical_restriction_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_ssquad4_hierarchical_restriction_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_ssquad4_hierarchical_restriction_kernel<From, To><<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_ssquad4_hierarchical_restriction(const int                           level,
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
            return cu_ssquad4_hierarchical_restriction_tpl(level,
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
            return cu_ssquad4_hierarchical_restriction_tpl(level,
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
            return cu_ssquad4_hierarchical_restriction_tpl(level,
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
                    "[Error]  cu_ssquad4_prolongation_tpl: not implemented for type "
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
__global__ void cu_ssquad4_hierarchical_restriction_SoA_kernel(const int                           level,
                                                               const ptrdiff_t                     nelements,
                                                               idx_t **const SFEM_RESTRICT         elements,
                                                               const uint16_t *const SFEM_RESTRICT e2n_count,
                                                               const int                           vec_size,
                                                               const ptrdiff_t                     from_stride,
                                                               const From *const SFEM_RESTRICT     from,
                                                               const ptrdiff_t                     to_stride,
                                                               To *const SFEM_RESTRICT             to) {
    const int corners[4] = {// Bottom
                            cu_ssquad4_lidx(level, 0, 0),
                            cu_ssquad4_lidx(level, level, 0),
                            cu_ssquad4_lidx(level, level, level),
                            cu_ssquad4_lidx(level, 0, level)};

    const scalar_t h = 1. / level;
    scalar_t       acc[4];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        for (int d = 0; d < vec_size; d++) {
            for (int i = 0; i < 4; i++) {
                acc[i] = 0;
            }

            for (int yi = 0; yi < level + 1; yi++) {
                for (int xi = 0; xi < level + 1; xi++) {
                    const int       lidx = cu_ssquad4_lidx(level, xi, yi);
                    const ptrdiff_t idx  = elements[lidx][e];

                    const scalar_t x = xi * h;
                    const scalar_t y = yi * h;

                    // Evaluate Quad4 basis functions at x, y, z
                    const scalar_t xm = (1 - x);
                    const scalar_t ym = (1 - y);

                    scalar_t f[4];
                    f[0] = xm * ym;  // (0, 0, 0)
                    f[1] = x * ym;   // (1, 0, 0)
                    f[2] = x * y;    // (1, 1, 0)
                    f[3] = xm * y;   // (0, 1, 0)

                    const ptrdiff_t global_from_idx = (idx * vec_size + d) * from_stride;
                    const scalar_t  val             = from[global_from_idx] / e2n_count[idx];

                    assert(from[global_from_idx] == from[global_from_idx]);
                    assert(e2n_count[idx] > 0);
                    assert(val == val);

                    for (int i = 0; i < 4; i++) {
                        acc[i] += f[i] * val;
                    }
                }
            }

            for (int v = 0; v < 4; v++) {
                const ptrdiff_t global_to_idx = (elements[corners[v]][e] * vec_size + d) * to_stride;
                atomicAdd(&to[global_to_idx], acc[v]);
            }
        }
    }
}

template <typename From, typename To>
static int cu_ssquad4_hierarchical_restriction_SoA_tpl(const int                           level,
                                                       const ptrdiff_t                     nelements,
                                                       idx_t **const SFEM_RESTRICT         elements,
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
                &min_grid_size, &block_size, cu_ssquad4_hierarchical_restriction_SoA_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_ssquad4_hierarchical_restriction_SoA_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                level, nelements, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_ssquad4_hierarchical_restriction_SoA_kernel<From, To><<<n_blocks, block_size, 0>>>(
                level, nelements, elements, element_to_node_incidence_count, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_ssquad4_hierarchical_restriction_SoA(const int                           level,
                                                   const ptrdiff_t                     nelements,
                                                   idx_t **const SFEM_RESTRICT         elements,
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
            return cu_ssquad4_hierarchical_restriction_SoA_tpl(level,
                                                               nelements,
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
            return cu_ssquad4_hierarchical_restriction_SoA_tpl(level,
                                                               nelements,
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
            return cu_ssquad4_hierarchical_restriction_SoA_tpl(level,
                                                               nelements,
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
                    "[Error]  cu_ssquad4_prolongation_tpl: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

static const int TILE_SIZE = 4;
#define ROUND_ROBIN(val, shift) ((val + shift) & (TILE_SIZE - 1))
#define ROUND_ROBIN_2(val, shift) ((val + shift) & (2 - 1))

static inline __device__ int is_even(const int v) { return !(v & 1); }
static inline __device__ int is_odd(const int v) { return (v & 1); }

template <typename T>
class ShapeInterpolation {
public:
    T     *data{nullptr};
    size_t nodes{0};
    int    stride{0};

    ShapeInterpolation(const int steps, const int padding = 0) {
        nodes  = (steps + 1);
        stride = nodes + padding;
        std::vector<T> S_host(2 * stride, 0);
        double         h = 1. / steps;
        for (int i = 0; i < nodes; i++) {
            S_host[0 * stride + i] = (1 - h * i);
            S_host[1 * stride + i] = h * i;
        }

        auto nbytes = S_host.size() * sizeof(T);

        SFEM_CUDA_CHECK(cudaMalloc((void **)&data, nbytes));
        SFEM_CUDA_CHECK(cudaMemcpy(data, S_host.data(), nbytes, cudaMemcpyHostToDevice));
    }

    ~ShapeInterpolation() { cudaFree(data); }
};

template <typename From, typename To>
__global__ void cu_ssquad4_restrict_kernel(const ptrdiff_t nelements,
                                           // const ptrdiff_t                     stride,
                                           const int                           from_level,
                                           const int                           from_level_stride,
                                           idx_t **const SFEM_RESTRICT         from_elements,
                                           const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                                           const int                           to_level,
                                           const int                           to_level_stride,
                                           idx_t **const SFEM_RESTRICT         to_elements,
                                           const To *const SFEM_RESTRICT       S,
                                           const int                           vec_size,
                                           const ptrdiff_t                     from_stride,
                                           const From *const SFEM_RESTRICT     from,
                                           const ptrdiff_t                     to_stride,
                                           To *const SFEM_RESTRICT             to) {
    static_assert(TILE_SIZE == 4,
                  "This only works with tile size 4 because the implementation assumes a fixed tile size for shared memory "
                  "layout and indexing.");

    // Unsigned char necessary for multiple template instantiations of this kernel
    extern __shared__ unsigned char cu_buff[];

    const int step_factor = from_level / to_level;

    // Tile number in group
    const int tile    = threadIdx.x >> 4;
    const int n_tiles = blockDim.x >> 2;
    const int sub_idx = threadIdx.x & 0x3;

    From *in = (From *)&cu_buff[tile * TILE_SIZE * sizeof(From)];

    // hex8 idx
    const int xi = sub_idx & 0x1;
    const int yi = (sub_idx >> 1) & 0x1;

    assert(n_tiles * TILE_SIZE == blockDim.x);

    // // 1 macro element per tile
    const ptrdiff_t e         = blockIdx.x * n_tiles + tile;
    const int       from_even = is_even(from_level);
    const int       to_nloops = to_level + from_even;

    // Add padding (Make sure that S has the correct padding)
    const int S_stride = step_factor + 1 + from_even;

    // // Vector loop
    for (int d = 0; d < vec_size; d++) {
        // loop on all TO micro elements

        for (int to_yi = 0; to_yi < to_nloops; to_yi++) {
            for (int to_xi = 0; to_xi < to_nloops; to_xi++) {
                // Attention: parallelism in tile using xi, yi, zi
                To acc = 0;

                const int y_start = to_yi * step_factor;
                const int x_start = to_xi * step_factor;

                const int y_odd = is_odd(y_start);
                const int x_odd = is_odd(x_start);

                // Figure out local index range

                const int y_end = (y_start == to_level ? 1 : step_factor) + y_odd;
                const int x_end = (x_start == to_level ? 1 : step_factor) + x_odd;

                for (int from_yi = y_odd; from_yi < y_end; from_yi += 2) {
                    for (int from_xi = x_odd; from_xi < x_end; from_xi += 2) {
                        // Parallel read from global mem on 4 fine nodes
                        {
                            const int yy = y_start + from_yi;
                            const int xx = x_start + from_xi;

                            const int off_from_yi = (yy + yi);
                            const int off_from_xi = (xx + xi);

                            const bool from_exists = e < nelements &&              // Check element exists
                                                     off_from_yi <= from_level &&  //
                                                     off_from_xi <= from_level;

                            __syncwarp();
                            if (from_exists) {
                                const int idx_from = cu_ssquad4_lidx(from_level * from_level_stride,
                                                                     off_from_xi * from_level_stride,
                                                                     off_from_yi * from_level_stride);

                                const ptrdiff_t gidx = from_elements[idx_from][e];
                                const ptrdiff_t idx  = (gidx * vec_size + d) * from_stride;
                                in[sub_idx]          = from[idx] / from_element_to_node_incidence_count[gidx];
                            } else {
                                in[sub_idx] = 0;
                            }
                            __syncwarp();
                        }

                        const To *const Sy = &S[yi * S_stride + from_yi];
                        const To *const Sx = &S[xi * S_stride + from_xi];

                        for (int dy = 0; dy < 2; dy++) {
                            const int rrdy = ROUND_ROBIN_2(dy, yi);
                            for (int dx = 0; dx < 2; dx++) {
                                const int rrdx = ROUND_ROBIN_2(dx, xi);
                                // No bank conflicts due to round robin for single precision
                                const To c = in[rrdy * 2 + rrdx];
                                const To f = Sx[rrdx] * Sy[rrdy];
                                assert(f >= 0);
                                assert(f <= 1);
                                acc += c * f;
                            }
                        }
                    }
                }

                // Parallel accumulate on 4 coarse nodes
                const int off_to_yi = to_yi + yi;
                const int off_to_xi = to_xi + xi;

                const bool exists = e < nelements &&          // Check element exists
                                    off_to_yi <= to_level &&  //
                                    off_to_xi <= to_level;
                if (exists) {
                    const int idx_to =
                            cu_ssquad4_lidx(to_level * to_level_stride, off_to_xi * to_level_stride, off_to_yi * to_level_stride);

                    const ptrdiff_t gidx = to_elements[idx_to][e];
                    const ptrdiff_t idx  = (gidx * vec_size + d) * to_stride;
                    atomicAdd(&to[idx], acc);
                }
            }
        }
    }
}

template <typename From, typename To>
int cu_ssquad4_restrict_tpl(const ptrdiff_t nelements,
                            // const ptrdiff_t                     stride,
                            const int                           from_level,
                            const int                           from_level_stride,
                            idx_t **const SFEM_RESTRICT         from_elements,
                            const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                            const int                           to_level,
                            const int                           to_level_stride,
                            idx_t **const SFEM_RESTRICT         to_elements,
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
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_ssquad4_restrict_kernel<From, To>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size / TILE_SIZE - 1) / (block_size / TILE_SIZE));

    ShapeInterpolation<To> S(from_level / to_level, from_level % 2 == 0);

    size_t shared_mem_size = block_size * sizeof(From);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_ssquad4_restrict_kernel<From, To><<<n_blocks, block_size, shared_mem_size, s>>>(nelements,
                                                                                           // stride,
                                                                                           from_level,
                                                                                           from_level_stride,
                                                                                           from_elements,
                                                                                           from_element_to_node_incidence_count,
                                                                                           to_level,
                                                                                           to_level_stride,
                                                                                           to_elements,
                                                                                           S.data,
                                                                                           vec_size,
                                                                                           from_stride,
                                                                                           from,
                                                                                           to_stride,
                                                                                           to);
    } else {
        cu_ssquad4_restrict_kernel<From, To><<<n_blocks, block_size, shared_mem_size>>>(nelements,
                                                                                        // stride,
                                                                                        from_level,
                                                                                        from_level_stride,
                                                                                        from_elements,
                                                                                        from_element_to_node_incidence_count,
                                                                                        to_level,
                                                                                        to_level_stride,
                                                                                        to_elements,
                                                                                        S.data,
                                                                                        vec_size,
                                                                                        from_stride,
                                                                                        from,
                                                                                        to_stride,
                                                                                        to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_ssquad4_restrict(const ptrdiff_t nelements,
                               // const ptrdiff_t                     stride,
                               const int                           from_level,
                               const int                           from_level_stride,
                               idx_t **const SFEM_RESTRICT         from_elements,
                               const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                               const int                           to_level,
                               const int                           to_level_stride,
                               idx_t **const SFEM_RESTRICT         to_elements,
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

    if (to_level == 1) {
        return cu_ssquad4_hierarchical_restriction_SoA(from_level,
                                                       nelements,
                                                       from_elements,
                                                       from_element_to_node_incidence_count,
                                                       vec_size,
                                                       from_type,
                                                       from_stride,
                                                       from,
                                                       to_type,
                                                       to_stride,
                                                       to,
                                                       stream);
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_ssquad4_restrict_tpl<real_t, real_t>(nelements,
                                                           // stride,
                                                           from_level,
                                                           from_level_stride,
                                                           from_elements,
                                                           from_element_to_node_incidence_count,
                                                           to_level,
                                                           to_level_stride,
                                                           to_elements,
                                                           vec_size,
                                                           from_stride,
                                                           (real_t *)from,
                                                           to_stride,
                                                           (real_t *)to,
                                                           stream);
        }
        case SFEM_FLOAT32: {
            return cu_ssquad4_restrict_tpl<float, float>(nelements,
                                                         // stride,
                                                         from_level,
                                                         from_level_stride,
                                                         from_elements,
                                                         from_element_to_node_incidence_count,
                                                         to_level,
                                                         to_level_stride,
                                                         to_elements,
                                                         vec_size,
                                                         from_stride,
                                                         (float *)from,
                                                         to_stride,
                                                         (float *)to,
                                                         stream);
        }
        case SFEM_FLOAT64: {
            return cu_ssquad4_restrict_tpl<double, double>(nelements,
                                                           // stride,
                                                           from_level,
                                                           from_level_stride,
                                                           from_elements,
                                                           from_element_to_node_incidence_count,
                                                           to_level,
                                                           to_level_stride,
                                                           to_elements,
                                                           vec_size,
                                                           from_type,
                                                           (double *)from,
                                                           to_type,
                                                           (double *)to,
                                                           stream);
        }

        default: {
            fprintf(stderr,
                    "[Error]  cu_ssquad4_prolongate: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

// Even TO sub-elements are used to interpolate from FROM sub-elements
template <typename From, typename To>
__global__ void cu_ssquad4_prolongate_kernel(const ptrdiff_t                 nelements,
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
    static_assert(TILE_SIZE == 4, "This only works with tile size 8!");

    // Uunsigned char necessary for multiple instantiations
    extern __shared__ unsigned char cu_buff[];

    const int step_factor = to_level / from_level;

    // Tile number in group
    const int tile    = threadIdx.x >> 4;
    const int n_tiles = blockDim.x >> 2;
    const int sub_idx = threadIdx.x & 0x3;

    From *in = (From *)&cu_buff[tile * TILE_SIZE * sizeof(From)];

    // quad4 idx
    const int xi = sub_idx & 0x1;         // equivalent to sub_idx % 2
    const int yi = (sub_idx >> 1) & 0x1;  // equivalent to (sub_idx / 2) % 2

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

        for (int from_yi = 0; from_yi < from_nloops; from_yi++) {
            const int off_from_yi = (from_yi + yi);

            for (int from_xi = 0; from_xi < from_nloops; from_xi++) {
                const int off_from_xi = (from_xi + xi);

                const bool from_exists = e < nelements && off_from_yi <= from_level && off_from_xi <= from_level;

                // Wait for shared memory transactions to be finished
                __syncwarp();

                // Gather
                if (from_exists) {
                    const int idx_from = cu_ssquad4_lidx(
                            from_level * from_level_stride, off_from_xi * from_level_stride, off_from_yi * from_level_stride);

                    const idx_t     gidx = from_elements[idx_from * stride + e];
                    const ptrdiff_t idx  = (gidx * vec_size + d) * from_stride;
                    in[sub_idx]          = from[idx];
                } else {
                    in[sub_idx] = 0;
                }

                // Wait for in to be filled
                __syncwarp();

                int start_yi = from_yi * step_factor;
                start_yi += is_odd(start_yi);  // Skip odd numbers

                int start_xi = from_xi * step_factor;
                start_xi += is_odd(start_xi);  // Skip odd numbers

                const int end_yi = MIN(to_nloops, start_yi + step_factor);
                const int end_xi = MIN(to_nloops, start_xi + step_factor);

                // sub-loop on even TO micro-elements

                for (int to_yi = start_yi; to_yi < end_yi; to_yi += 2) {
                    for (int to_xi = start_xi; to_xi < end_xi; to_xi += 2) {
                        const int off_to_yi = (to_yi + yi);
                        const int off_to_xi = (to_xi + xi);

                        const To x = (off_to_xi - from_xi * step_factor) * between_h;
                        const To y = (off_to_yi - from_yi * step_factor) * between_h;

                        assert(x >= 0);
                        assert(x <= 1);
                        assert(y >= 0);
                        assert(y <= 1);

                        // This requires 64 bytes on the stack frame
                        // Cartesian order
                        To f[4] = {// Bottom
                                   (1 - x) * (1 - y),
                                   x * (1 - y),
                                   (1 - x) * y,
                                   x * y};

#ifndef NDEBUG
                        To pou = 0;
                        for (int i = 0; i < 4; i++) {
                            pou += f[i];
                        }

                        assert(fabs(1 - pou) < 1e-8);
#endif

                        To out = 0;
                        for (int v = 0; v < 4; v++) {
                            const int round_robin = ROUND_ROBIN(v, sub_idx);
                            // There should be no bank conflicts due to round robin
                            out += f[round_robin] * in[round_robin];
                        }

                        // Check if not ghost nodes for scatter assign
                        const bool to_exists = e < nelements && off_to_yi <= to_level && off_to_xi <= to_level;

                        if (to_exists) {
                            // set the value to the output
                            const int idx_to = cu_ssquad4_lidx(
                                    to_level * to_level_stride, off_to_xi * to_level_stride, off_to_yi * to_level_stride);

                            to[(to_elements[idx_to * stride + e] * vec_size + d) * to_stride] = out;
                        }
                    }
                }
            }
        }
    }
}

template <typename From, typename To>
int cu_ssquad4_prolongate_tpl(const ptrdiff_t                 nelements,
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

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_ssquad4_prolongate_kernel<From, To><<<n_blocks, block_size, shared_mem_size, s>>>(nelements,
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
                                                                                             from,
                                                                                             to_type,
                                                                                             to_stride,
                                                                                             to);
    } else {
        cu_ssquad4_prolongate_kernel<From, To><<<n_blocks, block_size, shared_mem_size>>>(nelements,
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
                                                                                          from,
                                                                                          to_type,
                                                                                          to_stride,
                                                                                          to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_ssquad4_prolongate(const ptrdiff_t                 nelements,
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
            return cu_ssquad4_prolongate_tpl<real_t, real_t>(nelements,
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
            return cu_ssquad4_prolongate_tpl<float, float>(nelements,
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
            return cu_ssquad4_prolongate_tpl<double, double>(nelements,
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
                    "[Error]  cu_ssquad4_prolongate: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            return SFEM_FAILURE;
        }
    }
}
