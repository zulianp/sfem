#include "cu_tet4_prolongation_restriction.h"

#include <cassert>
#include <cstdio>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ------- PROLONGATION -------- //

template <typename From, typename To>
__global__ void cu_tet4_to_macrotet4_prolongation_kernel(
        const ptrdiff_t coarse_nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const idx_t *const SFEM_RESTRICT fine_node_map,
        const int vec_size,
        const From *const SFEM_RESTRICT from,
        To *const SFEM_RESTRICT to) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < coarse_nnodes;
         i += blockDim.x * gridDim.x) {
        const ptrdiff_t i_offset = i * vec_size;

        for (int v = 0; v < vec_size; v++) {
            to[i_offset + v] = from[i_offset + v];
        }

        const count_t start = coarse_rowptr[i];
        const count_t end = coarse_rowptr[i + 1];
        const int extent = end - start;
        const idx_t *const cols = &coarse_colidx[start];
        const idx_t *const verts = &fine_node_map[start];

        for (int k = 0; k < extent; k++) {
            const ptrdiff_t j = cols[k];
            const idx_t edge = verts[k];
            if (i < j) {
                assert(edge >= coarse_nnodes);

                const ptrdiff_t edge_offset = edge * vec_size;
                const ptrdiff_t j_offset = j * vec_size;
                for (int v = 0; v < vec_size; v++) {
                    const To edge_value = 0.5 * (from[i_offset + v] + from[j_offset + v]);
                    
                    to[edge_offset + v] = edge_value;
                }
            }
        }
    }
}

template <typename From, typename To>
static int cu_tet4_to_macrotet4_prolongation_tpl(const ptrdiff_t coarse_nnodes,
                                                 const count_t *const SFEM_RESTRICT coarse_rowptr,
                                                 const idx_t *const SFEM_RESTRICT coarse_colidx,
                                                 const idx_t *const SFEM_RESTRICT fine_node_map,
                                                 const int vec_size,
                                                 const From *const SFEM_RESTRICT from,
                                                 To *const SFEM_RESTRICT to,
                                                 void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           cu_tet4_to_macrotet4_prolongation_kernel<From, To>,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (coarse_nnodes + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_tet4_to_macrotet4_prolongation_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                coarse_nnodes, coarse_rowptr, coarse_colidx, fine_node_map, vec_size, from, to);
    } else {
        cu_tet4_to_macrotet4_prolongation_kernel<From, To><<<n_blocks, block_size, 0>>>(
                coarse_nnodes, coarse_rowptr, coarse_colidx, fine_node_map, vec_size, from, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_tet4_to_macrotet4_prolongation(const ptrdiff_t coarse_nnodes,
                                             const count_t *const SFEM_RESTRICT coarse_rowptr,
                                             const idx_t *const SFEM_RESTRICT coarse_colidx,
                                             const idx_t *const SFEM_RESTRICT fine_node_map,
                                             const int vec_size,
                                             const enum RealType from_type,
                                             const void *const SFEM_RESTRICT from,
                                             const enum RealType to_type,
                                             void *const SFEM_RESTRICT to,
                                             void *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet4_to_macrotet4_prolongation_tpl(coarse_nnodes,
                                                         coarse_rowptr,
                                                         coarse_colidx,
                                                         fine_node_map,
                                                         vec_size,
                                                         (real_t *)from,
                                                         (real_t *)to,
                                                         stream);
        }
        case SFEM_FLOAT32: {
            return cu_tet4_to_macrotet4_prolongation_tpl(coarse_nnodes,
                                                         coarse_rowptr,
                                                         coarse_colidx,
                                                         fine_node_map,
                                                         vec_size,
                                                         (float *)from,
                                                         (float *)to,
                                                         stream);
        }
        case SFEM_FLOAT64: {
            return cu_tet4_to_macrotet4_prolongation_tpl(coarse_nnodes,
                                                         coarse_rowptr,
                                                         coarse_colidx,
                                                         fine_node_map,
                                                         vec_size,
                                                         (double *)from,
                                                         (double *)to,
                                                         stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_to_macrotet4_prolongation_tpl: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }

    return SFEM_SUCCESS;
}

// ------- RESTRICTION -------- //

template <typename From, typename To>
__global__ void cu_macrotet4_to_tet4_restriction_kernel(
        const ptrdiff_t coarse_nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const idx_t *const SFEM_RESTRICT fine_node_map,
        const int vec_size,
        const From *const SFEM_RESTRICT from,
        To *const SFEM_RESTRICT to) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < coarse_nnodes;
         i += blockDim.x * gridDim.x) {
        for (int v = 0; v < vec_size; v++) {
            atomicAdd(&to[i * vec_size + v], from[i * vec_size + v]);
        }

        const count_t start = coarse_rowptr[i];
        const count_t end = coarse_rowptr[i + 1];
        const int extent = end - start;
        const idx_t *const cols = &coarse_colidx[start];
        const idx_t *const verts = &fine_node_map[start];

        for (int k = 0; k < extent; k++) {
            const idx_t j = cols[k];
            const idx_t edge = verts[k];
            if (i < j) {
                for (int v = 0; v < vec_size; v++) {
                    const real_t edge_value = 0.5 * from[edge * vec_size + v];

                    atomicAdd(&to[i * vec_size + v], edge_value);
                    atomicAdd(&to[j * vec_size + v], edge_value);
                }
            }
        }
    }
}

template <typename From, typename To>
static int cu_macrotet4_to_tet4_restriction_tpl(const ptrdiff_t coarse_nnodes,
                                                const count_t *const SFEM_RESTRICT coarse_rowptr,
                                                const idx_t *const SFEM_RESTRICT coarse_colidx,
                                                const idx_t *const SFEM_RESTRICT fine_node_map,
                                                const int vec_size,
                                                const From *const SFEM_RESTRICT from,
                                                To *const SFEM_RESTRICT to,
                                                void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           cu_macrotet4_to_tet4_restriction_kernel<From, To>,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (coarse_nnodes + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_macrotet4_to_tet4_restriction_kernel<From, To><<<n_blocks, block_size, 0, s>>>(
                coarse_nnodes, coarse_rowptr, coarse_colidx, fine_node_map, vec_size, from, to);
    } else {
        cu_macrotet4_to_tet4_restriction_kernel<From, To><<<n_blocks, block_size, 0>>>(
                coarse_nnodes, coarse_rowptr, coarse_colidx, fine_node_map, vec_size, from, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_macrotet4_to_tet4_restriction(const ptrdiff_t coarse_nnodes,
                                            const count_t *const SFEM_RESTRICT coarse_rowptr,
                                            const idx_t *const SFEM_RESTRICT coarse_colidx,
                                            const idx_t *const SFEM_RESTRICT fine_node_map,
                                            const int vec_size,
                                            const enum RealType from_type,
                                            const void *const SFEM_RESTRICT from,
                                            const enum RealType to_type,
                                            void *const SFEM_RESTRICT to,
                                            void *stream) {
    assert(from_type == to_type && "TODO mixed types!");
    if (from_type != to_type) {
        return SFEM_FAILURE;
    }

    switch (from_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_macrotet4_to_tet4_restriction_tpl(coarse_nnodes,
                                                        coarse_rowptr,
                                                        coarse_colidx,
                                                        fine_node_map,
                                                        vec_size,
                                                        (real_t *)from,
                                                        (real_t *)to,
                                                        stream);
        }
        case SFEM_FLOAT32: {
            return cu_macrotet4_to_tet4_restriction_tpl(coarse_nnodes,
                                                        coarse_rowptr,
                                                        coarse_colidx,
                                                        fine_node_map,
                                                        vec_size,
                                                        (float *)from,
                                                        (float *)to,
                                                        stream);
        }
        case SFEM_FLOAT64: {
            return cu_macrotet4_to_tet4_restriction_tpl(coarse_nnodes,
                                                        coarse_rowptr,
                                                        coarse_colidx,
                                                        fine_node_map,
                                                        vec_size,
                                                        (double *)from,
                                                        (double *)to,
                                                        stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_to_macrotet4_prolongation_tpl: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }

    return SFEM_SUCCESS;
}