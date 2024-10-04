#include "cu_tet4_prolongation_restriction.h"

#include "sfem_cuda_base.h"

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

////////////////////////////////////////////////////////////////////////
template <typename From, typename To>
__global__ void cu_macrotet4_to_tet4_prolongation_element_based_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const int vec_size,
        const ptrdiff_t from_stride,
        const From *const SFEM_RESTRICT from,
        const ptrdiff_t to_stride,
        To *const SFEM_RESTRICT to) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // P1
        const idx_t i0 = elements[0 * stride + e];
        const idx_t i1 = elements[1 * stride + e];
        const idx_t i2 = elements[2 * stride + e];
        const idx_t i3 = elements[3 * stride + e];

        // P2
        const idx_t i4 = elements[4 * stride + e];
        const idx_t i5 = elements[5 * stride + e];
        const idx_t i6 = elements[6 * stride + e];
        const idx_t i7 = elements[7 * stride + e];
        const idx_t i8 = elements[8 * stride + e];
        const idx_t i9 = elements[9 * stride + e];

#ifndef NDEBUG
        if(i0 == i4) {
            printf("[%d %d %d %d %d %d %d %d %d %d] (%ld, %ld)\n", i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, e, stride);
        }
#endif
        assert(i0 != i4);

        for (int v = 0; v < vec_size; v++) {
            const ptrdiff_t ii0 = i0 * vec_size + v;
            const ptrdiff_t ii1 = i1 * vec_size + v;
            const ptrdiff_t ii2 = i2 * vec_size + v;
            const ptrdiff_t ii3 = i3 * vec_size + v;
            const ptrdiff_t ii4 = i4 * vec_size + v;
            const ptrdiff_t ii5 = i5 * vec_size + v;
            const ptrdiff_t ii6 = i6 * vec_size + v;
            const ptrdiff_t ii7 = i7 * vec_size + v;
            const ptrdiff_t ii8 = i8 * vec_size + v;
            const ptrdiff_t ii9 = i9 * vec_size + v;

            to[ii0 * to_stride] = from[ii0 * from_stride];
            to[ii1 * to_stride] = from[ii1 * from_stride];
            to[ii2 * to_stride] = from[ii2 * from_stride];
            to[ii3 * to_stride] = from[ii3 * from_stride];

            to[ii4 * to_stride] = 0.5 * (from[ii0 * from_stride] + from[ii1 * from_stride]);
            to[ii5 * to_stride] = 0.5 * (from[ii1 * from_stride] + from[ii2 * from_stride]);
            to[ii6 * to_stride] = 0.5 * (from[ii0 * from_stride] + from[ii2 * from_stride]);
            to[ii7 * to_stride] = 0.5 * (from[ii0 * from_stride] + from[ii3 * from_stride]);
            to[ii8 * to_stride] = 0.5 * (from[ii1 * from_stride] + from[ii3 * from_stride]);
            to[ii9 * to_stride] = 0.5 * (from[ii2 * from_stride] + from[ii3 * from_stride]);
        }
    }
}

template <typename From, typename To>
static int cu_macrotet4_to_tet4_prolongation_element_based_tpl(const ptrdiff_t nelements,
                                                               const ptrdiff_t stride,
                                                               const idx_t *const SFEM_RESTRICT
                                                                       elements,
                                                               const int vec_size,
                                                               const ptrdiff_t from_stride,
                                                               const From *const SFEM_RESTRICT from,
                                                               const ptrdiff_t to_stride,
                                                               To *const SFEM_RESTRICT to,
                                                               void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_macrotet4_to_tet4_prolongation_element_based_kernel<From, To>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_macrotet4_to_tet4_prolongation_element_based_kernel<From, To>
                <<<n_blocks, block_size, 0, s>>>(
                        nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    } else {
        cu_macrotet4_to_tet4_prolongation_element_based_kernel<From, To>
                <<<n_blocks, block_size, 0>>>(
                        nelements, stride, elements, vec_size, from_stride, from, to_stride, to);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

int cu_macrotet4_to_tet4_prolongation_element_based(const ptrdiff_t nelements,
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
            return cu_macrotet4_to_tet4_prolongation_element_based_tpl(nelements,
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
            return cu_macrotet4_to_tet4_prolongation_element_based_tpl(nelements,
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
            return cu_macrotet4_to_tet4_prolongation_element_based_tpl(nelements,
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
                    "[Error] cu_tet4_to_macrotet4_prolongation_tpl: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

/////////////////////////

template <typename From, typename To>
__global__ void cu_macrotet4_to_tet4_restriction_element_based_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const uint16_t *const SFEM_RESTRICT e2n_count,
        const int vec_size,
        const ptrdiff_t from_stride,
        const From *const SFEM_RESTRICT from,
        const ptrdiff_t to_stride,
        To *const SFEM_RESTRICT to) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        const idx_t i0 = elements[0 * stride + e];
        const idx_t i1 = elements[1 * stride + e];
        const idx_t i2 = elements[2 * stride + e];
        const idx_t i3 = elements[3 * stride + e];

        // P2
        const idx_t i4 = elements[4 * stride + e];
        const idx_t i5 = elements[5 * stride + e];
        const idx_t i6 = elements[6 * stride + e];
        const idx_t i7 = elements[7 * stride + e];
        const idx_t i8 = elements[8 * stride + e];
        const idx_t i9 = elements[9 * stride + e];

        assert(i0 != i4);

        for (int v = 0; v < vec_size; v++) {
            const ptrdiff_t ii0 = i0 * vec_size + v;
            const ptrdiff_t ii1 = i1 * vec_size + v;
            const ptrdiff_t ii2 = i2 * vec_size + v;
            const ptrdiff_t ii3 = i3 * vec_size + v;
            const ptrdiff_t ii4 = i4 * vec_size + v;
            const ptrdiff_t ii5 = i5 * vec_size + v;
            const ptrdiff_t ii6 = i6 * vec_size + v;
            const ptrdiff_t ii7 = i7 * vec_size + v;
            const ptrdiff_t ii8 = i8 * vec_size + v;
            const ptrdiff_t ii9 = i9 * vec_size + v;

            atomicAdd(&to[ii0 * to_stride], from[ii0 * from_stride] / e2n_count[i0]);
            atomicAdd(&to[ii1 * to_stride], from[ii1 * from_stride] / e2n_count[i1]);
            atomicAdd(&to[ii2 * to_stride], from[ii2 * from_stride] / e2n_count[i2]);
            atomicAdd(&to[ii3 * to_stride], from[ii3 * from_stride] / e2n_count[i3]);
            atomicAdd(&to[ii0 * to_stride], from[ii4 * from_stride] * (0.5 / e2n_count[i4]));
            atomicAdd(&to[ii1 * to_stride], from[ii5 * from_stride] * (0.5 / e2n_count[i5]));
            atomicAdd(&to[ii0 * to_stride], from[ii6 * from_stride] * (0.5 / e2n_count[i6]));
            atomicAdd(&to[ii0 * to_stride], from[ii7 * from_stride] * (0.5 / e2n_count[i7]));
            atomicAdd(&to[ii1 * to_stride], from[ii8 * from_stride] * (0.5 / e2n_count[i8]));
            atomicAdd(&to[ii2 * to_stride], from[ii9 * from_stride] * (0.5 / e2n_count[i9]));
            atomicAdd(&to[ii1 * to_stride], from[ii4 * from_stride] * (0.5 / e2n_count[i4]));
            atomicAdd(&to[ii2 * to_stride], from[ii5 * from_stride] * (0.5 / e2n_count[i5]));
            atomicAdd(&to[ii2 * to_stride], from[ii6 * from_stride] * (0.5 / e2n_count[i6]));
            atomicAdd(&to[ii3 * to_stride], from[ii7 * from_stride] * (0.5 / e2n_count[i7]));
            atomicAdd(&to[ii3 * to_stride], from[ii8 * from_stride] * (0.5 / e2n_count[i8]));
            atomicAdd(&to[ii3 * to_stride], from[ii9 * from_stride] * (0.5 / e2n_count[i9]));
        }
    }
}

template <typename From, typename To>
static int cu_macrotet4_to_tet4_restriction_element_based_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
        const int vec_size,
        const ptrdiff_t from_stride,
        const From *const SFEM_RESTRICT from,
        const ptrdiff_t to_stride,
        To *const SFEM_RESTRICT to,
        void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_macrotet4_to_tet4_restriction_element_based_kernel<From, To>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_macrotet4_to_tet4_restriction_element_based_kernel<From, To>
                <<<n_blocks, block_size, 0, s>>>(nelements,
                                                 stride,
                                                 elements,
                                                 element_to_node_incidence_count,
                                                 vec_size,
                                                 from_stride,
                                                 from,
                                                 to_stride,
                                                 to);
    } else {
        cu_macrotet4_to_tet4_restriction_element_based_kernel<From, To>
                <<<n_blocks, block_size, 0>>>(nelements,
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

int cu_macrotet4_to_tet4_restriction_element_based(const ptrdiff_t nelements,
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
            return cu_macrotet4_to_tet4_restriction_element_based_tpl(
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
            return cu_macrotet4_to_tet4_restriction_element_based_tpl(
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
            return cu_macrotet4_to_tet4_restriction_element_based_tpl(
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
                    "[Error] cu_tet4_to_macrotet4_prolongation_tpl: not implemented for type "
                    "%s "
                    "(code %d)\n",
                    real_type_to_string(from_type),
                    from_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
