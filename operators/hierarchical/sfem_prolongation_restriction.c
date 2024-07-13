#include "sfem_prolongation_restriction.h"

#include "sfem_base.h"
#include "sfem_defs.h"

#include <mpi.h>
#include <stddef.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

ptrdiff_t max_node_id(const enum ElemType type,
                      const ptrdiff_t nelements,
                      idx_t **const SFEM_RESTRICT elements) {
    const int nxe = elem_num_nodes(type);
    ptrdiff_t ret = 0;
    for (int i = 0; i < nxe; i++) {
        for (ptrdiff_t e = 0; e < nelements; e++) {
            ret = MAX(ret, elements[i][e]);
        }
    }
    return ret;
}

// We assume that fine indices have a higher id
int hierarchical_create_coarse_indices(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       ptrdiff_t *n_coarse_indices,
                                       idx_t **SFEM_RESTRICT coarse_indices) {
    ptrdiff_t count = 0;
#pragma omp parallel for reduction(+ : count)
    for (ptrdiff_t i = 0; i < n_indices; i++) {
        count += fine_indices[i] <= max_coarse_idx;
    }

    *n_coarse_indices = count;
    *coarse_indices = (idx_t *)malloc(count * sizeof(idx_t));

    count = 0;
    for (ptrdiff_t i = 0; i < n_indices; i++) {
        if (fine_indices[i] <= max_coarse_idx) {
            (*coarse_indices)[count++] = fine_indices[i];
        }
    }

    return 0;
}

int hierarchical_collect_coarse_values(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       const real_t *const SFEM_RESTRICT fine_values,
                                       real_t *const SFEM_RESTRICT coarse_values) {
    ptrdiff_t count = 0;
    for (ptrdiff_t i = 0; i < n_indices; i++) {
        if (fine_indices[i] <= max_coarse_idx) {
            coarse_values[count++] = fine_values[i];
        }
    }

    return 0;
}

int hierarchical_prolongation(const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const int vec_size,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to) {
    if (from_element == TET4 && (to_element == TET10 || to_element == MACRO_TET4)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[0][e];
            const idx_t i1 = elements[1][e];
            const idx_t i2 = elements[2][e];
            const idx_t i3 = elements[3][e];

            // P2
            const idx_t i4 = elements[4][e];
            const idx_t i5 = elements[5][e];
            const idx_t i6 = elements[6][e];
            const idx_t i7 = elements[7][e];
            const idx_t i8 = elements[8][e];
            const idx_t i9 = elements[9][e];

            assert(i0 != i4);

            for (int v = 0; v < vec_size; v++) {
                to[i0 * vec_size + v] = from[i0 * vec_size + v];
                to[i1 * vec_size + v] = from[i1 * vec_size + v];
                to[i2 * vec_size + v] = from[i2 * vec_size + v];
                to[i3 * vec_size + v] = from[i3 * vec_size + v];

                to[i4 * vec_size + v] = 0.5 * (from[i0 * vec_size + v] + from[i1 * vec_size + v]);
                to[i5 * vec_size + v] = 0.5 * (from[i1 * vec_size + v] + from[i2 * vec_size + v]);
                to[i6 * vec_size + v] = 0.5 * (from[i0 * vec_size + v] + from[i2 * vec_size + v]);
                to[i7 * vec_size + v] = 0.5 * (from[i0 * vec_size + v] + from[i3 * vec_size + v]);
                to[i8 * vec_size + v] = 0.5 * (from[i1 * vec_size + v] + from[i3 * vec_size + v]);
                to[i9 * vec_size + v] = 0.5 * (from[i2 * vec_size + v] + from[i3 * vec_size + v]);
            }
        }
    } else if (from_element == TRI3 && (to_element == TRI6 || to_element == MACRO_TRI3)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[0][e];
            const idx_t i1 = elements[1][e];
            const idx_t i2 = elements[2][e];

            // P2
            const idx_t i3 = elements[3][e];
            const idx_t i4 = elements[4][e];
            const idx_t i5 = elements[5][e];

            for (int v = 0; v < vec_size; v++) {
                to[i0 * vec_size + v] = from[i0 * vec_size + v];
                to[i1 * vec_size + v] = from[i1 * vec_size + v];
                to[i2 * vec_size + v] = from[i2 * vec_size + v];

                to[i3 * vec_size + v] = 0.5 * (from[i0 * vec_size + v] + from[i1 * vec_size + v]);
                to[i4 * vec_size + v] = 0.5 * (from[i1 * vec_size + v] + from[i2 * vec_size + v]);
                to[i5 * vec_size + v] = 0.5 * (from[i0 * vec_size + v] + from[i2 * vec_size + v]);
            }
        }
    } else {
        assert(0);
        fprintf(stderr,
                "Unsupported element pair for hierarchical_prolongation %d, %d\n",
                from_element,
                to_element);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    return 0;
}

int build_p1_to_p2_edge_map(const ptrdiff_t nnodes,
                            const count_t *const SFEM_RESTRICT coarse_rowptr,
                            const idx_t *const SFEM_RESTRICT coarse_colidx,
                            idx_t *const SFEM_RESTRICT p2_vertices) {
    ptrdiff_t fine_idx = nnodes;
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t start = coarse_rowptr[i];
        const int extent = coarse_rowptr[i + 1] - start;
        const idx_t *const cols = &coarse_colidx[start];
        idx_t *const verts = &p2_vertices[start];

        for (int k = 0; k < extent; k++) {
            const idx_t j = cols[k];
            if (i < j) {
                verts[k] = fine_idx++;
                // } else if (i == j) {
                //     verts[k] = i;
            } else {
                verts[k] = INVALID_IDX;
            }
        }
    }

// Fill-up the transpose (very expensive)
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const count_t i_start = coarse_rowptr[i];
        const int i_extent = coarse_rowptr[i + 1] - i_start;
        const idx_t *const i_cols = &coarse_colidx[i_start];
        idx_t *const i_verts = &p2_vertices[i_start];

        for (int i_k = 0; i_k < i_extent; i_k++) {
            const idx_t j = i_cols[i_k];

            if (i > j) {
                const count_t j_start = coarse_rowptr[j];
                const int j_extent = coarse_rowptr[j + 1] - j_start;
                const idx_t *const j_cols = &coarse_colidx[j_start];
                const idx_t *const j_verts = &p2_vertices[j_start];

                for (int j_k = 0; j_k < j_extent; j_k++) {
                    const idx_t ii = j_cols[j_k];
                    if (ii == i) {
                        i_verts[i_k] = j_verts[j_k];
                        break;
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int hierarchical_restriction(
        // CRS-node-graph of the coarse mesh
        const ptrdiff_t nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const int vec_size,
        const real_t *const SFEM_RESTRICT from,
        real_t *const SFEM_RESTRICT to) {
    // Serial only
    ptrdiff_t fine_idx = nnodes;
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        for (int v = 0; v < vec_size; v++) {
            to[i * vec_size + v] += from[i * vec_size + v];
        }

        const count_t start = coarse_rowptr[i];
        const count_t end = coarse_rowptr[i + 1];
        const int extent = end - start;
        const idx_t *const cols = &coarse_colidx[start];

        for (int k = 0; k < extent; k++) {
            const idx_t j = cols[k];
            if (i < j) {
                for (int v = 0; v < vec_size; v++) {
                    const real_t edge_value = from[fine_idx * vec_size + v];
                    to[i * vec_size + v] += 0.5 * edge_value;
                    to[j * vec_size + v] += 0.5 * edge_value;
                }

                fine_idx++;
            }
        }
    }

    // For parallel we need some other data-structure

    // ptrdiff_t ncoarse = max_node_id(to_element, nelements, elements) + 1;

    // FIXME: Should we pass this from outside? What is the overhead of the allocation?
    //     int *temp = calloc(ncoarse, sizeof(int));
    //     if (to_element == TET4 && (from_element == TET10 || from_element == MACRO_TET4)) {
    // #pragma omp parallel for
    //         for (ptrdiff_t e = 0; e < nelements; e++) {
    //             // P1
    //             const idx_t i0 = elements[0][e];
    //             const idx_t i1 = elements[1][e];
    //             const idx_t i2 = elements[2][e];
    //             const idx_t i3 = elements[3][e];

    //             // P2
    //             const idx_t i4 = elements[4][e];
    //             const idx_t i5 = elements[5][e];
    //             const idx_t i6 = elements[6][e];
    //             const idx_t i7 = elements[7][e];
    //             const idx_t i8 = elements[8][e];
    //             const idx_t i9 = elements[9][e];

    // #pragma omp atomic update
    //             to[i0] += from[i0] + 0.5 * (from[i4] + from[i6] + from[i7]);

    // #pragma omp atomic update
    //             to[i1] += from[i1] + 0.5 * (from[i4] + from[i5] + from[i8]);

    // #pragma omp atomic update
    //             to[i2] += from[i2] + 0.5 * (from[i5] + from[i6] + from[i9]);

    // #pragma omp atomic update
    //             to[i3] += from[i3] + 0.5 * (from[i7] + from[i8] + from[i9]);

    // #pragma omp atomic update
    //             temp[i0] += 1;
    // #pragma omp atomic update
    //             temp[i1] += 1;
    // #pragma omp atomic update
    //             temp[i2] += 1;
    // #pragma omp atomic update
    //             temp[i3] += 1;
    //         }

    //     } else if (to_element == TRI3 && (from_element == TRI6 || from_element ==
    //     MACRO_TRI3)) {
    // #pragma omp parallel for
    //         for (ptrdiff_t e = 0; e < nelements; e++) {
    //             // P1
    //             const idx_t i0 = elements[0][e];
    //             const idx_t i1 = elements[1][e];
    //             const idx_t i2 = elements[2][e];

    //             // P2
    //             const idx_t i3 = elements[3][e];
    //             const idx_t i4 = elements[4][e];
    //             const idx_t i5 = elements[5][e];

    // #pragma omp atomic update
    //             to[i0] += from[i0] + 0.5 * (from[i3] + from[i5]);

    // #pragma omp atomic update
    //             to[i1] += from[i1] + 0.5 * (from[i3] + from[i4]);

    // #pragma omp atomic update
    //             to[i2] += from[i2] + 0.5 * (from[i4] + from[i5]);

    // #pragma omp atomic update
    //             temp[i0] += 1;
    // #pragma omp atomic update
    //             temp[i1] += 1;
    // #pragma omp atomic update
    //             temp[i2] += 1;
    //         }
    //     } else {
    //         assert(0);
    //         fprintf(stderr,
    //                 "Unsupported element pair for hierarchical_restriction %d, %d\n",
    //                 from_element,
    //                 to_element);
    //         MPI_Abort(MPI_COMM_WORLD, 1);
    //         return 1;
    //     }

    // #pragma omp parallel for
    //     for (ptrdiff_t i = 0; i < ncoarse; i++) {
    //         to[i] /= temp[i];
    //     }

    //     free(temp);
    return 0;
}


int hierarchical_restriction_with_edge_map(
    // CRS-node-graph of the coarse mesh
    const ptrdiff_t nnodes,
    const count_t *const SFEM_RESTRICT coarse_rowptr,
    const idx_t *const SFEM_RESTRICT coarse_colidx,
    const idx_t *const SFEM_RESTRICT p2_vertices,
    const int vec_size,
    const real_t *const SFEM_RESTRICT from,
    real_t *const SFEM_RESTRICT to)
{
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        for (int v = 0; v < vec_size; v++) {
#pragma omp atomic update
            to[i * vec_size + v] += from[i * vec_size + v];
        }

        const count_t start = coarse_rowptr[i];
        const count_t end = coarse_rowptr[i + 1];
        const int extent = end - start;
        const idx_t *const cols = &coarse_colidx[start];
        const idx_t *const verts = &p2_vertices[start];

        for (int k = 0; k < extent; k++) {
            const idx_t j = cols[k];
            const idx_t edge = verts[k];
            if (i < j) {
                for (int v = 0; v < vec_size; v++) {
                    const real_t edge_value = from[edge * vec_size + v];

#pragma omp atomic update
                    to[i * vec_size + v] += 0.5 * edge_value;

#pragma omp atomic update
                    to[j * vec_size + v] += 0.5 * edge_value;
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
