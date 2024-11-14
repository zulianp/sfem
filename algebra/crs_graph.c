#include "crs_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "sortreduce.h"

#include "sfem_defs.h"

// #include "bitonic.h"

// https://dirtyhandscoding.github.io/posts/vectorizing-small-fixed-size-sort.html
// https://xhad1234.github.io/Parallel-Sort-Merge-Join-in-Peloton/
// https://github.com/sid1607/avx2-merge-sort/blob/master/merge_sort.h
// https://onlinelibrary.wiley.com/doi/full/10.1002/spe.2922

SFEM_INLINE int ispow2(idx_t n) { return n && (!(n & (n - 1))); }

// SFEM_INLINE static int cmpfunc(const void *a, const void *b) { return (*(idx_t *)a - *(idx_t
// *)b); } SFEM_INLINE static void quicksort(idx_t *arr, idx_t size) { qsort(arr, size,
// sizeof(idx_t), cmpfunc); } SFEM_INLINE static idx_t unique(idx_t *arr, idx_t size) {
//     idx_t *first = arr;
//     idx_t *last = arr + size;

//     if (first == last) return 0;

//     idx_t *result = first;
//     while (++first != last)
//         if (*result != *first && ++result != first) *result = *first;

//     return (++result) - arr;
// }

idx_t find_idx(const idx_t target, const idx_t *x, idx_t n) {
    for (idx_t i = 0; i < n; ++i) {
        if (target == x[i]) {
            return i;
        }
    }

    return n;
}

int build_n2e(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              const int nnodesxelem,
              idx_t **const elems,
              count_t **out_n2eptr,
              element_idx_t **out_elindex) {
    double tick = MPI_Wtime();

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e: allocating %g GB\n", (nnodes + 1) * sizeof(count_t) * 1e-9);
#endif

    count_t *n2eptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    memset(n2eptr, 0, (nnodes + 1) * sizeof(count_t));

    int *book_keeping = (int *)malloc((nnodes) * sizeof(int));
    memset(book_keeping, 0, (nnodes) * sizeof(int));

    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            assert(elems[edof_i][i] < nnodes);
            assert(elems[edof_i][i] >= 0);

            ++n2eptr[elems[edof_i][i] + 1];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        n2eptr[i + 1] += n2eptr[i];
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e: allocating %g GB\n", n2eptr[nnodes] * sizeof(element_idx_t) * 1e-9);
#endif
    element_idx_t *elindex = (element_idx_t *)malloc(n2eptr[nnodes] * sizeof(element_idx_t));

    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            element_idx_t node = elems[edof_i][i];

            assert(n2eptr[node] + book_keeping[node] < n2eptr[node + 1]);

            elindex[n2eptr[node] + book_keeping[node]++] = i;
        }
    }

    free(book_keeping);

    *out_n2eptr = n2eptr;
    *out_elindex = elindex;

    double tock = MPI_Wtime();
    printf("crs_graph.c: build_n2e\t\t%g seconds\n", tock - tick);
    return 0;
}

int build_n2e_for_elem_type(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const elems,
                            count_t **out_n2eptr,
                            element_idx_t **out_elindex) {
    // TODO (maybe)
    if (element_type != MACRO_TET4 /*&& element_type != MACRO_TRI3*/) {
        return build_n2e(
                nelements, nnodes, elem_num_nodes(element_type), elems, out_n2eptr, out_elindex);
    }

    double tick = MPI_Wtime();

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e_for_elem_type: allocating %g GB\n", (nnodes + 1) * sizeof(count_t) * 1e-9);
#endif

    count_t *n2eptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    memset(n2eptr, 0, (nnodes + 1) * sizeof(count_t));

    int *book_keeping = (int *)malloc((nnodes) * sizeof(int));
    memset(book_keeping, 0, (nnodes) * sizeof(int));

    if (element_type == MACRO_TET4) {
        static const int tet4_refine_pattern[8][4] = {// Corner tests
                                                      {0, 4, 6, 7},
                                                      {4, 1, 5, 8},
                                                      {6, 5, 2, 9},
                                                      {7, 8, 9, 3},
                                                      // Octahedron tets
                                                      {4, 5, 6, 8},
                                                      {7, 4, 6, 8},
                                                      {6, 5, 9, 8},
                                                      {7, 6, 9, 8}};

        for (int sub_elem = 0; sub_elem < 8; sub_elem++) {
            for (int sub_elem_node = 0; sub_elem_node < 4; ++sub_elem_node) {
                int node_number = tet4_refine_pattern[sub_elem][sub_elem_node];

                for (ptrdiff_t i = 0; i < nelements; ++i) {
                    assert(elems[node_number][i] < nnodes);
                    assert(elems[node_number][i] >= 0);

                    ++n2eptr[elems[node_number][i] + 1];
                }
            }
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        n2eptr[i + 1] += n2eptr[i];
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e: allocating %g GB\n", n2eptr[nnodes] * sizeof(element_idx_t) * 1e-9);
#endif
    element_idx_t *elindex = (element_idx_t *)malloc(n2eptr[nnodes] * sizeof(element_idx_t));

    const int nnodesxelem = elem_num_nodes(element_type);
    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            element_idx_t node = elems[edof_i][i];

            assert(n2eptr[node] + book_keeping[node] < n2eptr[node + 1]);

            elindex[n2eptr[node] + book_keeping[node]++] = i;
        }
    }

    free(book_keeping);

    *out_n2eptr = n2eptr;
    *out_elindex = elindex;

    double tock = MPI_Wtime();
    printf("crs_graph.c: build_n2e\t\t%g seconds\n", tock - tick);
    return 0;
}

static int build_crs_graph_from_n2e(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    const int nnodesxelem,
                                    idx_t **const SFEM_RESTRICT elems,
                                    const count_t *const SFEM_RESTRICT n2eptr,
                                    const element_idx_t *const SFEM_RESTRICT elindex,
                                    count_t **out_rowptr,
                                    idx_t **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t *colidx = 0;

    {
        rowptr[0] = 0;

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                count_t ebegin = n2eptr[node];
                count_t eend = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    assert(eidx < nelements);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[edof_i][eidx];
                        assert(nneighs < 4096);
                        n2nbuff[nneighs++] = neighnode;
                    }
                }

                nneighs = sortreduce(n2nbuff, nneighs);
                rowptr[node + 1] = nneighs;
            }
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[nnodes];
        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                count_t ebegin = n2eptr[node];
                count_t eend = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    assert(eidx < nelements);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[edof_i][eidx];
                        assert(nneighs < 4096);
                        n2nbuff[nneighs++] = neighnode;
                    }
                }

                nneighs = sortreduce(n2nbuff, nneighs);

                for (idx_t i = 0; i < nneighs; ++i) {
                    colidx[rowptr[node] + i] = n2nbuff[i];
                }
            }
        }
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
}

static int build_crs_graph_mem_conservative(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            const int nnodesxelem,
                                            idx_t **const elems,
                                            count_t **out_rowptr,
                                            idx_t **out_colidx) {
    double tick = MPI_Wtime();

    count_t *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, nnodesxelem, elems, &n2eptr, &elindex);

    int err = build_crs_graph_from_n2e(
            nelements, nnodes, nnodesxelem, elems, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return err;
}

static int build_crs_graph_faster(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  const int nnodesxelem,
                                  idx_t **const elems,
                                  count_t **out_rowptr,
                                  idx_t **out_colidx) {
    ptrdiff_t nnz = 0;
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t *colidx = 0;

    {
        count_t *n2eptr;
        element_idx_t *elindex;
        build_n2e(nelements, nnodes, nnodesxelem, elems, &n2eptr, &elindex);

        double tick = MPI_Wtime();

        rowptr[0] = 0;

        ptrdiff_t overestimated_nnz = 0;
#pragma omp parallel for reduction(+ : overestimated_nnz)
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const count_t ebegin = n2eptr[node];
            const count_t eend = n2eptr[node + 1];
            idx_t nneighs = (eend - ebegin) * nnodesxelem;
            overestimated_nnz += nneighs;
        }

        colidx = (idx_t *)malloc(overestimated_nnz * sizeof(idx_t));

        double tock = MPI_Wtime();
        printf("crs_graph.c: overestimate nnz\t%g seconds\n", tock - tick);
        tick = tock;

        ptrdiff_t coloffset = 0;
        idx_t n2nbuff[2048];
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const count_t ebegin = n2eptr[node];
            const count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;
            for (count_t e = ebegin; e < eend; ++e) {
                const idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    const idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);

            nnz += nneighs;
            rowptr[node + 1] = nnz;

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[coloffset + i] = n2nbuff[i];
            }

            coloffset += nneighs;
        }

        free(n2eptr);
        free(elindex);

        tock = MPI_Wtime();
        printf("crs_graph.c: build nz structure\t%g seconds\n", tock - tick);
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
}

int build_crs_graph_for_elem_type(const int element_type,
                                  const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const elems,
                                  count_t **out_rowptr,
                                  idx_t **out_colidx) {
    int SFEM_CRS_FAST_SERIAL = 0;
    SFEM_READ_ENV(SFEM_CRS_FAST_SERIAL, atoi);

    if (SFEM_CRS_FAST_SERIAL) {
        return build_crs_graph_faster(
                nelements, nnodes, elem_num_nodes(element_type), elems, out_rowptr, out_colidx);
    }

    return build_crs_graph_mem_conservative(
            nelements, nnodes, elem_num_nodes(element_type), elems, out_rowptr, out_colidx);
}

int build_crs_graph_from_element(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 int nxe,
                                 idx_t **const elems,
                                 count_t **out_rowptr,
                                 idx_t **out_colidx) {
    int SFEM_CRS_FAST_SERIAL = 0;
    SFEM_READ_ENV(SFEM_CRS_FAST_SERIAL, atoi);

    if (SFEM_CRS_FAST_SERIAL) {
        return build_crs_graph_faster(nelements, nnodes, nxe, elems, out_rowptr, out_colidx);
    }

    return build_crs_graph_mem_conservative(nelements, nnodes, nxe, elems, out_rowptr, out_colidx);
}

int build_crs_graph(const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const elems,
                    count_t **out_rowptr,
                    idx_t **out_colidx) {
    return build_crs_graph_for_elem_type(4, nelements, nnodes, elems, out_rowptr, out_colidx);
}

int build_crs_graph_3(const ptrdiff_t nelements,
                      const ptrdiff_t nnodes,
                      idx_t **const elems,
                      count_t **out_rowptr,
                      idx_t **out_colidx) {
    return build_crs_graph_for_elem_type(3, nelements, nnodes, elems, out_rowptr, out_colidx);
}

int block_crs_to_crs(const ptrdiff_t nnodes,
                     const int block_size,
                     const count_t *const block_rowptr,
                     const idx_t *const block_colidx,
                     const real_t *const block_values,
                     count_t *const rowptr,
                     idx_t *const colidx,
                     real_t *const values) {
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        count_t k = block_rowptr[i] * (block_size * block_size);
        count_t ncols = block_rowptr[i + 1] - block_rowptr[i];

        for (int b = 0; b < block_size; ++b) {
            rowptr[i * block_size + b] = k + ncols * (b * block_size);
        }
    }

    rowptr[nnodes * block_size] =
            2 * rowptr[nnodes * block_size - 1] - rowptr[nnodes * block_size - 2];

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        // Block row
        const count_t bstart = block_rowptr[i];
        const count_t bend = block_rowptr[i + 1];

        for (int brow = 0; brow < block_size; ++brow) {
            const idx_t row = i * block_size + brow;
            // Scalar row
            const count_t start = rowptr[row];
#ifndef NDEBUG
            const count_t end = rowptr[row + 1];
#endif

            for (count_t bk = bstart, k = start; bk < bend; ++bk) {
                // Block column
                const idx_t bcolidx = block_colidx[bk];
                // Data block
                const real_t *block = &block_values[bk * block_size * block_size];

                for (int bcol = 0; bcol < block_size; ++bcol, ++k) {
                    assert(k < end);

                    colidx[k] = bcolidx * block_size + bcol;
                    values[k] = block[bcol * block_size + brow];
                }
            }
        }
    }

    return 0;
}

int crs_graph_block_to_scalar(const ptrdiff_t nnodes,
                              const int block_size,
                              const count_t *const block_rowptr,
                              const idx_t *const block_colidx,
                              count_t *const rowptr,
                              idx_t *const colidx) {
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        count_t k = block_rowptr[i] * (block_size * block_size);
        count_t ncols = block_rowptr[i + 1] - block_rowptr[i];

        for (int b = 0; b < block_size; ++b) {
            rowptr[i * block_size + b] = k + ncols * (b * block_size);
        }
    }

    rowptr[nnodes * block_size] =
            2 * rowptr[nnodes * block_size - 1] - rowptr[nnodes * block_size - 2];

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        // Block row
        const count_t bstart = block_rowptr[i];
        const count_t bend = block_rowptr[i + 1];

        for (int brow = 0; brow < block_size; ++brow) {
            const idx_t row = i * block_size + brow;
            // Scalar row
            const count_t start = rowptr[row];
#ifndef NDEBUG
            const count_t end = rowptr[row + 1];
#endif

            for (count_t bk = bstart, k = start; bk < bend; ++bk) {
                // Block column
                const idx_t bcolidx = block_colidx[bk];
                // Data block
                for (int bcol = 0; bcol < block_size; ++bcol, ++k) {
                    assert(k < end);

                    colidx[k] = bcolidx * block_size + bcol;
                }
            }
        }
    }

    return 0;
}

int create_dual_graph_mem_conservative(const ptrdiff_t n_elements,
                                       const ptrdiff_t n_nodes,
                                       const int element_type,
                                       idx_t **const elems,
                                       count_t **out_dual_eptr,
                                       element_idx_t **out_dual_eidx) {
    count_t *n2eptr = 0;
    element_idx_t *elindex = 0;

    if (element_type == TET10) {
        build_n2e(n_elements, n_nodes, elem_num_nodes(TET4), elems, &n2eptr, &elindex);
    } else {
        build_n2e(n_elements, n_nodes, elem_num_nodes(element_type), elems, &n2eptr, &elindex);
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("create_dual_graph_mem_conservative: allocating %g GB\n",
           n_elements * sizeof(int) * 1e-9);
#endif

    int *connection_counter = (int *)malloc(n_elements * sizeof(int));
    memset(connection_counter, 0, n_elements * sizeof(int));

    const int n_sides = elem_num_sides(element_type);
    int n_nodes_per_elem = elem_num_nodes(element_type);

    // Optimize for Tet10
    if (element_type == TET10) {
        n_nodes_per_elem = 4;
    }

    enum ElemType st = side_type(element_type);
    int n_nodes_per_side = elem_num_nodes(st);

    if (element_type == TET10) {
        n_nodes_per_side = 3;
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("create_dual_graph_mem_conservative: allocating %g GB\n",
           (n_elements + 1) * sizeof(count_t) * 1e-9);
#endif
    count_t *dual_e_ptr = (count_t *)calloc((n_elements + 1), sizeof(count_t));

    const ptrdiff_t n_overestimated_connections = n_elements * n_sides;
    // +1 more to avoid illegal access when counting self
    size_t extra_buffer_space = 1000;

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("create_dual_graph_mem_conservative: allocating %g GB\n",
           (n_overestimated_connections + extra_buffer_space) * sizeof(element_idx_t) * 1e-9);
#endif

    element_idx_t *dual_eidx = (element_idx_t *)calloc(
            n_overestimated_connections + extra_buffer_space, sizeof(element_idx_t));

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        count_t offset = dual_e_ptr[e];
        element_idx_t *elist = &dual_eidx[offset];

        int count_common = 0;
        for (int en = 0; en < n_nodes_per_elem; en++) {
            const idx_t node = elems[en][e];

            for (count_t eii = n2eptr[node]; eii < n2eptr[node + 1]; eii++) {
                const element_idx_t e_adj = elindex[eii];
                assert(e_adj < n_elements);

                if (connection_counter[e_adj] == 0) {
                    assert(offset + count_common <
                           n_overestimated_connections + extra_buffer_space);
                    elist[count_common++] = e_adj;
                }

                connection_counter[e_adj]++;
            }
        }

        // if(!connection_counter[e]) {
        //     elist[count_common++] = e;
        // }

        connection_counter[e] = 0;

        int actual_count = 0;
        for (int ec = 0; ec < count_common; ec++) {
            element_idx_t l = elist[ec];
            int overlap = connection_counter[l];
            assert(overlap <= n_nodes_per_elem);

            if (overlap == n_nodes_per_side) {
                elist[actual_count++] = l;
            }

            connection_counter[l] = 0;
        }

        dual_e_ptr[e + 1] = actual_count + offset;
    }

    free(n2eptr);
    free(elindex);
    free(connection_counter);

    *out_dual_eptr = dual_e_ptr;
    *out_dual_eidx = dual_eidx;
    return 0;
}

int create_dual_graph(const ptrdiff_t n_elements,
                      const ptrdiff_t n_nodes,
                      const int element_type,
                      idx_t **const elems,
                      count_t **out_rowptr,
                      element_idx_t **out_colidx) {
    double tick = MPI_Wtime();
    const int ret = create_dual_graph_mem_conservative(
            n_elements, n_nodes, element_type, elems, out_rowptr, out_colidx);

    double tock = MPI_Wtime();
    printf("crs_graph.c: create_dual_graph\t%g seconds\n", tock - tick);

    return ret;
}

static int build_crs_graph_upper_triangular_from_n2e(const ptrdiff_t nelements,
                                                     const ptrdiff_t nnodes,
                                                     const int nnodesxelem,
                                                     idx_t **const SFEM_RESTRICT elems,
                                                     const count_t *const SFEM_RESTRICT n2eptr,
                                                     const element_idx_t *const SFEM_RESTRICT
                                                             elindex,
                                                     count_t **out_rowptr,
                                                     idx_t **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t *colidx = 0;

    {
        rowptr[0] = 0;


        {
#pragma omp parallel for
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                idx_t n2nbuff[4096];

                count_t ebegin = n2eptr[node];
                count_t eend = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    assert(eidx < nelements);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[edof_i][eidx];
                        if (neighnode > node) {
                            assert(nneighs < 4096);
                            n2nbuff[nneighs++] = neighnode;
                        }
                    }

                    nneighs = sortreduce(n2nbuff, nneighs);
                    rowptr[node + 1] = nneighs;
                }
            }

            // Cumulative sum
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                rowptr[node + 1] += rowptr[node];
            }

            const ptrdiff_t nnz = rowptr[nnodes];
            colidx = (idx_t *)malloc(nnz * sizeof(idx_t));


            {
#pragma omp parallel for
                for (ptrdiff_t node = 0; node < nnodes; ++node) {
                    idx_t n2nbuff[4096];
                    count_t ebegin = n2eptr[node];
                    count_t eend = n2eptr[node + 1];

                    idx_t nneighs = 0;

                    for (count_t e = ebegin; e < eend; ++e) {
                        element_idx_t eidx = elindex[e];
                        assert(eidx < nelements);

                        for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                            idx_t neighnode = elems[edof_i][eidx];
                            if (neighnode > node) {
                                assert(nneighs < 4096);
                                n2nbuff[nneighs++] = neighnode;
                            }
                        }

                        nneighs = sortreduce(n2nbuff, nneighs);

                        for (idx_t i = 0; i < nneighs; ++i) {
                            colidx[rowptr[node] + i] = n2nbuff[i];
                        }
                    }
                }
            }
        }
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
}

int build_crs_graph_upper_triangular_from_element(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  int nxe,
                                                  idx_t **const elems,
                                                  count_t **out_rowptr,
                                                  idx_t **out_colidx) {
    double tick = MPI_Wtime();

    count_t *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, nxe, elems, &n2eptr, &elindex);

    int err = build_crs_graph_upper_triangular_from_n2e(
            nelements, nnodes, nxe, elems, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return err;
}
