#include "crs_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "sortreduce.h"

#include "sfem_defs.h"

// #include "bitonic.h"

// https://dirtyhandscoding.github.io/posts/vectorizing-small-fixed-size-sort.html
// https://xhad1234.github.io/Parallel-Sort-Merge-Join-in-Peloton/
// https://github.com/sid1607/avx2-merge-sort/blob/master/merge_sort.h
// https://onlinelibrary.wiley.com/doi/full/10.1002/spe.2922

SFEM_INLINE int ispow2(idx_t n) { return n && (!(n & (n - 1))); }

// SFEM_INLINE static int cmpfunc(const void *a, const void *b) { return (*(idx_t *)a - *(idx_t *)b); }
// SFEM_INLINE static void quicksort(idx_t *arr, idx_t size) { qsort(arr, size, sizeof(idx_t), cmpfunc); }
// SFEM_INLINE static idx_t unique(idx_t *arr, idx_t size) {
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
              idx_t **out_elindex) {
    double tick = MPI_Wtime();

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

    idx_t *elindex = (idx_t *)malloc(n2eptr[nnodes] * sizeof(idx_t));

    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
        for (idx_t i = 0; i < nelements; ++i) {
            idx_t node = elems[edof_i][i];

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

static int build_crs_graph_mem_conservative(const ptrdiff_t nelements,
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
        idx_t *elindex;
        build_n2e(nelements, nnodes, nnodesxelem, elems, &n2eptr, &elindex);

        rowptr[0] = 0;

        idx_t n2nbuff[2048];
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);

            nnz += nneighs;
            rowptr[node + 1] = nnz;
        }

        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

        ptrdiff_t coloffset = 0;
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[coloffset + i] = n2nbuff[i];
            }

            coloffset += nneighs;
        }

        free(n2eptr);
        free(elindex);
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
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
        idx_t *elindex;
        build_n2e(nelements, nnodes, nnodesxelem, elems, &n2eptr, &elindex);

        double tick = MPI_Wtime();

        rowptr[0] = 0;

        ptrdiff_t overestimated_nnz = 0;
        idx_t n2nbuff[2048];
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const count_t ebegin = n2eptr[node];
            const count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    const idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            overestimated_nnz += nneighs;
        }

        colidx = (idx_t *)malloc(overestimated_nnz * sizeof(idx_t));

        double tock = MPI_Wtime();
        printf("crs_graph.c: overestimate nnz\t%g seconds\n", tock - tick);
        tick = tock;

        ptrdiff_t coloffset = 0;
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
    int SFEM_CRS_MEM_CONSERVATIVE = 0;
    SFEM_READ_ENV(SFEM_CRS_MEM_CONSERVATIVE, atoi);

    if (SFEM_CRS_MEM_CONSERVATIVE) {
        return build_crs_graph_mem_conservative(nelements, nnodes, element_type, elems, out_rowptr, out_colidx);
    } else {
        return build_crs_graph_faster(nelements, nnodes, element_type, elems, out_rowptr, out_colidx);
    }
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

    rowptr[nnodes * block_size] = 2 * rowptr[nnodes * block_size - 1] - rowptr[nnodes * block_size - 2];

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

int create_dual_graph_mem_conservative(const ptrdiff_t n_elements,
                                       const ptrdiff_t n_nodes,
                                       const int element_type,
                                       idx_t **const elems,
                                       count_t **out_dual_eptr,
                                       idx_t **out_dual_eidx) {
    count_t *n2eptr = 0;
    idx_t *elindex = 0;

    if(element_type == TET10) {
        build_n2e(n_elements, n_nodes, TET4, elems, &n2eptr, &elindex);
    } else {
        build_n2e(n_elements, n_nodes, element_type, elems, &n2eptr, &elindex);
    }

    int *connection_counter = (int *)malloc(n_elements * sizeof(int));
    memset(connection_counter, 0, n_elements * sizeof(int));

    const int n_sides = elem_num_sides(element_type);
    int n_nodes_per_elem = elem_num_nodes(element_type);

    // Optimize for Tet10
    if(element_type == TET10) {
        n_nodes_per_elem = 4;
    }
   
    enum ElemType st = side_type(element_type);
    int n_nodes_per_side = elem_num_nodes(st);

    if(element_type == TET10) {
        n_nodes_per_side = 3;
    }

    count_t *dual_e_ptr = (count_t *)malloc((n_elements + 1) * sizeof(count_t));
    memset(dual_e_ptr, 0, (n_elements + 1) * sizeof(count_t));

    const ptrdiff_t n_overestimated_connections = n_elements * n_sides;

    // +1 more to avoid illegal access when counting self
    idx_t *dual_eidx = (idx_t *)malloc((n_overestimated_connections + 1) * sizeof(idx_t));
    memset(dual_eidx, 0, (n_overestimated_connections + 1) * sizeof(idx_t));

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        count_t offset = dual_e_ptr[e];
        idx_t *elist = &dual_eidx[offset];

        int count_common = 0;
        for (int en = 0; en < n_nodes_per_elem; en++) {
            const idx_t node = elems[en][e];

            for (idx_t eii = n2eptr[node]; eii < n2eptr[node + 1]; eii++) {
                const idx_t e_adj = elindex[eii];

                if (connection_counter[e_adj] == 0) {
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
            idx_t l = elist[ec];
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
                      idx_t **out_colidx) {
    if (1) {
        return create_dual_graph_mem_conservative(n_elements, n_nodes, element_type, elems, out_rowptr, out_colidx);
    } else {
        count_t *n2eptr = 0;
        idx_t *elindex = 0;
        build_n2e(n_elements, n_nodes, element_type, elems, &n2eptr, &elindex);

        count_t *e_ptr = (count_t *)malloc((n_elements + 1) * sizeof(count_t));
        memset(e_ptr, 0, (n_elements + 1) * sizeof(count_t));

        ptrdiff_t n_overestimated_connections = 0;
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            const count_t e_begin = n2eptr[node];
            const count_t e_end = n2eptr[node + 1];

            for (count_t e1 = e_begin; e1 < e_end; ++e1) {
                const idx_t e_idx_1 = elindex[e1];

                for (count_t e2 = e_begin; e2 < e_end; ++e2) {
                    const idx_t e_idx_2 = elindex[e2];
                    if (e_idx_1 == e_idx_2) continue;

                    n_overestimated_connections++;
                    e_ptr[e_idx_1 + 1]++;
                }
            }
        }

        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            e_ptr[e + 1] += e_ptr[e];
        }

        count_t *book_keeping = (count_t *)malloc((n_elements + 1) * sizeof(count_t));
        memset(book_keeping, 0, (n_elements + 1) * sizeof(count_t));

        idx_t *elem_1 = (idx_t *)malloc(element_type * sizeof(idx_t));
        idx_t *elem_2 = (idx_t *)malloc(element_type * sizeof(idx_t));

        idx_t *connections = (idx_t *)malloc(n_overestimated_connections * sizeof(idx_t));
        memset(connections, 0, n_overestimated_connections * sizeof(idx_t));

        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            const count_t e_begin = n2eptr[node];
            const count_t e_end = n2eptr[node + 1];

            for (count_t e1 = e_begin; e1 < e_end; ++e1) {
                const idx_t e_idx_1 = elindex[e1];

                for (count_t e2 = e_begin; e2 < e_end; ++e2) {
                    const idx_t e_idx_2 = elindex[e2];
                    if (e_idx_1 == e_idx_2) continue;

                    const count_t idx = e_ptr[e_idx_1] + book_keeping[e_idx_1]++;
                    connections[idx] = e_idx_2;
                }
            }
        }

        memset(book_keeping, 0, (n_elements + 1) * sizeof(count_t));

        {
            free(elindex);
            free(n2eptr);

            elindex = 0;
            n2eptr = 0;
        }

        ptrdiff_t offset = 0;
        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            const count_t e_begin = e_ptr[e];
            const count_t e_end = e_ptr[e + 1];
            const count_t e_range = e_end - e_begin;

            const count_t n_neighs = sortreduce(&connections[e_begin], e_range);

            // Compress
            for (count_t k = 0; k < n_neighs; k++) {
                connections[offset++] = connections[e_begin + k];
            }

            book_keeping[e + 1] = n_neighs + book_keeping[e];
        }

        memset(e_ptr, 0, (n_elements + 1) * sizeof(count_t));
        idx_t *dual_connections = (idx_t *)malloc(book_keeping[n_elements] * sizeof(idx_t));
        memset(dual_connections, 0, book_keeping[n_elements] * sizeof(idx_t));

        for (ptrdiff_t e = 0; e < n_elements; ++e) {
            const count_t e_begin = book_keeping[e];
            const count_t e_end = book_keeping[e + 1];
            const count_t e_extent = e_end - e_begin;

            e_ptr[e + 1] += e_ptr[e];

            for (int d = 0; d < element_type; ++d) {
                elem_1[d] = elems[d][e];
            }

            sort_idx(elem_1, element_type);

            for (count_t k = 0; k < e_extent; ++k) {
                const count_t ke = connections[e_begin + k];
                if (e == ke) continue;

                for (int d = 0; d < element_type; ++d) {
                    elem_2[d] = elems[d][ke];
                }

                sort_idx(elem_2, element_type);

                int count_same = 0;
                for (int k1 = 0; k1 < element_type; k1++) {
                    for (int k2 = 0; k2 < element_type; k2++) {
                        count_same += (elem_1[k1] == elem_2[k2]);
                    }
                }

                assert(count_same < element_type);
                if (count_same == (element_type - 1)) {
                    dual_connections[e_ptr[e + 1]++] = ke;
                }
            }
        }

        {
            free(elem_1);
            free(elem_2);

            free(book_keeping);
            free(connections);
        }

        *out_rowptr = e_ptr;
        *out_colidx = dual_connections;
        return 0;
    }
}
