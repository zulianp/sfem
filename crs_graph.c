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

// #include "bitonic.h"

// https://dirtyhandscoding.github.io/posts/vectorizing-small-fixed-size-sort.html
// https://xhad1234.github.io/Parallel-Sort-Merge-Join-in-Peloton/
// https://github.com/sid1607/avx2-merge-sort/blob/master/merge_sort.h
// https://onlinelibrary.wiley.com/doi/full/10.1002/spe.2922

SFEM_INLINE int ispow2(idx_t n) { return n && (!(n & (n - 1))); }

SFEM_INLINE static int cmpfunc(const void *a, const void *b) { return (*(idx_t *)a - *(idx_t *)b); }
SFEM_INLINE static void quicksort(idx_t *arr, idx_t size) { qsort(arr, size, sizeof(idx_t), cmpfunc); }
SFEM_INLINE static idx_t unique(idx_t *arr, idx_t size) {
    idx_t *first = arr;
    idx_t *last = arr + size;

    if (first == last) return 0;

    idx_t *result = first;
    while (++first != last)
        if (*result != *first && ++result != first) *result = *first;

    return (++result) - arr;
}

// idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
//     idx_t *ptr = bsearch(&key, arr, size, sizeof(idx_t), cmpfunc);
//     if (!ptr) return -1;
//     return (idx_t)(ptr - arr);
// }

// SFEM_INLINE static int choose(int condition, int valTrue, int valFalse) {
//     return (condition * valTrue) | (!condition * valFalse);
// }

// idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t n) {
//     idx_t start = 0;
//     idx_t end = n - 1;
//     while ((end - start) > 1) {
//         const int midPoint = (end + start) / 2;
//         const int c = key < arr[midPoint];
//         end = choose(c, midPoint, end);
//         start = choose(!c, midPoint, start);
//     }
//     return choose(key < arr[end], start, end);
// }

idx_t find_idx(const idx_t target, const idx_t * x, idx_t n) {
    for (idx_t i = 0; i < n; ++i) {
        if (target == x[i]) {
            return i;
        }
    }

    return n;
}

int build_n2e(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              idx_t *const elems[4],
              idx_t **out_n2eptr,
              idx_t **out_elindex) {
    double tick = MPI_Wtime();

    idx_t *n2eptr = (idx_t *)malloc((nnodes + 1) * sizeof(idx_t));
    memset(n2eptr, 0, nnodes * sizeof(idx_t));

    int *bookkepping = (int *)malloc((nnodes) * sizeof(int));
    memset(bookkepping, 0, (nnodes) * sizeof(int));

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        for (idx_t i = 0; i < nelements; ++i) {
            assert(elems[edof_i][i] < nnodes);
            assert(elems[edof_i][i] >= 0);

            ++n2eptr[elems[edof_i][i] + 1];
        }
    }

    for (idx_t i = 0; i < nnodes; ++i) {
        n2eptr[i + 1] += n2eptr[i];
    }

    idx_t *elindex = (idx_t *)malloc(n2eptr[nnodes] * sizeof(idx_t));

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        for (idx_t i = 0; i < nelements; ++i) {
            idx_t node = elems[edof_i][i];

            assert(n2eptr[node] + bookkepping[node] < n2eptr[node + 1]);

            elindex[n2eptr[node] + bookkepping[node]++] = i;
        }
    }

    free(bookkepping);

    *out_n2eptr = n2eptr;
    *out_elindex = elindex;

    double tock = MPI_Wtime();
    printf("crs_graph.c: build_n2e\t\t%g seconds\n", tock - tick);
    return 0;
}

int build_crs_graph_mem_conservative(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t *const elems[4],
                                     idx_t **out_rowptr,
                                     idx_t **out_colidx) {
    ptrdiff_t nnz = 0;
    idx_t *rowptr = (idx_t *)malloc((nnodes + 1) * sizeof(idx_t));
    idx_t *colidx = 0;

    {
        idx_t *n2eptr;
        idx_t *elindex;
        build_n2e(nelements, nnodes, elems, &n2eptr, &elindex);

        rowptr[0] = 0;

        idx_t n2nbuff[2048];
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = n2eptr[node];
            idx_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            // quicksort(n2nbuff, nneighs);
            // nneighs = unique(n2nbuff, nneighs);

            nneighs = sortreduce(n2nbuff, nneighs);

            nnz += nneighs;
            rowptr[node + 1] = nnz;
        }

        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

        ptrdiff_t coloffset = 0;
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = n2eptr[node];
            idx_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            quicksort(n2nbuff, nneighs);
            nneighs = unique(n2nbuff, nneighs);

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

int build_crs_graph_faster(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t *const elems[4],
                           idx_t **out_rowptr,
                           idx_t **out_colidx) {


    ptrdiff_t nnz = 0;
    idx_t *rowptr = (idx_t *)malloc((nnodes + 1) * sizeof(idx_t));
    idx_t *colidx = 0;

    {
        idx_t *n2eptr;
        idx_t *elindex;
        build_n2e(nelements, nnodes, elems, &n2eptr, &elindex);

        double tick = MPI_Wtime();

        rowptr[0] = 0;

        ptrdiff_t overestimated_nnz = 0;
        idx_t n2nbuff[2048];
        for (idx_t node = 0; node < nnodes; ++node) {
            const idx_t ebegin = n2eptr[node];
            const idx_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
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
        for (idx_t node = 0; node < nnodes; ++node) {
            const idx_t ebegin = n2eptr[node];
            const idx_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                const idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            // quicksort(n2nbuff, nneighs);
            // nneighs = unique(n2nbuff, nneighs);
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

int build_crs_graph(const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t *const elems[4],
                    idx_t **out_rowptr,
                    idx_t **out_colidx) {
    int SFEM_CRS_MEM_CONSERVATIVE = 0;
    SFEM_READ_ENV(SFEM_CRS_MEM_CONSERVATIVE, atoi);

    if (SFEM_CRS_MEM_CONSERVATIVE) {
        return build_crs_graph_mem_conservative(nelements, nnodes, elems, out_rowptr, out_colidx);
    } else {
        return build_crs_graph_faster(nelements, nnodes, elems, out_rowptr, out_colidx);
    }
}
