#include "crs_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

// https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/

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

idx_t binarysearch(const idx_t key, const idx_t *arr, idx_t size) {
    idx_t *ptr = bsearch(&key, arr, size, sizeof(idx_t), cmpfunc);
    if (!ptr) return -1;
    return (idx_t)(ptr - arr);
}

int build_n2e(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              idx_t *const elems[4],
              idx_t **out_e2nptr,
              idx_t **out_elindex) {
    idx_t *e2nptr = malloc((nnodes + 1) * sizeof(idx_t));
    memset(e2nptr, 0, nnodes * sizeof(idx_t));

    int *bookkepping = malloc((nnodes) * sizeof(int));
    memset(bookkepping, 0, (nnodes) * sizeof(int));

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        for (idx_t i = 0; i < nelements; ++i) {
            assert(elems[edof_i][i] < nnodes);
            assert(elems[edof_i][i] >= 0);

            ++e2nptr[elems[edof_i][i] + 1];
        }
    }

    for (idx_t i = 0; i < nnodes; ++i) {
        e2nptr[i + 1] += e2nptr[i];
    }

    idx_t *elindex = (idx_t *)malloc(e2nptr[nnodes] * sizeof(idx_t));

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        for (idx_t i = 0; i < nelements; ++i) {
            idx_t node = elems[edof_i][i];

            assert(e2nptr[node] + bookkepping[node] < e2nptr[node + 1]);

            elindex[e2nptr[node] + bookkepping[node]++] = i;
        }
    }

    free(bookkepping);

    *out_e2nptr = e2nptr;
    *out_elindex = elindex;
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
        idx_t *e2nptr;
        idx_t *elindex;
        build_n2e(nelements, nnodes, elems, &e2nptr, &elindex);

        rowptr[0] = 0;

        idx_t n2nbuff[2048];
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

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

            nnz += nneighs;
            rowptr[node + 1] = nnz;
        }

        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

        ptrdiff_t coloffset = 0;
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

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

        free(e2nptr);
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
        idx_t *e2nptr;
        idx_t *elindex;
        build_n2e(nelements, nnodes, elems, &e2nptr, &elindex);

        rowptr[0] = 0;

        ptrdiff_t overestimated_nnz = 0;
        idx_t n2nbuff[2048];
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

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

            overestimated_nnz += nneighs;
        }

        colidx = (idx_t *)malloc(overestimated_nnz * sizeof(idx_t));

        ptrdiff_t coloffset = 0;
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

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

            nnz += nneighs;
            rowptr[node + 1] = nnz;

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[coloffset + i] = n2nbuff[i];
            }

            coloffset += nneighs;
        }

        free(e2nptr);
        free(elindex);
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
