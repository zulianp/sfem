#include "sfem_hex8_mesh_graph.h"

#include "crs_graph.h"
#include "sortreduce.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

// PROTEUS
// static idx_t hex8_edge_connectivity[8][3] =
//         {{1, 2, 4},
//          {0, 3, 5},
//          {0, 3, 6},
//          {1, 2, 7},
//          {0, 5, 6},
//          {1, 4, 7},
//          {2, 4, 7},
//          {3, 5, 6}};

// LAGRANGE
static idx_t hex8_edge_connectivity[8][3] = {
        // BOTTOM
        {1, 3, 4},
        {0, 2, 5},
        {1, 3, 6},
        {0, 2, 7},
        // TOP
        {0, 5, 7},
        {1, 4, 6},
        {2, 5, 7},
        {3, 4, 6}};

static int hex8_build_edge_graph_from_n2e(const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          const count_t *const SFEM_RESTRICT n2eptr,
                                          const element_idx_t *const SFEM_RESTRICT elindex,
                                          count_t **out_rowptr,
                                          idx_t **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t *colidx = 0;

    static const int nnodesxelem = 8;

    {
        rowptr[0] = 0;

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                element_idx_t eidx = elindex[e];
                assert(eidx < nelements);

                int lidx = -1;
                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    if (elems[edof_i][eidx] == node) {
                        lidx = edof_i;
                        break;
                    }
                }

                assert(lidx != -1);
                assert(lidx < 8);

                for (int d = 0; d < 3; d++) {
                    idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);
            rowptr[node + 1] = nneighs;
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[nnodes];
        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                element_idx_t eidx = elindex[e];
                assert(eidx < nelements);

                int lidx = 0;
                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    if (elems[edof_i][eidx] == node) {
                        lidx = edof_i;
                        break;
                    }
                }

                for (int d = 0; d < 3; d++) {
                    idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[rowptr[node] + i] = n2nbuff[i];
            }
        }
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
}

int hex8_build_edge_graph(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          count_t **out_rowptr,
                          idx_t **out_colidx) {
    double tick = MPI_Wtime();

    count_t *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, 8, elems, &n2eptr, &elindex);

    int err = hex8_build_edge_graph_from_n2e(
            nelements, nnodes, elems, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return err;
}
