#include "multiblock_crs_graph.h"

#include "sortreduce.h"

int build_multiblock_n2e(const uint16_t      n_blocks,
                         const enum ElemType element_types[],
                         const ptrdiff_t     n_elements[],
                         idx_t **const       elements[],
                         const ptrdiff_t     n_nodes,
                         uint16_t          **out_block_number,
                         count_t           **out_n2eptr,
                         element_idx_t     **out_elindex) {
    count_t *n2eptr = (count_t *)malloc((n_nodes + 1) * sizeof(count_t));
    memset(n2eptr, 0, (n_nodes + 1) * sizeof(count_t));

    int *book_keeping = (int *)malloc((n_nodes) * sizeof(int));
    memset(book_keeping, 0, (n_nodes) * sizeof(int));

    for (uint16_t i = 0; i < n_blocks; i++) {
        enum ElemType element_type = element_types[i];
        int           nnodesxelem  = elem_num_nodes(element_type);

        for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
            for (int j = 0; j < n_elements[i]; j++) {
                n2eptr[elements[i][edof_i][j] + 1]++;
            }
        }
    }

    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        n2eptr[i + 1] += n2eptr[i];
    }

    element_idx_t *elindex      = (element_idx_t *)malloc(n2eptr[n_nodes] * sizeof(element_idx_t));
    uint16_t      *block_number = (uint16_t *)malloc(n2eptr[n_nodes] * sizeof(uint16_t));

    for (uint16_t i = 0; i < n_blocks; i++) {
        enum ElemType element_type = element_types[i];
        int           nnodesxelem  = elem_num_nodes(element_type);

        for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
            for (ptrdiff_t j = 0; j < n_elements[i]; ++j) {
                element_idx_t node = elements[i][edof_i][j];

                assert(n2eptr[node] + book_keeping[node] < n2eptr[node + 1]);

                elindex[n2eptr[node] + book_keeping[node]]      = j;
                block_number[n2eptr[node] + book_keeping[node]++] = i;
            }
        }
    }

    free(book_keeping);

    *out_n2eptr       = n2eptr;
    *out_elindex      = elindex;
    *out_block_number = block_number;

    return SFEM_SUCCESS;
}

int build_multiblock_crs_graph_from_n2e(const uint16_t                           n_blocks,
                                        const enum ElemType                      element_types[],
                                        const ptrdiff_t                          n_elements[],
                                        const ptrdiff_t                          n_nodes,
                                        idx_t **const SFEM_RESTRICT              elems[],
                                        const count_t *const SFEM_RESTRICT       n2eptr,
                                        const element_idx_t *const SFEM_RESTRICT elindex,
                                        const uint16_t *const SFEM_RESTRICT      block_number,
                                        count_t                                **out_rowptr,
                                        idx_t                                  **out_colidx) {
    count_t *rowptr = (count_t *)malloc((n_nodes + 1) * sizeof(count_t));
    idx_t   *colidx = 0;

    {
        rowptr[0] = 0;

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                count_t ebegin = n2eptr[node];
                count_t eend   = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    uint16_t      b    = block_number[e];
                    assert(eidx < n_elements[b]);

                    int nnodesxelem = elem_num_nodes(element_types[b]);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[b][edof_i][eidx];
                        assert(nneighs < 4096);
                        n2nbuff[nneighs++] = neighnode;
                    }
                }

                nneighs          = sortreduce(n2nbuff, nneighs);
                rowptr[node + 1] = nneighs;
            }
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[n_nodes];
        colidx              = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                count_t ebegin = n2eptr[node];
                count_t eend   = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    uint16_t      b    = block_number[e];
                    assert(eidx < n_elements[b]);

                    int nnodesxelem = elem_num_nodes(element_types[b]);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[b][edof_i][eidx];
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

int build_multiblock_crs_graph(const uint16_t      n_blocks,
                               const enum ElemType element_types[],
                               const ptrdiff_t     n_elements[],
                               idx_t **const       elems[],
                               const ptrdiff_t     n_nodes,
                               count_t           **out_rowptr,
                               idx_t             **out_colidx) {
    uint16_t      *block_number = 0;
    count_t       *n2eptr       = 0;
    element_idx_t *elindex      = 0;

    build_multiblock_n2e(n_blocks, element_types, n_elements, elems, n_nodes, &block_number, &n2eptr, &elindex);
    build_multiblock_crs_graph_from_n2e(
            n_blocks, element_types, n_elements, n_nodes, elems, n2eptr, elindex, block_number, out_rowptr, out_colidx);

    free(block_number);
    free(n2eptr);
    free(elindex);

    return SFEM_SUCCESS;
}

static int build_multiblock_crs_graph_upper_triangular_from_n2e(const uint16_t                           n_blocks,
                                                                const enum ElemType                      element_types[],
                                                                const ptrdiff_t                          n_elements[],
                                                                const ptrdiff_t                          n_nodes,
                                                                idx_t **const SFEM_RESTRICT              elems[],
                                                                const count_t *const SFEM_RESTRICT       n2eptr,
                                                                const element_idx_t *const SFEM_RESTRICT elindex,
                                                                const uint16_t *const SFEM_RESTRICT      block_number,
                                                                count_t                                **out_rowptr,
                                                                idx_t                                  **out_colidx) {
    count_t *rowptr = (count_t *)malloc((n_nodes + 1) * sizeof(count_t));
    idx_t   *colidx = 0;

    {
        rowptr[0] = 0;

        {
#pragma omp parallel for
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                idx_t n2nbuff[4096];

                count_t ebegin = n2eptr[node];
                count_t eend   = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    element_idx_t eidx = elindex[e];
                    uint16_t      b    = block_number[e];
                    assert(eidx < n_elements[b]);

                    int nnodesxelem = elem_num_nodes(element_types[b]);

                    for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                        idx_t neighnode = elems[b][edof_i][eidx];
                        if (neighnode > node) {
                            assert(nneighs < 4096);
                            n2nbuff[nneighs++] = neighnode;
                        }
                    }

                    nneighs          = sortreduce(n2nbuff, nneighs);
                    rowptr[node + 1] = nneighs;
                }
            }

            // Cumulative sum
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                rowptr[node + 1] += rowptr[node];
            }

            const ptrdiff_t nnz = rowptr[n_nodes];
            colidx              = (idx_t *)malloc(nnz * sizeof(idx_t));

            {
#pragma omp parallel for
                for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                    idx_t   n2nbuff[4096];
                    count_t ebegin = n2eptr[node];
                    count_t eend   = n2eptr[node + 1];

                    idx_t nneighs = 0;

                    for (count_t e = ebegin; e < eend; ++e) {
                        element_idx_t eidx = elindex[e];
                        uint16_t      b    = block_number[e];
                        assert(eidx < n_elements[b]);

                        int nnodesxelem = elem_num_nodes(element_types[b]);

                        for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                            idx_t neighnode = elems[b][edof_i][eidx];
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

int build_multiblock_crs_graph_upper_triangular(const uint16_t      n_blocks,
                                                const enum ElemType element_types[],
                                                const ptrdiff_t     n_elements[],
                                                idx_t **const       elems[],
                                                const ptrdiff_t     n_nodes,
                                                count_t           **out_rowptr,
                                                idx_t             **out_colidx) {
    uint16_t      *block_number = 0;
    count_t       *n2eptr       = 0;
    element_idx_t *elindex      = 0;

    build_multiblock_n2e(n_blocks, element_types, n_elements, elems, n_nodes, &block_number, &n2eptr, &elindex);
    build_multiblock_crs_graph_upper_triangular_from_n2e(
            n_blocks, element_types, n_elements, n_nodes, elems, n2eptr, elindex, block_number, out_rowptr, out_colidx);

    free(block_number);
    free(n2eptr);
    free(elindex);

    return SFEM_SUCCESS;
}
