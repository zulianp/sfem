#include "sfem_multiblock_adj_table.h"
#include "sfem_API.hpp"

#include "multiblock_crs_graph.h"
#include "sortreduce.h"

#include <vector>

std::vector<enum ElemType> simple_element_types(const ptrdiff_t n_blocks, const enum ElemType element_types[]) {
    std::vector<enum ElemType> element_type_for_algo(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        element_type_for_algo[b] = element_types[b];
        if (element_type_for_algo[b] == TET10) {
            element_type_for_algo[b] = TET4;
        } else if (element_type_for_algo[b] == TRI6) {
            element_type_for_algo[b] = TRI3;
        }
    }
    return element_type_for_algo;
}

extern "C" int multiblock_create_dual_graph(const ptrdiff_t       n_blocks,
                                            const ptrdiff_t       n_elements[],
                                            const ptrdiff_t       n_nodes,
                                            const enum ElemType   element_types[],
                                            idx_t **const         elems[],
                                            count_t **const       adj_ptr_out,
                                            element_idx_t **const adj_idx_out,
                                            block_idx_t **const   adj_block_number_out) {
    std::vector<enum ElemType> element_type_for_algo = simple_element_types(n_blocks, element_types);

    count_t       *n2eptr;
    element_idx_t *elindex;
    block_idx_t   *block_number;

    build_multiblock_n2e(n_blocks, element_type_for_algo.data(), n_elements, elems, n_nodes, &block_number, &n2eptr, &elindex);

    std::vector<int *> connection_counter(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        connection_counter[b] = (int *)calloc(n_elements[b], sizeof(int));
    }

    std::vector<int> n_sides(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        n_sides[b] = elem_num_sides(element_type_for_algo[b]);
    }

    std::vector<int> n_nodes_per_elem(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        n_nodes_per_elem[b] = elem_num_nodes(element_type_for_algo[b]);
    }

    std::vector<int> n_nodes_per_side(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        enum ElemType st    = side_type(element_type_for_algo[b]);
        n_nodes_per_side[b] = elem_num_nodes(st);
    }

    std::vector<count_t *> dual_e_ptr(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        dual_e_ptr[b] = (count_t *)calloc((n_elements[b] + 1), sizeof(count_t));
    }

    std::vector<ptrdiff_t> n_overestimated_connections(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        n_overestimated_connections[b] = n_elements[b] * n_sides[b];
    }

    std::vector<element_idx_t *> dual_eidx(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        dual_eidx[b] = (element_idx_t *)calloc(n_overestimated_connections[b] + 1000, sizeof(element_idx_t));
    }

    std::vector<block_idx_t *> adj_block_number(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        adj_block_number[b] = (block_idx_t *)calloc(n_overestimated_connections[b] + 1000, sizeof(block_idx_t));
    }

    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        for (ptrdiff_t e = 0; e < n_elements[b]; e++) {
            count_t        offset = dual_e_ptr[b][e];
            element_idx_t *elist  = &dual_eidx[b][offset];
            block_idx_t   *blist  = &adj_block_number[b][offset];

            int count_common = 0;
            for (int en = 0; en < n_nodes_per_elem[b]; en++) {
                const idx_t node = elems[b][en][e];

                for (count_t eii = n2eptr[node]; eii < n2eptr[node + 1]; eii++) {
                    const element_idx_t e_adj = elindex[eii];
                    assert(e_adj < n_elements[b]);

                    if (connection_counter[b][e_adj] == 0) {
                        assert(offset + count_common < n_overestimated_connections[b] + extra_buffer_space);
                        elist[count_common]   = e_adj;
                        blist[count_common++] = block_number[eii];
                    }

                    connection_counter[b][e_adj]++;
                }
            }

            connection_counter[b][e] = 0;

            int actual_count = 0;
            for (int ec = 0; ec < count_common; ec++) {
                element_idx_t l       = elist[ec];
                int           overlap = connection_counter[b][l];
                assert(overlap <= n_nodes_per_elem[b]);

                if (overlap == n_nodes_per_side[b]) {
                    elist[actual_count++] = l;
                }

                connection_counter[b][l] = 0;
            }

            dual_e_ptr[b][e + 1] = actual_count + offset;
        }
    }

    // free memory
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        free(connection_counter[b]);
        free(dual_e_ptr[b]);
        free(dual_eidx[b]);
    }

    free(n2eptr);
    free(elindex);

    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        adj_ptr_out[b]          = dual_e_ptr[b];
        adj_idx_out[b]          = dual_eidx[b];
        adj_block_number_out[b] = adj_block_number[b];
    }

    return SFEM_SUCCESS;
}

#define LST(b, i, j) local_side_table[b][(i)*n_nodes_per_elem[b] + (j)]
extern "C" int multiblock_create_element_adj_table_from_dual_graph(const ptrdiff_t       n_blocks,
                                                                   const ptrdiff_t       n_elements[],
                                                                   const ptrdiff_t       n_nodes,
                                                                   const enum ElemType   element_types[],
                                                                   idx_t **const         elems[],
                                                                   count_t **const       adj_ptr,
                                                                   element_idx_t **const adj_idx,
                                                                   block_idx_t **const adj_block_number,
                                                                   element_idx_t **const table_element,
                                                                   block_idx_t **const   table_block) {
    std::vector<enum ElemType> element_type_for_algo = simple_element_types(n_blocks, element_types);

    std::vector<int>           n_nodes_per_side(n_blocks);
    std::vector<int>           n_nodes_per_elem(n_blocks);

    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        auto st      = side_type(element_type_for_algo[b]);
        n_nodes_per_side[b] = elem_num_nodes(st);
        n_nodes_per_elem[b] = elem_num_nodes(element_type_for_algo[b]);
    }

    std::vector<std::vector<int>> local_side_table(n_blocks);
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        local_side_table[b].resize(n_nodes_per_side[b] * n_nodes_per_elem[b]);
        fill_local_side_table(element_type_for_algo[b], local_side_table[b].data());
    }

    #pragma omp parallel for collapse(2)
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        for (ptrdiff_t e = 0; e < n_elements[b]; e++) {
            idx_t nodes1[SFEM_MAX_NUM_NODES_PER_SIDE];
            idx_t nodes2[SFEM_MAX_NUM_NODES_PER_SIDE];
            int   assigned[SFEM_MAX_NUM_SIDES];

            const count_t begin = adj_ptr[b][e];
            const count_t end   = adj_ptr[b][e + 1];
            const count_t range = end - begin;

            memset(assigned, 0, range * sizeof(int));

            for (int s1 = 0; s1 < n_nodes_per_side[b]; s1++) {
                table_element[b][e * n_nodes_per_side[b] + s1] = SFEM_ELEMENT_IDX_INVALID;


                for (int j = 0; j < n_nodes_per_elem[b]; j++) {
                    nodes1[j] = elems[b][LST(b, s1, j)][e];
                }

                sort_idx(nodes1, n_nodes_per_elem[b]);

                for (count_t k = 0; k < range; k++) {
                    if (assigned[k]) continue;
                    const element_idx_t e_adj = adj_idx[b][begin + k];

                    for (int s2 = 0; s2 < n_nodes_per_side[b]; s2++) {
                        for (int j = 0; j < n_nodes_per_elem[b]; j++) {
                            nodes2[j] = elems[b][LST(b, s2, j)][e_adj];
                        }

                        sort_idx(nodes2, n_nodes_per_elem[b]);

                        int diffs = 0;
                        for (int j = 0; j < n_nodes_per_elem[b]; j++) {
                            diffs += nodes1[j] != nodes2[j];
                        }

                        if (!diffs) {
                            // Array of structures
                            table_element[b][e * n_nodes_per_side[b] + s1] = e_adj;
                            table_block[b][e * n_nodes_per_side[b] + s1] = adj_block_number[b][begin + k];
                            assigned[k]        = 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

extern "C" int multiblock_create_element_adj_table(const ptrdiff_t              n_blocks,
                                                   const ptrdiff_t              n_elements[],
                                                   const ptrdiff_t              n_nodes,
                                                   const enum ElemType          element_types[],
                                                   idx_t **const                elems[],
                                                   element_idx_t *SFEM_RESTRICT table_element_out[],
                                                   block_idx_t *SFEM_RESTRICT   table_block_out[]) {
    std::vector<enum ElemType>   element_type_for_algo = simple_element_types(n_blocks, element_types);
    std::vector<count_t *>       adj_ptr(n_blocks);
    std::vector<element_idx_t *> adj_idx(n_blocks);
    std::vector<block_idx_t *>   adj_block_number(n_blocks);

    // Memory allocated inside multiblock_create_dual_graph
    multiblock_create_dual_graph(n_blocks,
                                 n_elements,
                                 n_nodes,
                                 element_type_for_algo.data(),
                                 elems,
                                 adj_ptr.data(),
                                 adj_idx.data(),
                                 adj_block_number.data());

    std::vector<element_idx_t *> table_element(n_blocks);
    std::vector<block_idx_t *>   table_block(n_blocks);

    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        table_element[b] =
                (element_idx_t *)malloc(n_elements[b] * elem_num_sides(element_type_for_algo[b]) * sizeof(element_idx_t));
        table_block[b] = (block_idx_t *)malloc(n_elements[b] * elem_num_sides(element_type_for_algo[b]) * sizeof(block_idx_t));
    }

    multiblock_create_element_adj_table_from_dual_graph(n_blocks,
                                                        n_elements,
                                                        n_nodes,
                                                        element_type_for_algo.data(),
                                                        elems,
                                                        adj_ptr.data(),
                                                        adj_idx.data(),
                                                        adj_block_number.data(),
                                                        table_element.data(),
                                                        table_block.data());

    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        table_element_out[b] = table_element[b];
        table_block_out[b]   = table_block[b];
    }

    // Free temporaries
    for (ptrdiff_t b = 0; b < n_blocks; b++) {
        free(adj_ptr[b]);
        free(adj_idx[b]);
        free(adj_block_number[b]);
    }

    return SFEM_SUCCESS;
}
