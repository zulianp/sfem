#include "sfem_base.h"
#include "sfem_defs.h"

#include "sortreduce.h"

#include "crs_graph.h"

#include <stdio.h>
#include <string.h>

#define SFEM_MAX_NUM_SIDES 8
#define SFEM_MAX_NUM_NODES_PER_SIDE 6
#define SFEM_INVALID_IDX (-1)

#define LST(i, j) local_side_table[(i)*nn + (j)]

static void fill_local_side_table(enum ElemType element_type, int *local_side_table) {
    enum ElemType st = side_type(element_type);
    const int nn = elem_num_nodes(st);

    if (element_type == TET10 || element_type == TET4) {
        LST(0, 0) = 1 - 1;
        LST(0, 1) = 2 - 1;
        LST(0, 2) = 4 - 1;

        LST(1, 0) = 2 - 1;
        LST(1, 1) = 3 - 1;
        LST(1, 2) = 4 - 1;

        LST(2, 0) = 1 - 1;
        LST(2, 1) = 4 - 1;
        LST(2, 2) = 3 - 1;

        LST(3, 0) = 1 - 1;
        LST(3, 1) = 3 - 1;
        LST(3, 2) = 2 - 1;

        if (element_type == TET10) {
            LST(0, 3) = 5 - 1;
            LST(0, 4) = 9 - 1;
            LST(0, 5) = 8 - 1;

            LST(1, 3) = 6 - 1;
            LST(1, 4) = 10 - 1;
            LST(1, 5) = 9 - 1;

            LST(2, 3) = 8 - 1;
            LST(2, 4) = 10 - 1;
            LST(2, 5) = 7 - 1;

            LST(3, 3) = 7 - 1;
            LST(3, 4) = 6 - 1;
            LST(3, 5) = 5 - 1;
        }

    } else if(element_type == TRI3) {
        LST(0, 0) = 1 - 1;
        LST(0, 1) = 2 - 1;

        LST(1, 0) = 2 - 1;
        LST(1, 1) = 3 - 1;

        LST(2, 0) = 3 - 1;
        LST(2, 1) = 1 - 1;
    } else {
        assert(0);
    }
}

void fill_element_adj_table(const ptrdiff_t n_elements,
                            const ptrdiff_t n_nodes,
                            enum ElemType element_type,
                            idx_t **const SFEM_RESTRICT elems,
                            ptrdiff_t *const SFEM_RESTRICT table) {
    int element_type_for_algo = element_type;

    if (element_type == TET10) {
        // This is enough for many operations
        element_type_for_algo = TET4;
    }

    count_t *adj_ptr = 0;
    element_idx_t *adj_idx = 0;
    create_dual_graph(n_elements, n_nodes, element_type_for_algo, elems, &adj_ptr, &adj_idx);

    int local_side_table[SFEM_MAX_NUM_SIDES * SFEM_MAX_NUM_NODES_PER_SIDE];
    fill_local_side_table(element_type_for_algo, local_side_table);

    idx_t nodes1[SFEM_MAX_NUM_NODES_PER_SIDE];
    idx_t nodes2[SFEM_MAX_NUM_NODES_PER_SIDE];
    int assigned[SFEM_MAX_NUM_SIDES];

    enum ElemType st = side_type(element_type_for_algo);
    const int nn = elem_num_nodes(st);
    const int ns = elem_num_sides(element_type_for_algo);

    for (ptrdiff_t e = 0; e < n_elements; e++) {
        const count_t begin = adj_ptr[e];
        const count_t end = adj_ptr[e + 1];
        const count_t range = end - begin;

        memset(assigned, 0, range * sizeof(int));

        for (int s1 = 0; s1 < ns; s1++) {
            table[e * ns + s1] = SFEM_INVALID_IDX;

            for (int j = 0; j < nn; j++) {
                nodes1[j] = elems[LST(s1, j)][e];
            }

            sort_idx(nodes1, nn);

            for (count_t k = 0; k < range; k++) {
                if (assigned[k]) continue;
                const element_idx_t e_adj = adj_idx[begin + k];

                for (int s2 = 0; s2 < ns; s2++) {
                    for (int j = 0; j < nn; j++) {
                        nodes2[j] = elems[LST(s2, j)][e_adj];
                    }

                    sort_idx(nodes2, nn);

                    int diffs = 0;
                    for (int j = 0; j < nn; j++) {
                        diffs += nodes1[j] != nodes2[j];
                    }

                    if (!diffs) {
                        // Array of structures
                        table[e * ns + s1] = e_adj;
                        assigned[k] = 1;
                        break;
                    }
                }
            }
        }
    }

    free(adj_ptr);
    free(adj_idx);
}

void extract_surface_connectivity_with_adj_table(const ptrdiff_t n_elements,
                                                 const ptrdiff_t n_nodes,
                                                 const int element_type,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 ptrdiff_t *n_surf_elements,
                                                 element_idx_t **surf_elems,
                                                 element_idx_t **parent_element)

{
    const int ns = elem_num_sides(element_type);
    ptrdiff_t *table = (ptrdiff_t *)malloc(n_elements * ns * sizeof(ptrdiff_t));
    fill_element_adj_table(n_elements, n_nodes, element_type, elems, table);

    int local_side_table[SFEM_MAX_NUM_SIDES * SFEM_MAX_NUM_NODES_PER_SIDE];
    fill_local_side_table(element_type, local_side_table);
    enum ElemType st = side_type(element_type);
    const int nn = elem_num_nodes(st);

    // for (ptrdiff_t e = 0; e < n_elements; e++) {
    //     for (int s = 0; s < ns; s++) {
    //         printf("%ld ", (long)table[e * ns + s]);
    //     }
    //     printf("\n");
    // }

    *n_surf_elements = 0;
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const ptrdiff_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                (*n_surf_elements)++;
            }
        }
    }

    *parent_element = malloc((*n_surf_elements) * sizeof(element_idx_t));
    for (int s = 0; s < nn; s++) {
        surf_elems[s] = malloc((*n_surf_elements) * sizeof(element_idx_t));
    }

    ptrdiff_t side_offset = 0;
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const ptrdiff_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                for (int n = 0; n < nn; n++) {
                    idx_t node = elems[LST(s, n)][e];
                    surf_elems[n][side_offset] = node;
                }

                (*parent_element)[side_offset] = e;
                side_offset++;
            }
        }
    }

    free(table);
}

#undef LST
