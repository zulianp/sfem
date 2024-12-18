#include "sfem_sshex8_skin.h"

#include "adj_table.h"
#include "proteus_hex8.h"
#include "sfem_hex8_mesh_graph.h"

#include <assert.h>

static SFEM_INLINE int ssquad4_lidx(const int L, const int x, const int y) {
    int Lp1 = L + 1;
    int ret = y * Lp1 + x;

    assert(ret < Lp1 * Lp1);
    assert(ret >= 0);
    return ret;
}

static SFEM_INLINE int ssquad4_txe(int level) { return level * level; }

static SFEM_INLINE int ssquad4_nxe(int level) {
    const int corners    = 4;
    const int edge_nodes = 4 * (level - 1);
    const int area_nodes = (level - 1) * (level - 1);
    return corners + edge_nodes + area_nodes;
}

#define A3SET(a, x, y, z) \
    do {                  \
        a[0] = x;         \
        a[1] = y;         \
        a[2] = z;         \
    } while (0)

int sshex8_surface_from_sideset(const int                                L,
                                const ptrdiff_t                          nelements,
                                idx_t **const SFEM_RESTRICT              elements,
                                const ptrdiff_t                          n_surf_elements,
                                const element_idx_t *const SFEM_RESTRICT parents,
                                const int16_t *const SFEM_RESTRICT       side_idx,
                                idx_t **SFEM_RESTRICT                    sides) {
    const int ns = elem_num_sides(HEX8);
    for (ptrdiff_t i = 0; i < n_surf_elements; i++) {
        int start[3]     = {0, 0, 0};
        int end[3]       = {L + 1, L + 1, L + 1};
        int increment[3] = {1, 1, 1};

        const ptrdiff_t e = parents[i];
        const int       s = side_idx[e];

        switch (s) {
            case HEX8_LEFT: {
                A3SET(start, 0, L, 0);
                A3SET(end, 1, -1, L + 1);
                A3SET(increment, 1, -1, 1);
                break;
            }
            case HEX8_RIGHT: {
                start[0] = L;
                end[0]   = L + 1;
                break;
            }
            case HEX8_BOTTOM: {
                start[2] = 0;
                end[2]   = 1;
                A3SET(start, 0, L, 0);
                A3SET(end, L + 1, -1, 1);
                A3SET(increment, 1, -1, 1);
                break;
            }
            case HEX8_TOP: {
                start[2] = L;
                end[2]   = L + 1;
                break;
            }
            case HEX8_FRONT: {
                start[1] = 0;
                end[1]   = 1;
                break;
            }
            case HEX8_BACK: {
                A3SET(start, L, L, 0);
                A3SET(end, -1, L + 1, L + 1);
                A3SET(increment, -1, 1, 1);
                break;
            }
            default: {
                assert(0);
                break;
            }
        }

        int n = 0;
        for (int zi = start[2]; zi != end[2]; zi += increment[2]) {
            for (int yi = start[1]; yi != end[1]; yi += increment[1]) {
                for (int xi = start[0]; xi != end[0]; xi += increment[0]) {
                    const int   lidx = proteus_hex8_lidx(L, xi, yi, zi);
                    const idx_t node = elements[lidx][e];
                    sides[n++][i]    = node;
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_skin(const int       L,
                const ptrdiff_t nelements,
                idx_t         **SFEM_RESTRICT elements,
                ptrdiff_t      *SFEM_RESTRICT n_surf_elements,
                idx_t **const   SFEM_RESTRICT surf_elements,
                element_idx_t **SFEM_RESTRICT parent_element) {
    const int proteus_to_std[8] = {// Bottom
                                   proteus_hex8_lidx(L, 0, 0, 0),
                                   proteus_hex8_lidx(L, L, 0, 0),
                                   proteus_hex8_lidx(L, L, L, 0),
                                   proteus_hex8_lidx(L, 0, L, 0),
                                   // Top
                                   proteus_hex8_lidx(L, 0, 0, L),
                                   proteus_hex8_lidx(L, L, 0, L),
                                   proteus_hex8_lidx(L, L, L, L),
                                   proteus_hex8_lidx(L, 0, L, L)};

    idx_t *hex8_elements[8] = {elements[proteus_to_std[0]],
                               elements[proteus_to_std[1]],
                               elements[proteus_to_std[2]],
                               elements[proteus_to_std[3]],
                               elements[proteus_to_std[4]],
                               elements[proteus_to_std[5]],
                               elements[proteus_to_std[6]],
                               elements[proteus_to_std[7]]};

    ptrdiff_t hex8_n_nodes = nxe_max_node_id(nelements, 8, hex8_elements) + 1;

    const int      ns    = elem_num_sides(HEX8);
    element_idx_t *table = 0;
    create_element_adj_table(nelements, hex8_n_nodes, HEX8, hex8_elements, &table);

    // Num side-nodes
    const int nn = (L + 1) * (L + 1);

    *n_surf_elements = 0;
    for (ptrdiff_t e = 0; e < nelements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const element_idx_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                (*n_surf_elements)++;
            }
        }
    }

    *parent_element = malloc((*n_surf_elements) * sizeof(element_idx_t));
    for (int s = 0; s < nn; s++) {
        surf_elements[s] = malloc((*n_surf_elements) * sizeof(idx_t));
    }

    ptrdiff_t side_offset = 0;
    for (ptrdiff_t e = 0; e < nelements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const element_idx_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                int start[3]     = {0, 0, 0};
                int end[3]       = {L + 1, L + 1, L + 1};
                int increment[3] = {1, 1, 1};

                switch (s) {
                    case HEX8_LEFT: {
                        A3SET(start, 0, L, 0);
                        A3SET(end, 1, -1, L + 1);
                        A3SET(increment, 1, -1, 1);
                        break;
                    }
                    case HEX8_RIGHT: {
                        start[0] = L;
                        end[0]   = L + 1;
                        break;
                    }
                    case HEX8_BOTTOM: {
                        start[2] = 0;
                        end[2]   = 1;
                        A3SET(start, 0, L, 0);
                        A3SET(end, L + 1, -1, 1);
                        A3SET(increment, 1, -1, 1);
                        break;
                    }
                    case HEX8_TOP: {
                        start[2] = L;
                        end[2]   = L + 1;
                        break;
                    }
                    case HEX8_FRONT: {
                        start[1] = 0;
                        end[1]   = 1;
                        break;
                    }
                    case HEX8_BACK: {
                        A3SET(start, L, L, 0);
                        A3SET(end, -1, L + 1, L + 1);
                        A3SET(increment, -1, 1, 1);
                        break;
                    }
                    default: {
                        assert(0);
                        break;
                    }
                }

                int n = 0;
                for (int zi = start[2]; zi != end[2]; zi += increment[2]) {
                    for (int yi = start[1]; yi != end[1]; yi += increment[1]) {
                        for (int xi = start[0]; xi != end[0]; xi += increment[0]) {
                            const int   lidx                = proteus_hex8_lidx(L, xi, yi, zi);
                            const idx_t node                = elements[lidx][e];
                            surf_elements[n++][side_offset] = node;
                        }
                    }
                }

                (*parent_element)[side_offset] = e;
                side_offset++;
            }
        }
    }

    free(table);
    return SFEM_SUCCESS;
}
