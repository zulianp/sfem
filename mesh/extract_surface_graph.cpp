
#include "sfem_base.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#if 0

#include "crs_graph.h"

extern "C" void extract_surface_connectivity(const int element_type,
                                             const ptrdiff_t n_elements,
                                             const ptrdiff_t n_nodes,
                                             idx_t** const SFEM_RESTRICT elems,
                                             ptrdiff_t* n_surf_elements,
                                             idx_t** surf_elems,
                                             idx_t** parent_element) {
    count_t* n2eptr = 0;
    idx_t* elindex = 0;

    const int el_n_nodes = elem_num_nodes(element_type);
    const int el_n_sides = elem_num_sides(element_type);

    ElemType st = side_type(element_type);
    const int side_num_nodes = elem_num_nodes(st);

    if (element_type == TET10) {
        // Use edge nodes?
        build_n2e(n_elements, n_nodes, 6, elems + 4, &n2eptr, &elindex);

        // Or use corners?
        // build_n2e(n_elements, n_nodes, 4, elems, &n2eptr, &elindex);
    } else if (element_type == TRI6) {
        // This will give us an element adjaciency graph directly
        build_n2e(n_elements, n_nodes, 3, elems + 3, &n2eptr, &elindex);
    } else {
        // Generic case (more work?)
        build_n2e(n_elements, n_nodes, el_n_nodes, elems, &n2eptr, &elindex);
    }

    int max_n2e = 0;

    // Use N2E connectivity to find adjacient elements
    for (ptrdiff_t node = 0; node < n_nodes; ++node) {
        const count_t begin = n2eptr[node];
        const count_t end = n2eptr[node + 1];
        max_n2e = MAX(max_n2e, end - begin);
    }

    idx_t* etable = (idx_t*)malloc(el_n_nodes * max_n2e);

    for (ptrdiff_t node = 0; node < n_nodes; ++node) {
        const count_t begin = n2eptr[node];
        const count_t end = n2eptr[node + 1];
        const count_t range = end - begin;

        // Fill table
        for (count_t k = 0; k < range; k++) {
            idx_t e = elindex[begin + k];
            idx_t * trow = &etable[k*el_n_nodes];

            for(int i = 0; i < el_n_nodes; i++) {
                trow[i] = elems[i][e];
            }

            // Sort for comparison
            std::sort(trow, trow + el_n_nodes);
        }

        // find adjacient elements
        for (count_t k1 = 0; k1 < range; k1++) {
            idx_t e1 = elindex[begin + k1];
            idx_t * trow1 = &etable[k1*el_n_nodes];

            for (count_t k2 = k1+1; k2 < range; k2++) {
                idx_t e2 = elindex[begin + k2];
                idx_t * trow2 = &etable[k2*el_n_nodes];

                // See if common nodes are exactly equal to side_num_nodes

                int count_common = 0;
                for(int i1 = 0, i2 = 0; i1 < el_n_nodes && i2 < el_n_nodes;) {
                    if(trow1[i1] < trow2[i2]) {
                        i1++;
                        continue;
                    }

                    if(trow1[i1] > trow2[i2]) {
                        i2++;
                        continue;
                    }

                    count_common++;
                    i1++;
                    i2++;
                }

                if(count_common == side_num_nodes) {
                    // We have a pair of adj-elements
                    // Face is not boundary
                    continue;
                }

            }
        }

    }


    free(etable);
    free(n2eptr);
    free(elindex);
}

#else

#define PARENT_ID(sideidx) (sideidx) / (4)

extern "C" void extract_surface_connectivity(const ptrdiff_t n_elements,
                                             idx_t** const elems,
                                             ptrdiff_t* n_surf_elements,
                                             idx_t** surf_elems,
                                             idx_t** parent_element) {
    const ptrdiff_t n_sides = 4 * n_elements;
    std::vector<idx_t> buff(n_sides * 3);

    ptrdiff_t face_idx = 0;
    for (ptrdiff_t i = 0; i < n_elements; ++i) {
        buff[face_idx + 0] = elems[0][i];
        buff[face_idx + 1] = elems[1][i];
        buff[face_idx + 2] = elems[3][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[1][i];
        buff[face_idx + 1] = elems[2][i];
        buff[face_idx + 2] = elems[3][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[2][i];
        buff[face_idx + 1] = elems[3][i];
        buff[face_idx + 2] = elems[0][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[0][i];
        buff[face_idx + 1] = elems[1][i];
        buff[face_idx + 2] = elems[2][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;
    }

    std::vector<ptrdiff_t> sideidx(n_sides);

    for (ptrdiff_t i = 0; i < n_sides; ++i) {
        sideidx[i] = i;
    }

    std::sort(sideidx.begin(), sideidx.end(), [&](const ptrdiff_t l, const ptrdiff_t r) {
        for (int d = 0; d < 3; ++d) {
            const idx_t lidx = buff[l * 3 + d];
            const idx_t ridx = buff[r * 3 + d];

            if (lidx < ridx) {
                return true;
            } else if (lidx > ridx) {
                return false;
            }
        }

        return false;
    });

    ptrdiff_t n_surface = 0;

    for (ptrdiff_t i = 0; i < n_sides - 1;) {
        const ptrdiff_t l = sideidx[i];
        const ptrdiff_t r = sideidx[i + 1];

        assert(l >= 0);
        assert(r >= 0);

        bool same = true;

        for (int d = 0; d < 3; ++d) {
            const idx_t lidx = buff[l * 3 + d];
            const idx_t ridx = buff[r * 3 + d];

            if (lidx < ridx) {
                same = false;
                break;
            } else if (lidx > ridx) {
                same = false;
                break;
            }
        }

        if (same) {
            // Not surface face remove
            sideidx[i] = -1;
            sideidx[i + 1] = -1;
            i += 2;
        } else {
            i += 1;
            n_surface += 1;
        }
    }

    if (sideidx[n_sides - 1] >= 0) {
        n_surface += 1;
    }

    for (int d = 0; d < 3; ++d) {
        surf_elems[d] = (idx_t*)malloc(n_surface * sizeof(idx_t));
    }

    *parent_element = (idx_t*)malloc(n_surface * sizeof(idx_t));

    face_idx = 0;
    for (ptrdiff_t i = 0; i < n_sides; i++) {
        if (sideidx[i] < 0) continue;

        for (int d = 0; d < 3; ++d) {
            surf_elems[d][face_idx] = buff[sideidx[i] * 3 + d];
        }

        idx_t parent_id = PARENT_ID(sideidx[i]);
        (*parent_element)[face_idx] = parent_id;

        face_idx++;
    }

    *n_surf_elements = n_surface;
}

#undef PARENT_ID
#endif