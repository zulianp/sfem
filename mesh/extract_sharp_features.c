#include "extract_sharp_features.h"

#include <math.h>

static SFEM_INLINE void normalize3(real_t *const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

static SFEM_INLINE real_t dot3(const real_t *const a, const real_t *const b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

int extract_sharp_edges(const enum ElemType element_type,
                        const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elements,
                        geom_t **const SFEM_RESTRICT points,
                        // CRS-graph (node to node)
                        const count_t *const SFEM_RESTRICT rowptr,
                        const idx_t *const SFEM_RESTRICT colidx,
                        const geom_t angle_threshold,
                        ptrdiff_t *out_n_sharp_edges,
                        idx_t **out_e0,
                        idx_t **out_e1) {
    const count_t nedges = rowptr[nnodes];

    geom_t *normal[3];
    for (int d = 0; d < 3; d++) {
        normal[d] = calloc(nedges, sizeof(geom_t));
    }

    const int nxe = elem_num_nodes(element_type);

    count_t *opposite = malloc(nedges * sizeof(count_t));
    {
        // Opposite edge index
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t begin = rowptr[i];
            const count_t extent = rowptr[i + 1] - begin;
            const idx_t *cols = &colidx[begin];

            for (count_t k = 0; k < extent; k++) {
                const idx_t o = cols[k];
                if (i > o) continue;

                const count_t o_begin = rowptr[o];
                const count_t o_extent = rowptr[o + 1] - o_begin;
                const idx_t *o_cols = &colidx[o_begin];

                for (count_t o_k = 0; o_k < o_extent; o_k++) {
                    if (i == o_cols[o_k]) {
                        opposite[begin + k] = o_begin + o_k;
                        opposite[o_begin + o_k] = begin + k;
                        break;
                    }
                }
            }
        }
    }

    {
        // Compute normals
        for (ptrdiff_t e = 0; e < nelements; e++) {
            const idx_t i0 = elements[0][e];
            const idx_t i1 = elements[1][e];
            const idx_t i2 = elements[2][e];

            real_t u[3] = {points[0][i1] - points[0][i0],
                           points[1][i1] - points[1][i0],
                           points[2][i1] - points[2][i0]};
            real_t v[3] = {points[0][i2] - points[0][i0],
                           points[1][i2] - points[1][i0],
                           points[2][i2] - points[2][i0]};

            normalize3(u);
            normalize3(v);

            real_t n[3] = {u[1] * v[2] - u[2] * v[1],  //
                           u[2] * v[0] - u[0] * v[2],  //
                           u[0] * v[1] - u[1] * v[0]};

            normalize3(n);

            for (int ln = 0; ln < nxe; ln++) {
                const int lnp1 = (ln + 1 == nxe) ? 0 : (ln + 1);

                const idx_t node_from = elements[ln][e];
                const idx_t node_to = elements[lnp1][e];

                const count_t extent = rowptr[node_from + 1] - rowptr[node_from];
                const idx_t *cols = &colidx[rowptr[node_from]];

                ptrdiff_t edge_id = SFEM_PTRDIFF_INVALID;
                for (count_t k = 0; k < extent; k++) {
                    if (cols[k] == node_to) {
                        edge_id = rowptr[node_from] + k;
                        break;
                    }
                }

                assert(edge_id != SFEM_PTRDIFF_INVALID);
                for (int d = 0; d < 3; d++) {
                    normal[d][edge_id] = n[d];
                }
            }
        }
    }

    ptrdiff_t n_sharp_edges = 0;
    idx_t *e0 = malloc(nedges * sizeof(idx_t));
    idx_t *e1 = malloc(nedges * sizeof(idx_t));

    {
        geom_t *dihedral_angle = calloc(nedges, sizeof(geom_t));
        ptrdiff_t edge_count = 0;
        {
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                const count_t begin = rowptr[i];
                const count_t extent = rowptr[i + 1] - begin;
                const idx_t *cols = &colidx[begin];

                for (count_t k = 0; k < extent; k++) {
                    if (i >= cols[k]) continue;

                    ptrdiff_t edge_id = begin + k;
                    ptrdiff_t o_edge_id = opposite[edge_id];

                    assert(edge_id != o_edge_id);

                    // Higher precision computation
                    real_t n[3] = {normal[0][edge_id], normal[1][edge_id], normal[2][edge_id]};
                    real_t on[3] = {
                        normal[0][o_edge_id], normal[1][o_edge_id], normal[2][o_edge_id]};
                    real_t da = dot3(n, on);

                    // Store for minimum edge for exporting data
                    dihedral_angle[edge_count] = (geom_t)da;
                    e0[edge_count] = i;
                    e1[edge_count] = cols[k];
                    edge_count++;
                }
            }
        }

        // 1) select sharp edges create edge selection index
        // 2) create face islands index (for contact integral separation)
        // 3) export edge and face selection
        // TODO Future work: detect sharp corners

        {
            // Select edges
            for (ptrdiff_t i = 0; i < edge_count; i++) {
                if (dihedral_angle[i] <= angle_threshold) {
                    e0[n_sharp_edges] = e0[i];
                    e1[n_sharp_edges] = e1[i];
                    dihedral_angle[n_sharp_edges] = dihedral_angle[i];
                    n_sharp_edges++;
                }
            }
        }

        free(dihedral_angle);
    }

    *out_n_sharp_edges = n_sharp_edges;
    *out_e0 = e0;
    *out_e1 = e1;

    free(opposite);
    for (int d = 0; d < 3; d++) {
        free(normal[d]);
    }

    return 0;
}

int extract_sharp_corners(const ptrdiff_t nnodes,
                          const ptrdiff_t n_sharp_edges,
                           idx_t *const SFEM_RESTRICT e0,
                           idx_t *const SFEM_RESTRICT e1,
                          ptrdiff_t *out_ncorners,
                          idx_t **out_corners,
                          int edge_clean_up) {
    ptrdiff_t n_corners = 0;
    idx_t *corners = 0;

    ptrdiff_t out_n_sharp_edges = n_sharp_edges;
    {
        int *incidence_count = calloc(nnodes, sizeof(int));

        for (ptrdiff_t i = 0; i < n_sharp_edges; i++) {
            incidence_count[e0[i]]++;
            incidence_count[e1[i]]++;
        }

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            if (incidence_count[i] >= 3) {
                n_corners++;
            }
        }

        corners = malloc(n_corners * sizeof(idx_t));
        for (ptrdiff_t i = 0, n_corners = 0; i < nnodes; i++) {
            if (incidence_count[i] >= 3) {
                corners[n_corners] = i;
                n_corners++;
            }
        }

        if(edge_clean_up) {
        	out_n_sharp_edges = 0;
        	for (ptrdiff_t i = 0; i < n_sharp_edges; i++) {
        	    if(incidence_count[e0[i]] < 3 &&
        	       incidence_count[e1[i]] < 3) {
        	    	e0[out_n_sharp_edges] = e0[i];
        	    	e1[out_n_sharp_edges] = e1[i];

        	    	out_n_sharp_edges++;
        	    }
        	}
        }

        free(incidence_count);
    }

    *out_ncorners = n_corners;
    *out_corners = corners;

    return out_n_sharp_edges;
}

int extract_disconnected_faces(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               const ptrdiff_t n_sharp_edges,
                               const idx_t *const SFEM_RESTRICT e0,
                               const idx_t *const SFEM_RESTRICT e1,
                               ptrdiff_t *out_n_disconnected_elements,
                               element_idx_t **out_disconnected_elements) {
    ptrdiff_t n_disconnected_elements = 0;
    element_idx_t *disconnected_elements = 0;
    {
        // Select unconnected faces
        short *checked = calloc(nnodes, sizeof(short));

        const int nxe = elem_num_nodes(element_type);
        for (ptrdiff_t i = 0; i < n_sharp_edges; i++) {
            checked[e0[i]] = 1;
            checked[e1[i]] = 1;
        }

        for (ptrdiff_t e = 0; e < nelements; e++) {
            short connected_to_sharp_edge = 0;
            for (int ln = 0; ln < nxe; ln++) {
                connected_to_sharp_edge += checked[elements[ln][e]];
            }

            n_disconnected_elements += connected_to_sharp_edge == 0;
        }

        disconnected_elements = malloc(n_disconnected_elements * sizeof(element_idx_t));

        ptrdiff_t eidx = 0;
        for (ptrdiff_t e = 0; e < nelements; e++) {
            short connected_to_sharp_edge = 0;
            for (int ln = 0; ln < nxe; ln++) {
                connected_to_sharp_edge += checked[elements[ln][e]];
            }

            if (connected_to_sharp_edge == 0) {
                disconnected_elements[eidx++] = e;
            }
        }

        free(checked);
    }

    *out_n_disconnected_elements = n_disconnected_elements;
    *out_disconnected_elements = disconnected_elements;

    return 0;
}
