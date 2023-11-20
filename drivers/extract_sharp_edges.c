#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh_write.h"

#include "sortreduce.h"

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
                        count_t **out_e0,
                        count_t **out_e1) {
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

                ptrdiff_t edge_id = -1;
                for (count_t k = 0; k < extent; k++) {
                    if (cols[k] == node_to) {
                        edge_id = rowptr[node_from] + k;
                        break;
                    }
                }

                assert(edge_id >= 0);
                for (int d = 0; d < 3; d++) {
                    normal[d][edge_id] = n[d];
                }
            }
        }
    }

    ptrdiff_t n_sharp_edges = 0;
    count_t *e0 = malloc(nedges * sizeof(count_t));
    count_t *e1 = malloc(nedges * sizeof(count_t));

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
                          const count_t *const SFEM_RESTRICT e0,
                          const count_t *const SFEM_RESTRICT e1,
                          ptrdiff_t *out_ncorners,
                          idx_t **out_corners) {
    ptrdiff_t n_corners = 0;
    idx_t *corners = 0;

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

        free(incidence_count);
    }

    *out_ncorners = n_corners;
    *out_corners = corners;

    return 0;
}

int extract_disconnected_faces(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               const ptrdiff_t n_sharp_edges,
                               const count_t *const SFEM_RESTRICT e0,
                               const count_t *const SFEM_RESTRICT e1,
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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 4) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <angle_threshold> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const geom_t angle_threshold = atof(argv[2]);
    const char *output_folder = argv[3];

    if (!rank) {
        printf("%s %s %s %s\n", argv[0], argv[1], argv[2], output_folder);
    }

    double tick = MPI_Wtime();

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (shell_type(mesh.element_type) != TRISHELL3) {
        fprintf(stderr, "%s this driver only supports triangle meshes", argv[0]);
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t n_sharp_edges = 0;
    count_t *e0 = 0;
    count_t *e1 = 0;

    {  // Extract sharp edges!
        count_t *rowptr = 0;
        idx_t *colidx = 0;
        build_crs_graph_for_elem_type(
            mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

        extract_sharp_edges(mesh.element_type,
                            mesh.nelements,
                            mesh.nnodes,
                            mesh.elements,
                            mesh.points,
                            // CRS-graph (node to node)
                            rowptr,
                            colidx,
                            angle_threshold,
                            &n_sharp_edges,
                            &e0,
                            &e1);

        free(rowptr);
        free(colidx);
    }

    ptrdiff_t n_corners = 0;
    idx_t *corners = 0;
    extract_sharp_corners(mesh.nnodes, n_sharp_edges, e0, e1, &n_corners, &corners);

    ptrdiff_t n_disconnected_elements = 0;
    element_idx_t *disconnected_elements = 0;

    extract_disconnected_faces(mesh.element_type,
                               mesh.nelements,
                               mesh.nnodes,
                               mesh.elements,
                               n_sharp_edges,
                               e0,
                               e1,
                               &n_disconnected_elements,
                               &disconnected_elements);

    {
        char path[1024 * 10];
        sprintf(path, "%s/i0.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e0, n_sharp_edges, n_sharp_edges);

        sprintf(path, "%s/i1.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e1, n_sharp_edges, n_sharp_edges);

        sprintf(path, "%s/corners", output_folder);

        struct stat st = {0};
        if (stat(path, &st) == -1) {
            mkdir(path, 0700);
        }

        sprintf(path, "%s/corners/i0.raw", output_folder);

        array_write(comm, path, SFEM_MPI_COUNT_T, corners, n_corners, n_corners);

        sprintf(path, "%s/e." dtype_ELEMENT_IDX_T ".raw", output_folder);
        array_write(comm,
                    path,
                    SFEM_MPI_ELEMENT_IDX_T,
                    disconnected_elements,
                    n_disconnected_elements,
                    n_disconnected_elements);
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("extract_sharp_edges.c: #elements %ld, #nodes %ld, #n_sharp_edges %ld\n",
               (long)mesh.nelements,
               (long)mesh.nnodes,
               (long)n_sharp_edges);
        printf("----------------------------------------\n");
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free Resources
    ///////////////////////////////////////////////////////////////////////////////

    mesh_destroy(&mesh);

    free(e0);
    free(e1);
    free(disconnected_elements);
    free(corners);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
