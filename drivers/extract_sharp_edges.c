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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
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

    count_t *rowptr = 0;
    idx_t *colidx = 0;
    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

    const count_t nedges = rowptr[mesh.nnodes];

    geom_t *normal[3];
    for (int d = 0; d < 3; d++) {
        normal[d] = calloc(nedges, sizeof(geom_t));
    }

    short *checked = calloc(nedges, sizeof(short));
    const int nxe = elem_num_nodes(mesh.element_type);

    count_t *opposite = malloc(nedges * sizeof(count_t));
    {
        // Opposite edge index
        for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
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
        for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
            const idx_t i0 = mesh.elements[0][e];
            const idx_t i1 = mesh.elements[1][e];
            const idx_t i2 = mesh.elements[2][e];

            real_t u[3] = {mesh.points[0][i1] - mesh.points[0][i0],
                           mesh.points[1][i1] - mesh.points[1][i0],
                           mesh.points[2][i1] - mesh.points[2][i0]};
            real_t v[3] = {mesh.points[0][i2] - mesh.points[0][i0],
                           mesh.points[1][i2] - mesh.points[1][i0],
                           mesh.points[2][i2] - mesh.points[2][i0]};

            normalize3(u);
            normalize3(v);

            real_t n[3] = {u[1] * v[2] - u[2] * v[1],  //
                           u[2] * v[0] - u[0] * v[2],  //
                           u[0] * v[1] - u[1] * v[0]};

            normalize3(n);

            for (int ln = 0; ln < nxe; ln++) {
                const int lnp1 = (ln + 1 == nxe) ? 0 : (ln + 1);

                const idx_t node_from = mesh.elements[ln][e];
                const idx_t node_to = mesh.elements[lnp1][e];

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
                assert(checked[edge_id] == 0);
                for (int d = 0; d < 3; d++) {
                    normal[d][edge_id] = n[d];
                }
            }
        }
    }

    ptrdiff_t n_undirected_edges = (nedges - mesh.nnodes) / 2;
    assert(n_undirected_edges * 2 + mesh.nnodes == nedges);

    geom_t *dihedral_angle = calloc(n_undirected_edges, sizeof(geom_t));
    count_t *e1 = malloc(n_undirected_edges * sizeof(count_t));
    count_t *e2 = malloc(n_undirected_edges * sizeof(count_t));

    {
        for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
            const count_t begin = rowptr[i];
            const count_t extent = rowptr[i + 1] - begin;
            const idx_t *cols = &colidx[begin];

            for (count_t k = 0; k < extent; k++) {
                if (i >= cols[k]) continue;

                ptrdiff_t edge_id = begin + k;
                ptrdiff_t o_edge_id = opposite[edge_id];

                // Higher precision computation
                real_t n[3] = {normal[0][edge_id], normal[1][edge_id], normal[2][edge_id]};
                real_t on[3] = {normal[0][o_edge_id], normal[1][o_edge_id], normal[2][o_edge_id]};
                real_t da = dot3(n, on);

                // Store for minimum edge for exporting data
                ptrdiff_t min_edge_id = MIN(edge_id, o_edge_id);
                
                dihedral_angle[min_edge_id] = (geom_t)da;
                
                e1[min_edge_id] = i;
                e2[min_edge_id] = cols[k];
            }
        }
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("extract_sharp_edges.c: #elements %ld, #nodes %ldn",
               (long)mesh.nelements,
               (long)mesh.nnodes);
        printf("----------------------------------------\n");
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free Resources
    ///////////////////////////////////////////////////////////////////////////////

    mesh_destroy(&mesh);

    free(rowptr);
    free(colidx);
    free(checked);
    free(dihedral_angle);

    for (int d = 0; d < 3; d++) {
        free(normal[d]);
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
