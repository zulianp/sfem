#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 6) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <x> <y> <z> <max_nodes> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    geom_t roi[3] = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
    ptrdiff_t max_nodes = atol(argv[5]);

    const char *output_folder = "./";
    if (argc > 6) {
        output_folder = argv[6];
    }

    if (!rank) {
        printf("%s %s %g %g %g %ld %s\n",
               argv[0],
               argv[1],
               (double)roi[0],
               (double)roi[1],
               (double)roi[2],
               (long)max_nodes,
               output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (max_nodes > mesh.nnodes) {
        MPI_Abort(comm, -1);
    }

    // double tack = MPI_Wtime();

    geom_t closest_sq_dist = 1000000;
    ptrdiff_t closest_node = -1;

    const int dim = mesh.spatial_dim;
    geom_t *sq_dists = (geom_t *)malloc(mesh.nnodes * sizeof(geom_t));

    for (ptrdiff_t node = 0; node < mesh.nnodes; ++node) {
        geom_t sq_dist = 0;
        for (int d = 0; d < dim; ++d) {
            const real_t m_x = mesh.points[d][node];
            const real_t roi_x = roi[d];
            const real_t diff = m_x - roi_x;
            sq_dist += diff * diff;
        }

        sq_dists[node] = sq_dist;

        if (sq_dist < closest_sq_dist) {
            closest_sq_dist = sq_dist;
            closest_node = node;
        }
    }

    printf("found: %ld %g\n", closest_node, closest_sq_dist);

    if (closest_node < 0) {
        MPI_Abort(comm, -1);
    }

    idx_t *selected_nodes = (idx_t *)malloc((mesh.nnodes + 1) * sizeof(idx_t));
    idx_t *additional_nodes = (idx_t *)malloc((mesh.nnodes + 1) * sizeof(idx_t));

    memset(selected_nodes, 0, (mesh.nnodes + 1) * sizeof(idx_t));
    memset(additional_nodes, 0, (mesh.nnodes + 1) * sizeof(idx_t));

    int euclidean = 1;
    if (euclidean) {

        idx_t *args = (idx_t *)malloc(mesh.nnodes * sizeof(idx_t));
        argsort_f(mesh.nnodes, sq_dists, args);

        for (ptrdiff_t i = 0; i < max_nodes; ++i) {
            const idx_t idx = args[i];
            selected_nodes[idx + 1] = 1;
        }

        free(args);

    } else {
        count_t *rowptr;
        idx_t *colidx;
        build_crs_graph_3(mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

        // TODO
    }

    free(sq_dists);

    idx_t *selected_elements = (idx_t *)malloc((mesh.nelements + 1) * sizeof(idx_t));
    memset(selected_elements, 0, (mesh.nelements + 1) * sizeof(idx_t));

    for (ptrdiff_t i = 0; i < mesh.nelements; ++i) {
        for (int d = 0; d < mesh.element_type; ++d) {
            idx_t node = mesh.elements[d][i];
            idx_t sn = selected_nodes[node + 1];

            if(sn != 0) {
                selected_elements[i + 1] = 1;
            }
        }

        if(selected_elements[i + 1]) {
            for (int d = 0; d < mesh.element_type; ++d) {
                idx_t node = mesh.elements[d][i];
                idx_t sn = selected_nodes[node + 1];
                
                if(sn == 0) {
                    additional_nodes[node] = 1;
                }
            }
        }
    }

    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        selected_nodes[i + 1] += selected_nodes[i] + additional_nodes[i];
    }

    for (ptrdiff_t i = 0; i < mesh.nelements; ++i) {
        selected_elements[i + 1] += selected_elements[i];
    }

    ptrdiff_t n_selected_nodes = selected_nodes[mesh.nnodes];
    ptrdiff_t n_selected_elements = selected_elements[mesh.nelements];

    idx_t *mapping = (idx_t *)malloc(n_selected_nodes * sizeof(idx_t));
    idx_t **elems = (idx_t **)malloc(mesh.element_type * sizeof(idx_t *));

    for (int d = 0; d < mesh.element_type; ++d) {
        elems[d] = (idx_t *)malloc(n_selected_elements * sizeof(idx_t));
    }

    geom_t **points = (geom_t **)malloc(mesh.spatial_dim * sizeof(geom_t *));
    for (int d = 0; d < mesh.spatial_dim; ++d) {
        points[d] = (geom_t *)malloc(n_selected_nodes * sizeof(geom_t));
    }

    for (ptrdiff_t e = 0; e < mesh.nelements; ++e) {
        const idx_t offset = selected_elements[e];
        if (offset == selected_elements[e + 1]) continue;

        for (int d = 0; d < mesh.element_type; ++d) {
            assert(selected_nodes[mesh.elements[d][e]] != selected_nodes[mesh.elements[d][e] + 1]);
            elems[d][offset] = selected_nodes[mesh.elements[d][e]];
        }
    }

    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        const idx_t offset = selected_nodes[i];
        if(offset == selected_nodes[i+1]) continue;

        for (int d = 0; d < mesh.spatial_dim; ++d) {
            points[d][offset] = mesh.points[d][i];
        }

        mapping[offset] = i;
    }

    // for (ptrdiff_t i = 0; i < mesh.nnodes + 1; ++i) {
    //     const idx_t offset = selected_nodes[i];
    //     printf("%ld\n", (long)offset);
    // }
    // printf("--------------\n");


    // for (ptrdiff_t i = 0; i < mesh.nelements + 1; ++i) {
    //     const idx_t offset = selected_elements[i];
    //     printf("%ld\n", (long)offset);
    // }

    if(!rank) {
        printf("select_submesh.c: nelements=%ld npoints=%ld\n", (long)n_selected_elements, n_selected_nodes);
    }

    free(selected_nodes);
    free(additional_nodes);
    free(selected_elements);


    mesh_t selection;
    selection.comm = mesh.comm;
    selection.mem_space = mesh.mem_space;

    selection.spatial_dim = mesh.spatial_dim;
    selection.element_type = mesh.element_type;

    selection.nelements = n_selected_elements;
    selection.nnodes = n_selected_nodes;

    selection.elements = elems;
    selection.points = points;

    selection.node_mapping = mapping;
    selection.node_owner = 0;

    mesh_write(output_folder, &selection);

    mesh_destroy(&mesh);
    mesh_destroy(&selection);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
