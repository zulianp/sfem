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

#include "extract_sharp_features.h"
#include "mesh_utils.h"
#include "sfem_glob.hpp"

#include "sfem_API.hpp"

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
    sfem::create_directory(output_folder);

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    if (shell_type(mesh->element_type()) != TRISHELL3) {
        fprintf(stderr, "%s this driver only supports triangle meshes", argv[0]);
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build graphs
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t n_sharp_edges = 0;
    idx_t *e0 = 0;
    idx_t *e1 = 0;

    {  // Extract sharp edges!
        count_t *rowptr = 0;
        idx_t *colidx = 0;
        build_crs_graph_for_elem_type(
            mesh->element_type(), n_elements, n_nodes, mesh->elements()->data(), &rowptr, &colidx);

        extract_sharp_edges(mesh->element_type(),
                            n_elements,
                            n_nodes,
                            mesh->elements()->data(),
                            mesh->points()->data(),
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

    ptrdiff_t n_disconnected_elements = 0;
    element_idx_t *disconnected_elements = 0;

    extract_disconnected_faces(mesh->element_type(),
                               n_elements,
                               n_nodes,
                               mesh->elements()->data(),
                               n_sharp_edges,
                               e0,
                               e1,
                               &n_disconnected_elements,
                               &disconnected_elements);

    ptrdiff_t n_corners = 0;
    idx_t *corners = 0;
    n_sharp_edges =
        extract_sharp_corners(n_nodes, n_sharp_edges, e0, e1, &n_corners, &corners, 1);

    {
        char path[1024 * 10];
        snprintf(path, sizeof(path), "%s/i0.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e0, n_sharp_edges, n_sharp_edges);

        snprintf(path, sizeof(path), "%s/i1.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e1, n_sharp_edges, n_sharp_edges);

        snprintf(path, sizeof(path), "%s/corners", output_folder);

        sfem::create_directory(path);

        snprintf(path, sizeof(path), "%s/corners/i0.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, corners, n_corners, n_corners);

        snprintf(path, sizeof(path), "%s/e." dtype_ELEMENT_IDX_T ".raw", output_folder);
        array_write(comm,
                    path,
                    SFEM_MPI_ELEMENT_IDX_T,
                    disconnected_elements,
                    n_disconnected_elements,
                    n_disconnected_elements);

        {
            const int nxe = elem_num_nodes(mesh->element_type());
            idx_t **delems = allocate_elements(nxe, n_disconnected_elements);
            select_elements(
                nxe, n_disconnected_elements, disconnected_elements, mesh->elements()->data(), delems);

            snprintf(path, sizeof(path), "%s/disconnected", output_folder);
            sfem::create_directory(path);

            for (int d = 0; d < nxe; d++) {
                snprintf(path, sizeof(path), "%s/disconnected/i%d.raw", output_folder, d);
                array_write(comm,
                            path,
                            SFEM_MPI_IDX_T,
                            delems[d],
                            n_disconnected_elements,
                            n_disconnected_elements);
            }

            free_elements(nxe, delems);
        }
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("extract_sharp_edges.c: #elements %ld, #nodes %ld, #n_sharp_edges %ld\n",
               (long)n_elements,
               (long)n_nodes,
               (long)n_sharp_edges);
        printf("----------------------------------------\n");
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free Resources
    ///////////////////////////////////////////////////////////////////////////////

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
