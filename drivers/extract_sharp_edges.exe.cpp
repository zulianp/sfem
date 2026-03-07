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
#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_mesh_write.hpp"

#include "sortreduce.h"

#include "smesh_extractions.hpp"
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

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    if (shell_type(mesh->element_type(0)) != smesh::TRISHELL3) {
        fprintf(stderr, "%s this driver only supports triangle meshes", argv[0]);
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build graphs
    ///////////////////////////////////////////////////////////////////////////////

    auto sharp_edges = smesh::extract_sharp_edges(*mesh, angle_threshold);
    auto disconnected_elements = smesh::extract_disconnected_faces(*mesh, sharp_edges);
    auto corners = smesh::extract_sharp_corners(n_nodes, sharp_edges, true);

    const ptrdiff_t n_sharp_edges = sharp_edges->extent(1);
    const ptrdiff_t n_disconnected_elements = disconnected_elements->size();
    const ptrdiff_t n_corners = corners->size();
    auto e0 = sharp_edges->data()[0];
    auto e1 = sharp_edges->data()[1];

    {
        char path[1024 * 10];
        snprintf(path, sizeof(path), "%s/i0.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e0, n_sharp_edges, n_sharp_edges);

        snprintf(path, sizeof(path), "%s/i1.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, e1, n_sharp_edges, n_sharp_edges);

        snprintf(path, sizeof(path), "%s/corners", output_folder);

        sfem::create_directory(path);

        snprintf(path, sizeof(path), "%s/corners/i0.raw", output_folder);
        array_write(comm, path, SFEM_MPI_COUNT_T, corners->data(), n_corners, n_corners);

        snprintf(path, sizeof(path), "%s/e." dtype_ELEMENT_IDX_T ".raw", output_folder);
        array_write(comm,
                    path,
                    SFEM_MPI_ELEMENT_IDX_T,
                    disconnected_elements->data(),
                    n_disconnected_elements,
                    n_disconnected_elements);

        {
            const int nxe = elem_num_nodes(mesh->element_type(0));
            auto delems = sfem::create_host_buffer<idx_t>(nxe, n_disconnected_elements);
            auto src = mesh->elements(0)->data();

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < n_disconnected_elements; i++) {
                    delems->data()[d][i] = src[d][disconnected_elements->data()[i]];
                }
            }

            snprintf(path, sizeof(path), "%s/disconnected", output_folder);
            sfem::create_directory(path);

            delems->to_files(smesh::Path(std::string(output_folder) + "/disconnected/i%d." + std::string(smesh::TypeToString<idx_t>::value())));
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

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
