#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "utils.h"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "sortreduce.hpp"

#include "smesh_extractions.hpp"
#include "smesh_glob.hpp"

#include "sfem_API.hpp"

int extract_sharp_edges_driver(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (argc != 4) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <folder> <angle_threshold> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const geom_t  angle_threshold = atof(argv[2]);
    const char   *output_folder   = argv[3];

    if (!comm->rank()) {
        printf("%s %s %s %s\n", argv[0], argv[1], argv[2], output_folder);
    }

    const double tick = smesh::time_seconds();
    smesh::create_directory(output_folder);

    const char *folder = argv[1];

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    if (shell_type(mesh->element_type(0)) != smesh::TRISHELL3) {
        fprintf(stderr, "%s this driver only supports triangle meshes", argv[0]);
        return EXIT_FAILURE;
    }

    auto sharp_edges             = smesh::extract_sharp_edges(*mesh, angle_threshold);
    auto disconnected_elements   = smesh::extract_disconnected_faces(*mesh, sharp_edges);
    auto corners                 = smesh::extract_sharp_corners(n_nodes, sharp_edges, true);

    const ptrdiff_t n_sharp_edges           = sharp_edges->extent(1);
    const ptrdiff_t n_disconnected_elements = disconnected_elements->size();
    const ptrdiff_t n_corners               = corners->size();
    auto              e0                    = sharp_edges->data()[0];
    auto              e1                    = sharp_edges->data()[1];

    {
        char path[1024 * 10];
        snprintf(path, sizeof(path), "%s/i0.raw", output_folder);
        sfem::Buffer<idx_t>::wrap(n_sharp_edges, e0)->to_file(smesh::Path(path));

        snprintf(path, sizeof(path), "%s/i1.raw", output_folder);
        sfem::Buffer<idx_t>::wrap(n_sharp_edges, e1)->to_file(smesh::Path(path));

        snprintf(path, sizeof(path), "%s/corners", output_folder);

        smesh::create_directory(path);

        snprintf(path, sizeof(path), "%s/corners/i0.raw", output_folder);
        corners->to_file(smesh::Path(path));

        snprintf(path, sizeof(path), "%s/e." dtype_ELEMENT_IDX_T ".raw", output_folder);
        disconnected_elements->to_file(smesh::Path(path));

        {
            const int nxe = elem_num_nodes(mesh->element_type(0));
            auto      delems = sfem::create_host_buffer<idx_t>(nxe, n_disconnected_elements);
            auto      src    = mesh->elements(0)->data();

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < n_disconnected_elements; i++) {
                    delems->data()[d][i] = src[d][disconnected_elements->data()[i]];
                }
            }

            snprintf(path, sizeof(path), "%s/disconnected", output_folder);
            smesh::create_directory(path);

            delems->to_files(smesh::Path(std::string(output_folder) + "/disconnected/i%d." + std::string(smesh::TypeToString<idx_t>::value())));
        }
    }

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("extract_sharp_edges.c: #elements %ld, #nodes %ld, #n_sharp_edges %ld\n",
               (long)n_elements,
               (long)n_nodes,
               (long)n_sharp_edges);
        printf("----------------------------------------\n");
    }

    const double tock = smesh::time_seconds();
    if (!comm->rank()) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return extract_sharp_edges_driver(ctx->communicator(), argc, argv);
}
