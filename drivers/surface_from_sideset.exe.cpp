#include "sfem_API.hpp"

#include "matrixio_array.h"
#include "sfem_defs.h"

#include "adj_table.h"
#include "sfem_hex8_mesh_graph.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 4) {
        fprintf(stderr, "usage: %s <mesh> <sideset> <output_folder>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_EXTRACT_NODESET = 0;
    SFEM_READ_ENV(SFEM_EXTRACT_NODESET, atoi);

    int SFEM_CONVERT_TO_STD_MESH = 0;
    SFEM_READ_ENV(SFEM_CONVERT_TO_STD_MESH, atoi);

    const char *path_mesh    = argv[1];
    auto        m            = sfem::Mesh::create_from_file(comm, path_mesh);
    const char *path_sideset = argv[2];
    auto        s            = sfem::Sideset::create_from_file(comm, path_sideset);
    auto       &mesh         = *((mesh_t *)m->impl_mesh());

    // Make sure the folder exists
    struct stat st = {0};
    if (stat(argv[3], &st) == -1) {
        mkdir(argv[3], 0700);
    }

    enum ElemType element_type       = (enum ElemType)mesh.element_type;
    std::string   path_output_format = argv[3];
    path_output_format += "/i%d.raw";

    if (SFEM_ELEMENT_REFINE_LEVEL <= 1) {
        int  nnxs       = elem_num_nodes(side_type(element_type));
        auto surf_elems = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size());

        {
            SFEM_TRACE_SCOPE("extract_surface_from_sideset");
            if (extract_surface_from_sideset(element_type,
                                             mesh.elements,
                                             s->parent()->size(),
                                             s->parent()->data(),
                                             s->lfi()->data(),
                                             surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }
        }

        if (surf_elems->to_files(path_output_format.c_str()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to write files!\n");
        }

        if (SFEM_EXTRACT_NODESET) {
            ptrdiff_t n_nodes{0};
            idx_t    *nodes{nullptr};

            {
                SFEM_TRACE_SCOPE("extract_nodeset_from_sideset");
                if (extract_nodeset_from_sideset(element_type,
                                                 mesh.elements,
                                                 s->parent()->size(),
                                                 s->parent()->data(),
                                                 s->lfi()->data(),
                                                 &n_nodes,
                                                 &nodes) != SFEM_SUCCESS) {
                    SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                }
            }

            std::string path_nodes = argv[3];
            path_nodes += "/nodeset.raw";
            auto nodeset = sfem::manage_host_buffer(n_nodes, nodes);
            nodeset->to_file(path_nodes.c_str());
        }

    } else {
        if (element_type != HEX8) {
            SFEM_ERROR("Element %s not supported for semi-structured discretization\n", type_to_string(element_type));
        }

        auto ss = sfem::SemiStructuredMesh::create(m, SFEM_ELEMENT_REFINE_LEVEL);

        std::shared_ptr<sfem::Buffer<idx_t *>> surf_elems;

        if (SFEM_CONVERT_TO_STD_MESH) {
            const int nnxs = 4;
            const int nexs = ss->level() * ss->level();
            surf_elems     = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size() * nexs);

            SFEM_TRACE_SCOPE("sshex8_extract_quadshell4_surface_from_sideset");
            if (sshex8_extract_quadshell4_surface_from_sideset(ss->level(),
                                                               ss->element_data(),
                                                               s->parent()->size(),
                                                               s->parent()->data(),
                                                               s->lfi()->data(),
                                                               surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }

        } else {
            int nnxs   = (ss->level() + 1) * (ss->level() + 1);
            surf_elems = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size());

            SFEM_TRACE_SCOPE("sshex8_extract_surface_from_sideset");
            if (sshex8_extract_surface_from_sideset(ss->level(),
                                                    ss->element_data(),
                                                    s->parent()->size(),
                                                    s->parent()->data(),
                                                    s->lfi()->data(),
                                                    surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }
        }

        if (surf_elems->to_files(path_output_format.c_str()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to write files!\n");
        }

        if (SFEM_EXTRACT_NODESET) {
            ptrdiff_t n_nodes{0};
            idx_t    *nodes{nullptr};

            {
                SFEM_TRACE_SCOPE("sshex8_extract_nodeset_from_sideset");
                if (sshex8_extract_nodeset_from_sideset(ss->level(),
                                                        ss->element_data(),
                                                        s->parent()->size(),
                                                        s->parent()->data(),
                                                        s->lfi()->data(),
                                                        &n_nodes,
                                                        &nodes) != SFEM_SUCCESS) {
                    SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                }
            }

            std::string path_nodes = argv[3];
            path_nodes += "/nodeset.raw";
            auto nodeset = sfem::manage_host_buffer(n_nodes, nodes);
            nodeset->to_file(path_nodes.c_str());
        }
    }

    return MPI_Finalize();
}
