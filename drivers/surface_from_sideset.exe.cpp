#include "sfem_API.hpp"

#include "matrixio_array.h"
#include "sfem_defs.hpp"

#include "adj_table.hpp"
#include "sfem_hex8_mesh_graph.hpp"
#include "sshex8_mesh.hpp"
#include "smesh_sidesets.hpp"
#include "smesh_sshex8_graph.hpp"

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
    auto m = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(path_mesh));
    const char *path_sideset = argv[2];
    auto s = sfem::Sideset::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(path_sideset));
    const auto elements = m->elements(0)->data();

    // Make sure the folder exists
    sfem::create_directory(argv[3]);

    smesh::ElemType element_type       = m->element_type(0);
    std::string path_output_format = std::string(argv[3]) + "/i%d." + std::string(smesh::TypeToString<idx_t>::value());

    if (SFEM_ELEMENT_REFINE_LEVEL <= 1) {
        int  nnxs       = elem_num_nodes(side_type(element_type));
        auto surf_elems = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size());

        {
            SFEM_TRACE_SCOPE("extract_surface_from_sideset");
            if (smesh::extract_surface_from_sideset(element_type,
                                                    elements,
                                                    s->parent()->size(),
                                                    s->parent()->data(),
                                                    s->lfi()->data(),
                                                    surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }
        }

        if (surf_elems->to_files(smesh::Path(path_output_format)) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to write files!\n");
        }

        if (SFEM_EXTRACT_NODESET) {
            ptrdiff_t n_nodes{0};
            idx_t    *nodes{nullptr};

            {
                SFEM_TRACE_SCOPE("extract_nodeset_from_sideset");
                if (smesh::extract_nodeset_from_sideset(element_type,
                                                        elements,
                                                        s->parent()->size(),
                                                        s->parent()->data(),
                                                        s->lfi()->data(),
                                                        &n_nodes,
                                                        &nodes) != SFEM_SUCCESS) {
                    SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                }
            }

            std::string path_nodes = std::string(argv[3]) + "/nodeset." + std::string(smesh::TypeToString<idx_t>::value());
            auto nodeset = sfem::manage_host_buffer(n_nodes, nodes);
            nodeset->to_file(smesh::Path(path_nodes));
        }

    } else {
        if (element_type != smesh::HEX8) {
            SFEM_ERROR("Element %s not supported for semi-structured discretization\n", type_to_string(element_type));
        }

        auto ss = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);
        const int level = sfem::semi_structured_level(*ss);

        std::shared_ptr<sfem::Buffer<idx_t *>> surf_elems;

        if (SFEM_CONVERT_TO_STD_MESH) {
            const int nnxs = 4;
            const int nexs = level * level;
            surf_elems     = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size() * nexs);
            auto ss_surf_elems = sfem::create_host_buffer<idx_t>((level + 1) * (level + 1), s->parent()->size());

            SFEM_TRACE_SCOPE("sshex8_extract_surface_from_sideset");
            if (smesh::sshex8_extract_surface_from_sideset(level,
                                                           ss->elements(0)->data(),
                                                           s->parent()->size(),
                                                           s->parent()->data(),
                                                           s->lfi()->data(),
                                                           ss_surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }

            ssquad4_to_standard_quad4_mesh(level, s->parent()->size(), ss_surf_elems->data(), surf_elems->data());

        } else {
            int nnxs   = (level + 1) * (level + 1);
            surf_elems = sfem::create_host_buffer<idx_t>(nnxs, s->parent()->size());

            SFEM_TRACE_SCOPE("sshex8_extract_surface_from_sideset");
            if (smesh::sshex8_extract_surface_from_sideset(level,
                                                           ss->elements(0)->data(),
                                                           s->parent()->size(),
                                                           s->parent()->data(),
                                                           s->lfi()->data(),
                                                           surf_elems->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }
        }

        if (surf_elems->to_files(smesh::Path(path_output_format)) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to write files!\n");
        }

        if (SFEM_EXTRACT_NODESET) {
            ptrdiff_t n_nodes{0};
            idx_t    *nodes{nullptr};

            {
                SFEM_TRACE_SCOPE("sshex8_extract_nodeset_from_sideset");
                if (smesh::sshex8_extract_nodeset_from_sideset(level,
                                                               ss->elements(0)->data(),
                                                               s->parent()->size(),
                                                               s->parent()->data(),
                                                               s->lfi()->data(),
                                                               &n_nodes,
                                                               &nodes) != SFEM_SUCCESS) {
                    SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                }
            }

            std::string path_nodes = std::string(argv[3]) + "/nodeset." + std::string(smesh::TypeToString<idx_t>::value());
            auto nodeset = sfem::manage_host_buffer(n_nodes, nodes);
            nodeset->to_file(smesh::Path(path_nodes));
        }
    }

    return MPI_Finalize();
}
