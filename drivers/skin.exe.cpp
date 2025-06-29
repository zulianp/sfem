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
#include "sfem_mesh_write.h"

#include "extract_surface_graph.h"

#include "sfem_defs.h"

#include "argsort.h"

#include "adj_table.h"

#include "sfem_hex8_mesh_graph.h"
#include "sfem_sshex8_skin.h"
#include "sshex8.h"  // FIXME

#include "sfem_glob.hpp"

#include <fstream>

#include "sfem_API.hpp"

static SFEM_INLINE void normalize(real_t *const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

void correct_side_orientation(const ptrdiff_t                          nsides,
                              idx_t **const SFEM_RESTRICT              sides,
                              const element_idx_t *const SFEM_RESTRICT parent,
                              idx_t **const SFEM_RESTRICT              elements,
                              geom_t **const SFEM_RESTRICT             xyz) {
    double tick = MPI_Wtime();

    for (ptrdiff_t i = 0; i < nsides; ++i) {
        const idx_t i0 = sides[0][i];
        const idx_t i1 = sides[1][i];
        const idx_t i2 = sides[2][i];

        real_t u[3] = {xyz[0][i1] - xyz[0][i0], xyz[1][i1] - xyz[1][i0], xyz[2][i1] - xyz[2][i0]};
        real_t v[3] = {xyz[0][i2] - xyz[0][i0], xyz[1][i2] - xyz[1][i0], xyz[2][i2] - xyz[2][i0]};

        normalize(u);
        normalize(v);

        real_t n[3] = {u[1] * v[2] - u[2] * v[1],  //
                       u[2] * v[0] - u[0] * v[2],  //
                       u[0] * v[1] - u[1] * v[0]};

        normalize(n);

        // Compute element barycenter
        real_t              b[3] = {0, 0, 0};
        const element_idx_t p    = parent[i];

        for (int d = 0; d < 4; ++d) {
            b[0] += xyz[0][elements[d][p]];
            b[1] += xyz[1][elements[d][p]];
            b[2] += xyz[2][elements[d][p]];
        }

        b[0] /= 4;
        b[1] /= 4;
        b[2] /= 4;

        b[0] -= xyz[0][i0];
        b[1] -= xyz[1][i0];
        b[2] -= xyz[2][i0];

        real_t cos_angle = n[0] * b[0] + n[1] * b[1] + n[2] * b[2];
        assert(cos_angle != 0.);

        if (cos_angle > 0) {
            // Normal pointing inside
            // Switch order of nodes
            sides[1][i] = i2;
            sides[2][i] = i1;
        }
    }

    double tock = MPI_Wtime();
    printf("skin.c: correct_side_orientation\t%g seconds\n", tock - tick);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    sfem::create_directory(output_folder);

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    auto points      = mesh->points()->data();
    const ptrdiff_t n_nodes     = mesh->n_nodes();
    const ptrdiff_t n_elements  = mesh->n_elements();
    const int       spatial_dim = mesh->spatial_dimension();

    const char *SFEM_ELEMENT_TYPE = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_TYPE, );
    enum ElemType override_element_type = INVALID;
    if (SFEM_ELEMENT_TYPE) {
        override_element_type = type_from_string(SFEM_ELEMENT_TYPE);
    }

    if (override_element_type == SSHEX8) {
        int SFEM_ELEMENT_REFINE_LEVEL = 0;
        SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

        if (!SFEM_ELEMENT_REFINE_LEVEL) {
            fprintf(stderr,
                    "ElemType sshex8 requires SFEM_ELEMENT_REFINE_LEVEL to be defined and >= "
                    "2!\n");
            return EXIT_FAILURE;
        }

        if (mesh->element_type() == HEX8) {
            // Generate proteus mesh on the fly!

            const int nxe = sshex8_nxe(SFEM_ELEMENT_REFINE_LEVEL);
            const int txe = sshex8_txe(SFEM_ELEMENT_REFINE_LEVEL);

            auto elements   = sfem::create_host_buffer<idx_t>(nxe, n_elements);
            auto d_elements = mesh->elements()->data();

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    d_elements[d][i] = SFEM_IDX_INVALID;
                }
            }

            ptrdiff_t n_unique_nodes, interior_start;
            sshex8_generate_elements(SFEM_ELEMENT_REFINE_LEVEL,
                                     n_elements,
                                     n_nodes,
                                     mesh->elements()->data(),
                                     d_elements,
                                     &n_unique_nodes,
                                     &interior_start);

            const int nnxs = (SFEM_ELEMENT_REFINE_LEVEL + 1) * (SFEM_ELEMENT_REFINE_LEVEL + 1);

            ptrdiff_t n_surf_elements = 0;
            idx_t   **d_surf_elems    = (idx_t **)calloc(nnxs, sizeof(idx_t *));

            element_idx_t *parent = 0;
            sshex8_skin(SFEM_ELEMENT_REFINE_LEVEL, n_elements, d_elements, &n_surf_elements, d_surf_elems, &parent);
            auto surf_elements = sfem::manage_host_buffer<idx_t>(nnxs, n_surf_elements, d_surf_elems);
            auto surf_parent   = sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent);

            char path[2048];
            for (int d = 0; d < nnxs; d++) {
                snprintf(path, sizeof(path), "%s/i%d.raw", output_folder, d);
                array_write(comm, path, SFEM_MPI_IDX_T, d_surf_elems[d], n_surf_elements, n_surf_elements);
            }

            // TODO
            return SFEM_SUCCESS;

        } else {
            assert(0);
            return SFEM_FAILURE;
        }
    }

    enum ElemType st   = side_type(mesh->element_type());
    const int     nnxs = elem_num_nodes(st);

    ptrdiff_t      n_surf_elements = 0;
    idx_t        **d_surf_elems    = (idx_t **)malloc(nnxs * sizeof(idx_t *));
    element_idx_t *parent          = 0;
    int16_t       *side_idx        = 0;

    if (extract_skin_sideset(
                n_elements, n_nodes, mesh->element_type(), mesh->elements()->data(), &n_surf_elements, &parent, &side_idx) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Failed to extract skin!\n");
    }

    for (int s = 0; s < nnxs; s++) {
        d_surf_elems[s] = (idx_t *)malloc(n_surf_elements * sizeof(idx_t));
    }

    if (extract_surface_from_sideset(
                mesh->element_type(), mesh->elements()->data(), n_surf_elements, parent, side_idx, d_surf_elems) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Unable to extract surface from sideset!\n");
    }

    idx_t *vol2surf = (idx_t *)malloc(n_nodes * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        vol2surf[i] = SFEM_IDX_INVALID;
    }

    ptrdiff_t next_id = 0;
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < nnxs; ++d) {
            idx_t idx = d_surf_elems[d][i];
            if (vol2surf[idx] == SFEM_IDX_INVALID) {
                vol2surf[idx] = next_id++;
            }
        }
    }

    ptrdiff_t n_surf_nodes = next_id;
    auto surf_points = sfem::create_host_buffer<geom_t>(spatial_dim, n_surf_nodes);
    auto d_surf_points = surf_points->data();
  
    idx_t *mapping = (idx_t *)malloc(n_surf_nodes * sizeof(idx_t));

 
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        if (vol2surf[i] < 0) continue;
        mapping[vol2surf[i]] = i;

        for (int d = 0; d < spatial_dim; ++d) {
            d_surf_points[d][vol2surf[i]] = points[d][i];
        }
    }

    // Correct normal orientation using elements with orginal indexing (only with P1 for the moment)
    if (mesh->element_type() == TET4) {
        correct_side_orientation(n_surf_elements, d_surf_elems, parent, mesh->elements()->data(), points);
    }

    // Re-index elements
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < nnxs; ++d) {
            d_surf_elems[d][i] = vol2surf[d_surf_elems[d][i]];
        }
    }

    free(vol2surf);

    auto surf_elements = sfem::manage_host_buffer<idx_t>(nnxs, n_surf_elements, d_surf_elems);
    auto surf_parent   = sfem::manage_host_buffer<element_idx_t>(n_surf_elements, parent);
    auto surf_side_idx = sfem::manage_host_buffer<int16_t>(n_surf_elements, side_idx);

    auto        sideset      = sfem::Sideset::create(mesh->comm(), surf_parent, surf_side_idx);
    std::string sideset_path = output_folder;
    sideset_path += "/sidesets";
    sfem::create_directory(sideset_path.c_str());
    sideset->write(sideset_path.c_str());

    auto surf = std::make_shared<sfem::Mesh>(
            sfem::Communicator::wrap(comm), spatial_dim, shell_type(side_type(mesh->element_type())), n_surf_elements, surf_elements, n_surf_nodes, surf_points);

    surf->write(output_folder);

    // Clean-up

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("Surface: #elements %ld #nodes %ld\n", (long)n_surf_elements, (long)n_surf_nodes);
    }
    
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
