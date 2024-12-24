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

#include "sshex8.h"  // FIXME
#include "sfem_hex8_mesh_graph.h"
#include "sfem_sshex8_skin.h"

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

    {
        struct stat st = {0};
        if (stat(output_folder, &st) == -1) {
            mkdir(output_folder, 0700);
        }
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

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

        if (mesh.element_type == HEX8) {
            // Generate proteus mesh on the fly!

            const int nxe      = sshex8_nxe(SFEM_ELEMENT_REFINE_LEVEL);
            const int txe      = sshex8_txe(SFEM_ELEMENT_REFINE_LEVEL);
            idx_t   **elements = 0;

            elements = malloc(nxe * sizeof(idx_t *));
            for (int d = 0; d < nxe; d++) {
                elements[d] = malloc(mesh.nelements * sizeof(idx_t));
            }

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
                    elements[d][i] = -1;
                }
            }

            ptrdiff_t n_unique_nodes, interior_start;
            sshex8_generate_elements(SFEM_ELEMENT_REFINE_LEVEL,
                                         mesh.nelements,
                                         mesh.nnodes,
                                         mesh.elements,
                                         elements,
                                         &n_unique_nodes,
                                         &interior_start);

            const int nnxs = (SFEM_ELEMENT_REFINE_LEVEL + 1) * (SFEM_ELEMENT_REFINE_LEVEL + 1);

            ptrdiff_t      n_surf_elements = 0;
            idx_t        **surf_elems      = (idx_t **)calloc(nnxs, sizeof(idx_t *));
            element_idx_t *parent          = 0;

            sshex8_skin(SFEM_ELEMENT_REFINE_LEVEL, mesh.nelements, elements, &n_surf_elements, surf_elems, &parent);

            char path[2048];
            for (int d = 0; d < nnxs; d++) {
                sprintf(path, "%s/i%d.raw", output_folder, d);
                array_write(comm, path, SFEM_MPI_IDX_T, surf_elems[d], n_surf_elements, n_surf_elements);
            }

            // TODO
            return SFEM_SUCCESS;

        } else {
            assert(0);
            return SFEM_FAILURE;
        }
    }

    enum ElemType st   = side_type(mesh.element_type);
    const int     nnxs = elem_num_nodes(st);

    ptrdiff_t      n_surf_elements = 0;
    idx_t        **surf_elems      = (idx_t **)malloc(nnxs * sizeof(idx_t *));
    element_idx_t *parent          = 0;
    int16_t       *side_idx        = 0;

    if (extract_skin_sideset(
                mesh.nelements, mesh.nnodes, mesh.element_type, mesh.elements, &n_surf_elements, &parent, &side_idx) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Failed to extract skin!\n");
    }

    for (int s = 0; s < nnxs; s++) {
        surf_elems[s] = malloc(n_surf_elements * sizeof(idx_t));
    }

    if (extract_surface_from_sideset(mesh.element_type, mesh.elements, n_surf_elements, parent, side_idx, surf_elems) !=
        SFEM_SUCCESS) {
        SFEM_ERROR("Unable to extract surface from sideset!\n");
    }

    idx_t *vol2surf = (idx_t *)malloc(mesh.nnodes * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        vol2surf[i] = -1;
    }

    ptrdiff_t next_id = 0;
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < nnxs; ++d) {
            idx_t idx = surf_elems[d][i];
            if (vol2surf[idx] < 0) {
                vol2surf[idx] = next_id++;
            }
        }
    }

    ptrdiff_t n_surf_nodes = next_id;
    geom_t  **points       = (geom_t **)malloc(mesh.spatial_dim * sizeof(geom_t *));
    for (int d = 0; d < mesh.spatial_dim; d++) {
        points[d] = 0;
    }

    idx_t *mapping = (idx_t *)malloc(n_surf_nodes * sizeof(idx_t));

    for (int d = 0; d < mesh.spatial_dim; ++d) {
        points[d] = (geom_t *)malloc(n_surf_nodes * sizeof(geom_t));
    }

    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        if (vol2surf[i] < 0) continue;
        mapping[vol2surf[i]] = i;

        for (int d = 0; d < mesh.spatial_dim; ++d) {
            points[d][vol2surf[i]] = mesh.points[d][i];
        }
    }

    // Correct normal orientation using elements with orginal indexing (only with P1 for the moment)
    if (mesh.element_type == TET4) {
        correct_side_orientation(n_surf_elements, surf_elems, parent, mesh.elements, mesh.points);
    }

    // Re-index elements
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < nnxs; ++d) {
            surf_elems[d][i] = vol2surf[surf_elems[d][i]];
        }
    }

    free(vol2surf);

    mesh_t surf;
    mesh_init(&surf);

    surf.comm      = mesh.comm;
    surf.mem_space = mesh.mem_space;

    surf.spatial_dim  = mesh.spatial_dim;
    surf.element_type = shell_type(side_type(mesh.element_type));

    surf.nelements = n_surf_elements;
    surf.nnodes    = n_surf_nodes;

    surf.elements = surf_elems;
    surf.points   = points;

    surf.node_mapping    = mapping;
    surf.element_mapping = 0;
    surf.node_owner      = 0;

    mesh_write(output_folder, &surf);

    char path[2048];
    sprintf(path, "%s/parent.raw", output_folder);
    array_write(comm, path, SFEM_MPI_ELEMENT_IDX_T, parent, n_surf_elements, n_surf_elements);

    sprintf(path, "%s/lfi.int16.raw", output_folder);
    array_write(comm, path, MPI_SHORT, side_idx, n_surf_elements, n_surf_elements);

    // Clean-up

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("Surface: #elements %ld #nodes %ld\n", (long)surf.nelements, (long)surf.nnodes);
    }

    mesh_destroy(&mesh);
    mesh_destroy(&surf);
    free(parent);
    free(side_idx);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
