
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
#include "sfem_glob.hpp"
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
    printf("create_surface_from_element_adjaciency_table.c: correct_side_orientation\t%g seconds\n", tock - tick);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 4) {
        if (!rank) {
            fprintf(stderr, "usage: %s <mesh_folder> <adj_table_pattern> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *mesh_folder       = argv[1];
    const char *adj_table_pattern = argv[2];
    const char *output_folder     = argv[3];

    sfem::create_directory(output_folder);

    if (!rank) {
        printf("%s %s %s %s\n", argv[0], mesh_folder, adj_table_pattern, output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read and process table data
    ///////////////////////////////////////////////////////////////////////////////

    auto files   = sfem::find_files(adj_table_pattern);
    int  n_sides = files.size();

    element_idx_t **table            = (element_idx_t **)malloc(n_sides * sizeof(element_idx_t *));
    ptrdiff_t       n_local_elements = 0, _dump_ = 0;

    printf("n_sides (%d):\n", n_sides);
    for (int s = 0; s < n_sides; s++) {
        printf("Reading %s\n", files[s].c_str());
        array_create_from_file(
                comm, files[s].c_str(), SFEM_MPI_ELEMENT_IDX_T, (void **)&table[s], &n_local_elements, &_dump_);
    }

    // Compute vol surface map
    ptrdiff_t n_surf_elements = 0;
    for (int s = 0; s < n_sides; s++) {
        for (ptrdiff_t e = 0; e < n_local_elements; e++) {
            n_surf_elements += table[s][e] < 0;
        }
    }

    element_idx_t *parent         = (element_idx_t *)malloc(n_surf_elements * sizeof(element_idx_t));
    int           *local_side_idx = (int *)malloc(n_surf_elements * sizeof(int));

    ptrdiff_t surf_element_idx = 0;
    for (ptrdiff_t e = 0; e < n_local_elements; e++) {
        for (int s = 0; s < n_sides; s++) {
            if (table[s][e] < 0) {
                parent[surf_element_idx]         = e;
                local_side_idx[surf_element_idx] = s;
                surf_element_idx++;
            }
        }
    }

    // Free resources to leave space for mesh
    for (int s = 0; s < n_sides; s++) {
        free(table[s]);
    }

    free(table);

    ///////////////////////////////////////////////////////////////////////////////
    // Read mesh data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    enum ElemType st   = side_type(mesh->element_type());
    const int     nnxs = elem_num_nodes(st);

    idx_t **surf_elems = 0;
    {
        // Extract surface index
        int  nnxs             = elem_num_nodes(side_type(mesh->element_type()));
        int *local_side_table = (int *)malloc(elem_num_sides(mesh->element_type()) * nnxs * sizeof(int));
        surf_elems            = (idx_t **)malloc(nnxs * sizeof(idx_t *));
        for (int d = 0; d < nnxs; d++) {
            surf_elems[d] = (idx_t *)malloc(n_surf_elements * sizeof(idx_t));
        }

        fill_local_side_table(mesh->element_type(), local_side_table);

        auto elements = mesh->elements()->data();

        for (ptrdiff_t e = 0; e < n_surf_elements; e++) {
            const element_idx_t p = parent[e];
            const int           s = local_side_idx[e];

            for (int d = 0; d < nnxs; d++) {
                int   node_num   = local_side_table[s * nnxs + d];
                idx_t node       = elements[node_num][p];
                surf_elems[d][e] = node;
            }
        }

        free(local_side_table);
        free(local_side_idx);
    }

    idx_t *vol2surf = (idx_t *)malloc(n_nodes * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        vol2surf[i] = SFEM_IDX_INVALID;
    }

    ptrdiff_t next_id = 0;
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < nnxs; ++d) {
            idx_t idx = surf_elems[d][i];
            if (vol2surf[idx] == SFEM_IDX_INVALID) {
                vol2surf[idx] = next_id++;
            }
        }
    }

    ptrdiff_t n_surf_nodes = next_id;
    geom_t  **points       = (geom_t **)malloc(mesh->spatial_dimension() * sizeof(geom_t *));
    for (int d = 0; d < mesh->spatial_dimension(); d++) {
        points[d] = 0;
    }

    idx_t *mapping = (idx_t *)malloc(n_surf_nodes * sizeof(idx_t));

    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        points[d] = (geom_t *)malloc(n_surf_nodes * sizeof(geom_t));
    }

    auto original_points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        if (vol2surf[i] < 0) continue;

        mapping[vol2surf[i]] = i;

        for (int d = 0; d < mesh->spatial_dimension(); ++d) {
            points[d][vol2surf[i]] = original_points[d][i];
        }
    }

    // Correct normal orientation using elements with orginal indexing (only with P1 for the moment)
    if (mesh->element_type() == TET4) {
        correct_side_orientation(n_surf_elements, surf_elems, parent, mesh->elements()->data(), mesh->points()->data());
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

    surf.comm      = mesh->comm();
    // surf.mem_space = mesh->mem_space();

    surf.spatial_dim  = mesh->spatial_dimension();
    surf.element_type = side_type(mesh->element_type());

    surf.nelements = n_surf_elements;
    surf.nnodes    = n_surf_nodes;

    surf.elements = surf_elems;
    surf.points   = points;

    surf.node_mapping    = mapping;
    surf.element_mapping = 0;
    surf.node_owner      = 0;

    mesh_write(output_folder, &surf);

    char path[2048];
    snprintf(path, sizeof(path), "%s/parent.raw", output_folder);
    array_write(comm, path, SFEM_MPI_ELEMENT_IDX_T, parent, n_surf_elements, n_surf_elements);

    // Clean-up

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("Surface: #elements %ld #nodes %ld\n", (long)surf.nelements, (long)surf.nnodes);
    }

    mesh_destroy(&surf);
    free(parent);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
