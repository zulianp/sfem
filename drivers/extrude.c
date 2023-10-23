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

static SFEM_INLINE void normalize(real_t* const vec3)
{
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

void extrude(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT sides,
    geom_t** const SFEM_RESTRICT xyz,
    const geom_t thickness,
    idx_t** const SFEM_RESTRICT extruded_elements,
    geom_t** const SFEM_RESTRICT extruded_xyz)
{
    double tick = MPI_Wtime();

    geom_t **pseudo_normals = malloc(3 * sizeof(geom_t *));
    for (int d = 0; d < 3; d++) {
        pseudo_normals[d] = calloc(nnodes, sizeof(geom_t));
    }

    for (ptrdiff_t i = 0; i < nsides; ++i) {

        const idx_t i0 = sides[0][i];
        const idx_t i1 = sides[1][i];
        const idx_t i2 = sides[2][i];

        real_t u[3] = { xyz[0][i1] - xyz[0][i0], xyz[1][i1] - xyz[1][i0], xyz[2][i1] - xyz[2][i0] };
        real_t v[3] = { xyz[0][i2] - xyz[0][i0], xyz[1][i2] - xyz[1][i0], xyz[2][i2] - xyz[2][i0] };

        normalize(u);
        normalize(v);

        real_t n[3] = {
            u[1] * v[2] - u[2] * v[1], //
            u[2] * v[0] - u[0] * v[2], //
            u[0] * v[1] - u[1] * v[0] //
        };

        normalize(n);

        for(int d = 0; d < 3; d++) {
            pseudo_normals[d][i0] += n[d];
            pseudo_normals[d][i1] += n[d];
            pseudo_normals[d][i2] += n[d];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        real_t n[3];

        for(int d = 0; d < 3; d++) {
            n[d] = pseudo_normals[d][i];
        }

        normalize(n);

        for(int d = 0; d < 3; d++) {
            pseudo_normals[d][i] = n[d];
        }
    }   

    // Construct wedge elements
    for (ptrdiff_t i = 0; i < nsides; ++i) {
        for (int d = 0; d < 3; d++) {
            idx_t node = sides[d][i];
            idx_t extruded_node = nnodes + node;

            // Original face (copy node id)
            extruded_elements[d][i] = node;

            // extruded face store extuded node id
            extruded_elements[3 + d][i] = extruded_node;
        }
    }


    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        for (int dd = 0; dd < 3; dd++) {
            extruded_xyz[dd][i] = xyz[dd][i];
        }

        for (int dd = 0; dd < 3; dd++) {
            extruded_xyz[dd][nnodes + i] = xyz[dd][i] + thickness * pseudo_normals[dd][i];
        }
     }

    double tock = MPI_Wtime();
    printf("extrude.c: correct_side_orientation\t%g seconds\n", tock - tick);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 4) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <thickness> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const geom_t thickness = atof(argv[2]);
    const char* output_folder = argv[3];

    {
        struct stat st = { 0 };
        if (stat(output_folder, &st) == -1) {
            mkdir(output_folder, 0700);
        }
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char* folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    mesh_t extruded;
    mesh_init(&extruded);

    extruded.comm = mesh.comm;
    extruded.mem_space = mesh.mem_space;

    extruded.spatial_dim = mesh.spatial_dim;
    extruded.element_type = WEDGE6;

    extruded.nelements = mesh.nelements;
    extruded.nnodes = mesh.nnodes * 2;

    extruded.node_mapping = 0;
    extruded.element_mapping = 0;
    extruded.node_owner = 0;

    int nnxe_extruded = elem_num_nodes(extruded.element_type);
    extruded.elements = malloc(nnxe_extruded * sizeof(idx_t*));
    for (int d = 0; d < nnxe_extruded; d++) {
        extruded.elements[d] = malloc(extruded.nelements * sizeof(idx_t));
    }

    extruded.points = malloc(extruded.spatial_dim * sizeof(geom_t*));
    for (int d = 0; d < extruded.spatial_dim; d++) {
        extruded.points[d] = malloc(extruded.nnodes * sizeof(geom_t));
    }

    extrude(
        mesh.nelements,
        mesh.nnodes,
        mesh.elements,
        mesh.points,
        thickness,
        extruded.elements,
        extruded.points);

    mesh_write(output_folder, &extruded);

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("Surface: #elements %ld #nodes %ld\n", (long)extruded.nelements, (long)extruded.nnodes);
    }

    // Clean-up
    mesh_destroy(&mesh);
    mesh_destroy(&extruded);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
