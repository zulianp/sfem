#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

static SFEM_INLINE void normal(real_t u[3], real_t v[3], real_t *n) {
    const real_t u_len = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    const real_t v_len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    for (int d = 0; d < 3; ++d) {
        u[d] /= u_len;
        v[d] /= v_len;
    }

    n[0] = (u[1] * v[2]) - (u[2] * v[1]);
    n[1] = (u[2] * v[0]) - (u[0] * v[2]);
    n[2] = (u[0] * v[1]) - (u[1] * v[0]);

    const real_t n_len = sqrt((n[0] * n[0]) + (n[1] * n[1]) + (n[2] * n[2]));

    for (int d = 0; d < 3; ++d) {
        n[d] /= n_len;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *output_folder = argv[2];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], folder, output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    count_t *adj_ptr = 0;
    idx_t *adj_idx = 0;

    ///////////////////////////////////////////////////////////////////////////////
    // Create dual-graph for navigating neighboring elements
    ///////////////////////////////////////////////////////////////////////////////

    create_dual_graph(mesh.nelements, mesh.nnodes, mesh.element_type, mesh.elements, &adj_ptr, &adj_idx);

    uint8_t *color = (uint8_t *)malloc(mesh.nelements * sizeof(uint8_t));
    memset(color, 0, mesh.nelements * sizeof(uint8_t));

    const real_t angle_threshold = 0.99;

    ptrdiff_t size_queue = (mesh.nelements + 1);
    ptrdiff_t *elem_queue = (ptrdiff_t *)malloc(size_queue * sizeof(ptrdiff_t));

    elem_queue[0] = 0;
    for (ptrdiff_t e = 1; e < size_queue; ++e) {
        elem_queue[e] = -1;
    }

    // Next slot
    ptrdiff_t next_slot = 1;
    uint8_t next_color = 1;

    ///////////////////////////////////////////////////////////////////////////////
    // Create marker for different faces based on dihedral angles
    ///////////////////////////////////////////////////////////////////////////////

    for (ptrdiff_t q = 0; elem_queue[q] >= 0; q = (q + 1) % size_queue) {
        const ptrdiff_t e = elem_queue[q];

        if(color[e]) continue;

        real_t u[3];
        real_t v[3];
        real_t n[3];

        {
            const idx_t idx0 = mesh.elements[0][e];
            const idx_t idx1 = mesh.elements[1][e];
            const idx_t idx2 = mesh.elements[2][e];

            for (int d = 0; d < 3; ++d) {
                u[d] = mesh.points[d][idx1] - mesh.points[d][idx0];
                v[d] = mesh.points[d][idx2] - mesh.points[d][idx0];
            }

            normal(u, v, n);
        }

        const count_t e_begin = adj_ptr[e];
        const count_t e_end = adj_ptr[e + 1];

        real_t current_thres = angle_threshold;

        for (count_t k = e_begin; k < e_end; ++k) {
            const idx_t e_adj = adj_idx[k];

            if(!color[e_adj]) {
                elem_queue[next_slot++ % size_queue] = e_adj;
                continue;
            }

            real_t ua[3];
            real_t va[3];
            real_t na[3];

            {
                const idx_t idx0 = mesh.elements[0][e_adj];
                const idx_t idx1 = mesh.elements[1][e_adj];
                const idx_t idx2 = mesh.elements[2][e_adj];

                for (int d = 0; d < 3; ++d) {
                    ua[d] = mesh.points[d][idx1] - mesh.points[d][idx0];
                    va[d] = mesh.points[d][idx2] - mesh.points[d][idx0];
                }

                normal(ua, va, na);
            }

            const real_t cos_angle = fabs((n[0] * na[0]) + (n[1] * na[1]) + (n[2] * na[2]));

            if (cos_angle > current_thres) {
                color[e] = color[e_adj];
                current_thres = cos_angle;
            }
        }

        if (!color[e]) {
            color[e] = next_color++;
        }

        elem_queue[q] = -1;
    }

    printf("num_colors = %d, sides touched %ld / %ld\n", (int)next_color, (long)next_slot, (long)mesh.nelements);

    array_write(comm, "color.raw", MPI_CHAR, color, mesh.nelements, mesh.nelements);

    ///////////////////////////////////////////////////////////////////////////////
    // Create separate meshes for the different parts
    ///////////////////////////////////////////////////////////////////////////////

    {
        // TODO
    }

    if (0) {
        // handle mapping
        char path[2048];
        sprintf(path, "%s/node_mapping.raw", folder);

        idx_t *node_mapping = 0;
        ptrdiff_t nlocal, nglobal;
        array_create_from_file(comm, path, SFEM_MPI_IDX_T, (void **)&node_mapping, &nlocal, &nglobal);

        // create parts
        // TODO

        // clean-up
        free(node_mapping);
    }

    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
