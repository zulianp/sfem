#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "mesh_aura.h"
#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_gap.h"

#include "mesh_utils.h"

#include "mass.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 13) {
        fprintf(stderr,
                "usage: %s <folder> <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <output_folder>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_INTERPOLATE = 1;
    SFEM_READ_ENV(SFEM_INTERPOLATE, atoi);

    double tick = MPI_Wtime();

    const char* folder = argv[1];
    ptrdiff_t nglobal[3] = {atol(argv[2]), atol(argv[3]), atol(argv[4])};
    geom_t origin[3] = {atof(argv[5]), atof(argv[6]), atof(argv[7])};
    geom_t delta[3] = {atof(argv[8]), atof(argv[9]), atof(argv[10])};
    const char* data_path = argv[11];
    const char* output_folder = argv[12];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t* sdf = 0;
    ptrdiff_t nlocal[3];

    {
        double ndarray_read_tick = MPI_Wtime();

        if (ndarray_create_from_file(
                comm, data_path, SFEM_MPI_GEOM_T, 3, (void**)&sdf, nlocal, nglobal)) {
            return EXIT_FAILURE;
        }

        double ndarray_read_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] ndarray_create_from_file %g (seconds)\n",
                   rank,
                   ndarray_read_tock - ndarray_read_tick);
        }
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

    if (size > 1) {
        geom_t* psdf;
        sdf_view(comm,
                 mesh.nnodes,
                 mesh.points[2],
                 nlocal,
                 nglobal,
                 stride,
                 origin,
                 delta,
                 sdf,
                 &psdf,
                 &nlocal[2],
                 &origin[2]);

        free(sdf);
        sdf = psdf;
    }

    real_t* g = calloc(mesh.nnodes, sizeof(real_t));
    real_t* xnormal = calloc(mesh.nnodes, sizeof(real_t));
    real_t* ynormal = calloc(mesh.nnodes, sizeof(real_t));
    real_t* znormal = calloc(mesh.nnodes, sizeof(real_t));
    {
        double resample_tick = MPI_Wtime();

        if (SFEM_INTERPOLATE) {
            interpolate_gap(
                // Mesh
                mesh.n_owned_nodes,
                mesh.points,
                // SDF
                nlocal,
                stride,
                origin,
                delta,
                sdf,
                // Output
                g,
                xnormal,
                ynormal,
                znormal);
        } else {
            if (size == 1) {
                resample_gap(
                    // Mesh
                    mesh.element_type,
                    mesh.nelements,
                    mesh.n_owned_nodes,
                    mesh.elements,
                    mesh.points,
                    // SDF
                    nlocal,
                    stride,
                    origin,
                    delta,
                    sdf,
                    // Output
                    g,
                    xnormal,
                    ynormal,
                    znormal);
            } else {
                resample_gap_local(
                    // Mesh
                    mesh.element_type,
                    mesh.nelements,
                    mesh.nnodes,
                    mesh.elements,
                    mesh.points,
                    // SDF
                    nlocal,
                    stride,
                    origin,
                    delta,
                    sdf,
                    // Output
                    g,
                    xnormal,
                    ynormal,
                    znormal);

                real_t* mass_vector = calloc(mesh.nnodes, sizeof(real_t));

                assemble_lumped_mass(shell_type(mesh.element_type),
                                     mesh.nelements,
                                     mesh.nnodes,
                                     mesh.elements,
                                     mesh.points,
                                     mass_vector);

                // exchange ghost nodes and add contribution
                if (size > 1) {
                    send_recv_t slave_to_master;
                    mesh_create_nodal_send_recv(&mesh, &slave_to_master);

                    ptrdiff_t count = mesh_exchange_master_buffer_count(&slave_to_master);
                    real_t* real_buffer = malloc(count * sizeof(real_t));

                    exchange_add(&mesh, &slave_to_master, mass_vector, real_buffer);
                    exchange_add(&mesh, &slave_to_master, g, real_buffer);
                    exchange_add(&mesh, &slave_to_master, xnormal, real_buffer);
                    exchange_add(&mesh, &slave_to_master, ynormal, real_buffer);
                    exchange_add(&mesh, &slave_to_master, znormal, real_buffer);
                    free(real_buffer);
                    send_recv_destroy(&slave_to_master);
                }

                // divide by the mass vector
                for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                    if (mass_vector[i] == 0) {
                        fprintf(stderr,
                                "Found 0 mass at %ld, info (%ld, %ld)\n",
                                i,
                                mesh.n_owned_nodes,
                                mesh.nnodes);
                    }

                    assert(mass_vector[i] != 0);
                    g[i] /= mass_vector[i];
                }

                for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                    const real_t xn = xnormal[i];
                    const real_t yn = ynormal[i];
                    const real_t zn = znormal[i];
                    const real_t ln = sqrt(xn * xn + yn * yn + zn * zn);

                    assert(ln != 0.);

                    xnormal[i] /= ln;
                    ynormal[i] /= ln;
                    znormal[i] /= ln;
                }

                free(mass_vector);
            }
        }

        double resample_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] resample %g (seconds)\n", rank, resample_tock - resample_tick);
        }
    }

    // Write result to disk
    {
        double io_tick = MPI_Wtime();

        char path[1024 * 10];
        sprintf(path, "%s/gap.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, g);

        sprintf(path, "%s/xnormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, xnormal);

        sprintf(path, "%s/ynormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, ynormal);

        sprintf(path, "%s/znormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, znormal);

        if (0) {
            for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                g[i] = rank;
            }

            sprintf(path, "%s/rank.float64.raw", output_folder);
            mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, g);
        }

        double io_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] write %g (seconds)\n", rank, io_tock - io_tick);
        }
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    // Free resources
    {
        free(sdf);
        free(g);
        free(xnormal);
        free(ynormal);
        free(znormal);
        mesh_destroy(&mesh);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #grid (%ld x %ld x %ld)\n",
               (long)nelements,
               (long)nnodes,
               nglobal[0],
               nglobal[1],
               nglobal[2]);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
