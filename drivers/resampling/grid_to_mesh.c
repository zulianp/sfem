#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "mesh_aura.h"
#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_field.h"

#include "tet10_resample_field.h"

#include "mesh_utils.h"

#include "mass.h"

double calculate_flops(const ptrdiff_t nelements, const ptrdiff_t quad_nodes, double time_sec) {
    const double flops = (nelements * (35 + 166 * quad_nodes)) / time_sec;
    return flops;
}

int main(int argc, char* argv[]) {
    // printf("========================================\n");
    // printf("Starting grid_to_mesh\n");
    // printf("========================================\n\n");
    PRINT_CURRENT_FUNCTION;

    sfem_resample_field_info info;

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, mpi_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);

    // print argv
    // if (!rank) {
    //     printf("argc: %d\n", argc);
    //     printf("argv: \n");
    //     for (int i = 0; i < argc; i++) {
    //         printf(" %s", argv[i]);
    //     }
    //     printf("\n");
    // }

    if (argc != 13) {
        fprintf(stderr, "Error: Invalid number of arguments\n\n");

        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_INTERPOLATE = 1;
    SFEM_READ_ENV(SFEM_INTERPOLATE, atoi);

    double tick = MPI_Wtime();

    ptrdiff_t nglobal[3] = {atol(argv[1]), atol(argv[2]), atol(argv[3])};
    geom_t origin[3] = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
    geom_t delta[3] = {atof(argv[7]), atof(argv[8]), atof(argv[9])};
    const char* data_path = argv[10];
    const char* folder = argv[11];
    const char* output_path = argv[12];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    // ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    real_t* field = NULL;
    ptrdiff_t nlocal[3];

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    {
        double ndarray_read_tick = MPI_Wtime();

        if (SFEM_READ_FP32) {
            
            float* temp = NULL;

            if (ndarray_create_from_file(
                        comm, data_path, MPI_FLOAT, 3, (void**)&temp, nlocal, nglobal)) {
                exit(EXIT_FAILURE);
            }

            // {  /// DEBUG ///
            // printf("temp (ptr): %p, %s:%d\n", (void *)temp, __FILE__, __LINE__);

            // double norm_temp = 0.0;
            // double max_temp = temp[0];
            // double min_temp = temp[0];

            ptrdiff_t n_zyx = nlocal[0] * nlocal[1] * nlocal[2];
            field = malloc(n_zyx * sizeof(real_t));

            // if (field == NULL) {
            //     fprintf(stderr, "Error: malloc failed\n");
            //     exit(EXIT_FAILURE);
            // }

            for (ptrdiff_t i = 0; i < n_zyx; i++) {
                field[i] = (real_t)(temp[i]);

                // norm_temp += (double)(temp[i] * temp[i]);
                // max_temp = (double)(fmax(max_temp, temp[i]));
                // min_temp = (double)(fmin(min_temp, temp[i]));
            }

            // norm_temp = sqrt(norm_temp);

            // printf("\n");
            // printf("norm_temp = %1.14e , %s:%d\n", norm_temp, __FILE__, __LINE__);
            // printf("max_temp  = %1.14e , %s:%d\n", max_temp, __FILE__, __LINE__);
            // printf("min_temp  = %1.14e , %s:%d\n", min_temp, __FILE__, __LINE__);
            // printf("n_zyx     = %ld , %s:%d\n", n_zyx, __FILE__, __LINE__);
            // printf("field == NULL: %s, %s:%d\n", field == NULL ? "true" : "false", __FILE__, __LINE__);
            // printf("size field = %ld MB , %s:%d\n", (n_zyx * sizeof(real_t) / 1024 / 1024), __FILE__, __LINE__);

            // } /// end DEBUG ///
            free(temp);

        } else {
            if (ndarray_create_from_file(
                        comm, data_path, SFEM_MPI_REAL_T, 3, (void**)&field, nlocal, nglobal)) {
                return EXIT_FAILURE;
            }
        }

        // { /// DEBUG ///
        //     double filed_norm = 0.0;
        //     double filed_max = field[0];
        //     double filed_min = field[0];

        //     ptrdiff_t n_zyx_private = nlocal[0] * nlocal[1] * nlocal[2];
        //     for(ptrdiff_t i = 0; i < n_zyx_private; i++) {
        //         // field[i] = sin((double)(i) / 10000.0);
        //         filed_norm += field[i] * field[i];
        //         filed_max = fmax(filed_max, field[i]);
        //         filed_min = fmin(filed_min, field[i]);
        //     }

        //     filed_norm = sqrt(filed_norm);
        //     printf("filed_norm = %1.14e , %s:%d\n", filed_norm, __FILE__, __LINE__);
        //     printf("filed_max  = %1.14e , %s:%d\n", filed_max, __FILE__, __LINE__);
        //     printf("filed_min  = %1.14e , %s:%d\n", filed_min, __FILE__, __LINE__);
        //     printf("n_zyx_private     = %ld , %s:%d\n", n_zyx_private, __FILE__, __LINE__);
        // }

        double ndarray_read_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] ndarray_create_from_file %g (seconds)\n",
                   rank,
                   ndarray_read_tock - ndarray_read_tick);
        }
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

    if (mpi_size > 1) {
        real_t* pfield;
        field_view(comm,
                   mesh.nnodes,
                   mesh.points[2],
                   nlocal,
                   nglobal,
                   stride,
                   origin,
                   delta,
                   field,
                   &pfield,
                   &nlocal[2],
                   &origin[2]);

        free(field);
        field = pfield;
    }

    real_t* g = calloc(mesh.nnodes, sizeof(real_t));

    {
        double resample_tick = MPI_Wtime();

        if (SFEM_INTERPOLATE) {
            interpolate_field(
                    // Mesh
                    mesh.n_owned_nodes,
                    mesh.points,
                    // discrete field
                    nlocal,
                    stride,
                    origin,
                    delta,
                    field,
                    // Output
                    g);
        } else {
            if (mpi_size == 1) {

                // { /// DEBUG ///
                //     printf("\nFunction: %s\n", __FUNCTION__);
                //     printf("\nMPI size = 1 DEBUG: %s:%d\n", __FILE__, __LINE__);
                //     printf("field (ptr): %p, %s:%d\n", (void *)field, __FILE__, __LINE__);
                //     printf("nlocal[0] = %ld, nlocal[1] = %ld, nlocal[2] = %ld, %s:%d\n",
                //            nlocal[0],
                //            nlocal[1],
                //            nlocal[2],
                //            __FILE__,
                //            __LINE__);

                //     double norm_data = 0.0;
                //     for (ptrdiff_t i = 0; i < nlocal[0] * nlocal[1] * nlocal[2]; i++) {
                //         norm_data += field[i] * field[i];
                //     }
                //     norm_data = sqrt(norm_data);
                //     printf("norm_data input = %g   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< %s:%d\n\n",
                //            norm_data,
                //            __FILE__,
                //            __LINE__);

                //     int indices[3] = {22, 55, 111};
                //     printf("field[%d] = %g, %s:%d\n", indices[0], field[indices[0]], __FILE__, __LINE__);
                //     printf("field[%d] = %g, %s:%d\n", indices[1], field[indices[1]], __FILE__, __LINE__);
                //     printf("field[%d] = %g, %s:%d\n", indices[2], field[indices[2]], __FILE__, __LINE__);

                // } /// end DEBUG ///

                resample_field(
                        // Mesh
                        mesh.element_type,
                        mesh.nelements,
                        mesh.n_owned_nodes,
                        mesh.elements,
                        mesh.points,
                        // discrete field
                        nlocal,
                        stride,
                        origin,
                        delta,
                        field,
                        // Output
                        g,
                        &info);

            } else {
                resample_field_local(
                        // Mesh
                        mesh.element_type,
                        mesh.nelements,
                        mesh.nnodes,
                        mesh.elements,
                        mesh.points,
                        // discrete field
                        nlocal,
                        stride,
                        origin,
                        delta,
                        field,
                        // Output
                        g,
                        &info);

                real_t* mass_vector = calloc(mesh.nnodes, sizeof(real_t));

                if (mesh.element_type == TET10) {
                    // FIXME (we should wrap mass vector assembly in sfem_resample_field.c)
                    tet10_assemble_dual_mass_vector(
                            mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mass_vector);
                } else {
                    enum ElemType st = shell_type(mesh.element_type);

                    if (st == INVALID) {
                        assemble_lumped_mass(mesh.element_type,
                                             mesh.nelements,
                                             mesh.nnodes,
                                             mesh.elements,
                                             mesh.points,
                                             mass_vector);

                    } else {
                        assemble_lumped_mass(st,
                                             mesh.nelements,
                                             mesh.nnodes,
                                             mesh.elements,
                                             mesh.points,
                                             mass_vector);
                    }
                }

                // exchange ghost nodes and add contribution
                if (mpi_size > 1) {
                    send_recv_t slave_to_master;
                    mesh_create_nodal_send_recv(&mesh, &slave_to_master);

                    ptrdiff_t count = mesh_exchange_master_buffer_count(&slave_to_master);
                    real_t* real_buffer = malloc(count * sizeof(real_t));

                    exchange_add(&mesh, &slave_to_master, mass_vector, real_buffer);
                    exchange_add(&mesh, &slave_to_master, g, real_buffer);
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

                free(mass_vector);
            }
        }

        double resample_tock = MPI_Wtime();

        // get MPI world size
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);

        int* elements_v = malloc(mpi_size * sizeof(int));

        MPI_Gather(&mesh.nelements, 1, MPI_INT, elements_v, 1, MPI_INT, 0, comm);

        int tot_nelements = 0;
        if (!rank) {
            for (int i = 0; i < mpi_size; i++) {
                tot_nelements += elements_v[i];
            }
        }

        free(elements_v);

        double* flops_v = NULL;
        flops_v = malloc(mpi_size * sizeof(double));

        const double flops = calculate_flops(mesh.nelements,                    //
                                             info.quad_nodes_cnt,               //
                                             (resample_tock - resample_tick));  //

        MPI_Gather(&flops, 1, MPI_DOUBLE, flops_v, 1, MPI_DOUBLE, 0, comm);

        double tot_flops = 0.0;
        if (!rank) {
            for (int i = 0; i < mpi_size; i++) {
                tot_flops += flops_v[i];
            }
        }

        free(flops_v);

        if (!rank) {
            const int nelements = mesh.nelements;
            const double elements_second =
                    (double)tot_nelements / (double)(resample_tock - resample_tick);

            printf("\n");
            printf("===========================================\n");

            printf("Rank: [%d]  Nr of elements  %d\n",                    //
                   rank,                                                  //
                   nelements * mpi_size);                                 //
            printf("Rank: [%d]  Resample        %g (seconds)\n",          //
                   rank,                                                  //
                   resample_tock - resample_tick);                        //
            printf("Rank: [%d]  Throughput      %e (elements/second)\n",  //
                   rank,                                                  //
                   elements_second);                                      //
            printf("Rank: [%d]  FLOPS           %e (FLOP/S)\n",           //
                   rank,                                                  //
                   tot_flops);                                            //
            printf("===========================================\n");

            printf("\n");
        }
    }

    // Write result to disk
    {
        if (rank == 0) {
            printf("-------------------------------------------\n");
            printf("Writing result to disk\n");
            printf("Output path: %s\n", output_path);
            printf("-------------------------------------------\n");
        }

        double io_tick = MPI_Wtime();

        /// DEBUG ///
        // double norm = 1.0;
        // double max_g = g[0];
        // double min_g = g[0];

        // for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
        //     norm += g[i] * g[i];
        //     if (g[i] > max_g) {
        //         max_g = g[i];
        //     }
        //     if (g[i] < min_g) {
        //         min_g = g[i];
        //     }
        // }

        // printf("\nNorm: %1.14e  <<<< TEST NORM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n", norm);
        // printf("Max: %1.14e  <<<< TEST MAX <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n", max_g);
        // printf("Min: %1.14e  <<<< TEST MIN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n", min_g);
        // printf("Mesh nnodes: %ld\n", mesh.nnodes);
        // printf("SFEM_INTERPOLATE: %d\n\n", SFEM_INTERPOLATE);
        /// end DEBUG ///

        mesh_write_nodal_field(&mesh, output_path, SFEM_MPI_REAL_T, g);

        double io_tock = MPI_Wtime();

        if (!rank) {
            printf("[%d] write %g (seconds)\n", rank, io_tock - io_tick);
        }
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    // Free resources
    {
        free(field);
        free(g);
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

    const int return_value = MPI_Finalize();
    RETURN_FROM_FUNCTION(return_value);
}
