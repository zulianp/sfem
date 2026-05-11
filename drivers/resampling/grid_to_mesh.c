#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "field_mpi_domain.h"
#include "mass.h"
#include "matrixio_array.h"
#include "matrixio_ndarray.h"
#include "mesh_aura.h"
#include "mesh_utils.h"
#include "quadratures_rule.h"
#include "read_mesh.h"
#include "resample_adjoint_main.h"
#include "resampling_utils.h"
#include "sfem_mesh_read.h"
#include "sfem_mesh_write.h"
#include "sfem_queue.h"
#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "tet10_resample_field.h"

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// main ////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    // printf("========================================\n");
    // printf("Starting grid_to_mesh\n");
    // printf("========================================\n\n");
    PRINT_CURRENT_FUNCTION;

    int SFEM_ADJOINT = 0;
    SFEM_READ_ENV(SFEM_ADJOINT, atoi);

    int SFEM_TEST_CCELL = 0;
    SFEM_READ_ENV(SFEM_TEST_CCELL, atoi);

    int SFEM_RASTER_TRI_MESH = 0;
    SFEM_READ_ENV(SFEM_RASTER_TRI_MESH, atoi);

    if (SFEM_ADJOINT == 1) {
        return main_adjoint(argc, argv);
    }

    if (SFEM_RASTER_TRI_MESH == 1) {
        return main_raster_from_surface_mesh(argc, argv);
    }

    if (SFEM_TEST_CCELL == 1) {
        return main_test_ccell(argc, argv);
    }

    printf("========================================\n");
    printf("Starting sfem_resample_field_adjoint_hex_quad test\n");
    printf("========================================\n\n");

    printf("<sizeof_real_t> %zu\n", sizeof(real_t));

    // return test_field_mpi_domain(argc, argv);

    // sfem_queue_test();
    // return EXIT_SUCCESS;

    sfem_resample_field_info info;

    info.element_type = TET10;

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // print argv
    if (mpi_rank == 0) {
        printf("argc: %d\n", argc);
        printf("argv: \n");
        for (int i = 0; i < argc; i++) {
            printf(" %s", argv[i]);
        }
        printf("\n");
    }

    if (argc < 13 && argc > 14) {
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
    geom_t    origin[3]  = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
    geom_t    delta[3]   = {atof(argv[7]), atof(argv[8]), atof(argv[9])};

    const char* data_path   = argv[10];
    const char* folder      = argv[11];
    const char* output_path = argv[12];

    if (check_string_in_args(argc, (const char**)argv, "TET4", mpi_rank == 0)) {
        info.element_type = TET4;
    } else if (check_string_in_args(argc, (const char**)argv, "TET10", mpi_rank == 0)) {
        info.element_type = TET10;
    } else {
        fprintf(stderr, "Error: Invalid element type\n\n");
        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path> <element_type>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    info.use_accelerator = SFEM_ACCELERATOR_TYPE_CPU;

#ifdef SFEM_ENABLE_CUDA

    if (check_string_in_args(argc, (const char**)argv, "CUDA", mpi_rank == 0)) {
        info.use_accelerator = SFEM_ACCELERATOR_TYPE_CUDA;
        if (mpi_rank == 0) printf("info.use_accelerator = 1\n");

    } else if (check_string_in_args(argc, (const char**)argv, "CPU", mpi_rank == 0)) {
        info.use_accelerator = SFEM_ACCELERATOR_TYPE_CPU;
        if (mpi_rank == 0) printf("info.use_accelerator = 0\n");

    } else {
        fprintf(stderr, "Error: Invalid accelerator type\n\n");
        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path> <element_type> <accelerator_type>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

#endif

    if (info.element_type == TET4) {
        if (mpi_rank == 0) printf("info.element_type = TET4,    %s:%d\n", __FILE__, __LINE__);
    } else if (info.element_type == TET10) {
        if (mpi_rank == 0) printf("info.element_type = TET10,   %s:%d\n", __FILE__, __LINE__);
    } else {
        if (mpi_rank == 0) printf("info.element_type = UNKNOWN, %s:%d\n", __FILE__, __LINE__);
    }

    info.quad_nodes_cnt = TET_QUAD_NQP;

    printf("Reading mesh from folder: %s\n", folder);

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        fprintf(stderr, "Error: mesh_read failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    // ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    real_t* field = NULL;

    ptrdiff_t nlocal[3];

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    ptrdiff_t n_zyx = 0;
    {
        double ndarray_read_tick = MPI_Wtime();

        if (SFEM_READ_FP32) {
            float* temp = NULL;

            if (ndarray_create_from_file(comm,           //
                                         data_path,      //
                                         MPI_FLOAT,      //
                                         3,              //
                                         (void**)&temp,  //
                                         nlocal,         //
                                         nglobal)) {     //

                fprintf(stderr, "Error: ndarray_create_from_file failed, data_path: %s  %s:%d\n", data_path, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

            n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

            printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

            field = malloc(n_zyx * sizeof(real_t));

            for (ptrdiff_t i = 0; i < n_zyx; i++) {
                field[i] = (real_t)(temp[i]);
            }

            free(temp);

        } else {
            if (ndarray_create_from_file(comm, data_path, SFEM_MPI_REAL_T, 3, (void**)&field, nlocal, nglobal)) {
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

        if (mpi_rank == 0) {
            printf("[%d] ndarray_create_from_file %g (seconds)\n", mpi_rank, ndarray_read_tock - ndarray_read_tick);
        }
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

    // used to perform the assembly of the dual mass vector in the kernel
    // for TET10 elements
    // 0: do not assemble the dual mass vector in the kernel if the memory model is host and mpi_size > 1
    // 1: assemble the dual mass vector in the kernel
    // int assemble_dual_mass_vector_cuda = 0;

    // if (info.element_type == TET10 && SFEM_TET10_CUDA == ON) {
    //     if (SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY && mpi_size > 1) {
    //         assemble_dual_mass_vector_cuda = 0;
    //     } else {
    //         assemble_dual_mass_vector_cuda = 1;
    //     }
    // }

    // real_t* test_field = calloc(nlocal[0] * nlocal[1] * nlocal[2], sizeof(real_t));  /// DEBUG

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

        n_zyx = nlocal[0] * nlocal[1] * nlocal[2];  // Update n_zyx after field_view
        printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

        free(field);
        field = pfield;
    }

    // const int multi_field = 3;

    real_t* g = calloc(mesh.nnodes, sizeof(real_t));
    // real_t* multi_g = calloc(mesh.nnodes * multi_field, sizeof(real_t));

    {  // begin resample_field_mesh
        /////////////////////////////////
        MPI_Barrier(MPI_COMM_WORLD);
        double resample_tick = MPI_Wtime();

        if (SFEM_INTERPOLATE) {
            printf("SFEM_INTERPOLATE = 1, %s:%d\n", __FILE__, __LINE__);
            interpolate_field(mesh.n_owned_nodes,  // Mesh:
                              mesh.points,         // Mesh:
                              nlocal,              // discrete field
                              stride,              //
                              origin,              //
                              delta,               //
                              field,               //
                              g);                  // Output
        } else {
            int ret_resample = 1;

            switch (info.element_type) {                         //
                case TET10:                                      // TET10 case
                    ret_resample =                               //
                            resample_field_mesh_tet10(mpi_size,  //
                                                      mpi_rank,  //
                                                      &mesh,     //
                                                      nlocal,    //
                                                      stride,    //
                                                      origin,    //
                                                      delta,     //
                                                      field,     //
                                                      g,         //
                                                      &info);    //
                    break;                                       //

                case TET4:                                      // TET4 case
                    ret_resample =                              //
                            resample_field_mesh_tet4(mpi_size,  //
                                                     mpi_rank,  //
                                                     &mesh,     //
                                                     nlocal,    //
                                                     stride,    //
                                                     origin,    //
                                                     delta,     //
                                                     field,     //
                                                     g,         //
                                                     &info);    //

                    break;
                default:
                    fprintf(stderr, "Error: Invalid element type: %s:%d\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                    break;
            }

            if (ret_resample) {
                fprintf(stderr, "Error: resample_field_mesh failed %s:%d\n", __FILE__, __LINE__);
                return EXIT_FAILURE;
            }

        }

        // end if SFEM_INTERPOLATE
        /////////////////////////////////
        // END resample_field_mesh
        /////////////////////////////////

        MPI_Barrier(MPI_COMM_WORLD);
        double resample_tock = MPI_Wtime();

        // int* elements_v = malloc(mpi_size * sizeof(int));

        // MPI_Gather(&mesh.nelements, 1, MPI_INT, elements_v, 1, MPI_INT, 0, comm);

        // int tot_nelements = 0;
        // if (mpi_rank == 0) {
        //     for (int i = 0; i < mpi_size; i++) {
        //         tot_nelements += elements_v[i];
        //     }
        // }
        // free(elements_v);

        int tot_nelements = 0;
        MPI_Reduce(&mesh.nelements, &tot_nelements, 1, MPI_INT, MPI_SUM, 0, comm);

        int tot_nnodes = 0;
        MPI_Reduce(&mesh.n_owned_nodes, &tot_nnodes, 1, MPI_INT, MPI_SUM, 0, comm);

        double* flops_v = NULL;
        flops_v         = malloc(mpi_size * sizeof(double));

        const double flops = calculate_flops(mesh.nelements,                    //
                                             info.quad_nodes_cnt,               //
                                             (resample_tock - resample_tick));  //

        MPI_Gather(&flops, 1, MPI_DOUBLE, flops_v, 1, MPI_DOUBLE, 0, comm);

        double tot_flops = 0.0;
        if (mpi_rank == 0) {
            for (int i = 0; i < mpi_size; i++) {
                tot_flops += flops_v[i];
            }
        }

        free(flops_v);

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        handle_print_performance_metrics_cpu(&info,                          //
                                             mpi_rank,                       //
                                             mpi_size,                       //
                                             resample_tock - resample_tick,  //
                                             __FILE__,                       //
                                             __LINE__,                       //
                                             "grid_to_mesh",                 //
                                             mesh.nnodes,                    //
                                             info.quad_nodes_cnt,            //
                                             &mesh,                          //
                                             1);                             //

    }  // end resample_field_mesh

    // Write result to disk
    {
        if (mpi_rank == 0) {
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

        if (!mpi_rank) {
            printf("[%d] write %g (seconds)\n", mpi_rank, io_tock - io_tick);
        }
    }  // end write result to disk

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes    = mesh.nnodes;

    // Free resources
    {
        free(field);
        free(g);
        mesh_destroy(&mesh);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double tock = MPI_Wtime();

    if (!mpi_rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #grid (%ld x %ld x %ld)\n",
               (long)nelements,
               (long)nnodes,
               nglobal[0],
               nglobal[1],
               nglobal[2]);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    // free(test_field);   /// DEBUG
    // test_field = NULL;  /// DEBUG

    const int return_value = MPI_Finalize();
    RETURN_FROM_FUNCTION(return_value);
}  // END: main
