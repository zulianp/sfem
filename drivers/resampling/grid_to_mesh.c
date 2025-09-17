#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
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
#include "sfem_mesh_write.h"
#include "sfem_queue.h"
#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "tet10_resample_field.h"

#define RED_TEXT "\x1b[31m"
#define GREEN_TEXT "\x1b[32m"
#define RESET_TEXT "\x1b[0m"

/**
 * @brief Get the option argument
 * @note This function is used to get the argument and its unique option from the command line
 *
 * @param argc
 * @param argv
 * @param option
 * @param arg
 * @param arg_size
 * @return int
 */
int                                      //
get_option_argument(int         argc,    //
                    char*       argv[],  //
                    const char* option,  //
                    char**      arg,     //
                    size_t*     arg_size) {  //

    // check if option start with "--"
    if (strncmp(option, "--", 2) != 0) {
        fprintf(stderr, RED_TEXT "Error: option must start with '--'\n" RESET_TEXT);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], option, strlen(option)) == 0) {
            if (i + 1 < argc) {
                *arg      = argv[i + 1];
                *arg_size = strlen(argv[i + 1]);
                return 0;  // Success
            } else {
                *arg      = NULL;
                *arg_size = 0;
                return -1;  // Option found but no argument
            }
        }
    }
    *arg      = NULL;
    *arg_size = 0;
    return -2;  // Option not found
}

/**
 * @brief Handle the option result object
 *
 * @param result
 * @param option
 * @param arg
 * @param arg_size
 * @param mandatory
 */
void  //
handle_option_result(const int result, const char* option, const char* arg, const size_t arg_size, const int mandatory,
                     const int print_result) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        if (result == 0) {
            if (print_result) printf("Option: %s: %s\n", option, arg);
        } else if (result == -1) {
            if (print_result) fprintf(stderr, "\x1b[31mOption: %s found but no argument provided\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        } else {
            if (print_result) fprintf(stderr, "\x1b[31mOption: %s not found\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        }
    } else if (result != 0 && mpi_rank == 0) {
        if (result == -1) {
            if (print_result) fprintf(stderr, "\x1b[31mOption: %s found but no argument provided\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        } else {
            if (print_result) fprintf(stderr, "\x1b[31mOption: %s not found\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        }
    }
}

void                                                                      //
print_performance_metrics_cpu(sfem_resample_field_info* info,             //
                              FILE*                     output_file,      //
                              const int                 mpi_rank,         //
                              const int                 mpi_size,         //
                              const double              seconds,          //
                              const char*               file,             //
                              const int                 line,             //
                              const char*               function,         //
                              const int                 n_points_struct,  //
                              const int                 quad_nodes_cnt,   //
                              const mesh_t*             mesh) {                       //

    MPI_Comm comm = MPI_COMM_WORLD;

    int tot_npoints_struct = 0;
    MPI_Reduce(&n_points_struct, &tot_npoints_struct, 1, MPI_INT, MPI_SUM, 0, comm);

    int tot_nelements = 0;
    MPI_Reduce(&mesh->nelements, &tot_nelements, 1, MPI_INT, MPI_SUM, 0, comm);

    int tot_nnodes = 0;
    MPI_Reduce(&mesh->n_owned_nodes, &tot_nnodes, 1, MPI_INT, MPI_SUM, 0, comm);

    if (mpi_rank != 0) return;

    char tet_model[100];

    if (info->element_type == TET4) {
        snprintf(tet_model, 100, "TET4");
    } else if (info->element_type == TET10) {
        snprintf(tet_model, 100, "TET10");
    } else {
        snprintf(tet_model, 100, "UNKNOWN");
    }

    const double elements_per_second          = (double)(tot_nelements) / seconds;
    const double nodes_per_second             = (double)(tot_nnodes) / seconds;
    const double quadrature_points_per_second = (double)(tot_nelements * quad_nodes_cnt) / seconds;
    const double nodes_struc_second           = (double)(tot_npoints_struct) / seconds;

    const int real_t_bits = sizeof(real_t) * 8;

    fprintf(output_file, "============================================================================\n");
    fprintf(output_file, "CPU:    file: %s:%d \n", file, line);
    fprintf(output_file, "CPU:    MPI rank: %d\n", mpi_rank);
    fprintf(output_file, "CPU:    MPI size: %d\n", mpi_size);
    fprintf(output_file, "CPU:    %d-bit real_t\n", real_t_bits);
    fprintf(output_file, "CPU:    Element type:              %s\n", tet_model);
    fprintf(output_file, "CPU:    Clock                      %f seconds\n", seconds);
    fprintf(output_file, "CPU:    function:                  %s\n", function);
    fprintf(output_file, "CPU:    Number of elements:        %d.\n", tot_nelements);
    fprintf(output_file, "CPU:    Number of nodes:           %d.\n", tot_nnodes);
    fprintf(output_file, "CPU:    Number of points struct:   %d.\n", tot_npoints_struct);
    fprintf(output_file, "CPU:    Throughput for the kernel: %e elements/second\n", elements_per_second);
    fprintf(output_file, "CPU:    Throughput for the kernel: %e points_struct/second\n", nodes_struc_second);
    fprintf(output_file, "CPU:    Throughput for the kernel: %e nodes/second\n", nodes_per_second);
    fprintf(output_file, "CPU:    Throughput for the kernel: %e quadrature_points/second\n", quadrature_points_per_second);
    fprintf(output_file, "============================================================================\n\n");
    fprintf(output_file,
            "<BenchH> mpi_rank, mpi_size, real_t_bits, tot_nelements, tot_nnodes, npoint_struc, clock, elements_second, "
            "nodes_second, "
            "nodes_struc_second, quadrature_points_second\n");
    fprintf(output_file,
            "<BenchR> %d,   %d,  %d,   %d,   %d,   %d,   %g,   %g,   %g,   %g,  %g\n",  //
            mpi_rank,                                                                   //
            mpi_size,                                                                   //
            real_t_bits,                                                                //
            tot_nelements,                                                              //
            tot_nnodes,                                                                 //
            tot_npoints_struct,                                                         //
            seconds,                                                                    //
            elements_per_second,                                                        //
            nodes_per_second,                                                           //
            nodes_struc_second,                                                         //
            quadrature_points_per_second);                                              //
    fprintf(output_file, "============================================================================\n");
}

// Function to handle printing performance metrics
void                                                                             //
handle_print_performance_metrics_cpu(sfem_resample_field_info* info,             //
                                     int                       mpi_rank,         //
                                     int                       mpi_size,         //
                                     double                    seconds,          //
                                     const char*               file,             //
                                     int                       line,             //
                                     const char*               function,         //
                                     int                       n_points_struct,  //
                                     int                       npq,              //
                                     mesh_t*                   mesh,             //
                                     int                       print_to_file) {                        //

    FILE* output_file_print = NULL;

    char tet_model[100];

    if (info->element_type == TET4) {
        snprintf(tet_model, 100, "TET4");
    } else if (info->element_type == TET10) {
        snprintf(tet_model, 100, "TET10");
    } else {
        snprintf(tet_model, 100, "UNKNOWN");
    }

    if (print_to_file == 1 && mpi_rank == 0) {
        char      filename[1000];
        const int real_t_bits = sizeof(real_t) * 8;
        snprintf(filename, 1000, "resampling_cpu_%s_mpi_size_%d_%dbit.log", tet_model, mpi_size, real_t_bits);
        output_file_print = fopen(filename, "w");
    }

    // This function must be called by all ranks
    // Internally it will check if the rank is 0
    // All ranks are used to calculate the performance metrics
    print_performance_metrics_cpu(info,  //
                                  stdout,
                                  mpi_rank,
                                  mpi_size,
                                  seconds,
                                  file,
                                  line,
                                  function,
                                  n_points_struct,
                                  npq,
                                  mesh);

    if (print_to_file == 1) {
        print_performance_metrics_cpu(info,  //
                                      output_file_print,
                                      mpi_rank,
                                      mpi_size,
                                      seconds,
                                      file,
                                      line,
                                      function,
                                      n_points_struct,
                                      npq,
                                      mesh);

        if (output_file_print != NULL) fclose(output_file_print);
    }
}

double calculate_flops(const ptrdiff_t nelements, const ptrdiff_t quad_nodes, double time_sec) {
    const double flops = (nelements * (35 + 166 * quad_nodes)) / time_sec;
    return flops;
}

// Function prototype
// int check_string_in_args(int argc, char* argv[], const char* target);

// Function definition
int check_string_in_args(const int argc, const char* argv[], const char* target, int print_message) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], target) == 0) {
            if (print_message) printf("Found %s in argv[%d]\n", target, i);
            return 1;
        }
    }
    return 0;
}

// /**
//  * @brief Builds a field_mpi_domain_t structure.
//  *
//  * @param mpi_rank The MPI rank associated with this domain.
//  * @param n_zyx Total number of elements in the z, y, and x directions for this rank.
//  * @param nlocal Number of local elements in the z, y, and x directions.
//  * @param origin Local origin coordinates in the z, y, and x directions.
//  * @param delta Grid spacing in the z, y, and x directions.
//  * @return field_mpi_domain_t The populated structure.
//  */
// field_mpi_domain_t make_field_mpi_domain(const int mpi_rank, const ptrdiff_t n_zyx, const ptrdiff_t* nlocal,
//                                           const geom_t* origin, const geom_t* delta) {
//     field_mpi_domain_t domain;

//     domain.mpi_rank = mpi_rank;
//     domain.n_zyx    = n_zyx;

//     memcpy(domain.nlocal, nlocal, 3 * sizeof(ptrdiff_t));
//     memcpy(domain.origin, origin, 3 * sizeof(geom_t));

//     // Calculate start indices based on the logic from print_rank_info
//     domain.start_indices[0] = 0;  // Assuming x index starts at 0 for all ranks
//     domain.start_indices[1] = 0;  // Assuming y index starts at 0 for all ranks
//     // Ensure delta[2] is not zero to avoid division by zero
//     if (delta[2] != 0) {
//         // Use round to get the nearest integer index, handle potential floating point inaccuracies
//         domain.start_indices[2] = (int)round(origin[2] / delta[2]);
//     } else {
//         // Handle the case where delta[2] is zero, perhaps set to 0 or report an error
//         domain.start_indices[2] = 0;
//         // Optionally print an error or warning
//         // fprintf(stderr, "Warning: delta[2] is zero, cannot calculate start_index_z accurately.\n");
//     }

//     return domain;
// }

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// print_rank_info ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void                                         //
print_rank_info(int              mpi_rank,   //
                int              mpi_size,   //
                real_t           max_field,  //
                real_t           min_field,  //
                ptrdiff_t        n_zyx,      //
                const ptrdiff_t* nlocal,     //
                const geom_t*    origin,     //
                const geom_t*    delta,      //
                const ptrdiff_t* nglobal) {  //
                                             //
    MPI_Barrier(MPI_COMM_WORLD);

    int z_size_local = nlocal[2];
    int z_size       = 0;
    MPI_Reduce(&z_size_local, &z_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    field_mpi_domain_t field_mpi_domain;
    field_mpi_domain.mpi_rank = mpi_rank;
    field_mpi_domain.n_zyx    = n_zyx;
    memcpy(field_mpi_domain.nlocal, nlocal, 3 * sizeof(ptrdiff_t));
    memcpy(field_mpi_domain.origin, origin, 3 * sizeof(geom_t));

    for (int print_rank = 0; print_rank < mpi_size; ++print_rank) {
        if (mpi_rank == print_rank) {
            real_t origin_z = origin[2];
            real_t delta_z  = (real_t)(delta[2]) * (real_t)(nlocal[2] - 1);
            real_t max_z    = origin_z + delta_z;

            const int start_index_z   = origin_z / delta[2];
            const int end_index_z     = max_z / delta[2];
            const int delta_indices_z = end_index_z - start_index_z;
            const int size_z          = end_index_z - start_index_z + 1;

            field_mpi_domain.start_indices[0] = 0;
            field_mpi_domain.start_indices[1] = 0;
            field_mpi_domain.start_indices[2] = start_index_z;

            printf("Rank %d: max_field = %1.14e\n", mpi_rank, max_field);
            printf("Rank %d: min_field = %1.14e\n", mpi_rank, min_field);
            printf("Rank %d: n_zyx = %ld\n", mpi_rank, n_zyx);
            if (mpi_rank == 0) {
                printf("Rank %d: global_z_size = %d\n", mpi_rank, z_size);
            } else {
                // Other ranks don't have the reduced value
                printf("Rank %d: global_z_size = N/A (not root)\n", mpi_rank);
            }
            printf("Rank %d: nlocal = %ld %ld %ld\n", mpi_rank, nlocal[0], nlocal[1], nlocal[2]);
            printf("Rank %d: origin = %1.5e %1.5e %1.5e\n", mpi_rank, origin[0], origin[1], origin[2]);
            printf("Rank %d: delta = %1.5e %1.5e %1.5e\n", mpi_rank, delta[0], delta[1], delta[2]);
            printf("Rank %d: origin_z = %1.5e, delta_z = %1.5e, max_z = %1.5e\n", mpi_rank, origin_z, delta_z, max_z);
            printf("Rank %d: start_index_z = %d, end_index_z = %d, delta_indices_z = %d, size_z = %d\n",
                   mpi_rank,
                   start_index_z,
                   end_index_z,
                   delta_indices_z,
                   size_z);
            printf("Rank %d: nglobal = %ld %ld %ld\n\n", mpi_rank, nglobal[0], nglobal[1], nglobal[2]);

            fflush(stdout);  // Ensure output is flushed before the next rank prints
        }
        // Barrier ensures that only one rank prints at a time in order
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

/// Test functions for mesh field

real_t mesh_fun_par(real_t x, real_t y, real_t z) { return x * x + y * y + z * z; }

real_t mesh_fun_lin_x(real_t x, real_t y, real_t z) { return x; }

real_t mesh_fun_lin_hs_x(real_t x, real_t y, real_t z) { return x > 0.4 ? 1.0 : 0.0; }

real_t mesh_fun_lin_hs_y(real_t x, real_t y, real_t z) { return y > 0.4 ? 1.0 : 0.0; }

real_t mesh_fun_lin_hs_z(real_t x, real_t y, real_t z) { return z > 0.0 ? 1.0 : 0.0; }

real_t mesh_fun_trig(real_t x, real_t y, real_t z) { return 2.0 * (sin(6.0 * x) + cos(6.0 * y) + sin(6.0 * z)); }

real_t mesh_fun_trig_pos(real_t x, real_t y, real_t z) { return 8.0 + mesh_fun_trig(x, y, z); }

real_t mesh_fun_ones(real_t x, real_t y, real_t z) { return 1.0; }

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

    int SFEM_ADJOINT = 0;
    SFEM_READ_ENV(SFEM_ADJOINT, atoi);

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

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    // ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    real_t*       field        = NULL;
    unsigned int* field_cnt    = NULL;  // TESTING used to count the number of times a field is updated
    real_t*       field_alpha  = NULL;  // TESTING used to store the alpha field
    real_t*       filed_volume = NULL;  // TESTING used to store the volume field

    ptrdiff_t nlocal[3];

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    ptrdiff_t n_zyx = 0;
    {
        double ndarray_read_tick = MPI_Wtime();

        if (SFEM_READ_FP32) {
            float* temp = NULL;

            // int ndarray_create_from_file_segmented(
            //                MPI_Comm comm,
            //                const char *path,
            //                MPI_Datatype type,
            //                int ndims,
            //                void **data_ptr,
            //                int segment_size, // INT_MAX (ignored) in this case
            //                ptrdiff_t *const nlocal,
            //                const ptrdiff_t *const nglobal);

            if (ndarray_create_from_file(comm,           //
                                         data_path,      //
                                         MPI_FLOAT,      //
                                         3,              //
                                         (void**)&temp,  //
                                         nlocal,         //
                                         nglobal)) {     //

                fprintf(stderr, "Error: ndarray_create_from_file failed %s:%d\n", __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

            n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

            printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

            field = malloc(n_zyx * sizeof(real_t));

            // TODO: are data to analyze the results
            field_cnt    = calloc(n_zyx, sizeof(unsigned int));
            field_alpha  = calloc(n_zyx, sizeof(real_t));
            filed_volume = calloc(n_zyx, sizeof(real_t));

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
    int assemble_dual_mass_vector_cuda = 0;

    if (info.element_type == TET10 && SFEM_TET10_CUDA == ON) {
        if (SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY && mpi_size > 1) {
            assemble_dual_mass_vector_cuda = 0;
        } else {
            assemble_dual_mass_vector_cuda = 1;
        }
    }

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

    real_t* g = calloc(mesh.nnodes, sizeof(real_t));

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
        } else if (SFEM_ADJOINT == 0) {
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

        } else if (SFEM_ADJOINT == 1) {
            /// Adjoint case /////////////////////////////////////////////////

            int ret_resample_adjoint = 1;

            // DEBUG: fill g with ones
            // for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
            //     g[i] = 1.0;
            // }

            // TESTING: apply mesh_fun_b to g

            apply_fun_to_mesh(mesh.nnodes,                  //
                              (const geom_t**)mesh.points,  //
                              mesh_fun_ones,                //
                              g);                           //

            const real_t alpha_th_tet10 = 2.5;

            switch (info.element_type) {
                case TET10:

                    ret_resample_adjoint =                                //
                            resample_field_mesh_adjoint_tet10(mpi_size,   //
                                                              mpi_rank,   //
                                                              &mesh,      //
                                                              nlocal,     //
                                                              stride,     //
                                                              origin,     //
                                                              delta,      //
                                                              g,          //
                                                              field,      //
                                                              field_cnt,  //
                                                              &info);     //

                    real_t max_field_tet10 = -(__DBL_MAX__);
                    real_t min_field_tet10 = (__DBL_MAX__);

                    normalize_field_and_find_min_max(field, n_zyx, delta, &min_field_tet10, &max_field_tet10);

                    print_rank_info(mpi_rank, mpi_size, max_field_tet10, min_field_tet10, n_zyx, nlocal, origin, delta, nglobal);

                    ndarray_write(MPI_COMM_WORLD,
                                  "/home/sriva/git/sfem/workflows/resample/test_field_t10.raw",
                                  MPI_FLOAT,
                                  3,
                                  field,
                                  nlocal,
                                  nglobal);

                    break;

                case TET4:
                    ///////////////////////////////////// Case TEt4 /////////////////////////////////////

                    // ret_resample_adjoint =                                //
                    //         resample_field_TEST_adjoint_tet4(mpi_size,    //
                    //                                          mpi_rank,    //
                    //                                          &mesh,       //
                    //                                          nlocal,      //
                    //                                          stride,      //
                    //                                          origin,      //
                    //                                          delta,       //
                    //                                          field,       //
                    //                                          test_field,  //
                    //                                          g,           //
                    //                                          &info);      //

                    info.alpha_th            = 1.5;
                    info.adjoint_refine_type = ADJOINT_REFINE_ITERATIVE;
                    // info.adjoint_refine_type = ADJOINT_REFINE_ITERATIVE_QUEUE;
                    info.adjoint_refine_type = ADJOINT_BASE;
                    info.adjoint_refine_type = ADJOINT_REFINE_HYTEG_REFINEMENT;

                    mini_tet_parameters_t mini_tet_parameters;
                    {
                        mini_tet_parameters.alpha_min_threshold = 1.0;
                        mini_tet_parameters.alpha_max_threshold = 8.0;
                        mini_tet_parameters.min_refinement_L    = 1;
                        mini_tet_parameters.max_refinement_L    = 22;

                        const char* max_refinement_L_str = getenv("MAX_REFINEMENT_L");
                        if (max_refinement_L_str) {
                            mini_tet_parameters.max_refinement_L = atoi(max_refinement_L_str);
                        }
                    }

#if SFEM_LOG_LEVEL >= 5
                    printf("info.adjoint_refine_type = %d, %s:%d\n", info.adjoint_refine_type, __FILE__, __LINE__);
                    // print as a string
                    if (info.adjoint_refine_type == ADJOINT_REFINE_ITERATIVE) {
                        printf("info.adjoint_refine_type = ADJOINT_REFINE_ITERATIVE\n");
                    } else if (info.adjoint_refine_type == ADJOINT_REFINE_ITERATIVE_QUEUE) {
                        printf("info.adjoint_refine_type = ADJOINT_REFINE_ITERATIVE_QUEUE\n");
                    } else if (info.adjoint_refine_type == ADJOINT_BASE) {
                        printf("info.adjoint_refine_type =  ADJOINT_BASE\n");
                    } else if (info.adjoint_refine_type == ADJOINT_REFINE_HYTEG_REFINEMENT) {
                        printf("info.adjoint_refine_type = ADJOINT_REFINE_HYTEG_REFINEMENT\n");
                    } else {
                        printf("info.adjoint_refine_type = UNKNOWN\n");
                    }
#endif

                    ret_resample_adjoint =                                     //
                            resample_field_adjoint_tet4(mpi_size,              //
                                                        mpi_rank,              //
                                                        &mesh,                 //
                                                        nlocal,                //
                                                        stride,                //
                                                        origin,                //
                                                        delta,                 //
                                                        g,                     //
                                                        field,                 //
                                                        field_cnt,             //
                                                        field_alpha,           //
                                                        filed_volume,          //
                                                        &info,                 //
                                                        mini_tet_parameters);  //

                    // BitArray bit_array_in_out = create_bit_array(nlocal[0] * nlocal[1] * nlocal[2]);

                    // ret_resample_adjoint = in_out_field_mesh_tet4(mpi_size,           //
                    //                                               mpi_rank,           //
                    //                                               &mesh,              //
                    //                                               nlocal,             //
                    //                                               stride,             //
                    //                                               origin,             //
                    //                                               delta,              //
                    //                                               &bit_array_in_out,  //
                    //                                               &info);             //

                    unsigned int max_field_cnt = 0;
                    unsigned int max_in_out    = 0;

                    // unsigned int min_non_zero_field_cnt = UINT_MAX;
                    // unsigned int min_non_zero_in_out    = 0;

                    MPI_Barrier(MPI_COMM_WORLD);

                    real_t min_field_tet4 = 0.0;
                    real_t max_field_tet4 = 0.0;

                    normalize_field_and_find_min_max(field,             //
                                                     n_zyx,             //
                                                     delta,             //
                                                     &min_field_tet4,   //
                                                     &max_field_tet4);  //

                    print_rank_info(mpi_rank,        //
                                    mpi_size,        //
                                    max_field_tet4,  //
                                    min_field_tet4,  //
                                    n_zyx,           //
                                    nlocal,          //
                                    origin,          //
                                    delta,           //
                                    nglobal);        //

                    // printf("max_field = %1.14e\n", max_field);
                    // printf("min_field = %1.14e\n", min_field);
                    // printf("\n");

                    // // TEST: write the in out field and the field_cnt
                    // real_t* bit_array_in_out_real = to_real_array(bit_array_in_out);

                    // TEST: write the in out field and the field_cnt
                    real_t* field_cnt_real = (real_t*)malloc(n_zyx * sizeof(real_t));
                    for (ptrdiff_t i = 0; i < n_zyx; i++) {
                        field_cnt_real[i] = (real_t)(field_cnt[i]);
                    }

                    char out_filename_raw[1000];

                    const char* env_out_filename = getenv("OUT_FILENAME_RAW");
                    if (env_out_filename && strlen(env_out_filename) > 0) {
                        snprintf(out_filename_raw, 1000, "%s", env_out_filename);
                    } else {
                        snprintf(out_filename_raw, 1000, "/home/sriva/git/sfem/workflows/resample/test_field.raw", mpi_rank);
                    }

                    ndarray_write(MPI_COMM_WORLD,
                                  out_filename_raw,
                                  ((SFEM_REAL_T_IS_FLOAT32) ? MPI_FLOAT : MPI_DOUBLE),
                                  3,
                                  field,
                                  nlocal,
                                  nglobal);

                    // // TEST: write the in out field and the field_cnt
                    // ndarray_write(MPI_COMM_WORLD,
                    //               "/home/sriva/git/sfem/workflows/resample/bit_array.raw",
                    //               MPI_FLOAT,
                    //               3,
                    //               bit_array_in_out_real,
                    //               nlocal,
                    //               nglobal);

                    // TEST: write the in out field and the field_cnt
                    // ndarray_write(MPI_COMM_WORLD,
                    //               "/home/sriva/git/sfem/workflows/resample/field_cnt.raw",
                    //               MPI_FLOAT,
                    //               3,
                    //               field_cnt_real,
                    //               nlocal,
                    //               nglobal);

                    // ndarray_write(MPI_COMM_WORLD,
                    //               "/home/sriva/git/sfem/workflows/resample/test_field_alpha.raw",
                    //               MPI_FLOAT,
                    //               3,
                    //               field_alpha,
                    //               nlocal,
                    //               nglobal);

                    // ndarray_write(MPI_COMM_WORLD,
                    //               "/home/sriva/git/sfem/workflows/resample/test_field_volume.raw",
                    //               MPI_FLOAT,
                    //               3,
                    //               filed_volume,
                    //               nlocal,
                    //               nglobal);

                    // // TEST: write the in out field and the field_cnt
                    // free(bit_array_in_out_real);
                    // bit_array_in_out_real = NULL;

                    free(field_cnt_real);
                    field_cnt_real = NULL;

                    break;

                default:
                    fprintf(stderr, "Error: Invalid element type: %s:%d\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                    break;
            }

            if (ret_resample_adjoint) {
                fprintf(stderr, "Error: resample_field_mesh_adjoint failed %s:%d\n", __FILE__, __LINE__);
                return EXIT_FAILURE;
            }
        }

        // end if SFEM_INTERPOLATE
        /////////////////////////////////
        // END resample_field_mesh
        /////////////////////////////////

        MPI_Barrier(MPI_COMM_WORLD);
        double resample_tock = MPI_Wtime();

        // get MPI world size
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);

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
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes    = mesh.nnodes;

    // Free resources
    {
        free(field);
        free(g);
        mesh_destroy(&mesh);
    }

    if (field_cnt != NULL) {
        free(field_cnt);
        field_cnt = NULL;
    }

    if (field_alpha) {
        free(field_alpha);
        field_alpha = NULL;
    }

    if (filed_volume) {
        free(filed_volume);
        filed_volume = NULL;
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
}
