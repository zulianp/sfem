#include <ctype.h>
#include <limits.h>
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

#include "mass.h"
#include "mesh_utils.h"
#include "quadratures_rule.h"
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

real_t mesh_fun_a(real_t x, real_t y, real_t z) { return x * x + y * y + z * z; }

real_t mesh_fun_b(real_t x, real_t y, real_t z) { return 2.0 * (sin(3.0 * x) + cos(3.0 * y) + sin(3.0 * z)); }

real_t mesh_fun_c(real_t x, real_t y, real_t z) { return 1.0; }

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

    ptrdiff_t   nglobal[3]  = {atol(argv[1]), atol(argv[2]), atol(argv[3])};
    geom_t      origin[3]   = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
    geom_t      delta[3]    = {atof(argv[7]), atof(argv[8]), atof(argv[9])};
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
    real_t*       field     = NULL;
    unsigned int* field_cnt = NULL;
    ptrdiff_t     nlocal[3];

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
                exit(EXIT_FAILURE);
            }

            // {  /// DEBUG ///
            // printf("temp (ptr): %p, %s:%d\n", (void *)temp, __FILE__, __LINE__);

            // double norm_temp = 0.0;
            // double max_temp = temp[0];
            // double min_temp = temp[0];

            n_zyx     = nlocal[0] * nlocal[1] * nlocal[2];
            field     = malloc(n_zyx * sizeof(real_t));
            field_cnt = calloc(n_zyx, sizeof(unsigned int));

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
            // printf("field == NULL: %s, %s:%d\n", field == NULL ? "true" : "false", __FILE__,
            // __LINE__); printf("size field = %ld MB , %s:%d\n", (n_zyx * sizeof(real_t) / 1024 /
            // 1024), __FILE__, __LINE__);

            // } /// end DEBUG ///
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
            int ret_resample_adjoint = 1;

            // DEBUG: fill g with ones
            // for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
            //     g[i] = 1.0;
            // }

            // TESTING: apply mesh_fun_b to g
            apply_fun_to_mesh(0, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mesh_fun_b, g);

            switch (info.element_type) {
                case TET10:
                    /* code */
                    break;

                case TET4:

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

                    ret_resample_adjoint =                          //
                            resample_field_adjoint_tet4(mpi_size,   //
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

                    BitArray bit_array = create_bit_array(nlocal[0] * nlocal[1] * nlocal[2]);

                    ret_resample_adjoint = in_out_field_mesh_tet4(mpi_size,    //
                                                                  mpi_rank,    //
                                                                  &mesh,       //
                                                                  nlocal,      //
                                                                  stride,      //
                                                                  origin,      //
                                                                  delta,       //
                                                                  &bit_array,  //
                                                                  &info);      //

                    unsigned int max_field_cnt = 0;
                    unsigned int max_in_out    = 0;

                    unsigned int min_non_zero_field_cnt = UINT_MAX;
                    unsigned int min_non_zero_in_out    = 0;

                    for (ptrdiff_t i = 0; i < n_zyx; i++) {
                        if (field_cnt[i] > max_field_cnt) {
                            max_field_cnt = field_cnt[i];
                            max_in_out    = get_bit(bit_array, i);
                        }

                        if (field_cnt[i] > 0 && field_cnt[i] < min_non_zero_field_cnt) {
                            min_non_zero_field_cnt = field_cnt[i];
                            min_non_zero_in_out    = get_bit(bit_array, i);
                        }
                    }

                    printf("\n");
                    printf("max_field_cnt = %u, in_out = %u\n", max_field_cnt, max_in_out);
                    printf("min_non_zero_field_cnt = %u, in_out = %u\n", min_non_zero_field_cnt, min_non_zero_in_out);
                    printf("\n");

                    ndarray_write(MPI_COMM_WORLD,
                                  "/home/sriva/git/sfem/workflows/resample/test_field.raw",
                                  MPI_FLOAT,
                                  3,
                                  field,
                                  nlocal,
                                  nglobal);

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

//             if (mpi_size == 1) {
//                 resample_field(
//                         // Mesh
//                         mesh.element_type,
//                         mesh.nelements,
//                         mesh.nnodes,
//                         mesh.elements,
//                         mesh.points,
//                         // discrete field
//                         nlocal,
//                         stride,
//                         origin,
//                         delta,
//                         field,
//                         // Output
//                         g,
//                         &info);

//                 // end if mpi_size == 1

//             } else {
//                 // mpi_size > 1

//                 if (info.element_type == TET10 && SFEM_TET10_CUDA == ON) {
// #if SFEM_TET10_CUDA == ON
//                     const int ret =                                           //
//                             hex8_to_tet10_resample_field_local_CUDA_wrapper(  //
//                                     mpi_size,                                 // MPI size
//                                     mpi_rank,                                 // MPI rank
//                                     &mesh,                                    // Mesh
//                                     assemble_dual_mass_vector_cuda,           // assemble dual mass vector in
//                                     the kernel nlocal,                                   // number of nodes in
//                                     each direction stride,                                   // stride of the
//                                     data origin,                                   // origin of the domain
//                                     delta,                                    // delta of the domain
//                                     field,                                    // filed
//                                     g);                                       // output
// #endif
//                 } else {  // Other cases and CPU
//                     resample_field_local(
//                             // Mesh
//                             mesh.element_type,
//                             mesh.nelements,
//                             mesh.nnodes,
//                             mesh.elements,
//                             mesh.points,
//                             // discrete field
//                             nlocal,
//                             stride,
//                             origin,
//                             delta,
//                             field,
//                             // Output
//                             g,
//                             &info);
//                 }  // END if info.element_type == TET10 && SFEM_TET10_CUDA == ON /////

//                 real_t* mass_vector = calloc(mesh.nnodes, sizeof(real_t));

//                 if (mesh.element_type == TET10 && assemble_dual_mass_vector_cuda == 0) {
//                     // FIXME (we should wrap mass vector assembly in sfem_resample_field.c)

//                     // In case of CUDA == ON this is calculated in the CUDA kernels calls
//                     // Directely in the hex8_to_tet10_resample_field_local_CUDA function

//                     tet10_assemble_dual_mass_vector(mesh.nelements,  //
//                                                     mesh.nnodes,
//                                                     mesh.elements,
//                                                     mesh.points,
//                                                     mass_vector);

//                     // end if mesh.element_type == TET10
//                 } else if (mesh.element_type == TET4) {  // mesh.element_type == TET4
//                     enum ElemType st = shell_type(mesh.element_type);

//                     if (st == INVALID) {
//                         assemble_lumped_mass(
//                                 mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points,
//                                 mass_vector);

//                     } else {
//                         assemble_lumped_mass(st, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points,
//                         mass_vector);
//                     }

//                     // end if mesh.element_type == TET4

//                 }  // end if mesh.element_type == TET10

//                 if (assemble_dual_mass_vector_cuda == 0) {
//                     //// TODO In CPU must be called.
//                     //// TODO In GPU should be calculated in the kernel calls in case of unified and Managed
//                     memory
//                     //// TODO In GPU is calculated here in case of host memory and more than one MPI rank (at
//                     the moment)

//                     // exchange ghost nodes and add contribution
//                     if (mpi_size > 1) {
//                         send_recv_t slave_to_master;
//                         mesh_create_nodal_send_recv(&mesh, &slave_to_master);

//                         ptrdiff_t count       = mesh_exchange_master_buffer_count(&slave_to_master);
//                         real_t*   real_buffer = malloc(count * sizeof(real_t));

//                         exchange_add(&mesh, &slave_to_master, mass_vector, real_buffer);
//                         exchange_add(&mesh, &slave_to_master, g, real_buffer);

//                         free(real_buffer);
//                         send_recv_destroy(&slave_to_master);
//                     }  // end if mpi_size > 1

//                     // divide by the mass vector
//                     for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
//                         if (mass_vector[i] == 0) {
//                             fprintf(stderr, "Found 0 mass at %ld, info (%ld, %ld)\n", i, mesh.n_owned_nodes,
//                             mesh.nnodes);
//                         }

//                         assert(mass_vector[i] != 0);
//                         g[i] /= mass_vector[i];
//                     }  // end for i < mesh.n_owned_nodes
//                 }      // end if SFEM_TET10_CUDA == OFF || SFEM_CUDA_MEMORY_MODEL == UNIFIED

//                 free(mass_vector);
//             }  // end if mpi_size > 1
