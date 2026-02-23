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

#include "cell_list_bench.h"
#include "cell_tet2box.h"
#include "field_mpi_domain.h"
#include "mass.h"
#include "matrixio_array.h"
#include "matrixio_ndarray.h"
#include "mesh_aura.h"
#include "mesh_utils.h"
#include "quadratures_rule.h"
#include "read_mesh.h"
#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_mesh_read.h"
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

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_command_line_arguments
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_command_line_arguments(int argc, char* argv[], int mpi_rank) {
    if (mpi_rank == 0) {
        printf("argc: %d\n", argc);
        printf("argv: \n");
        for (int i = 0; i < argc; i++) {
            printf(" %s", argv[i]);
        }
        printf("\n");
    }  // END if (mpi_rank == 0)
    RETURN_FROM_FUNCTION();
}  // END Function: print_command_line_arguments

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// parse_element_type_from_args
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int parse_element_type_from_args(int argc, char* argv[], sfem_resample_field_info* info, int mpi_rank) {
    if (check_string_in_args(argc, (const char**)argv, "TET4", mpi_rank == 0)) {
        info->element_type = TET4;
    } else if (check_string_in_args(argc, (const char**)argv, "TET10", mpi_rank == 0)) {
        info->element_type = TET10;
    } else {
        fprintf(stderr, "Error: Invalid element type\n\n");
        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path> <element_type>\n",
                argv[0]);
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (element type)
    RETURN_FROM_FUNCTION(0);
}  // END Function: parse_element_type_from_args

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// get_output_base_directory
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void get_output_base_directory(char* out_base_directory, size_t buffer_size) {
    if (getenv("SFEM_OUT_BASE_DIRECTORY") != NULL) {
        snprintf(out_base_directory, buffer_size, "%s", getenv("SFEM_OUT_BASE_DIRECTORY"));
    } else {
        snprintf(out_base_directory, buffer_size, "/tmp/");
    }  // END if (getenv("SFEM_OUT_BASE_DIRECTORY"))
    RETURN_FROM_FUNCTION();
}  // END Function: get_output_base_directory

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// setup_grid_normalization
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void                                                //
setup_grid_normalization(ptrdiff_t    nnodes,       //
                         geom_t**     mesh_points,  //
                         ptrdiff_t*   nlocal,       //
                         geom_t*      origin,       //
                         geom_t*      delta,        //
                         real_t*      new_origin,   //
                         real_t*      new_side,     //
                         real_t*      new_delta,    //
                         const real_t margin,       //
                         int          enable_normalize,
                         int          mpi_rank) {  //

    PRINT_CURRENT_FUNCTION;

#if SFEM_LOG_LEVEL >= 5
    if (mpi_rank == 0) {
        printf("Setting up grid normalization, enable_normalize = %d, %s:%d\n", enable_normalize, __FILE__, __LINE__);
    }  // END if (mpi_rank == 0)
#endif  // SFEM_LOG_LEVEL >= 5

    if (enable_normalize) {
        // Normalize mesh bounding box
        normalize_mesh_BB(nnodes,          //
                          mesh_points,     //
                          nlocal[0],       //
                          1.0,             //
                          margin,          //
                          &new_origin[0],  //
                          &new_origin[1],  //
                          &new_origin[2],  //
                          &new_side[0],    //
                          &new_side[1],    //
                          &new_side[2]);   //

        delta[0] = 1.0;
        delta[1] = 1.0;
        delta[2] = 1.0;

        origin[0] = 0.0;
        origin[1] = 0.0;
        origin[2] = 0.0;

        new_delta[0] = new_side[0] / (real_t)(nlocal[0] - 1);
        new_delta[1] = new_side[1] / (real_t)(nlocal[1] - 1);
        new_delta[2] = new_side[2] / (real_t)(nlocal[2] - 1);

#if SFEM_LOG_LEVEL >= 5
        if (mpi_rank == 0) {
            printf("Normalized bounding box for refinement:\n new_origin = (%.5f %.5f %.5f),\n new_side = (%.5f %.5f "
                   "%.5f), \n%s:%d\n",
                   new_origin[0],
                   new_origin[1],
                   new_origin[2],
                   new_side[0],
                   new_side[1],
                   new_side[2],
                   __FILE__,
                   __LINE__);
            printf("  delta  = (%.5f %.5f %.5f)\n", delta[0], delta[1], delta[2]);
            printf("  origin = (%.5f %.5f %.5f)\n", origin[0], origin[1], origin[2]);
            printf("  nlocal = (%ld %ld %ld)\n", nlocal[0], nlocal[1], nlocal[2]);
        }  // END if (mpi_rank == 0)
#endif  // SFEM_LOG_LEVEL >= 5

    } else {
        // Use original grid parameters without normalization
        new_origin[0] = origin[0];
        new_origin[1] = origin[1];
        new_origin[2] = origin[2];

        new_side[0] = delta[0] * (real_t)(nlocal[0] - 1);
        new_side[1] = delta[1] * (real_t)(nlocal[1] - 1);
        new_side[2] = delta[2] * (real_t)(nlocal[2] - 1);

        new_delta[0] = delta[0];
        new_delta[1] = delta[1];
        new_delta[2] = delta[2];

    }  // END if (enable_normalize)

    RETURN_FROM_FUNCTION();
}  // END Function: setup_grid_normalization

void get_3d_coordinates(int              index,   //
                        const ptrdiff_t* nlocal,  //
                        const geom_t*    origin,  //
                        const geom_t*    delta,   //
                        int*             coords) {            //
    // Convert linear index to 3D grid indices
    const ptrdiff_t k = index / (nlocal[0] * nlocal[1]);
    const ptrdiff_t j = (index % (nlocal[0] * nlocal[1])) / nlocal[0];
    const ptrdiff_t i = index % nlocal[0];

    // Convert grid indices to spatial coordinates
    if (coords != NULL) {
        coords[0] = i;
        coords[1] = j;
        coords[2] = k;
    }
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

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// print_rank_info ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void                                               //
print_rank_info(int              mpi_rank,         //
                int              mpi_size,         //
                real_t           max_field,        //
                real_t           min_field,        //
                int              max_field_index,  //
                int              min_field_index,  //
                ptrdiff_t        n_zyx,            //
                const ptrdiff_t* nlocal,           //
                const geom_t*    origin,           //
                const geom_t*    delta,            //
                const ptrdiff_t* nglobal) {        //
    //

    PRINT_CURRENT_FUNCTION;

    MPI_Barrier(MPI_COMM_WORLD);

    int z_size_local = nlocal[2];
    int z_size       = 0;
    MPI_Reduce(&z_size_local, &z_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // field_mpi_domain_t field_mpi_domain;
    // field_mpi_domain.mpi_rank = mpi_rank;
    // field_mpi_domain.n_zyx    = n_zyx;
    // memcpy(field_mpi_domain.nlocal, nlocal, 3 * sizeof(ptrdiff_t));
    // memcpy(field_mpi_domain.origin, origin, 3 * sizeof(geom_t));

    for (int print_rank = 0; print_rank < mpi_size; ++print_rank) {
        if (mpi_rank == print_rank) {
            real_t origin_z = origin[2];
            real_t delta_z  = (real_t)(delta[2]) * (real_t)(nlocal[2] - 1);
            real_t max_z    = origin_z + delta_z;

            const int start_index_z   = origin_z / delta[2];
            const int end_index_z     = max_z / delta[2];
            const int delta_indices_z = end_index_z - start_index_z;
            const int size_z          = end_index_z - start_index_z + 1;

            // field_mpi_domain.start_indices[0] = 0;
            // field_mpi_domain.start_indices[1] = 0;
            // field_mpi_domain.start_indices[2] = start_index_z;

            const int max_index_0 = max_field_index % nlocal[0];
            const int max_index_1 = (max_field_index / nlocal[0]) % nlocal[1];
            const int max_index_2 = max_field_index / (nlocal[0] * nlocal[1]);

            printf("Rank %d: max_field = %1.14e, max index = %d, (%d, %d, %d)\n",
                   mpi_rank,
                   max_field,
                   max_field_index,
                   max_index_0,
                   max_index_1,
                   max_index_2);

            printf("Rank %d: min_field = %1.14e, min index = %d\n", mpi_rank, min_field, min_field_index);
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

    RETURN_FROM_FUNCTION();
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

real_t mesh_fun_zeros(real_t x, real_t y, real_t z) { return 0.0; }

real_t mesh_fun_linear_step(real_t x, real_t y, real_t z) {
    real_t dd = -0.2;
    if (x - dd < 0.0)
        return 0.0;
    else if ((x - dd) < 0.5)
        return 2.0 * (x - dd);
    else
        return 1.0;
}

real_t mesh_fun_chainsaw_x(real_t x, real_t y, real_t z) {
    real_t period = 0.2;
    real_t amp    = 1.0;
    return amp * (x / period - floor(0.5 + x / period));
}

real_t mesh_fun_chainsaw_xyz(real_t x, real_t y, real_t z) {
    const real_t period = 0.15;
    const real_t amp    = 1.0;
    const real_t xyz    = (copysign(1, x) * copysign(1, y) * copysign(1, z)) * sqrt(x * x + y * y + z * z);

    return amp * (xyz / period - floor(0.5 + xyz / period));
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_mesh_function_name
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_mesh_function_name(const function_XYZ_t mesh_fun_XYZ, const int mpi_rank) {
    if (mpi_rank != 0) {
        RETURN_FROM_FUNCTION();
    }  // END if (mpi_rank != 0)

    if (mesh_fun_XYZ == mesh_fun_par) {
        printf("Using: mesh_fun_par\n");
    } else if (mesh_fun_XYZ == mesh_fun_lin_x) {
        printf("Using: mesh_fun_lin_x\n");
    } else if (mesh_fun_XYZ == mesh_fun_lin_hs_x) {
        printf("Using: mesh_fun_lin_hs_x\n");
    } else if (mesh_fun_XYZ == mesh_fun_lin_hs_y) {
        printf("Using: mesh_fun_lin_hs_y\n");
    } else if (mesh_fun_XYZ == mesh_fun_lin_hs_z) {
        printf("Using: mesh_fun_lin_hs_z\n");
    } else if (mesh_fun_XYZ == mesh_fun_trig) {
        printf("Using: mesh_fun_trig\n");
    } else if (mesh_fun_XYZ == mesh_fun_trig_pos) {
        printf("Using: mesh_fun_trig_pos\n");
    } else if (mesh_fun_XYZ == mesh_fun_ones) {
        printf("Using: mesh_fun_ones\n");
    } else if (mesh_fun_XYZ == mesh_fun_zeros) {
        printf("Using: mesh_fun_zeros\n");
    } else if (mesh_fun_XYZ == mesh_fun_linear_step) {
        printf("Using: mesh_fun_linear_step\n");
    } else if (mesh_fun_XYZ == mesh_fun_chainsaw_x) {
        printf("Using: mesh_fun_chainsaw_x\n");
    } else if (mesh_fun_XYZ == mesh_fun_chainsaw_xyz) {
        printf("Using: mesh_fun_chainsaw_xyz\n");
    } else {
        printf("Using: UNKNOWN function\n");
    }  // END if-else chain

    RETURN_FROM_FUNCTION();
}  // END Function: print_mesh_function_name

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// make_metadata
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int make_metadata(ptrdiff_t nglobal[3], real_t delta[3], real_t origin[3], const char* folder) {
    char metadata_path[1000];
    snprintf(metadata_path, sizeof(metadata_path), "%s/metadata_sdf.float32.yml", folder);

    FILE* metadata_file = fopen(metadata_path, "w");
    if (metadata_file == NULL) {
        fprintf(stderr, "Error: Could not open metadata file for writing: %s\n", metadata_path);
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (metadata_file == NULL)

    char raw_path[1000];
    snprintf(raw_path, sizeof(raw_path), "%s/data.float32.raw", folder);

    fprintf(metadata_file, "nx: %ld\n", (long)nglobal[0]);
    fprintf(metadata_file, "ny: %ld\n", (long)nglobal[1]);
    fprintf(metadata_file, "nz: %ld\n", (long)nglobal[2]);
    fprintf(metadata_file, "block_size: 1\n");
    fprintf(metadata_file, "type: float\n");
    fprintf(metadata_file, "ox: %.17g\n", (double)origin[0]);
    fprintf(metadata_file, "oy: %.17g\n", (double)origin[1]);
    fprintf(metadata_file, "oz: %.17g\n", (double)origin[2]);
    fprintf(metadata_file, "dx: %.17g\n", (double)delta[0]);
    fprintf(metadata_file, "dy: %.17g\n", (double)delta[1]);
    fprintf(metadata_file, "dz: %.17g\n", (double)delta[2]);
    fprintf(metadata_file, "path: %s\n", raw_path);

    fclose(metadata_file);
    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END Function: make_metadata

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// benchmark_cdf_ratio_scan
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int benchmark_cdf_ratio_scan(const side_length_histograms_t* histograms,          //
                             const boxes_t*                  bounding_boxes_ptr,  //
                             const mesh_tet_geom_t*          geom,                //
                             const real_t                    min_grid_x,          //
                             const real_t                    max_grid_x,          //
                             const real_t                    min_grid_y,          //
                             const real_t                    max_grid_y,          //
                             const real_t                    min_grid_z,          //
                             const real_t                    max_grid_z,          //
                             const real_t                    min_ratio,           //
                             const real_t                    max_ratio,           //
                             const real_t                    step,                //
                             const int                       num_z,               //
                             const int                       num_queries,         //
                             const char*                     output_dir) {                            //
    PRINT_CURRENT_FUNCTION;

    if (histograms == NULL || bounding_boxes_ptr == NULL || geom == NULL) {
        fprintf(stderr, "ERROR: Invalid arguments to benchmark_cdf_ratio_scan\n");
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (histograms == NULL || bounding_boxes_ptr == NULL || geom == NULL)

    if (min_ratio < 0.0 || max_ratio > 1.0 || min_ratio >= max_ratio || step <= 0.0) {
        fprintf(stderr,
                "ERROR: Invalid ratio parameters (min=%g, max=%g, step=%g)\n",
                (double)min_ratio,
                (double)max_ratio,
                (double)step);
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (min_ratio < 0.0 || max_ratio > 1.0 || min_ratio >= max_ratio || step <= 0.0)

    printf("\n========================================\n");
    printf("Starting CDF Ratio Scan Benchmark\n");
    printf("========================================\n");
    printf("Ratio range: [%g, %g], step: %g\n", (double)min_ratio, (double)max_ratio, (double)step);
    printf("Benchmark parameters: num_z=%d, num_queries=%d\n", num_z, num_queries);
    printf("========================================\n\n");

    // Open CSV file for results if output_dir is provided
    FILE* csv_fp = NULL;
    if (output_dir != NULL) {
        char csv_path[4096];
        snprintf(csv_path, sizeof(csv_path), "%s/cdf_ratio_benchmark.csv", output_dir);
        csv_fp = fopen(csv_path, "w");
        if (csv_fp == NULL) {
            fprintf(stderr, "WARNING: Could not open file %s for writing\n", csv_path);
        } else {
            fprintf(csv_fp, "cdf_ratio,threshold_x,threshold_y,threshold_z,time_seconds\n");
            printf("Results will be written to: %s\n\n", csv_path);
        }  // END if (csv_fp == NULL)
    }  // END if (output_dir != NULL)

    int iteration = 0;
    for (real_t cdf_ratio = min_ratio; cdf_ratio <= max_ratio; cdf_ratio += step) {
        iteration++;

        printf("========================================\n");
        printf("Iteration %d: CDF ratio = %g\n", iteration, (double)cdf_ratio);
        printf("========================================\n");

        // Calculate thresholds for current CDF ratio
        side_length_cdf_thresholds_t thresholds = calculate_cdf_thresholds(histograms, cdf_ratio, cdf_ratio, cdf_ratio);

        printf("Thresholds: X=%g, Y=%g, Z=%g\n",
               (double)thresholds.threshold_x,
               (double)thresholds.threshold_y,
               (double)thresholds.threshold_z);

        // Build split map with current thresholds
        cell_list_split_3d_2d_map_t* split_map = NULL;

        double build_tick   = MPI_Wtime();
        int    build_result =                                                   //
                build_cell_list_3d_2d_split_map(&split_map,                     //
                                                thresholds.threshold_x,         //
                                                thresholds.threshold_y,         //
                                                bounding_boxes_ptr->min_x,      //
                                                bounding_boxes_ptr->min_y,      //
                                                bounding_boxes_ptr->min_z,      //
                                                bounding_boxes_ptr->max_x,      //
                                                bounding_boxes_ptr->max_y,      //
                                                bounding_boxes_ptr->max_z,      //
                                                bounding_boxes_ptr->num_boxes,  //
                                                min_grid_x,                     //
                                                max_grid_x,                     //
                                                min_grid_y,                     //
                                                max_grid_y,                     //
                                                min_grid_z,                     //
                                                max_grid_z);                    //
        double build_tock = MPI_Wtime();

        if (build_result != 0 || split_map == NULL) {
            fprintf(stderr, "ERROR: build_cell_list_3d_2d_split_map failed for cdf_ratio=%g\n", (double)cdf_ratio);
            if (split_map != NULL) {
                free_cell_list_split_3d_2d_map(split_map);
            }  // END if (split_map != NULL)
            continue;
        }  // END if (build_result != 0 || split_map == NULL)

        printf("Split map built in %g seconds (split_x=%g, split_y=%g)\n",
               build_tock - build_tick,
               (double)split_map->split_x,
               (double)split_map->split_y);

        // Run benchmark
        double bench_tick = MPI_Wtime();
        int    bench_result =
                query_tet_cell_list_split_3d_2d_given_xy_bench(split_map, bounding_boxes_ptr, geom, num_z, num_queries);
        double bench_tock = MPI_Wtime();
        double bench_time = bench_tock - bench_tick;

        if (bench_result != 0) {
            fprintf(stderr,
                    "WARNING: Benchmark returned non-zero status (%d) for cdf_ratio=%g\n",
                    bench_result,
                    (double)cdf_ratio);
        }  // END if (bench_result != 0)

        printf("Benchmark completed in %g seconds\n", bench_time);

        // Write results to CSV
        if (csv_fp != NULL) {
            fprintf(csv_fp,
                    "%.15e,%.15e,%.15e,%.15e,%.15e\n",
                    (double)cdf_ratio,
                    (double)thresholds.threshold_x,
                    (double)thresholds.threshold_y,
                    (double)thresholds.threshold_z,
                    bench_time);
            fflush(csv_fp);
        }  // END if (csv_fp != NULL)

        // Clean up split map
        free_cell_list_split_3d_2d_map(split_map);
        split_map = NULL;

        printf("\n");
    }  // END: for cdf_ratio

    if (csv_fp != NULL) {
        fclose(csv_fp);
        printf("Benchmark results written to: %s/cdf_ratio_benchmark.csv\n", output_dir);
    }  // END if (csv_fp != NULL)

    printf("========================================\n");
    printf("CDF Ratio Scan Benchmark Completed\n");
    printf("========================================\n\n");

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END Function: benchmark_cdf_ratio_scan