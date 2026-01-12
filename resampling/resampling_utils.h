#ifndef RESAMPLING_UTILS_H
#define RESAMPLING_UTILS_H

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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the option argument from command line
 * @note This function is used to get the argument and its unique option from the command line
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param option Option string starting with "--"
 * @param arg Pointer to store the argument value
 * @param arg_size Pointer to store the argument size
 * @return 0 on success, -1 if option found but no argument, -2 if option not found
 */
int get_option_argument(int         argc,    //
                        char*       argv[],  //
                        const char* option,  //
                        char**      arg,     //
                        size_t*     arg_size);

/**
 * @brief Convert linear index to 3D grid coordinates
 *
 * @param index Linear index in the grid
 * @param nlocal Local grid dimensions [nx, ny, nz]
 * @param origin Grid origin [ox, oy, oz]
 * @param delta Grid spacing [dx, dy, dz]
 * @param coords Output 3D coordinates [i, j, k]
 */
void get_3d_coordinates(int              index,   //
                        const ptrdiff_t* nlocal,  //
                        const geom_t*    origin,  //
                        const geom_t*    delta,   //
                        int*             coords);

/**
 * @brief Handle the option result and print/exit accordingly
 *
 * @param result Result code from get_option_argument
 * @param option Option string
 * @param arg Argument value
 * @param arg_size Argument size
 * @param mandatory Whether the option is mandatory
 * @param print_result Whether to print the result
 */
void handle_option_result(const int result, const char* option, const char* arg, const size_t arg_size, const int mandatory,
                          const int print_result);

/**
 * @brief Print CPU performance metrics
 *
 * @param info Resampling field information
 * @param output_file Output file stream
 * @param mpi_rank MPI rank
 * @param mpi_size MPI size
 * @param seconds Execution time in seconds
 * @param file Source file name
 * @param line Source line number
 * @param function Function name
 * @param n_points_struct Number of structure points
 * @param quad_nodes_cnt Quadrature nodes count
 * @param mesh Mesh structure
 */
void print_performance_metrics_cpu(sfem_resample_field_info* info,             //
                                   FILE*                     output_file,      //
                                   const int                 mpi_rank,         //
                                   const int                 mpi_size,         //
                                   const double              seconds,          //
                                   const char*               file,             //
                                   const int                 line,             //
                                   const char*               function,         //
                                   const int                 n_points_struct,  //
                                   const int                 quad_nodes_cnt,   //
                                   const mesh_t*             mesh);

/**
 * @brief Handle printing of CPU performance metrics to stdout and optional file
 *
 * @param info Resampling field information
 * @param mpi_rank MPI rank
 * @param mpi_size MPI size
 * @param seconds Execution time in seconds
 * @param file Source file name
 * @param line Source line number
 * @param function Function name
 * @param n_points_struct Number of structure points
 * @param npq Number of quadrature points
 * @param mesh Mesh structure
 * @param print_to_file Whether to write to file
 */
void handle_print_performance_metrics_cpu(sfem_resample_field_info* info,             //
                                          int                       mpi_rank,         //
                                          int                       mpi_size,         //
                                          double                    seconds,          //
                                          const char*               file,             //
                                          int                       line,             //
                                          const char*               function,         //
                                          int                       n_points_struct,  //
                                          int                       npq,              //
                                          mesh_t*                   mesh,             //
                                          int                       print_to_file);

/**
 * @brief Calculate FLOPs for resampling operation
 *
 * @param nelements Number of elements
 * @param quad_nodes Number of quadrature nodes
 * @param time_sec Execution time in seconds
 * @return FLOP count
 */
double calculate_flops(const ptrdiff_t nelements, const ptrdiff_t quad_nodes, double time_sec);

/**
 * @brief Check if a string exists in command line arguments
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param target Target string to find
 * @param print_message Whether to print if found
 * @return 1 if found, 0 otherwise
 */
int check_string_in_args(const int argc, const char* argv[], const char* target, int print_message);

/**
 * @brief Print rank information for debugging
 *
 * @param mpi_rank MPI rank
 * @param mpi_size MPI size
 * @param max_field Maximum field value
 * @param min_field Minimum field value
 * @param max_field_index Index of maximum field value
 * @param min_field_index Index of minimum field value
 * @param n_zyx Total number of points
 * @param nlocal Local grid dimensions
 * @param origin Grid origin
 * @param delta Grid spacing
 * @param nglobal Global grid dimensions
 */
void print_rank_info(int              mpi_rank,         //
                     int              mpi_size,         //
                     real_t           max_field,        //
                     real_t           min_field,        //
                     int              max_field_index,  //
                     int              min_field_index,  //
                     ptrdiff_t        n_zyx,            //
                     const ptrdiff_t* nlocal,           //
                     const geom_t*    origin,           //
                     const geom_t*    delta,            //
                     const ptrdiff_t* nglobal);

// ===========================================================================================
// Test mesh functions
// ===========================================================================================

/**
 * @brief Paraboloid function: f(x,y,z) = x^2 + y^2 + z^2
 */
real_t mesh_fun_par(real_t x, real_t y, real_t z);

/**
 * @brief Linear function in x: f(x,y,z) = x
 */
real_t mesh_fun_lin_x(real_t x, real_t y, real_t z);

/**
 * @brief Step function in x: f(x,y,z) = x > 0.4 ? 1.0 : 0.0
 */
real_t mesh_fun_lin_hs_x(real_t x, real_t y, real_t z);

/**
 * @brief Step function in y: f(x,y,z) = y > 0.4 ? 1.0 : 0.0
 */
real_t mesh_fun_lin_hs_y(real_t x, real_t y, real_t z);

/**
 * @brief Step function in z: f(x,y,z) = z > 0.0 ? 1.0 : 0.0
 */
real_t mesh_fun_lin_hs_z(real_t x, real_t y, real_t z);

/**
 * @brief Trigonometric function: f(x,y,z) = 2.0 * (sin(6*x) + cos(6*y) + sin(6*z))
 */
real_t mesh_fun_trig(real_t x, real_t y, real_t z);

/**
 * @brief Positive trigonometric function: f(x,y,z) = 8.0 + mesh_fun_trig(x,y,z)
 */
real_t mesh_fun_trig_pos(real_t x, real_t y, real_t z);

/**
 * @brief Constant function: f(x,y,z) = 1.0
 */
real_t mesh_fun_ones(real_t x, real_t y, real_t z);

/**
 * @brief Zero function: f(x,y,z) = 0.0
 */
real_t mesh_fun_zeros(real_t x, real_t y, real_t z);

/**
 * @brief Linear step function with smooth transition
 */
real_t mesh_fun_linear_step(real_t x, real_t y, real_t z);

/**
 * @brief Chainsaw function in x direction
 */
real_t mesh_fun_chainsaw_x(real_t x, real_t y, real_t z);

/**
 * @brief Chainsaw function in xyz direction
 */
real_t mesh_fun_chainsaw_xyz(real_t x, real_t y, real_t z);

/**
 * @brief Print the name of the active mesh function
 *
 * @param mesh_fun_XYZ Function pointer to identify
 * @param mpi_rank MPI rank (only rank 0 prints)
 */
void print_mesh_function_name(const function_XYZ_t mesh_fun_XYZ, const int mpi_rank);

/**
 * @brief Create metadata YAML file for a structured grid
 *
 * @param nglobal Global grid dimensions [nx, ny, nz]
 * @param delta Grid spacing [dx, dy, dz]
 * @param origin Grid origin [ox, oy, oz]
 * @param folder Folder path for metadata file
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on failure
 */
int make_metadata(ptrdiff_t nglobal[3], float_t delta[3], float_t origin[3], const char* folder);

#ifdef __cplusplus
}
#endif

#endif  // RESAMPLING_UTILS_H
