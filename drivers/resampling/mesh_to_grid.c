

#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "matrixio_ndarray.h"
#include "utils.h"

#include "mesh_aura.h"
#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_field.h"

#include "tet10_resample_field.h"

#include "mesh_utils.h"

#include "mass.h"

#define RED_TEXT "\x1b[31m"
#define GREEN_TEXT "\x1b[32m"
#define RESET_TEXT "\x1b[0m"

/**
 * @brief Get the option argument
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
                    char       *argv[],  //
                    const char *option,  //
                    char      **arg,     //
                    size_t     *arg_size) {  //

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
handle_option_result(const int result, const char *option, const char *arg, const size_t arg_size, const int mandatory,
                     const int print_result) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0 && print_result) {
        if (result == 0) {
            printf("Option: %s: %s\n", option, arg);
        } else if (result == -1) {
            fprintf(stderr, "\x1b[31mOption: %s found but no argument provided\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "\x1b[31mOption: %s not found\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        }
    } else if (result != 0 && mpi_rank == 0) {
        if (result == -1) {
            fprintf(stderr, "\x1b[31mOption: %s found but no argument provided\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "\x1b[31mOption: %s not found\n\x1b[0m", option);
            if (mandatory) {
                exit(EXIT_FAILURE);
            }
        }
    }
}

void  //
read_grid_options(int argc, char *argv[], ptrdiff_t *nglobal, geom_t *origin, geom_t *delta, const int print_result) {
    //
    int    result;
    size_t arg_size;

    {
        char *nx_ch = NULL;
        char *ny_ch = NULL;
        char *nz_ch = NULL;

        result = get_option_argument(argc, argv, "--nx", &nx_ch, &arg_size);
        handle_option_result(result, "--nx", nx_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--ny", &ny_ch, &arg_size);
        handle_option_result(result, "--ny", ny_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--nz", &nz_ch, &arg_size);
        handle_option_result(result, "--nz", nz_ch, arg_size, 1, print_result);

        nglobal[0] = atol(nx_ch);
        nglobal[1] = atol(ny_ch);
        nglobal[2] = atol(nz_ch);
    }

    {
        char *ox_ch = NULL;
        char *oy_ch = NULL;
        char *oz_ch = NULL;

        result = get_option_argument(argc, argv, "--ox", &ox_ch, &arg_size);
        handle_option_result(result, "--ox", ox_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--oy", &oy_ch, &arg_size);
        handle_option_result(result, "--oy", oy_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--oz", &oz_ch, &arg_size);
        handle_option_result(result, "--oz", oz_ch, arg_size, 1, print_result);

        origin[0] = atof(ox_ch);
        origin[1] = atof(oy_ch);
        origin[2] = atof(oz_ch);
    }

    {
        char *dx_ch = NULL;
        char *dy_ch = NULL;
        char *dz_ch = NULL;

        result = get_option_argument(argc, argv, "--dx", &dx_ch, &arg_size);
        handle_option_result(result, "--dx", dx_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--dy", &dy_ch, &arg_size);
        handle_option_result(result, "--dy", dy_ch, arg_size, 1, print_result);

        result = get_option_argument(argc, argv, "--dz", &dz_ch, &arg_size);
        handle_option_result(result, "--dz", dz_ch, arg_size, 1, print_result);

        delta[0] = atof(dx_ch);
        delta[1] = atof(dy_ch);
        delta[2] = atof(dz_ch);
    }
}

/**
 * @brief Allocate an ndarray
 *
 * @param comm
 * @param type
 * @param ndims
 * @param data_ptr
 * @param segment_size
 * @param nlocal
 * @param nglobal
 * @return int
 */
int                                                    //
ndarray_allocate(MPI_Comm               comm,          //
                 MPI_Datatype           type,          //
                 int                    ndims,         //
                 void                 **data_ptr,      //
                 int                    segment_size,  //
                 ptrdiff_t *const       nlocal,        //
                 const ptrdiff_t *const nglobal) {     //

    // nlocal is ignored and overridden for now
    int mpi_rank, mpi_size;

    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    int mpi_type_size;
    MPI_CATCH_ERROR(MPI_Type_size(type, &mpi_type_size));

    ptrdiff_t ntotal_user = 1;

    for (int d = 0; d < ndims; d++) {
        ntotal_user *= nglobal[d];
    }

    const MPI_Offset nbytes = ntotal_user * mpi_type_size;

    ptrdiff_t nlast = nglobal[ndims - 1];
    // ptrdiff_t nlast_uniform_split = nlast / mpi_size;
    ptrdiff_t nlast_local     = nlast / mpi_size;
    ptrdiff_t nlast_remainder = nlast % mpi_size;

    if (nlast_remainder > mpi_rank) {
        nlast_local += 1;
    }

    assert(mpi_size != 1 || nlast_local == nlast);

    for (int d = 0; d < ndims - 1; d++) {
        nlocal[d] = nglobal[d];
    }

    nlocal[ndims - 1] = nlast_local;

    ptrdiff_t stride = ntotal_user / nlast;
    long      offset = 0;
    long      nl     = nlast_local * stride;

    // void *data = malloc(nl * mpi_type_size);
    void *data = calloc(nl, mpi_type_size);  // it inits to 0

    int nrounds = nl / segment_size;
    nrounds += nrounds * ((ptrdiff_t)segment_size) < nl;

    MPI_CATCH_ERROR(MPI_Exscan(&nl, &offset, 1, MPI_LONG, MPI_SUM, comm));
    MPI_CATCH_ERROR(MPI_Exscan(MPI_IN_PLACE, &nrounds, 1, MPI_INT, MPI_MAX, comm));

    *data_ptr = data;
    return 0;
}

/**
 * @brief MAIN function
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * @return int
 */
int main(int argc, char *argv[]) {
    // Your code here

    PRINT_CURRENT_FUNCTION;

    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int      mpi_rank = -1;
    int      mpi_size = -1;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    sfem_resample_field_info info;
    info.element_type = TET4;

    printf("mesh_to_grid: mpi_rank = %d, mpi_size = %d\n", mpi_rank, mpi_size);
    // print argv
    if (mpi_rank == 0) {
        printf("argc: %d\n", argc);
        printf("argv: \n");
        for (int i = 0; i < argc; i++) {
            printf(" %s", argv[i]);
        }
        printf("\n");
    }

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    real_t   *field = NULL;
    ptrdiff_t nlocal[3];

    char  *mesh_folder = NULL;
    size_t arg_size;
    int    result = get_option_argument(argc, argv, "--mesh_folder", &mesh_folder, &arg_size);
    handle_option_result(result, "--mesh_folder", mesh_folder, arg_size, 1, mpi_rank == 0);

    mesh_t mesh;
    if (mesh_read(mpi_comm, mesh_folder, &mesh)) {
        fprintf(stderr, RED_TEXT "Error: mesh_read failed\n" RESET_TEXT);
        return EXIT_FAILURE;
    } else {
        if (mpi_rank == 0) printf(GREEN_TEXT "+ mesh_read succeeded\n" RESET_TEXT);
    }

    ptrdiff_t nglobal[3];
    geom_t    origin[3];
    geom_t    delta[3];

    read_grid_options(argc, argv, nglobal, origin, delta, mpi_rank == 0);

    char *output_path = NULL;
    result            = get_option_argument(argc, argv, "--output_path", &output_path, &arg_size);
    handle_option_result(result, "--output_path", output_path, arg_size, 1, mpi_rank == 0);

    ndarray_allocate(mpi_comm, MPI_FLOAT, 3, (void **)&field, 1, nlocal, nglobal);

    // for (int i = 0; i < 3; i++) {
    //     printf("nlocal[%d] = %ld\n", i, nlocal[i]);
    // }

    int recv_buff;
    int send_buff = (int)nlocal[2];

    MPI_Scan(&send_buff, &recv_buff, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == mpi_size - 1) {
        printf("recv_buff = %d\n", recv_buff);
        printf("send = %d\n", send_buff);
        printf("nglobal[2] = %ld\n", nglobal[2]);
    }

    MPI_Finalize();
    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}