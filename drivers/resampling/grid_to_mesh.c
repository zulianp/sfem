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

    const function_XYZ_t mesh_fun_XYZ = mesh_fun_ones;

    char out_base_directory[2048];

    if (getenv("SFEM_OUT_BASE_DIRECTORY") != NULL) {
        snprintf(out_base_directory, 2048, "%s", getenv("SFEM_OUT_BASE_DIRECTORY"));
    } else {
        snprintf(out_base_directory, 2048, "/tmp/");
    }

#if SFEM_LOG_LEVEL >= 5
    printf("Using SFEM_OUT_BASE_DIRECTORY: %s\n", out_base_directory);
#endif  // SFEM_LOG_LEVEL

#if SFEM_LOG_LEVEL >= 5
    print_mesh_function_name(mesh_fun_XYZ, mpi_rank);
#endif  // SFEM_LOG_LEVEL

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

    printf("Reading mesh from folder: %s\n", folder);

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        fprintf(stderr, "Error: mesh_read failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    // ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    real_t*       field         = NULL;
    unsigned int* field_cnt     = NULL;  // TESTING used to count the number of times a field is updated
    real_t*       field_alpha   = NULL;  // TESTING used to store the alpha field
    real_t*       field_volume  = NULL;  // TESTING used to store the volume field
    real_t*       field_fun_XYZ = NULL;  // TESTING used to store the analytical function

    ptrdiff_t nlocal[3];

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    ptrdiff_t n_zyx = 0;
    {
        double ndarray_read_tick = MPI_Wtime();

        if (SFEM_READ_FP32 && SFEM_ADJOINT == 0) {
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

                fprintf(stderr, "Error: ndarray_create_from_file failed, data_path: %s  %s:%d\n", data_path, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

            n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

            printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

            field = malloc(n_zyx * sizeof(real_t));

            // TODO: are data to analyze the results
            field_cnt    = calloc(n_zyx, sizeof(unsigned int));
            field_alpha  = calloc(n_zyx, sizeof(real_t));
            field_volume = calloc(n_zyx, sizeof(real_t));
            // field_fun_XYZ = calloc(n_zyx, sizeof(real_t));

            for (ptrdiff_t i = 0; i < n_zyx; i++) {
                field[i] = (real_t)(temp[i]);
            }

            free(temp);

        } else if (SFEM_ADJOINT == 0) {
            if (ndarray_create_from_file(comm, data_path, SFEM_MPI_REAL_T, 3, (void**)&field, nlocal, nglobal)) {
                return EXIT_FAILURE;
            }
        } else if (SFEM_ADJOINT == 1) {
            // In adjoint mode the field is not read from file
            // It is generated using the mesh_fun_XYZ function
            nlocal[0] = nglobal[0];
            nlocal[1] = nglobal[1];
            nlocal[2] = nglobal[2];

            n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

            printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

            field = calloc(n_zyx, sizeof(real_t));
            // field_fun_XYZ = calloc(n_zyx, sizeof(real_t));
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

    const int multi_field = 3;

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
                              mesh_fun_XYZ,                 //
                              g);                           //

            // {
            //     double* g_dbl = calloc(mesh.n_owned_nodes, sizeof(double));

            //     mesh_read_nodal_field(&mesh,                                                                               //
            //                           "/home/simone/git/sfem_d/sfem/workflows/resample/bone_raw/point_data/bone_edf.raw",  //
            //                           MPI_DOUBLE,                                                                          //
            //                           g_dbl);                                                                              //

            //     for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
            //         g[i] = (real_t)(g_dbl[i]);
            //     }
            //     free(g_dbl);
            // }
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
                    int    min_field_index = -1;
                    int    max_field_index = -1;

                    normalize_field_and_find_min_max(field,  //
                                                     n_zyx,
                                                     delta,
                                                     &min_field_tet10,
                                                     &max_field_tet10,
                                                     &max_field_index,
                                                     &min_field_index);

                    print_rank_info(mpi_rank,
                                    mpi_size,
                                    max_field_tet10,
                                    min_field_tet10,
                                    max_field_index,
                                    min_field_index,
                                    n_zyx,
                                    nlocal,
                                    origin,
                                    delta,
                                    nglobal);

                    ndarray_write(MPI_COMM_WORLD,
                                  "/home/simone/git/sfem_d/sfem/workflows/resample/test_field_t10.raw",
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
                        mini_tet_parameters.max_refinement_L    = 3;

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

// #define REDEFINE_BOUNDING_BOX_FOR_REFINE
#ifdef REDEFINE_BOUNDING_BOX_FOR_REFINE
                    {
                        real_t side;
                        real_t origin_bb[3];

                        mesh_cube_bounding_box(mesh.nnodes,            //
                                               (geom_t**)mesh.points,  //
                                               0.05,                   //
                                               &side,                  //
                                               &origin_bb[0],          //
                                               &origin_bb[1],          //
                                               &origin_bb[2]);         //

                        origin[0] = origin_bb[0];
                        origin[1] = origin_bb[1];
                        origin[2] = origin_bb[2];

                        delta[0] = side / (real_t)(nlocal[0] - 1);
                        delta[1] = side / (real_t)(nlocal[1] - 1);
                        delta[2] = side / (real_t)(nlocal[2] - 1);

#if SFEM_LOG_LEVEL >= 5
                        printf("Bounding box for refinement:\n origin = (%.5f %.5f %.5f),\n side = %.5f, \n%s:%d\n",
                               origin_bb[0],
                               origin_bb[1],
                               origin_bb[2],
                               side,
                               __FILE__,
                               __LINE__);

                        printf("  delta  = (%.5f %.5f %.5f)\n", delta[0], delta[1], delta[2]);
                        printf("  origin = (%.5f %.5f %.5f)\n", origin[0], origin[1], origin[2]);
                        printf("  nlocal = (%ld %ld %ld)\n", nlocal[0], nlocal[1], nlocal[2]);
#endif
                    }
#endif  // REDEFINE_BOUNDING_BOX_FOR_REFINE

#define ENABLE_NORMALIZE_MESH
#ifdef ENABLE_NORMALIZE_MESH

                    real_t new_origin[3];
                    real_t new_side[3];
                    real_t new_delta[3];

                    // const real_t norm_side = 100.0;

                    normalize_mesh_BB(mesh.nnodes,            //
                                      (geom_t**)mesh.points,  //
                                      nlocal[0],              //
                                      1.0,                    //
                                      0.02,                   //
                                      &new_origin[0],         //
                                      &new_origin[1],         //
                                      &new_origin[2],         //
                                      &new_side[0],           //
                                      &new_side[1],           //
                                      &new_side[2]);          //

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
#endif
#endif

// #define REDEFINE_BOUNDING_BOX
#ifdef REDEFINE_BOUNDING_BOX
                    {
                        real_t side;
                        real_t origin_bb[3];

                        mesh_cube_bounding_box(mesh.nnodes,            //
                                               (geom_t**)mesh.points,  //
                                               0.05,                   //
                                               &side,                  //
                                               &origin_bb[0],          //
                                               &origin_bb[1],          //
                                               &origin_bb[2]);         //

                        origin[0] = origin_bb[0];
                        origin[1] = origin_bb[1];
                        origin[2] = origin_bb[2];

                        delta[0] = side / (real_t)(nlocal[0] - 1);
                        delta[1] = side / (real_t)(nlocal[1] - 1);
                        delta[2] = side / (real_t)(nlocal[2] - 1);

#if SFEM_LOG_LEVEL >= 5
                        printf("Bounding box for refinement:\n origin = (%.5f %.5f %.5f),\n side = %.5f, \n%s:%d\n",
                               origin_bb[0],
                               origin_bb[1],
                               origin_bb[2],
                               side,
                               __FILE__,
                               __LINE__);

                        printf("  delta  = (%.5f %.5f %.5f)\n", delta[0], delta[1], delta[2]);
                        printf("  origin = (%.5f %.5f %.5f)\n", origin[0], origin[1], origin[2]);
                        printf("  nlocal = (%ld %ld %ld)\n", nlocal[0], nlocal[1], nlocal[2]);
#endif
                    }
#endif  // REDEFINE_BOUNDING_BOX_FOR_REFINE

                    printf("Stride: (%ld %ld %ld) \n", stride[0], stride[1], stride[2]);

                    ret_resample_adjoint =                                     //
                            resample_field_adjoint_tet4(mpi_size,              //
                                                        mpi_rank,              //
                                                        &mesh,                 //
                                                        nlocal,                //
                                                        stride,                //
                                                        origin,                //
                                                        delta,                 //
                                                        g,                     //
                                                        mesh_fun_XYZ,          //
                                                        field,                 //
                                                        field_cnt,             //
                                                        field_alpha,           //
                                                        field_volume,          //
                                                        field_fun_XYZ,         //
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

                    real_t min_field_tet4       = 0.0;
                    real_t max_field_tet4       = 0.0;
                    int    min_field_index_tet4 = -1;
                    int    max_field_index_tet4 = -1;

                    normalize_field_and_find_min_max(field,                   //
                                                     n_zyx,                   //
                                                     delta,                   //
                                                     &min_field_tet4,         //
                                                     &max_field_tet4,         //
                                                     &min_field_index_tet4,   //
                                                     &max_field_index_tet4);  //

                    int max_field_coords[3];

                    get_3d_coordinates(max_field_index_tet4,  //
                                       nlocal,
                                       origin,
                                       delta,
                                       max_field_coords);

                    printf("Max field coords: %d %d %d :: coord %d, max value: %1.14e\n",
                           max_field_coords[0],
                           max_field_coords[1],
                           max_field_coords[2],
                           (int)(stride[0] * max_field_coords[0] + stride[1] * max_field_coords[1] +
                                 stride[2] * max_field_coords[2]),
                           max_field_tet4);

                    print_rank_info(mpi_rank,              //
                                    mpi_size,              //
                                    max_field_tet4,        //
                                    min_field_tet4,        //
                                    max_field_index_tet4,  //
                                    min_field_index_tet4,  //
                                    n_zyx,                 //
                                    nlocal,                //
                                    origin,                //
                                    delta,                 //
                                    nglobal);              //

                    // printf("max_field = %1.14e\n", max_field);
                    // printf("min_field = %1.14e\n", min_field);
                    // printf("\n");

                    // // TEST: write the in out field and the field_cnt
                    // real_t* bit_array_in_out_real = to_real_array(bit_array_in_out);

                    // TEST: write the in out field and the field_cnt
                    // real_t* field_cnt_real = (real_t*)malloc(n_zyx * sizeof(real_t));
                    // for (ptrdiff_t i = 0; i < n_zyx; i++) {
                    //     field_cnt_real[i] = (real_t)(field_cnt[i]);
                    // }

                    char out_filename_raw[1000];

                    const char* env_out_filename = getenv("OUT_FILENAME_RAW");
                    if (env_out_filename && strlen(env_out_filename) > 0) {
                        snprintf(out_filename_raw, 1000, "%s", env_out_filename);
                    } else {
                        snprintf(out_filename_raw, 1000, "%s/test_field.raw", out_base_directory);
                    }

#if SFEM_LOG_LEVEL >= 5
                    printf("Writing output field to: %s, %s:%d\n", out_filename_raw, __FILE__, __LINE__);
#endif
                    make_metadata(nlocal, new_delta, new_origin, out_base_directory);
                    ndarray_write(MPI_COMM_WORLD,
                                  out_filename_raw,
                                  ((SFEM_REAL_T_IS_FLOAT32) ? MPI_FLOAT : MPI_DOUBLE),
                                  3,
                                  field,
                                  nlocal,
                                  nglobal);

#ifdef WRITE_FIELD_FUN_XYZ
                    if (field_fun_XYZ != NULL) {
                        char        out_filename_fun_xyz[1000];
                        const char* env_out_filename_fun_xyz = getenv("OUT_FILENAME_FUN_XYZ_RAW");
                        if (env_out_filename_fun_xyz && strlen(env_out_filename_fun_xyz) > 0) {
                            snprintf(out_filename_fun_xyz, 1000, "%s", env_out_filename_fun_xyz);
                        } else {
                            snprintf(out_filename_fun_xyz,
                                     1000,
                                     "/home/simone/git/sfem_d/sfem/workflows/resample/test_field_fun_XYZ.raw");
                        }
                        ndarray_write(MPI_COMM_WORLD,
                                      out_filename_fun_xyz,
                                      ((SFEM_REAL_T_IS_FLOAT32) ? MPI_FLOAT : MPI_DOUBLE),
                                      3,
                                      field_fun_XYZ,
                                      nlocal,
                                      nglobal);
                    }
#endif  // WRITE_FIELD_FUN_XYZ

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

                    // free(field_cnt_real);
                    // field_cnt_real = NULL;

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

        if (SFEM_ADJOINT == 0) mesh_write_nodal_field(&mesh, output_path, SFEM_MPI_REAL_T, g);

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

    if (field_cnt != NULL) {
        free(field_cnt);
        field_cnt = NULL;
    }

    if (field_alpha) {
        free(field_alpha);
        field_alpha = NULL;
    }

    if (field_volume) {
        free(field_volume);
        field_volume = NULL;
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
