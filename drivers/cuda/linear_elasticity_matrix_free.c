#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "linear_elasticity_incore_cuda.h"

#define CHECK_CUDA(func)                                               \
    do {                                                               \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                           \
                   cudaGetErrorString(status),                         \
                   status);                                            \
            return EXIT_FAILURE;                                       \
        }                                                              \
    } while (0)

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double tick = MPI_Wtime();

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 5) {
        fprintf(stderr, "usage: %s <mesh> <alpha> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mesh_folder = argv[1];
    const real_t alpha = atof(argv[2]);
    const char *x_path = argv[3];
    const char *output_path = argv[4];

    int SFEM_REPEAT = 1;
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    int SFEM_USE_MACRO = 0;
    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

    real_t SFEM_SHEAR_MODULUS = 1;
    real_t SFEM_FIRST_LAME_PARAMETER = 1;

    SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
    SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

    mesh_t mesh;
    mesh_read(comm, mesh_folder, &mesh);
    enum ElemType elem_type = mesh.element_type;

    if (SFEM_USE_MACRO) {
        elem_type = macro_type_variant(elem_type);
    }

    int block_size = mesh.spatial_dim;

    ptrdiff_t nnodes = mesh.nnodes;
    ptrdiff_t ndofs = nnodes * block_size;
    ptrdiff_t _nope_, x_n;
    real_t *x = 0;
    if (strcmp("gen:ones", x_path) == 0) {
        x = malloc(ndofs * sizeof(real_t));
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            x[i] = 1;
        }

    } else {
        array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);
    }

    real_t *y = calloc(ndofs, sizeof(real_t));

    {  // CUDA begin
        void *d_x, *d_y;

        // Create dense vectors
        CHECK_CUDA(cudaMalloc((void **)&d_x, ndofs * sizeof(real_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_y, ndofs * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_y, y, ndofs * sizeof(real_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, x, ndofs * sizeof(real_t), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaPeekAtLastError());

        cuda_incore_linear_elasticity_t ctx;
        cuda_incore_linear_elasticity_init(elem_type,
                                           &ctx,
                                           SFEM_SHEAR_MODULUS,
                                           SFEM_FIRST_LAME_PARAMETER,
                                           mesh.nelements,
                                           mesh.elements,
                                           mesh.points);

        cudaDeviceSynchronize();
        double spmv_tick = MPI_Wtime();

        for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
            cuda_incore_linear_elasticity_apply(&ctx, d_x, d_y);
        }

        cudaDeviceSynchronize();
        double spmv_tock = MPI_Wtime();
        printf("mf: %g (seconds)\n", (spmv_tock - spmv_tick) / SFEM_REPEAT);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, d_y, ndofs * sizeof(real_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));

        cuda_incore_linear_elasticity_destroy(&ctx);
    }

    array_write(comm, output_path, SFEM_MPI_REAL_T, y, ndofs, ndofs);
    free(x);
    free(y);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("linear_elasticity_matrix_free.c\n");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
