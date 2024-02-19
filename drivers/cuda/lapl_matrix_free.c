#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include "sfem_base.h"
#include "sfem_mesh.h"
#include "read_mesh.h"

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

#define CHECK_CUSPARSE(func)                                               \
    do {                                                                   \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                               \
                   cusparseGetErrorString(status),                         \
                   status);                                                \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    } while (0)

// make spmv cuda=1
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 5) {
        fprintf(
            stderr, "usage: %s <mesh> <alpha> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mesh_folder = argv[1];
    const real_t alpha = atof(argv[2]);
    const char *x_path = argv[3];
    const char *output_path = argv[4];

    mesh_t hmesh;
    mesh_read(comm, mesh_folder, &hmesh);

    ptrdiff_t nnodes = hmesh.nnodes;

    double tick = MPI_Wtime();

    ptrdiff_t _nope_, x_n;
    real_t *x = 0;
    array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);

    real_t *y = calloc(nnodes, sizeof(real_t));

    {  // CUDA begin
        void *dX, *dY;

        // Create dense vectors
        CHECK_CUDA(cudaMalloc((void **)&dX, nnodes * sizeof(real_t)));
        CHECK_CUDA(cudaMalloc((void **)&dY, nnodes * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(dY, y, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dX, x, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaPeekAtLastError());

        double spmv_tick = MPI_Wtime();

        // TODO MF kernel  

        cudaDeviceSynchronize();

        double spmv_tock = MPI_Wtime();
        printf("mf: %g (seconds)\n", spmv_tock - spmv_tick);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, dY, nnodes * sizeof(real_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
    }

    array_write(comm, output_path, SFEM_MPI_REAL_T, y, nnodes, nnodes);
    free(x);
    free(y);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("cuda_do_spmv.c\n");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
