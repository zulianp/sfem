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

#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh.h"
#include "sfem_defs.h"

#include "macro_tet4_laplacian_incore_cuda.h"
#include "tet4_laplacian_incore_cuda.h"

/*
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         48 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  32
  On-line CPU(s) list:   0-31
Vendor ID:               AuthenticAMD
  Model name:            AMD Ryzen 9 7950X 16-Core Processor
    CPU family:          25
    Model:               97
    Thread(s) per core:  2
    Core(s) per socket:  16
    Socket(s):           1
    Stepping:            2
    CPU(s) scaling MHz:  28%
    CPU max MHz:         5881.0000
    CPU min MHz:         400.0000
    BogoMIPS:            8986.18

    and

    NVIDIA GeForce RTX 3060
*/


// Comparisons with Vanilla MF code (seconds)
// 20,971,520 elements, 3,578,097 nodes
// OpenMP(16)           Cuda
// SPMV: 0.0301502  0.00534357 (cusparse) 5.64x speed up
// MF:   __         0.0111022             2.71x speed up wrt OpenMP spmv
//                  2.0x slower than cusparse
// 167,772,160 elements, 28,292,577  nodes
// OpenMP(16)           Cuda (CRS matrix assembly time 1.05465)
// SPMV: 0.184149       0.02335 (cusparse) 7.88x speed up
// MF:   __             0.0718077          2.56x speed up wrt OpenMP spmv
//                      3.0x slower than cusparse 
// Clearly there is room for improvements wrt Vanilla version
// Even  20 MF evaluations are time neutral if we consider assembly and 20 SpmV evaluations


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
        fprintf(stderr, "usage: %s <mesh> <alpha> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mesh_folder = argv[1];
    const real_t alpha = atof(argv[2]);
    const char *x_path = argv[3];
    const char *output_path = argv[4];

    int SFEM_REPEAT = 1;
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    mesh_t mesh;
    mesh_read(comm, mesh_folder, &mesh);

    ptrdiff_t nnodes = mesh.nnodes;

    double tick = MPI_Wtime();

    ptrdiff_t _nope_, x_n;
    real_t *x = 0;
    array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);

    real_t *y = calloc(nnodes, sizeof(real_t));

    {  // CUDA begin
        void *d_x, *d_y;

        // Create dense vectors
        CHECK_CUDA(cudaMalloc((void **)&d_x, nnodes * sizeof(real_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_y, nnodes * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_y, y, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, x, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaPeekAtLastError());

        cuda_incore_laplacian_t ctx;

        if (mesh.element_type == TET4) {
            tet4_cuda_incore_laplacian_init(&ctx, mesh);
        } else if (mesh.element_type == TET10) {
            // Go for macro just for testing
            macro_tet4_cuda_incore_laplacian_init(&ctx, mesh);
        }

        cudaDeviceSynchronize();

        double spmv_tick = MPI_Wtime();

        for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
            if (mesh.element_type == TET4) {
                tet4_cuda_incore_laplacian_apply(&ctx, d_x, d_y);
            } else if (mesh.element_type == TET10) {
                macro_tet4_cuda_incore_laplacian_apply(&ctx, d_x, d_y);
            }
        }

        cudaDeviceSynchronize();

        double spmv_tock = MPI_Wtime();
        printf("mf: %g (seconds)\n", (spmv_tock - spmv_tick) / SFEM_REPEAT);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, d_y, nnodes * sizeof(real_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));

        if (mesh.element_type == TET4) {
            tet4_cuda_incore_laplacian_destroy(&ctx);
        } else if (mesh.element_type == TET10) {
            macro_tet4_cuda_incore_laplacian_destroy(&ctx);
        }
    }

    array_write(comm, output_path, SFEM_MPI_REAL_T, y, nnodes, nnodes);
    free(x);
    free(y);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("lapl_matrix_free.c\n");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
