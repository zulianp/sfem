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

#include "laplacian_incore_cuda.h"

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
// MF(opt1): ___        0.0553568          3.326x speed up wrt OpenMP spmv
//                      2.4 slower than cusparse
// Clearly there is room for improvements wrt Vanilla version
// Even  20 MF evaluations are time neutral if we consider assembly and 20 SpmV evaluations
// with fp32 types in kernel 4x speed up (still not near peak)

// On CSCS Piz Daint P100 (cuda 11) matrix free with macro element is 1.7 times faster
// than SpMV and 3.1 times faster than standard matrix free

// Small experiment
// #elements 20,971,520 #nodes 3,578,097 #nz 53,007,057
// run 1) MF std: 0.00424194, macro: 0.00138402  SpMV (cusparse): 0.00242305

// Larger experiment (Naive OpenMP SpMV:  0.127337 seconds)
// #elements 167,772,160 #nodes 28,292,577 #nz 421,740,961
// run 1) MF std: 0.033685, macro: 0.010668, SpMV (cusparse): 0.0184639
// run 2) MF std: 0.033650, macro: 0.010675, SpMV (cusparse): 0.0184841

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

    mesh_t mesh;
    mesh_read(comm, mesh_folder, &mesh);
    enum ElemType elem_type = mesh.element_type;

    if (SFEM_USE_MACRO) {
        elem_type = macro_type_variant(elem_type);
    }

    ptrdiff_t nnodes = mesh.nnodes;

    real_t *x = 0;
    if (strcmp("gen:ones", x_path) == 0) {
        x = malloc(nnodes * sizeof(real_t));
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            x[i] = 1;
        }

    } else {
        ptrdiff_t _nope_, x_n;
        array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);
    }

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
        cuda_incore_laplacian_init(elem_type, &ctx, mesh.nelements, mesh.elements, mesh.points);

        cudaDeviceSynchronize();
        double mf_tick = MPI_Wtime();

        for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
            cuda_incore_laplacian_apply(&ctx, d_x, d_y);
        }

        cudaDeviceSynchronize();
        double mf_tock = MPI_Wtime();
        
        double avg_time = (mf_tock - mf_tick) / SFEM_REPEAT;
        double avg_throughput = (nnodes / avg_time) * (sizeof(real_t) * 1e-9);

        printf("mf: %g %g %ld %ld %ld\n", 
            avg_time, avg_throughput, mesh.nelements, nnodes, 0l);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, d_y, nnodes * sizeof(real_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));

        cuda_incore_laplacian_destroy(&ctx);
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
