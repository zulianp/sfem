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

#include "cu_proteus_tet4_laplacian.h"
#include "cu_tet4_fff.h"
#include "proteus_tet4_laplacian.h"
#include "sfem_cuda_blas.h"

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

    int L = 4;
    enum ElemType element_type = mesh.element_type;
    const ptrdiff_t nelements = mesh.nelements;
    const int nxe = proteus_tet4_nxe(L);
    const int txe = proteus_tet4_txe(L);
    ptrdiff_t nnodes_discont = nelements * nxe;

    real_t *x = 0;
    // if (strcmp("gen:ones", x_path) == 0) {
    x = malloc(nnodes_discont * sizeof(real_t));
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes_discont; ++i) {
        x[i] = 1;
    }

    // } else {
    //     ptrdiff_t _nope_, x_n;
    //     array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);
    // }

    real_t *y = calloc(nnodes_discont, sizeof(real_t));

    /////////////////////////////
    // CUDA begin
    void *d_x, *d_y;

    // Create dense vectors
    CHECK_CUDA(cudaMalloc((void **)&d_x, nnodes_discont * sizeof(real_t)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, nnodes_discont * sizeof(real_t)));
    CHECK_CUDA(cudaMemcpy(d_y, y, nnodes_discont * sizeof(real_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, nnodes_discont * sizeof(real_t), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    CHECK_CUDA(cudaPeekAtLastError());

    // FIXME hardcoded for tets
    void *d_fff = 0;
    cu_tet4_fff_allocate(nelements, &d_fff);
    cu_tet4_fff_fill(nelements, mesh.elements, mesh.points, d_fff);

    // idx_t *d_elements;
    // elements_to_device(nelements, elem_num_nodes(elem_type), mesh.elements,
    // &d_elements);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    double mf_tick = MPI_Wtime();

    // With CUDA
    cudaEventRecord(start, 0);

    // ---------------------------------------------------

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        cu_proteus_tet4_laplacian_apply(L,
                                        nelements,
                                        nelements,
                                        d_fff,
                                        SFEM_REAL_DEFAULT,
                                        d_x,
                                        d_y,
                                        SFEM_DEFAULT_STREAM);
    }

    // ---------------------------------------------------

    // With CUDA
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // With MPI Wtime
    cudaDeviceSynchronize();
    double mf_tock = MPI_Wtime();

    float cuda_elapsed;
    cudaEventElapsedTime(&cuda_elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    {  // Using MPI Wtime
        double avg_time = (mf_tock - mf_tick) / SFEM_REPEAT;
        double avg_throughput = (nnodes_discont / avg_time) * (sizeof(real_t) * 1e-9);
        printf("mf: %g %g %ld %ld %ld\n",
               avg_time,
               avg_throughput,
               nelements,
               nnodes_discont,
               0l);
        printf("proteus: %g [muE/s]\n", (nelements * txe) / avg_time);
    }

    {  // Using CUDA perf-counter (from ms to s)
        double avg_time = (cuda_elapsed / 1000) / SFEM_REPEAT;
        double avg_throughput = (nnodes_discont / avg_time) * (sizeof(real_t) * 1e-9);
        printf("cf: %g %g %ld %ld %ld\n",
               avg_time,
               avg_throughput,
               nelements,
               nnodes_discont,
               0l);
    }

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaMemcpy(y, d_y, nnodes_discont * sizeof(real_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    d_buffer_destroy(d_fff);
    // d_buffer_destroy(d_elements);
    /////////////////////////////

    // array_write(comm, output_path, SFEM_MPI_REAL_T, y, nnodes_discont, nnodes_discont);
    mesh_destroy(&mesh);
    free(x);
    free(y);

    double tock = MPI_Wtime();
    if (!rank) {
        float TTS = tock - tick;
        float TTS_op = (mf_tock - mf_tick) / SFEM_REPEAT;

        const int nxe = proteus_tet4_nxe(L);
        const int txe = proteus_tet4_txe(L);

        float mem_coeffs = 2 * nnodes_discont * sizeof(real_t) * 1e-9;
        float mem_jacs = 6 * nelements * sizeof(jacobian_t) * 1e-9;
        float mem_idx = nelements * nxe * sizeof(idx_t) * 1e-9;
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #microelements %ld #nodes %ld\n",
               nelements,
               nelements * txe,
               nnodes_discont);
        printf("#nodexelement %d #microelementsxelement %d\n", nxe, txe);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MmicroE/s]\n", 1e-6f * nelements * txe / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * nnodes_discont / TTS_op);
        printf("Operator memory %g (2 x coeffs) + %g (FFFs) + %g (index) = %g [GB]\n",
               mem_coeffs,
               mem_jacs,
               mem_idx,
               mem_coeffs + mem_jacs + mem_idx);
        printf("Total:\t\t\t%.4f\t[s]\n", TTS);
        printf("----------------------------------------\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
