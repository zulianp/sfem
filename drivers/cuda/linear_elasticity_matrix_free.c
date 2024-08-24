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

#include "cu_tet4_adjugate.h"
#include "cu_tet4_fff.h"

#include "cu_linear_elasticity.h"

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

        void *jacobian_adjugate = 0;
        void *jacobian_determinant = 0;
        cu_tet4_adjugate_allocate(mesh.nelements, &jacobian_adjugate, &jacobian_determinant);
        cu_tet4_adjugate_fill(mesh.nelements,
                              mesh.elements,
                              mesh.points,
                              jacobian_adjugate,
                              jacobian_determinant);

        idx_t *d_elements = 0;
        elements_to_device(mesh.nelements, elem_num_nodes(elem_type), mesh.elements, &d_elements);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // With MPI Wtime
        cudaDeviceSynchronize();
        double mf_tick = MPI_Wtime();

        // With CUDA
        cudaEventRecord(start, 0);

        // ---------------------------------------------------

        for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
            cu_linear_elasticity_apply(elem_type,
                                       mesh.nelements,
                                       mesh.nelements,
                                       d_elements,
                                       jacobian_adjugate,
                                       jacobian_determinant,
                                       SFEM_SHEAR_MODULUS,
                                       SFEM_FIRST_LAME_PARAMETER,
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
            double avg_throughput = (ndofs / avg_time) * (sizeof(real_t) * 1e-9);
            printf("mf: %g %g %ld %ld %ld\n", avg_time, avg_throughput, mesh.nelements, ndofs, 0l);
        }

        {  // Using CUDA perf-counter (from ms to s)
            double avg_time = (cuda_elapsed / 1000) / SFEM_REPEAT;
            double avg_throughput = (ndofs / avg_time) * (sizeof(real_t) * 1e-9);
            printf("cf: %g %g %ld %ld %ld\n", avg_time, avg_throughput, mesh.nelements, ndofs, 0l);
        }

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, d_y, ndofs * sizeof(real_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));

        cudaFree(jacobian_adjugate);
        cudaFree(jacobian_determinant);
        cudaFree(d_elements);
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
