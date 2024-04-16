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

#include "sfem_base.h"
// https://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-format-csr

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

    if (argc != 6) {
        fprintf(
            stderr, "usage: %s <alpha> <transpose> <crs_folder> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_REPEAT = 1;
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    const real_t alpha = atof(argv[1]);
    const int transpose = atoi(argv[2]);
    const char *crs_folder = argv[3];
    const char *x_path = argv[4];
    const char *output_path = argv[5];

    double tick = MPI_Wtime();

    char rowptr_path[1024 * 10];
    char colidx_path[1024 * 10];
    char values_path[1024 * 10];

    sprintf(rowptr_path, "%s/rowptr.raw", crs_folder);
    sprintf(colidx_path, "%s/colidx.raw", crs_folder);
    sprintf(values_path, "%s/values.raw", crs_folder);

    crs_t crs;
    crs_read(comm,
             rowptr_path,
             colidx_path,
             values_path,
             SFEM_MPI_COUNT_T,
             SFEM_MPI_IDX_T,
             SFEM_MPI_REAL_T,
             &crs);

    
    real_t *x = 0;
    if (strcmp("gen:ones", x_path) == 0) {
        ptrdiff_t ndofs = crs.lrows;
        x = malloc(ndofs * sizeof(real_t));
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            x[i] = 1;
        }

    } else {
        ptrdiff_t _nope_, x_n;
        array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &_nope_, &x_n);
    }

    real_t *y = calloc(crs.grows, sizeof(real_t));

    {  // CUDA begin
        cusparseHandle_t handle = NULL;
        CHECK_CUSPARSE(cusparseCreate(&handle));

        double beta = 0;

        cusparseDnVecDescr_t vecX, vecY;
        void *dX, *dY;

        count_t *csrRowOffsets;
        idx_t *csrColInd;
        real_t *csrValues;

        CHECK_CUDA(cudaMalloc((void **)&csrRowOffsets, (crs.lrows + 1) * sizeof(count_t)));
        CHECK_CUDA(cudaMalloc((void **)&csrColInd, crs.lnnz * sizeof(idx_t)));
        CHECK_CUDA(cudaMalloc((void **)&csrValues, crs.lnnz * sizeof(real_t)));

        CHECK_CUDA(cudaMemcpy(csrRowOffsets,
                              (count_t *)crs.rowptr,
                              (crs.lrows + 1) * sizeof(count_t),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(
            csrColInd, (idx_t *)crs.colidx, crs.lnnz * sizeof(idx_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(
            csrValues, (real_t *)crs.values, crs.lnnz * sizeof(real_t), cudaMemcpyHostToDevice));

        cusparseSpMatDescr_t d_matrix;

        cusparseIndexType_t csrRowOffsetsType = CUSPARSE_INDEX_32I;
        cusparseIndexType_t csrColIndType = CUSPARSE_INDEX_32I;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cudaDataType valueType = CUDA_R_64F;
        cusparseOperation_t op_type =
            transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

#if CUDART_VERSION < 12000
        cusparseSpMVAlg_t alg = CUSPARSE_MV_ALG_DEFAULT;
#else
        cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
#endif

        CHECK_CUSPARSE(cusparseCreateCsr(&d_matrix,
                                         crs.lrows,
                                         crs.lrows,
                                         crs.lnnz,
                                         csrRowOffsets,
                                         csrColInd,
                                         csrValues,
                                         csrRowOffsetsType,
                                         csrColIndType,
                                         idxBase,
                                         valueType));

        // Create dense vectors
        CHECK_CUDA(cudaMalloc((void **)&dX, crs.lrows * sizeof(real_t)));
        CHECK_CUDA(cudaMalloc((void **)&dY, crs.lrows * sizeof(real_t)));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, crs.lrows, dX, valueType));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, crs.lrows, dY, valueType));

        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle, op_type, &alpha, d_matrix, vecX, &beta, vecY, valueType, alg, &bufferSize));

        CHECK_CUDA(cudaMemcpy(dY, y, crs.lrows * sizeof(real_t), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(dX, x, crs.lrows * sizeof(real_t), cudaMemcpyHostToDevice));

        void *dBuffer = NULL;
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaPeekAtLastError());

        double spmv_tick = MPI_Wtime();

        for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
            CHECK_CUSPARSE(cusparseSpMV(
                handle, op_type, &alpha, d_matrix, vecX, &beta, vecY, valueType, alg, dBuffer));
        }

        cudaDeviceSynchronize();

        double spmv_tock = MPI_Wtime();
        printf("spmv: %g (seconds)\n", (spmv_tock - spmv_tick) / SFEM_REPEAT);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpy(y, dY, crs.lrows * sizeof(real_t), cudaMemcpyDeviceToHost));

        CHECK_CUSPARSE(cusparseDestroySpMat(d_matrix));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        CHECK_CUSPARSE(cusparseDestroy(handle));

        CHECK_CUDA(cudaFree(dBuffer));
        CHECK_CUDA(cudaFree(csrRowOffsets));
        CHECK_CUDA(cudaFree(csrColInd));
        CHECK_CUDA(cudaFree(csrValues));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
    }

    array_write(comm, output_path, SFEM_MPI_REAL_T, y, crs.grows, crs.grows);
    crs_free(&crs);
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
