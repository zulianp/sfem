#include <stdio.h>

#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

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
int main() {
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    double alpha = 1, beta = 0;

    cusparseDnVecDescr_t vecX, vecY;
    void *dX, *dY;

    cusparseSpMatDescr_t d_matrix;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    void* csrRowOffsets;
    void* csrColInd;
    void* csrValues;

    cusparseIndexType_t csrRowOffsetsType = CUSPARSE_INDEX_32I;
    cusparseIndexType_t csrColIndType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType = CUDA_R_64F;

    CHECK_CUSPARSE(cusparseCreateCsr(&d_matrix,
                                     rows,
                                     cols,
                                     nnz,
                                     csrRowOffsets,
                                     csrColInd,
                                     csrValues,
                                     csrRowOffsetsType,
                                     csrColIndType,
                                     idxBase,
                                     valueType));

    // Create dense vectors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, dX, valueType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, dY, valueType));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           d_matrix,
                                           vecX,
                                           &beta,
                                           vecY,
                                           valueType,
                                           CUSPARSE_SPMV_ALG_DEFAULT,
                                           &bufferSize));

    void* dBuffer = NULL;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                d_matrix,
                                vecX,
                                &beta,
                                vecY,
                                valueType,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));

    cudaDeviceSynchronize();

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
    return 0;
}