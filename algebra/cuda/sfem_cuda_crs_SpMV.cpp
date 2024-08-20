#include "sfem_cuda_crs_SpMV.hpp"

#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "sfem_Buffer.hpp"

#define CHECK_CUDA(func)                                               \
    do {                                                               \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                           \
                   cudaGetErrorString(status),                         \
                   status);                                            \
            abort();                                       \
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
            abort();                                           \
        }                                                                  \
    } while (0)

typedef idx_t cu_compat_count_t;
#define SFEM_CUSPARSE_COMPAT_COUNT_T SFEM_CUSPARSE_IDX_T

namespace sfem {

    static bool sfem_cusparse_initialized = false;
    static cusparseHandle_t cusparse_handle;
    void __attribute__((destructor)) sfem_cusparse_destroy() {
        if (sfem_cusparse_initialized) {
            // printf("Destroy CuBLAS\n");
            cusparseDestroy(cusparse_handle);
        }
    }

    static void sfem_cusparse_init() {
        if (!sfem_cusparse_initialized) {
            CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
            sfem_cusparse_initialized = true;
        }
    }

    class CRSSpMVImpl {
    public:
        cusparseSpMatDescr_t matrix;
        cusparseIndexType_t csrRowOffsetsType{SFEM_CUSPARSE_COMPAT_COUNT_T};
        cusparseIndexType_t csrColIndType{SFEM_CUSPARSE_IDX_T};
        cusparseIndexBase_t idxBase{CUSPARSE_INDEX_BASE_ZERO};
        cudaDataType valueType{SFEM_CUSPARSE_REAL_T};
        cusparseOperation_t op_type{CUSPARSE_OPERATION_NON_TRANSPOSE};

#if CUDART_VERSION < 12000
        cusparseSpMVAlg_t alg{CUSPARSE_MV_ALG_DEFAULT};
#else
        cusparseSpMVAlg_t alg{CUSPARSE_SPMV_ALG_DEFAULT};
#endif
        cusparseDnVecDescr_t vecX, vecY;
        size_t bufferSize{0};
        bool xy_init{false};
        ptrdiff_t rows;
        ptrdiff_t cols;
        void* dBuffer{NULL};
        double alpha{1};
        double beta{1};

        CRSSpMVImpl(const ptrdiff_t rows,
                    const ptrdiff_t cols,
                    const std::shared_ptr<Buffer<count_t>>& rowptr,
                    const std::shared_ptr<Buffer<idx_t>>& colidx,
                    const std::shared_ptr<Buffer<real_t>>& values,
                    const real_t scale_output)
            : rows(rows), cols(cols), beta(scale_output) {
            sfem_cusparse_init();

            CHECK_CUSPARSE(cusparseCreateCsr(&matrix,
                                             rows,
                                             rows,
                                             rowptr->data()[rows],
                                             rowptr->data(),
                                             colidx->data(),
                                             values->data(),
                                             csrRowOffsetsType,
                                             csrColIndType,
                                             idxBase,
                                             valueType));

            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle,
                                                   op_type,
                                                   &alpha,
                                                   matrix,
                                                   vecX,
                                                   &beta,
                                                   vecY,
                                                   valueType,
                                                   alg,
                                                   &bufferSize));

            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

            CHECK_CUDA(cudaPeekAtLastError());
        }

        ~CRSSpMVImpl() {
            CHECK_CUDA(cudaFree(dBuffer));
            CHECK_CUSPARSE(cusparseDestroySpMat(matrix));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        }

        void apply(const real_t* const x, real_t* const y) {
            if (!xy_init) {
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, (void*)x, valueType));
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, (void*)y, valueType));
                xy_init = true;
            } else {
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, (void*)x));
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, (void*)y));
            }

            CHECK_CUSPARSE(cusparseSpMV(
                    cusparse_handle, op_type, &alpha, matrix, vecX, &beta, vecY, valueType, alg, dBuffer));
        }
    };

    std::shared_ptr<CRSSpMV<count_t, idx_t, real_t>> d_crs_spmv(
            const ptrdiff_t rows,
            const ptrdiff_t cols,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values,
            const real_t scale_output) {
        auto ret = std::make_shared<CRSSpMV<count_t, idx_t, real_t>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values = values;
        ret->cols_ = cols;

        auto impl = std::make_shared<CRSSpMVImpl>(rows, cols, rowptr, colidx, values, scale_output);
        ret->apply_ = [=](const real_t* const x, real_t* const y) { impl->apply(x, y); };
        return ret;
    }

}  // namespace sfem
