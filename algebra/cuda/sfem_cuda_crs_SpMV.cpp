#include "sfem_cuda_crs_SpMV.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_config.h"
#include "sfem_cuda_base.h"
#include "sfem_cuda_blas.hpp"

#ifdef SFEM_ENABLE_CUSPARSE

#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#define CHECK_CUDA(func)                                               \
    do {                                                               \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                           \
                   cudaGetErrorString(status),                         \
                   status);                                            \
            abort();                                                   \
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
            abort();                                                       \
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
        bool initialized{false};
        ptrdiff_t rows;
        ptrdiff_t cols;
        void* dBuffer{NULL};
        double alpha{1};
        double beta{1};

        std::shared_ptr<Buffer<count_t>> rowptr;
        std::shared_ptr<Buffer<idx_t>> colidx;
        std::shared_ptr<Buffer<real_t>> values;

        std::shared_ptr<Buffer<cu_compat_count_t>> rowptr_compat;

        CRSSpMVImpl(const ptrdiff_t rows, const ptrdiff_t cols,
                    const std::shared_ptr<Buffer<count_t>>& rowptr,
                    const std::shared_ptr<Buffer<idx_t>>& colidx,
                    const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output)
            : rows(rows), cols(cols), colidx(colidx), values(values), beta(scale_output) {
            sfem_cusparse_init();
            assign_rowptr(rowptr, this->rowptr);
        }

        static const int needs_conversion = !std::is_same<cu_compat_count_t, count_t>::value;

        void assign_rowptr(const std::shared_ptr<Buffer<count_t>>& in,
                           std::shared_ptr<Buffer<cu_compat_count_t>>& out) {
            static_assert(!needs_conversion);
            out = in;
        }

        void initialize(const real_t* const x, real_t* const y) {
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, (void*)x, valueType));
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, (void*)y, valueType));

            CHECK_CUSPARSE(cusparseCreateCsr(&matrix,
                                             rows,
                                             rows,
                                             colidx->size(),
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
            if (!initialized) {
                initialize(x, y);
                initialized = true;
            } else {
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, (void*)x));
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, (void*)y));
            }

            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle,
                                        op_type,
                                        &alpha,
                                        matrix,
                                        vecX,
                                        &beta,
                                        vecY,
                                        valueType,
                                        alg,
                                        dBuffer));
        }
    };

    std::shared_ptr<CRSSpMV<count_t, idx_t, real_t>> d_crs_spmv(
            const ptrdiff_t rows, const ptrdiff_t cols,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output) {
        auto ret = std::make_shared<CRSSpMV<count_t, idx_t, real_t>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values = values;
        ret->cols_ = cols;
        ret->execution_space_ = EXECUTION_SPACE_DEVICE;

        auto impl = std::make_shared<CRSSpMVImpl>(rows, cols, rowptr, colidx, values, scale_output);
        ret->apply_ = [=](const real_t* const x, real_t* const y) { impl->apply(x, y); };
        return ret;
    }

    class SymCooSpMVImpl {
    public:
        cusparseSpMatDescr_t matrix;
        cusparseIndexType_t cooIndType{SFEM_CUSPARSE_IDX_T};
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
        bool initialized{false};
        ptrdiff_t ndofs;
        void* dBuffer{NULL};
        double alpha{1};
        double beta{1};
        double one{1};
        BLAS_Tpl<real_t> blas;

        std::shared_ptr<Buffer<count_t>> rowidx;
        std::shared_ptr<Buffer<idx_t>> colidx;
        std::shared_ptr<Buffer<real_t>> values;

        // probably should have alpha / beta args?
        SymCooSpMVImpl(const ptrdiff_t ndofs, const std::shared_ptr<Buffer<idx_t>>& rowidx,
                       const std::shared_ptr<Buffer<idx_t>>& colidx,
                       const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output)
            : ndofs(ndofs), rowidx(rowidx), colidx(colidx), values(values), beta(scale_output) {
            sfem_cusparse_init();

            assert(rowidx->size() == colidx->size());
            assert(values->size() == colidx->size());
        }

        void initialize(const real_t* const x, real_t* const y) {
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, ndofs, (void*)x, valueType));
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, ndofs, (void*)y, valueType));

            // printf("SymCooSpMVImpl::initialize: %ld %ld\n", (long)ndofs, (long)rowidx->size());
            // fflush(stdout);

            CHECK_CUSPARSE(cusparseCreateCoo(&matrix,
                                             ndofs,
                                             ndofs,
                                             rowidx->size(),
                                             rowidx->data(),
                                             colidx->data(),
                                             values->data(),
                                             cooIndType,
                                             idxBase,
                                             valueType));

            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matrix,
                                                   vecX,
                                                   &beta,
                                                   vecY,
                                                   valueType,
                                                   alg,
                                                   &bufferSize));

            size_t bufferSize_transpose = 0;
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle,
                                                   CUSPARSE_OPERATION_TRANSPOSE,
                                                   &alpha,
                                                   matrix,
                                                   vecX,
                                                   &beta,
                                                   vecY,
                                                   valueType,
                                                   alg,
                                                   &bufferSize_transpose));
            bufferSize = (bufferSize > bufferSize_transpose) ? bufferSize : bufferSize_transpose;
            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

            CHECK_CUDA(cudaPeekAtLastError());
        }

        ~SymCooSpMVImpl() {
            CHECK_CUSPARSE(cusparseDestroySpMat(matrix));
            CHECK_CUDA(cudaFree(dBuffer));

            CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        }

        void apply(const real_t* const x, real_t* const y) {
            SFEM_DEBUG_SYNCHRONIZE();
            if (!initialized) {
                initialize(x, y);
                initialized = true;
            } else {
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, (void*)x));
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, (void*)y));
            }

            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matrix,
                                        vecX,
                                        &beta,
                                        vecY,
                                        valueType,
                                        alg,
                                        dBuffer));

            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle,
                                        CUSPARSE_OPERATION_TRANSPOSE,
                                        &alpha,
                                        matrix,
                                        vecX,
                                        &one,
                                        vecY,
                                        valueType,
                                        alg,
                                        dBuffer));
            SFEM_DEBUG_SYNCHRONIZE();
        }
    };

    // Boundary conditions should be imposed before by hand... (if any) so cusparse API is valid
    std::shared_ptr<CooSymSpMV<idx_t, real_t>> d_sym_coo_spmv(
            const ptrdiff_t ndofs, const std::shared_ptr<Buffer<idx_t>>& rowidx,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values,
            const std::shared_ptr<Buffer<real_t>>& diag_values, const real_t scale_output) {
        auto ret = std::make_shared<CooSymSpMV<idx_t, real_t>>();
        ret->offdiag_colidx = colidx;
        ret->offdiag_rowidx = rowidx;
        ret->values = values;
        ret->diag_values = diag_values;
        ret->ndofs = ndofs;
        ret->execution_space_ = EXECUTION_SPACE_DEVICE;

        CUDA_BLAS<real_t>::build_blas(ret->blas);
        auto impl = std::make_shared<SymCooSpMVImpl>(ndofs, rowidx, colidx, values, scale_output);
        ret->apply_ = [=](const real_t* const x, real_t* const y) {
            impl->apply(x, y);
            ret->blas.xypaz(ndofs, diag_values->data(), x, 1, y);
        };
        return ret;
    }

#if CUDART_VERSION >= 12000
    class BSRSpMVImpl {
    public:
        cusparseSpMatDescr_t matrix;
        cusparseIndexType_t csrRowOffsetsType{SFEM_CUSPARSE_COMPAT_COUNT_T};
        cusparseIndexType_t csrColIndType{SFEM_CUSPARSE_IDX_T};
        cusparseIndexBase_t idxBase{CUSPARSE_INDEX_BASE_ZERO};
        cudaDataType valueType{SFEM_CUSPARSE_REAL_T};
        cusparseOperation_t op_type{CUSPARSE_OPERATION_NON_TRANSPOSE};

        cusparseSpMVAlg_t alg{CUSPARSE_SPMV_ALG_DEFAULT};

        cusparseDnVecDescr_t vecX, vecY;
        size_t bufferSize{0};
        bool initialized{false};
        ptrdiff_t block_rows;
        ptrdiff_t block_cols;
        int block_size;
        void* dBuffer{NULL};
        double alpha{1};
        double beta{1};
        cusparseOrder_t order{CUSPARSE_ORDER_ROW};

        std::shared_ptr<Buffer<count_t>> rowptr;
        std::shared_ptr<Buffer<idx_t>> colidx;
        std::shared_ptr<Buffer<real_t>> values;

        std::shared_ptr<Buffer<cu_compat_count_t>> rowptr_compat;

        BSRSpMVImpl(const ptrdiff_t block_rows, const ptrdiff_t block_cols, const int block_size,
                    const std::shared_ptr<Buffer<count_t>>& rowptr,
                    const std::shared_ptr<Buffer<idx_t>>& colidx,
                    const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output)
            : block_rows(block_rows),
              block_cols(block_cols),
              block_size(block_size),
              colidx(colidx),
              values(values),
              beta(scale_output) {
            sfem_cusparse_init();
            assign_rowptr(rowptr, this->rowptr);
        }

        static const int needs_conversion = !std::is_same<cu_compat_count_t, count_t>::value;

        void assign_rowptr(const std::shared_ptr<Buffer<count_t>>& in,
                           std::shared_ptr<Buffer<cu_compat_count_t>>& out) {
            static_assert(!needs_conversion);
            out = in;
        }

        void initialize(const real_t* const x, real_t* const y) {
            CHECK_CUSPARSE(
                    cusparseCreateDnVec(&vecX, block_cols * block_size, (void*)x, valueType));
            CHECK_CUSPARSE(
                    cusparseCreateDnVec(&vecY, block_rows * block_size, (void*)y, valueType));

            CHECK_CUSPARSE(cusparseCreateBsr(&matrix,
                                             block_rows,
                                             block_rows,
                                             colidx->size(),
                                             block_size,
                                             block_size,
                                             rowptr->data(),
                                             colidx->data(),
                                             values->data(),
                                             csrRowOffsetsType,
                                             csrColIndType,
                                             idxBase,
                                             valueType,
                                             order));

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

        ~BSRSpMVImpl() {
            CHECK_CUDA(cudaFree(dBuffer));
            CHECK_CUSPARSE(cusparseDestroySpMat(matrix));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        }

        void apply(const real_t* const x, real_t* const y) {
            SFEM_TRACE_SCOPE("BSRSpMV_CUDA::apply");

            if (!initialized) {
                initialize(x, y);
                initialized = true;
            } else {
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, (void*)x));
                CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, (void*)y));
            }

            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle,
                                        op_type,
                                        &alpha,
                                        matrix,
                                        vecX,
                                        &beta,
                                        vecY,
                                        valueType,
                                        alg,
                                        dBuffer));
        }
    };

    std::shared_ptr<BSRSpMV<count_t, idx_t, real_t>> d_bsr_spmv(
            const ptrdiff_t block_rows, const ptrdiff_t block_cols, const int block_size,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output) {
        auto ret = std::make_shared<BSRSpMV<count_t, idx_t, real_t>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values = values;
        ret->block_cols_ = block_cols;
        ret->execution_space_ = EXECUTION_SPACE_DEVICE;

        auto impl = std::make_shared<BSRSpMVImpl>(
                block_rows, block_cols, block_size, rowptr, colidx, values, scale_output);
        ret->apply_ = [=](const real_t* const x, real_t* const y) { impl->apply(x, y); };
        return ret;
    }

#else
    std::shared_ptr<BSRSpMV<count_t, idx_t, real_t>> d_bsr_spmv(
            const ptrdiff_t rows, const ptrdiff_t cols, const int block_size,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output) {
        assert(false);
        return nullptr;
    }
#endif

}  // namespace sfem

#else
#warning "No CUSPARSE installation!"

namespace sfem {
    std::shared_ptr<CRSSpMV<count_t, idx_t, real_t>> d_crs_spmv(
            const ptrdiff_t rows, const ptrdiff_t cols,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output) {
        assert(false);
        return nullptr;
    }

    std::shared_ptr<BSRSpMV<count_t, idx_t, real_t>> d_bsr_spmv(
            const ptrdiff_t rows, const ptrdiff_t cols, const int block_size,
            const std::shared_ptr<Buffer<count_t>>& rowptr,
            const std::shared_ptr<Buffer<idx_t>>& colidx,
            const std::shared_ptr<Buffer<real_t>>& values, const real_t scale_output) {
        assert(false);
        return nullptr;
    }
}  // namespace sfem

#endif
