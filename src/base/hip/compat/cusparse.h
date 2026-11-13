#ifndef SFEM_HIP_COMPAT_CUSPARSE_H
#define SFEM_HIP_COMPAT_CUSPARSE_H

// HIP build only (base/hip/compat is on the include path with SFEM_ENABLE_HIP): cuSPARSE's generic
// API is hipSPARSE's, whose calls take the same arguments in the same order (SFEM_ENABLE_CUSPARSE).
// hipSPARSE has no generic BSR matrix; sfem_crs_spmv_cuda.cpp's BSR product calls its bsrmv.
#include <hipsparse/hipsparse.h>

#include "sfem_gpu_runtime.hpp"

#define cusparseHandle_t hipsparseHandle_t
#define cusparseStatus_t hipsparseStatus_t
#define cusparseCreate hipsparseCreate
#define cusparseDestroy hipsparseDestroy
#define cusparseGetErrorString hipsparseGetErrorString

#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseDnVecDescr_t hipsparseDnVecDescr_t
#define cusparseCreateCsr hipsparseCreateCsr
#define cusparseCreateCoo hipsparseCreateCoo
#define cusparseDestroySpMat hipsparseDestroySpMat
#define cusparseCreateDnVec hipsparseCreateDnVec
#define cusparseDestroyDnVec hipsparseDestroyDnVec
#define cusparseDnVecSetValues hipsparseDnVecSetValues
#define cusparseSpMV hipsparseSpMV
#define cusparseSpMV_bufferSize hipsparseSpMV_bufferSize

#define cusparseSpMVAlg_t hipsparseSpMVAlg_t
#define cusparseIndexType_t hipsparseIndexType_t
#define cusparseIndexBase_t hipsparseIndexBase_t
#define cusparseOperation_t hipsparseOperation_t
#define cusparseOrder_t hipsparseOrder_t
#define cudaDataType hipDataType

#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_64I HIPSPARSE_INDEX_64I
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_ORDER_ROW HIPSPARSE_ORDER_ROW
#define CUSPARSE_MV_ALG_DEFAULT HIPSPARSE_MV_ALG_DEFAULT
#define CUSPARSE_SPMV_ALG_DEFAULT HIPSPARSE_SPMV_ALG_DEFAULT
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F

#endif  // SFEM_HIP_COMPAT_CUSPARSE_H
