#ifndef SFEM_HIP_COMPAT_CUBLAS_V2_H
#define SFEM_HIP_COMPAT_CUBLAS_V2_H

// HIP build only (base/hip/compat is on the include path with SFEM_ENABLE_HIP): cuBLAS is hipBLAS,
// whose calls take the same arguments in the same order (SFEM_ENABLE_CUBLAS).
#include <hipblas/hipblas.h>

#include "sfem_gpu_runtime.hpp"

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream

#define cublasSaxpy hipblasSaxpy
#define cublasDaxpy hipblasDaxpy
#define cublasSscal hipblasSscal
#define cublasDscal hipblasDscal
#define cublasSdot hipblasSdot
#define cublasDdot hipblasDdot
#define cublasSnrm2 hipblasSnrm2
#define cublasDnrm2 hipblasDnrm2

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR

#endif  // SFEM_HIP_COMPAT_CUBLAS_V2_H
