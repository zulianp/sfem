
#include "sfem_cuda_blas.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cassert>
#include <cstdio>

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

static const char *myCublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define CHECK_CUBLAS(func)                                               \
    do {                                                                 \
        cublasStatus_t status = (func);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                           \
            printf("CUBLAS API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                             \
                   myCublasGetErrorString(status),                       \
                   status);                                              \
            assert(false);                                               \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

static bool cublas_initialized = false;
static cublasHandle_t cublas_handle;
void __attribute__((destructor)) destroy_cublas() {
    if (cublas_initialized) {
        printf("Destroy CuBLAS\n");
        cublasDestroy(cublas_handle);
    }
}

namespace sfem {
    namespace device {

        void cublas_init() {
            if (!cublas_initialized) {
                CHECK_CUBLAS(cublasCreate(&cublas_handle));
            }
        }

        template <typename T>
        T *allocate(const std::size_t n) {
            T *ptr = nullptr;
            cudaMalloc((void **)&ptr, n * sizeof(T));
            cudaMemset(ptr, 0, n * sizeof(T));
            return ptr;
        }

        template <typename T>
        void destroy(T *a) {
            cudaFree(a);
        }

        template <typename T>
        void copy(const ptrdiff_t n, const T *const src, T *const dest) {
            cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        double dot(const ptrdiff_t n, const double *const l, const double *const r) {
            cublas_init();

            double ret = 0;
            CHECK_CUBLAS(cublasDdot(cublas_handle, n, l, 1, r, 1, &ret));
            return ret;
        }

        float dot(const ptrdiff_t n, const float *const l, const float *const r) {
            cublas_init();

            float ret = 0;
            CHECK_CUBLAS(cublasSdot(cublas_handle, n, l, 1, r, 1, &ret));
            return ret;
        }

        void axpby(const ptrdiff_t n,
                   const double alpha,
                   const double *const x,
                   const double beta,
                   double *const y) {
            cublas_init();

            if (beta != 1) {
                CHECK_CUBLAS(cublasDscal(cublas_handle, n, &beta, y, 1));
            }

            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
        }

        void axpby(const ptrdiff_t n,
                   const float alpha,
                   const float *const x,
                   const float beta,
                   float *const y) {
            cublas_init();

            if (beta != 1) {
                CHECK_CUBLAS(cublasSscal(cublas_handle, n, &beta, y, 1));
            }

            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
        }

    }  // namespace device
}  // namespace sfem

extern "C" {

real_t *d_allocate(const std::size_t n) { return sfem::device::allocate<real_t>(n); }

void d_destroy(real_t *a) { sfem::device::destroy(a); }

void d_copy(const ptrdiff_t n, const real_t *const src, real_t *const dest) {
    sfem::device::copy(n, src, dest);
}

real_t d_dot(const ptrdiff_t n, const real_t *const l, const real_t *const r) {
    return sfem::device::dot(n, l, r);
}

void d_axpby(const ptrdiff_t n,
             const real_t alpha,
             const real_t *const x,
             const real_t beta,
             real_t *const y) {
    sfem::device::axpby(n, alpha, x, beta, y);
}

void d_memset(void *ptr, int value, const std::size_t n) { cudaMemset(ptr, value, n); }
}
