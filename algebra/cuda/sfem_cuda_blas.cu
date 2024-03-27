
#include "sfem_cuda_blas.h"

#include "sfem_cuda_base.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(func)                                               \
    do {                                                               \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__,                                           \
                   cudaGetErrorString(status),                         \
                   status);                                            \
            assert(false);                                             \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define SFEM_ENABLE_CUBLAS
#ifdef SFEM_ENABLE_CUBLAS
#include "cublas_v2.h"

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

static bool sfem_blas_initialized = false;
static cublasHandle_t cublas_handle;
void __attribute__((destructor)) sfem_blas_destroy() {
    if (sfem_blas_initialized) {
        printf("Destroy CuBLAS\n");
        cublasDestroy(cublas_handle);
    }
}

static void sfem_blas_init() {
    if (!sfem_blas_initialized) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        sfem_blas_initialized = true;
    }
}

#else
static void sfem_blas_init() {}
#endif

namespace sfem {
    namespace device {

        template <typename T>
        __global__ void tscal(const ptrdiff_t n, const T alpha, T *const x) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += blockDim.x * gridDim.x) {
                x[i] *= alpha;
            }
        }

        template <typename T>
        __global__ void taxpby(const ptrdiff_t n,
                               const T alpha,
                               const T *const x,
                               const T beta,
                               T *const y) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += blockDim.x * gridDim.x) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }

        template <typename T>
        __global__ void tdot(const ptrdiff_t n, const T *const l, const T *const r, T *result) {
            T acc = 0;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += blockDim.x * gridDim.x) {
                acc += l[i] * r[i];
            }

            atomicAdd(result, acc);
        }

        template <typename T>
        T *allocate(const std::size_t n) {
            T *ptr = nullptr;
            cudaMalloc((void **)&ptr, n * sizeof(T));
            cudaMemset(ptr, 0, n * sizeof(T));
            assert(ptr);
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
            sfem_blas_init();

#ifdef SFEM_ENABLE_CUBLAS
            double ret = 0;
            CHECK_CUBLAS(cublasDdot(cublas_handle, n, l, 1, r, 1, &ret));
            return ret;
#else
            int kernel_block_size = 128;
            ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

            double *d_result = 0;
            cudaMalloc((void **)&d_result, sizeof(double));
            cudaMemset((void *)d_result, 0, sizeof(double));
            tdot<<<n_blocks, kernel_block_size>>>(n, l, r, d_result);

            double ret = 0;
            cudaMemcpy(&ret, d_result, sizeof(double), cudaMemcpyDeviceToHost);

            SFEM_DEBUG_SYNCHRONIZE();
            return ret;
#endif
        }

        float dot(const ptrdiff_t n, const float *const l, const float *const r) {
            sfem_blas_init();
#ifdef SFEM_ENABLE_CUBLAS
            float ret = 0;
            CHECK_CUBLAS(cublasSdot(cublas_handle, n, l, 1, r, 1, &ret));
            return ret;

#else
            int kernel_block_size = 128;
            ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

            float *d_result = 0;
            cudaMalloc((void **)&d_result, sizeof(float));
            cudaMemset((void *)d_result, 0, sizeof(float));
            tdot<<<n_blocks, kernel_block_size>>>(n, l, r, d_result);

            float ret = 0;
            cudaMemcpy(&ret, d_result, sizeof(float), cudaMemcpyDeviceToHost);

            SFEM_DEBUG_SYNCHRONIZE();
            return ret;
#endif
        }

        void axpby(const ptrdiff_t n,
                   const double alpha,
                   const double *const x,
                   const double beta,
                   double *const y) {
            sfem_blas_init();
#ifdef SFEM_ENABLE_CUBLAS

            if (beta != 1) {
                CHECK_CUBLAS(cublasDscal(cublas_handle, n, &beta, y, 1));
            }

            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, y, 1));

#else
            int kernel_block_size = 128;
            ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

            taxpby<<<n_blocks, kernel_block_size>>>(n, alpha, x, beta, y);

            SFEM_DEBUG_SYNCHRONIZE();

#endif
        }

        void axpby(const ptrdiff_t n,
                   const float alpha,
                   const float *const x,
                   const float beta,
                   float *const y) {
            sfem_blas_init();

#ifdef SFEM_ENABLE_CUBLAS

            if (beta != 1) {
                CHECK_CUBLAS(cublasSscal(cublas_handle, n, &beta, y, 1));
            }

            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, y, 1));

#else
            int kernel_block_size = 128;
            ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

            taxpby<<<n_blocks, kernel_block_size>>>(n, alpha, x, beta, y);

            SFEM_DEBUG_SYNCHRONIZE();
#endif
        }

    }  // namespace device
}  // namespace sfem

extern "C" {

real_t *d_allocate(const std::size_t n) { return sfem::device::allocate<real_t>(n); }

void device_to_host(const std::size_t n, const real_t *const d, real_t *h) {
    CHECK_CUDA(cudaMemcpy(h, d, n * sizeof(real_t), cudaMemcpyDeviceToHost));
}

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
