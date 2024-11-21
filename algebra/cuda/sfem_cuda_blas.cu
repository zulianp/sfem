
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"

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
        // printf("Destroy CuBLAS\n");
        cublasDestroy(cublas_handle);
    }
}

static void sfem_blas_init() {
    if (!sfem_blas_initialized) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        sfem_blas_initialized = true;
    }
}

template <typename T>
class BLASImpl {};

template <>
class BLASImpl<double> {
public:
    static double dot(const ptrdiff_t n, const double *const l, const double *const r) {
        sfem_blas_init();
        double ret = 0;
        CHECK_CUBLAS(cublasDdot(cublas_handle, n, l, 1, r, 1, &ret));
        return ret;
    }

    static void axpy(const ptrdiff_t n,
                     const double alpha,
                     const double *const x,
                     double *const y) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
    }

    static void axpby(const ptrdiff_t n,
                      const double alpha,
                      const double *const x,
                      const double beta,
                      double *const y) {
        sfem_blas_init();
        if (beta != 1) {
            CHECK_CUBLAS(cublasDscal(cublas_handle, n, &beta, y, 1));
        }

        CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
    }

    static void zaxpby(const ptrdiff_t n,
                       const double alpha,
                       const double *const x,
                       const double beta,
                       const double *const y,
                       double *const z) {
        sfem_blas_init();

        if (x == z) {
            CHECK_CUBLAS(cublasDscal(cublas_handle, n, &alpha, z, 1));
            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &beta, y, 1, z, 1));

        } else if (y == z) {
            CHECK_CUBLAS(cublasDscal(cublas_handle, n, &beta, z, 1));
            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, z, 1));
        } else {
            cudaMemset(z, 0, n * sizeof(double));
            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &alpha, x, 1, z, 1));
            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &beta, y, 1, z, 1));
        }
    }

    static void scal(const ptrdiff_t n, const double alpha, double *const x) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasDscal(cublas_handle, n, &alpha, x, 1));
    }

    static void nrm2(const ptrdiff_t n, const double *const x, double *const result) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, x, 1, result));
    }
};

template <>
class BLASImpl<float> {
public:
    static float dot(const ptrdiff_t n, const float *const l, const float *const r) {
        sfem_blas_init();
        float ret = 0;
        CHECK_CUBLAS(cublasSdot(cublas_handle, n, l, 1, r, 1, &ret));
        return ret;
    }

    static void axpy(const ptrdiff_t n, const float alpha, const float *const x, float *const y) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
    }

    static void axpby(const ptrdiff_t n,
                      const float alpha,
                      const float *const x,
                      const float beta,
                      float *const y) {
        sfem_blas_init();
        if (beta != 1) {
            CHECK_CUBLAS(cublasSscal(cublas_handle, n, &beta, y, 1));
        }

        CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, y, 1));
    }

    static void zaxpby(const ptrdiff_t n,
                       const float alpha,
                       const float *const x,
                       const float beta,
                       const float *const y,
                       float *const z) {
        sfem_blas_init();

        if (x == z) {
            CHECK_CUBLAS(cublasSscal(cublas_handle, n, &alpha, z, 1));
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &beta, y, 1, z, 1));

        } else if (y == z) {
            CHECK_CUBLAS(cublasSscal(cublas_handle, n, &beta, z, 1));
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, z, 1));
        } else {
            cudaMemset(z, 0, n * sizeof(float));
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &alpha, x, 1, z, 1));
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &beta, y, 1, z, 1));
        }
    }

    static void scal(const ptrdiff_t n, const float alpha, float *const x) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasSscal(cublas_handle, n, &alpha, x, 1));
    }

    static void nrm2(const ptrdiff_t n, const float *const x, float *const result) {
        sfem_blas_init();
        CHECK_CUBLAS(cublasSnrm2(cublas_handle, n, x, 1, result));
    }
};

template <typename T>
using BLAS = BLASImpl<T>;

#else

template <typename T>
__global__ void tscal(const ptrdiff_t n, const T alpha, T *const x) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] *= alpha;
    }
}

template <typename T>
__global__ void taxpby(const ptrdiff_t n,
                       const T alpha,
                       const T *const x,
                       const T beta,
                       T *const y) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

template <typename T>
__global__ void tzaxpby(const ptrdiff_t n,
                        const T alpha,
                        const T *const x,
                        const T beta,
                        const T *const y,
                        T *const z) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = alpha * x[i] + beta * y[i];
    }
}

inline __device__ unsigned int lane_id() { return threadIdx.x % SFEM_WARP_SIZE; }

template <typename T>
__device__ T warp_reduce_32(const T in) {
    static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
    T out = in;
    out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE);  // 0-16, 1-17, ..., 15-31
    out += __shfl_xor_sync(
            SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE);  // 0-8, ..., 1-7, ..., 23-31
    out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE);
    out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE);
    out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE);
    return out;
}

template <typename T>
__global__ void tdot(const ptrdiff_t n, const T *const l, const T *const r, T *result) {
    __shared__ T block_accumulator[SFEM_WARP_SIZE];

    T acc = 0;
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        acc += l[i] * r[i];
    }

    acc = warp_reduce_32(acc);

    const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
    const unsigned int lid = lane_id();

    if (!lid) {
        block_accumulator[warp_id] = acc;
    }

    __syncthreads();

    if (!warp_id) {
        acc = block_accumulator[lid];
        acc = warp_reduce_32(acc);

        if (!threadIdx.x) {
            atomicAdd(result, acc);
        }
    }
}

template <typename T>
class BLAS {
public:
    static float dot(const ptrdiff_t n, const T *const l, const T *const r) {
        int kernel_block_size = 128;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        double *d_result = 0;
        cudaMalloc((void **)&d_result, sizeof(double));
        cudaMemset((void *)d_result, 0, sizeof(double));

        tdot<<<n_blocks, kernel_block_size>>>(n, l, r, d_result);
        SFEM_DEBUG_SYNCHRONIZE();

        double ret = 0;
        cudaMemcpy(&ret, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return ret;
    }

    static void axpy(const ptrdiff_t n, const T alpha, const T *const x, T *const y) {
        int kernel_block_size = 128;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        taxpby<<<n_blocks, kernel_block_size>>>(n, alpha, x, (T)1, y);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    static void axpby(const ptrdiff_t n,
                      const T alpha,
                      const T *const x,
                      const T beta,
                      T *const y) {
        int kernel_block_size = 128;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        taxpby<<<n_blocks, kernel_block_size>>>(n, alpha, x, beta, y);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    static void zaxpby(const ptrdiff_t n,
                       const T alpha,
                       const T *const x,
                       const T beta,
                       const T *const y,
                       T *const z) {
        int kernel_block_size = 128;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        tzaxpby<<<n_blocks, kernel_block_size>>>(n, alpha, x, beta, y, z);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    static void scal(const ptrdiff_t n, const T alpha, T *const x) {
        int kernel_block_size = 128;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        tscal<<<n_blocks, kernel_block_size>>>(n, alpha, x);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    static void nrm2(const ptrdiff_t n, const T *const x, T *const result) {
        T sq_nrm = dot(n, x, x);
        *result = sqrt(sq_nrm);
    }
};

#endif

template <typename T>
__global__ void tvalues_kernel(const ptrdiff_t n, const T value, T *const x) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = value;
    }
}

template <typename T>
static void tvalues(const ptrdiff_t n, const T value, T *const x) {
    int kernel_block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

    tvalues_kernel<<<n_blocks, kernel_block_size>>>(n, value, x);
}

template <typename T>
__global__ void txypaz_kernel(const ptrdiff_t n, const T *const x, const T *const y, const T alpha, T *const z) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = x[i] * y[i] + alpha * z[i];
    }
}

template <typename T>
static void txypaz(const ptrdiff_t n, const T *const x, const T *const y, const T alpha, T *const z) {
    int kernel_block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

    txypaz_kernel<<<n_blocks, kernel_block_size>>>(n, x, y, alpha, z);
    SFEM_DEBUG_SYNCHRONIZE();
}

extern real_t *d_allocate(const std::size_t n) {
    real_t *ptr = nullptr;
    cudaMalloc((void **)&ptr, n * sizeof(real_t));
    cudaMemset(ptr, 0, n * sizeof(real_t));
    assert(ptr);
    return ptr;
}

extern void device_to_host(const std::size_t n, const real_t *const d, real_t *h) {
    CHECK_CUDA(cudaMemcpy(h, d, n * sizeof(real_t), cudaMemcpyDeviceToHost));
}

extern void host_to_device(const std::size_t n, const real_t *const h, real_t *d) {
    CHECK_CUDA(cudaMemcpy(d, h, n * sizeof(real_t), cudaMemcpyHostToDevice));
}

extern void d_destroy(void *a) { cudaFree(a); }

extern void d_copy(const ptrdiff_t n, const real_t *const src, real_t *const dest) {
    cudaMemcpy(dest, src, n * sizeof(real_t), cudaMemcpyDeviceToDevice);
}

extern real_t d_dot(const ptrdiff_t n, const real_t *const l, const real_t *const r) {
    return BLAS<real_t>::dot(n, l, r);
}

extern void d_axpby(const ptrdiff_t n,
                    const real_t alpha,
                    const real_t *const x,
                    const real_t beta,
                    real_t *const y) {
    BLAS<real_t>::axpby(n, alpha, x, beta, y);
}

extern void d_axpy(const ptrdiff_t n, const real_t alpha, const real_t *const x, real_t *const y) {
    BLAS<real_t>::axpy(n, alpha, x, y);
}

extern void d_zaxpby(const ptrdiff_t n,
                     const real_t alpha,
                     const real_t *const x,
                     const real_t beta,
                     const real_t *const y,
                     real_t *const z) {
    BLAS<real_t>::zaxpby(n, alpha, x, beta, y, z);
}

extern void d_scal(const ptrdiff_t n, const real_t alpha, real_t *const x) {
    BLAS<real_t>::scal(n, alpha, x);
}

extern real_t d_nrm2(const ptrdiff_t n, const real_t *const x) {
    real_t ret = 0;
    BLAS<real_t>::nrm2(n, x, &ret);
    return ret;
}

extern void d_memset(void *ptr, int value, const std::size_t n) { cudaMemset(ptr, value, n); }

extern void *d_buffer_alloc(const size_t n) {
    void *ptr = nullptr;
    cudaMalloc((void **)&ptr, n);
    cudaMemset(ptr, 0, n);

    SFEM_DEBUG_SYNCHRONIZE();
    return ptr;
}

extern void d_buffer_destroy(void *a) {
    cudaFree(a);
    SFEM_DEBUG_SYNCHRONIZE();
}

extern void buffer_device_to_host(const std::size_t n, const void *const d, void *h) {
    cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
}

extern void buffer_host_to_device(const std::size_t n, const void *const h, void *d) {
    cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
}

template <typename T>
__global__ void tdiv(const ptrdiff_t n, const T *const l, const T *const r, T *const result) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        assert(r[i] != 0);
        result[i] = l[i] / r[i];
    }
}

extern void d_ediv(const ptrdiff_t n,
                   const real_t *const l,
                   const real_t *const r,
                   real_t *const result) {
    int kernel_block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

    tdiv<<<n_blocks, kernel_block_size>>>(n, l, r, result);

    SFEM_DEBUG_SYNCHRONIZE();
}

namespace sfem {

    template <typename T>
    void CUDA_BLAS<T>::build_blas(struct BLAS_Tpl<T> &tpl) {
        tpl.allocate = [](const std::ptrdiff_t n) -> T * {
            T *ptr = nullptr;
            cudaMalloc((void **)&ptr, n * sizeof(T));
            cudaMemset(ptr, 0, n * sizeof(T));
            assert(ptr);
            return ptr;
        };

        tpl.copy = [](const ptrdiff_t n, const T *const src, T *const dest) {
            CHECK_CUDA(cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
        };

        tpl.zeros = [](const std::size_t size, T *const x) {
            CHECK_CUDA(cudaMemset(x, 0, size * sizeof(T)));
        };

        tpl.norm2 = [](const ptrdiff_t n, const T *const x) -> T {
            T ret = 0;
            BLASImpl<T>::nrm2(n, x, &ret);
            return ret;
        };

        tpl.xypaz = txypaz<T>;

        tpl.destroy = &d_destroy;
        tpl.values = &tvalues<T>;
        tpl.dot = &BLASImpl<T>::dot;
        tpl.axpby = &BLASImpl<T>::axpby;
        tpl.axpy = &BLASImpl<T>::axpy;
        tpl.scal = &BLASImpl<T>::scal;
        tpl.zaxpby = &BLASImpl<T>::zaxpby;
    }

    template struct CUDA_BLAS<double>;
    template struct CUDA_BLAS<float>;

    void device_synchronize() {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

}  // namespace sfem


