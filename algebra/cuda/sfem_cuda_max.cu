#include "sfem_cuda_max.hpp"

#include <algorithm>
#include <vector>

#include "sfem_cuda_base.hpp"

namespace sfem {
    namespace {
        const int block_size = 256;

        __global__ void max_kernel(const ptrdiff_t n, const real_t *const x, const real_t init, real_t *const partial) {
            __shared__ real_t cache[block_size];
            real_t            m = init;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) m = fmax(m, x[i]);
            cache[threadIdx.x] = m;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s /= 2) {
                if (threadIdx.x < s) cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + s]);
                __syncthreads();
            }
            if (threadIdx.x == 0) partial[blockIdx.x] = cache[0];
        }
    }  // namespace

    real_t d_max(const ptrdiff_t n, const real_t *const x, const real_t init) {
        if (n <= 0) return init;
        const int n_blocks = (int)std::min<ptrdiff_t>(1024, (n + block_size - 1) / block_size);
        real_t   *partial  = nullptr;
        SFEM_CUDA_CHECK(cudaMalloc(&partial, n_blocks * sizeof(real_t)));
        max_kernel<<<n_blocks, block_size>>>(n, x, init, partial);
        std::vector<real_t> host(n_blocks);
        SFEM_CUDA_CHECK(cudaMemcpy(host.data(), partial, n_blocks * sizeof(real_t), cudaMemcpyDeviceToHost));
        SFEM_CUDA_CHECK(cudaFree(partial));
        real_t m = init;
        for (const real_t v : host) m = std::max(m, v);
        return m;
    }
}  // namespace sfem
