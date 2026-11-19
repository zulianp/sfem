// STREAM triad a = b + s c on one Grace chip (OpenMP) or, built with nvcc, on the GPU: the bandwidth bound
// of the compressed partial assembly paper's cost model, measured on the node it runs on. Reports the best
// of `repeats` triads as 3 streams of n doubles (STREAM's convention; a write-allocate cache adds one more).
//     stream_triad [n=2^28] [repeats=20]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
__global__ void init(const long n, double *a, double *b, double *c) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        a[i] = 0;
        b[i] = 1;
        c[i] = 2;
    }
}
__global__ void triad(const long n, double *__restrict__ a, const double *__restrict__ b, const double *__restrict__ c, const double s) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) a[i] = b[i] + s * c[i];
}
#endif

int main(int argc, char **argv) {
    const long n       = argc > 1 ? atol(argv[1]) : (1l << 28);
    const int  repeats = argc > 2 ? atoi(argv[2]) : 20;
    const double s     = 3;
    double best = 1e30;
#ifdef __CUDACC__
    double *a, *b, *c;
    cudaMalloc(&a, n * sizeof(double));
    cudaMalloc(&b, n * sizeof(double));
    cudaMalloc(&c, n * sizeof(double));
    const int    blocks = 132 * 16, threads = 256;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init<<<blocks, threads, 0, stream>>>(n, a, b, c);
    cudaStreamSynchronize(stream);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(t0, stream);
        triad<<<blocks, threads, 0, stream>>>(n, a, b, c, s);
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        if (r > 0) best = std::min(best, 1e-3 * ms);
    }
    double check = 0;
    cudaMemcpy(&check, a + n / 2, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaStreamDestroy(stream);
    const char *where = "device";
#else
    // First touch by the threads that stream them.
    double *a = (double *)aligned_alloc(64, n * sizeof(double));
    double *b = (double *)aligned_alloc(64, n * sizeof(double));
    double *c = (double *)aligned_alloc(64, n * sizeof(double));
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        a[i] = 0;
        b[i] = 1;
        c[i] = 2;
    }
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r > 0) best = std::min(best, t);
    }
    const double check = a[n / 2];
    free(a);
    free(b);
    free(c);
    const char *where = "host";
#endif
    printf("stream_triad %s n=%ld (%.2f GiB per array) best of %d: %.4f ms, %.1f GB/s (3 streams), %.1f GB/s (4, write-allocate); check %g\n",
           where,
           n,
           n * 8.0 / (1 << 30),
           repeats - 1,
           1e3 * best,
           3 * 8.0 * n / best * 1e-9,
           4 * 8.0 * n / best * 1e-9,
           check);
    return check == 1 + s * 2 ? 0 : 1;
}
