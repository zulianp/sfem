#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void mma_fp64_m8n8k4_kernel(double *d_output) {
    const int lane_id = threadIdx.x % WARP_SIZE;

    double A[8 * 4] = {
        1., 2., 3., 4.,
        5., 6., 7., 8.,
        9., 10., 11., 12.,
        13., 14., 15., 16.,
        // 
        17., 18., 19., 20.,
        21., 22., 23., 24.,
        25., 26., 27., 28.,
        29., 30., 31., 32.
    };

    // Allocate fragment arrays in registers (per thread)
    double a_frag = A[lane_id];   // 1 FP64 per thread for A
    double b_frag = A[lane_id]+1;   // 1 FP64 per thread for B
    double acc_frag[2] = {0, 0}; // 2 FP64 per thread for accumulator C/D

    // Inline PTX mma.sync for FP64 m8n8k4 with aliasing C = D
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1},{%2},{%3},{%4,%5};\n"
        : "=d"(acc_frag[0]), "=d"(acc_frag[1])
        : "d"(a_frag), "d"(b_frag), "d"(acc_frag[0]), "d"(acc_frag[1]));

    for (int i = 0; i < 2; i++) {
        d_output[lane_id * 2 + i] = acc_frag[i];
    }
}

int main() {
    const int warp_threads = 32;
    const int elems_per_thread = 2;
    const int total_elems = warp_threads * elems_per_thread;

    double *d_output;
    cudaMalloc(&d_output, total_elems * sizeof(double));

    mma_fp64_m8n8k4_kernel<<<1, warp_threads>>>(d_output);
    cudaDeviceSynchronize();

    double host_output[total_elems];
    cudaMemcpy(host_output, d_output, total_elems * sizeof(double), cudaMemcpyDeviceToHost);

    printf("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 output:\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", host_output[i * 8 + j]);
        }
        printf("\n");
    }

    cudaFree(d_output);
    return 0;
}
