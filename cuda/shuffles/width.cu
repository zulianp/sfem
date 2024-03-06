#include <stdio.h>

// Using intrinsics to sum subsets of width 4 of the data
__global__ void k_warp_reduce() {
	const int width = 4;

    int laneId = threadIdx.x % 32;
    int subsetId = threadIdx.x / width;
    int value = subsetId + 1;//width - 1 - laneId;

    for (int i = width / 2; i >= 1; i /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    }

    printf("Thread %d, lane %d = %d\n", threadIdx.x, laneId, value);
}

// nvcc width.cu && ./a.out
int main() {
    k_warp_reduce<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}