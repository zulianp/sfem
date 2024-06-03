#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// nvcc -O3 --gpu-architecture=sm_80 baseline.cu
int main(void)
{
    long int N = 11 * (1 << 26);

    printf("N: %ld\n", N);
    const double saxpy_GB = (double)N * sizeof(float) * 3.0 / (1024.0 * 1024.0 * 1024.0);
    printf("saxpy_GB: %lf \n", saxpy_GB);

    float *x, *y, *yy, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    yy = (float *)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    printf("Init x, y.\n");
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t start, stop, start_pci, stop_pci, start_pci_out, stop_pci_out;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventCreate(&start_pci);
    cudaEventCreate(&stop_pci);

    cudaEventCreate(&start_pci_out);
    cudaEventCreate(&stop_pci_out);

    printf("Start cudaMemcpy \n");
    cudaEventRecord(start_pci);

    cudaMemcpyAsync(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_pci);
    cudaEventSynchronize(stop_pci);

    float ms_pci;
    cudaEventElapsedTime(&ms_pci, start_pci, stop_pci);

    cudaEventRecord(start);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 511) / 512, 512>>>(N, 2.0f, d_x, d_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    printf("Start cudaMemcpy out\n");
    cudaEventRecord(start_pci_out);

    cudaMemcpyAsync(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(yy, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_pci_out);
    cudaEventSynchronize(stop_pci_out);

    float ms_pci_out;
    cudaEventElapsedTime(&ms_pci_out, start_pci_out, stop_pci_out);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n");
    printf("Results: \n");
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = max(maxError, abs(2.0 * y[i] - yy[i]));
    }

    printf("\n");
    printf("N: %d\n", N);
    printf("milliseconds GPU: %f\n", milliseconds);
    printf("milliseconds PCI in: %f\n", ms_pci);
    printf("milliseconds PCI out: %f\n", ms_pci_out);

    printf("MBytes GPU: %lf\n", double(N * sizeof(float) * 3) / (1024.0 * 1024.0));
    printf("MBytes PCI: %lf\n", double(N * sizeof(float) * 2) / (1024.0 * 1024.0));
    printf("Max error:  %f\n", maxError);

    printf("Effective Bandwidth GPU (GB/s):     %f\n\n", (double)(N * sizeof(float) * 3.0) / milliseconds / (1024 * 1024));
    printf("Effective Bandwidth PCI (GB/s):     %f\n", (double)(N * sizeof(float) * 2.0) / ms_pci / (1024 * 1024));
    printf("Effective Bandwidth PCI out (GB/s): %f\n\n", (double)(N * sizeof(float) * 2.0) / ms_pci_out / (1024 * 1024));
}
