
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// #include <helper_cuda.h>
// #include <helper_functions.h>

using namespace nvcuda;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 8
#define N 8
#define K 4

using real_t = double;

#define checkKernelErrors(expr)                                                               \
    do {                                                                                      \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess) {                                                           \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

__global__ void wmma_ker(real_t *a, real_t *b, real_t *c) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, real_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, real_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, real_t> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, K);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

void print_matrix(const int rows,
                  const int cols,
                  const int strideI,
                  const int strideJ,
                  const real_t *const m) {

    printf("[\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f%c", m[i * strideI + j * strideJ], j < cols -1 ? ',' : ';');
        }
        printf("\n");
    }
    printf("]\n");
}

// nvcc  --compiler-options "-fPIC " -std=c++14 -arch=sm_86 -I/opt/cuda/include
// -I/opt/cuda/samples/Common/ -L/opt/cuda/lib64 -lcudart -lcusparse -lcusolver -lcublas
// wmma_example.cu
int main(int argc, char **argv) {
    real_t h_A[M * K];
    real_t h_B[N * K];
    real_t h_C[M * N];

    for (int i = 0; i < M * K; i++) {
        h_A[i] = i;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = i;
    }

    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0;
    }

    real_t *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(real_t) * M * K);
    cudaMalloc((void **)&d_B, sizeof(real_t) * K * N);
    cudaMalloc((void **)&d_C, sizeof(real_t) * M * N);

    cudaMemcpy(d_A, h_A, M * K * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(real_t), cudaMemcpyHostToDevice);

    int n_blocks = 1;
    int block_size = WARP_SIZE;
    wmma_ker<<<n_blocks, block_size, 0>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, M * N * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("%% A (%d x %d)\n A = \n", M, K);
    print_matrix(M, K, K, 1, h_A);
    printf("%% B (%d x %d)\n B = \n", K, N);
    print_matrix(K, N, 1, K, h_B);
    printf("%% C (%d x %d)\n C = \n", M, N);
    print_matrix(M, N, N, 1, h_C);

    return 0;
}
