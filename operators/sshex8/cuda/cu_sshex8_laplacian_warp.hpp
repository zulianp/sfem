

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_laplacian_apply_kernel_warp(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const T *const SFEM_RESTRICT x,
        T *const SFEM_RESTRICT y) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    assert(blockDim.x == BLOCK_SIZE);
    assert(blockDim.y == BLOCK_SIZE);
    assert(blockDim.z == BLOCK_SIZE);

    __shared__ T x_block[BLOCK_SIZE_3];
    __shared__ T y_block[BLOCK_SIZE_3];

#define CU_SSHEX8_WARP_USE_ELEMENTAL_MATRIX
#ifdef CU_SSHEX8_WARP_USE_ELEMENTAL_MATRIX
    T laplacian_matrix[8 * 8];
#endif

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const int lidx = threadIdx.z * BLOCK_SIZE_2 + threadIdx.y * BLOCK_SIZE + threadIdx.x;
        const ptrdiff_t idx = elements[lidx * stride + e];

        x_block[lidx] = x[idx];  // Copy coeffs to shared mem
        y_block[lidx] = 0;       // Reset

#ifdef CU_SSHEX8_WARP_USE_ELEMENTAL_MATRIX
        {
            T sub_fff[6];
            const T h = 1. / LEVEL;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }
#else
        T sub_fff[6];
        {
            const T h = 1. / LEVEL;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
        }
#endif

        const bool is_element = threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        T element_vector[8] = {0};
        if (is_element) {
            // gather
            T element_u[8] = {x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                              x_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)]};

#ifdef CU_SSHEX8_WARP_USE_ELEMENTAL_MATRIX
            for (int i = 0; i < 8; i++) {
                const T *const row = &laplacian_matrix[i * 8];
                const T ui = element_u[i];
                assert(ui == ui);
                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }
#else
            cu_hex8_laplacian_apply_fff_integral(sub_fff, element_u, element_vector);
#endif

            // TODO With stencil version atomics can be avoided
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z)], element_vector[0]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z)], element_vector[1]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
                      element_vector[2]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z)], element_vector[3]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y, threadIdx.z + 1)], element_vector[4]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
                      element_vector[5]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                      element_vector[6]);
            atomicAdd(&y_block[cu_sshex8_lidx(LEVEL, threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)],
                      element_vector[7]);
        }

        const int interior = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0 &&
                             threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        if (interior)
            y[idx] += y_block[lidx];
        else
            atomicAdd(&y[idx], y_block[lidx]);
    }
}

template <typename T, int LEVEL>
static int cu_affine_sshex8_laplacian_apply_warp_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const T *const x,
        T *const y,
        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    static const int BLOCK_SIZE = LEVEL + 1;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_laplacian_apply_kernel_warp<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(
                        nelements, stride, interior_start, elements, fff, x, y);
    } else {
        cu_affine_sshex8_laplacian_apply_kernel_warp<T, LEVEL><<<n_blocks, block_size, 0>>>(
                nelements, stride, interior_start, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
