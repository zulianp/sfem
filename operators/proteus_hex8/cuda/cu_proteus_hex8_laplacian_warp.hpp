

template <typename real_t, int LEVEL>
__global__ void cu_proteus_affine_hex8_laplacian_apply_kernel_warp(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const real_t *const SFEM_RESTRICT x,
        real_t *const SFEM_RESTRICT y) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    __shared__ real_t x_block[BLOCK_SIZE_3];
    __shared__ real_t y_block[BLOCK_SIZE_3];
    scalar_t laplacian_matrix[8 * 8];

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const int lidx = threadIdx.z * BLOCK_SIZE_2 + threadIdx.y * BLOCK_SIZE + threadIdx.x;
        const ptrdiff_t idx = elements[lidx * stride + e];

        x_block[lidx] = x[idx];  // Copy coeffs to shared mem
        y_block[lidx] = 0;       // Reset

        {
            scalar_t sub_fff[6];
            const scalar_t h = 1. / LEVEL;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }

        const bool is_element = threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        scalar_t element_vector[8] = {0};
        if (is_element) {
            // gather
            scalar_t element_u[8] = {x_block[B_(threadIdx.x, threadIdx.y, threadIdx.z)],
                                     x_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)],
                                     x_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
                                     x_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)],
                                     x_block[B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)],
                                     x_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
                                     x_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                                     x_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)]};

            for (int i = 0; i < 8; i++) {
                const scalar_t *const row = &laplacian_matrix[i * 8];
                const scalar_t ui = element_u[i];
                assert(ui == ui);
                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }

            // TODO With stencil version atomics can be avoided
            atomicAdd(&y_block[B_(threadIdx.x, threadIdx.y, threadIdx.z)], element_vector[0]);
            atomicAdd(&y_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)], element_vector[1]);
            atomicAdd(&y_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)],
                      element_vector[2]);
            atomicAdd(&y_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)], element_vector[3]);
            atomicAdd(&y_block[B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)], element_vector[4]);
            atomicAdd(&y_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)],
                      element_vector[5]);
            atomicAdd(&y_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                      element_vector[6]);
            atomicAdd(&y_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)],
                      element_vector[7]);
        }

        int interior = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0 &&
                       threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        __syncthreads();  //

        if (interior)
            y[idx] += y_block[lidx];
        else
            atomicAdd(&y[idx], y_block[lidx]);
    }
}

template <typename T, int LEVEL>
static int cu_proteus_affine_hex8_laplacian_apply_warp_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,          // Stride for elements and fff
        const ptrdiff_t interior_start,  // Stride for elements and fff
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
        cu_proteus_affine_hex8_laplacian_apply_kernel_warp<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(
                        nelements, stride, interior_start, elements, fff, x, y);
    } else {
        cu_proteus_affine_hex8_laplacian_apply_kernel_warp<T, LEVEL><<<n_blocks, block_size, 0>>>(
                nelements, stride, interior_start, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
