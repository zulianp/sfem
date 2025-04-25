#include "cu_hex8_laplacian_inline.hpp"
#include "cu_sshex8_inline.hpp"

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_laplacian_apply_local_mem_kernel(const ptrdiff_t nelements,
                                                               const ptrdiff_t stride,  // Stride for elements and fff
                                                               const ptrdiff_t interior_start,
                                                               const idx_t *const SFEM_RESTRICT         elements,
                                                               const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                               const T *const SFEM_RESTRICT             x,
                                                               T *const SFEM_RESTRICT                   y) {
#ifndef NDEBUG
    const int nxe = cu_sshex8_nxe(LEVEL);
#endif

    static const int BLOCK_SIZE   = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    // Uses "local" memory
    T x_block[BLOCK_SIZE_3];
    T y_block[BLOCK_SIZE_3];

#define CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_VOLGEN_USE_ELEMENTAL_MATRIX
#ifdef CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_VOLGEN_USE_ELEMENTAL_MATRIX
    T laplacian_matrix[8 * 8];
#endif

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
#ifdef CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_VOLGEN_USE_ELEMENTAL_MATRIX
        // Build operator
        {
            T       sub_fff[6];
            const T h = (T)1. / LEVEL;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }
#else
        const T h = (T)1. / LEVEL;
        T       sub_fff[6];
        cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
#endif

        // Gather
        cu_sshex8_gather<T, LEVEL>(nelements, stride, e, elements, 1, x, x_block);

        // Reset
        for (int i = 0; i < BLOCK_SIZE_3; i++) {
            y_block[i] = 0;
        }

        // Compute
        for (int zi = 0; zi < LEVEL; zi++) {
            for (int yi = 0; yi < LEVEL; yi++) {
                for (int xi = 0; xi < LEVEL; xi++) {
                    assert(cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1) < BLOCK_SIZE_3);

                    int ev[8] = {cu_sshex8_lidx(LEVEL, xi, yi, zi),
                                 cu_sshex8_lidx(LEVEL, xi + 1, yi, zi),
                                 cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi),
                                 cu_sshex8_lidx(LEVEL, xi, yi + 1, zi),
                                 cu_sshex8_lidx(LEVEL, xi, yi, zi + 1),
                                 cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1),
                                 cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1),
                                 cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1)};

                    T element_u[8];
                    for (int v = 0; v < 8; v++) {
                        element_u[v] = x_block[ev[v]];
                    }

#ifdef CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_VOLGEN_USE_ELEMENTAL_MATRIX
                    T element_vector[8] = {0};
                    for (int c = 0; c < 8; c++) {
                        const T *const col = &laplacian_matrix[c * 8];
                        const T        u_c = element_u[c];
                        assert(u_c == u_c);
                        for (int r = 0; r < 8; r++) {
                            assert(col[r] == col[r]);
                            element_vector[r] += u_c * col[r];
                        }
                    }

#else
                    T element_vector[8];
                    cu_hex8_laplacian_apply_fff_integral(sub_fff, element_u, element_vector);
#endif

                    for (int v = 0; v < 8; v++) {
                        y_block[ev[v]] += element_vector[v];
                    }
                }
            }
        }

        // Scatter
        cu_sshex8_scatter_add<T, LEVEL>(nelements, stride, e, elements, y_block, 1, y);
    }
}

template <typename T, int LEVEL>
static int acu_affine_sshex8_laplacian_apply_local_mem_tpl(const ptrdiff_t nelements,
                                                       const ptrdiff_t stride,          // Stride for elements and fff
                                                       const ptrdiff_t interior_start,  // Stride for elements and fff
                                                       const idx_t *const SFEM_RESTRICT         elements,
                                                       const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                       const T *const                           x,
                                                       T *const                                 y,
                                                       void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_affine_sshex8_laplacian_apply_local_mem_kernel<T, LEVEL>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_laplacian_apply_local_mem_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements, stride, interior_start, elements, fff, x, y);
    } else {
        cu_affine_sshex8_laplacian_apply_local_mem_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements, stride, interior_start, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
