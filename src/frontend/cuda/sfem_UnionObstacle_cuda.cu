#include "sfem_UnionObstacle_cuda.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_union_obstacle_kernels.hpp"

namespace sfem {
    namespace {
        const int block_size = 128;

        ptrdiff_t blocks_for(const ptrdiff_t n) { return n > 0 ? (n + block_size - 1) / block_size : 1; }

#define SFEM_UNION_LOOP(i, n) for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

        // Row pointers by value: the kernels cannot read the host's row arrays.
        struct Rows3 {
            real_t *r[3];
        };

        struct Rows6 {
            real_t *r[6];
        };

        __global__ void reset_kernel(const ptrdiff_t n, idx_t *const which) {
            SFEM_UNION_LOOP(i, n) { which[i] = 0; }
        }

        __global__ void select_kernel(const ptrdiff_t     n,
                                      const int           k,
                                      const real_t *const gk,
                                      const Rows3         nk,
                                      real_t *const       g,
                                      const Rows3         normals,
                                      idx_t *const        which) {
            SFEM_UNION_LOOP(i, n) {
                ::union_obstacle_select(
                        k, gk[i], nk.r[0][i], nk.r[1][i], nk.r[2][i], &g[i], &normals.r[0][i], &normals.r[1][i], &normals.r[2][i], &which[i]);
            }
        }

        __global__ void select_curvature_kernel(const ptrdiff_t n, const int k, const idx_t *const which, const Rows6 ck, const Rows6 curvature) {
            SFEM_UNION_LOOP(i, n) {
                if (which[i] == k) {
                    for (int c = 0; c < 6; ++c) curvature.r[c][i] = ck.r[c][i];
                }
            }
        }

#undef SFEM_UNION_LOOP
    }  // namespace

    void d_union_obstacle_reset(const ptrdiff_t n, idx_t *const which, void *const stream) {
        reset_kernel<<<blocks_for(n), block_size, 0, (cudaStream_t)stream>>>(n, which);
        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    void d_union_obstacle_select(const ptrdiff_t      n,
                                 const int            k,
                                 const real_t *const  gk,
                                 real_t *const *const nk,
                                 real_t *const        g,
                                 real_t *const *const normals,
                                 idx_t *const         which,
                                 void *const          stream) {
        const Rows3 rk = {{nk[0], nk[1], nk[2]}};
        const Rows3 rn = {{normals[0], normals[1], normals[2]}};
        select_kernel<<<blocks_for(n), block_size, 0, (cudaStream_t)stream>>>(n, k, gk, rk, g, rn, which);
        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    void d_union_obstacle_select_curvature(const ptrdiff_t      n,
                                           const int            k,
                                           const idx_t *const   which,
                                           real_t *const *const ck,
                                           real_t *const *const curvature,
                                           void *const          stream) {
        Rows6 rk, rc;
        for (int c = 0; c < 6; ++c) {
            rk.r[c] = ck[c];
            rc.r[c] = curvature[c];
        }
        select_curvature_kernel<<<blocks_for(n), block_size, 0, (cudaStream_t)stream>>>(n, k, which, rk, rc);
        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

}  // namespace sfem
