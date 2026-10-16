#include "sfem_NodeObstacleContact_cuda.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_cuda_max.hpp"
#include "sfem_node_obstacle_kernels.hpp"

namespace sfem {

    namespace {
        const int block_size = 128;

        ptrdiff_t blocks(const ptrdiff_t n) { return n > 0 ? (n + block_size - 1) / block_size : 1; }

#define SFEM_NODE_LOOP(i, n) for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

        __global__ void frames_kernel(const NodeObstacleDevice c, const int dir) {
            SFEM_NODE_LOOP(i, c.size) { node_obstacle_frame(c.normals[0][i], c.normals[1][i], c.normals[2][i], dir, &c.frames[9 * i]); }
        }

        __global__ void to_frame_kernel(const NodeObstacleDevice c, real_t *const v) {
            SFEM_NODE_LOOP(i, c.size) { node_obstacle_to_frame(&c.frames[9 * i], &v[(ptrdiff_t)c.idx[i] * 3]); }
        }

        __global__ void from_frame_kernel(const NodeObstacleDevice c, real_t *const v) {
            SFEM_NODE_LOOP(i, c.size) { node_obstacle_from_frame(&c.frames[9 * i], &v[(ptrdiff_t)c.idx[i] * 3]); }
        }

        __global__ void fill_kernel(const ptrdiff_t n, const real_t value, real_t *const x) {
            SFEM_NODE_LOOP(i, n) { x[i] = value; }
        }

        __global__ void upper_bound_kernel(const NodeObstacleDevice c, real_t *const ub) {
            SFEM_NODE_LOOP(i, c.size) { ub[(ptrdiff_t)c.idx[i] * 3] = c.gap[i]; }
        }

        __global__ void kkt_kernel(const NodeObstacleDevice c,
                                   const real_t *const      k,
                                   real_t *const            projected,
                                   real_t *const            complementarity,
                                   real_t *const            penetration) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    n[3] = {c.normals[0][i], c.normals[1][i], c.normals[2][i]};
                real_t          ci;
                c.mu[i]            = node_obstacle_kkt(n, &projected[node * 3], c.gap[i], k[node * 3], &ci);
                complementarity[i] = ci;
                penetration[i]     = -c.gap[i];
            }
        }

        __global__ void curvature_kernel(const NodeObstacleDevice c,
                                         const real_t *const      r,
                                         const real_t *const      k,
                                         const bool               capped,
                                         const real_t             sign,
                                         const count_t *const     rowptr,
                                         const idx_t *const       colidx,
                                         real_t *const            values) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    n[3] = {c.normals[0][i], c.normals[1][i], c.normals[2][i]};
                const real_t    C[6] = {c.curvature[0][i], c.curvature[1][i], c.curvature[2][i], c.curvature[3][i], c.curvature[4][i], c.curvature[5][i]};
                // The node's diagonal block.
                real_t *B = nullptr;
                for (count_t kk = rowptr[node]; kk < rowptr[node + 1]; ++kk) {
                    if (colidx[kk] == node) B = &values[kk * 9];
                }
                if (B) node_obstacle_curvature(n, C, &r[node * 3], c.gap[i], k[node * 3], capped, sign, B);
            }
        }

        __global__ void reaction_kernel(const NodeObstacleDevice c, real_t *const out) {
            SFEM_NODE_LOOP(i, c.size) {
                for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = -c.mu[i] * c.normals[d][i];
            }
        }

        __global__ void gap_field_kernel(const NodeObstacleDevice c, real_t *const out) {
            SFEM_NODE_LOOP(i, c.size) {
                for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = c.gap[i] * c.normals[d][i];
            }
        }

        __global__ void project_velocity_kernel(const NodeObstacleDevice c, real_t *const v, const real_t dt) {
            SFEM_NODE_LOOP(i, c.size) {
                if (c.gap[i] < 0) {
                    for (int d = 0; d < 3; ++d) v[(ptrdiff_t)c.idx[i] * 3 + d] += c.gap[i] * c.normals[d][i] / dt;
                }
            }
        }
#undef SFEM_NODE_LOOP
    }  // namespace

    void d_node_obstacle_frames(const NodeObstacleDevice &c, const int dir) {
        frames_kernel<<<blocks(c.size), block_size>>>(c, dir);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_to_frame(const NodeObstacleDevice &c, real_t *const v) {
        to_frame_kernel<<<blocks(c.size), block_size>>>(c, v);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_from_frame(const NodeObstacleDevice &c, real_t *const v) {
        from_frame_kernel<<<blocks(c.size), block_size>>>(c, v);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_upper_bound(const NodeObstacleDevice &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub) {
        fill_kernel<<<blocks(ndofs), block_size>>>(ndofs, unbounded, ub);
        upper_bound_kernel<<<blocks(c.size), block_size>>>(c, ub);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_kkt(const NodeObstacleDevice &c,
                             const real_t *const       k,
                             real_t *const             projected,
                             real_t *const             complementarity,
                             real_t *const             penetration) {
        kkt_kernel<<<blocks(c.size), block_size>>>(c, k, projected, complementarity, penetration);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_curvature(const NodeObstacleDevice &c,
                                   const real_t *const       r,
                                   const real_t *const       k,
                                   const bool                capped,
                                   const real_t              sign,
                                   const count_t *const      rowptr,
                                   const idx_t *const        colidx,
                                   real_t *const             values) {
        curvature_kernel<<<blocks(c.size), block_size>>>(c, r, k, capped, sign, rowptr, colidx, values);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_reaction(const NodeObstacleDevice &c, real_t *const out) {
        reaction_kernel<<<blocks(c.size), block_size>>>(c, out);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_gap_field(const NodeObstacleDevice &c, real_t *const out) {
        gap_field_kernel<<<blocks(c.size), block_size>>>(c, out);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    void d_node_obstacle_project_velocity(const NodeObstacleDevice &c, real_t *const v, const real_t dt) {
        project_velocity_kernel<<<blocks(c.size), block_size>>>(c, v, dt);
        SFEM_DEBUG_SYNCHRONIZE();
    }

    real_t d_max_or_zero(const ptrdiff_t n, const real_t *const x) {
        return d_max(n, x, real_t(0));
    }

}  // namespace sfem
