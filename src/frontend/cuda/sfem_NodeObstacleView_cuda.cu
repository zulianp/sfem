#include "sfem_NodeObstacleView_cuda.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_TangentValues.hpp"
#include "sfem_max_cuda.hpp"
#include "sfem_node_obstacle_kernels.hpp"

namespace sfem {

    namespace {
        const int block_size = 128;

        ptrdiff_t blocks_for(const ptrdiff_t n) { return n > 0 ? (n + block_size - 1) / block_size : 1; }

#define SFEM_NODE_LOOP(i, n) for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

        __global__ void frames_kernel(const NodeObstacleView c, const int dir) {
            SFEM_NODE_LOOP(i, c.size) { ::node_obstacle_frame(c.normals[0][i], c.normals[1][i], c.normals[2][i], dir, &c.frames[9 * i]); }
        }

        __global__ void to_frame_kernel(const NodeObstacleView c, const real_t *const s, const real_t *const in, real_t *const out) {
            SFEM_NODE_LOOP(node, c.n_nodes) {
                ::node_obstacle_scaled_to_frame(c.frame_of[node] < 0 ? nullptr : &c.frames[9 * (ptrdiff_t)c.frame_of[node]], s ? &s[node * 3] : nullptr, &in[node * 3], &out[node * 3]);
            }
        }

        __global__ void from_frame_kernel(const NodeObstacleView c, const real_t *const s, const real_t *const in, real_t *const out) {
            SFEM_NODE_LOOP(node, c.n_nodes) {
                ::node_obstacle_scaled_from_frame(c.frame_of[node] < 0 ? nullptr : &c.frames[9 * (ptrdiff_t)c.frame_of[node]], s ? &s[node * 3] : nullptr, &in[node * 3], &out[node * 3]);
            }
        }

        __global__ void fill_kernel(const ptrdiff_t n, const real_t value, real_t *const x) {
            SFEM_NODE_LOOP(i, n) { x[i] = value; }
        }

        __global__ void upper_bound_kernel(const NodeObstacleView c, real_t *const ub) {
            SFEM_NODE_LOOP(i, c.size) { ub[(ptrdiff_t)c.idx[i] * 3] = c.gap[i]; }
        }

        __global__ void node_mean_kernel(const NodeObstacleView c, real_t *const d) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    mean = (d[node * 3] + d[node * 3 + 1] + d[node * 3 + 2]) / 3;
                for (int k = 0; k < 3; ++k) d[node * 3 + k] = mean;
            }
        }

        __global__ void scaling_kernel(const ptrdiff_t     ndofs,
                                       const real_t *const ub,
                                       real_t *const       inverse_scale,
                                       real_t *const       scale,
                                       real_t *const       scaled_bound) {
            SFEM_NODE_LOOP(i, ndofs) {
                const real_t d    = inverse_scale[i];
                const real_t root = d > 0 ? sqrt(d) : real_t(1);
                scale[i]          = 1 / root;
                inverse_scale[i]  = root;
                scaled_bound[i]   = ub[i] * root;
            }
        }

        __global__ void offset_gap_kernel(const ptrdiff_t     size,
                                          const real_t        shift,
                                          const real_t        cutoff,
                                          const real_t        release,
                                          const real_t *const p0,
                                          real_t *const       gap,
                                          real_t *const       n0,
                                          real_t *const       n1,
                                          real_t *const       n2) {
            SFEM_NODE_LOOP(i, size) {
                ::node_obstacle_offset_gap(shift, cutoff, p0 ? release * p0[i] : real_t(0), &gap[i], &n0[i], &n1[i], &n2[i]);
            }
        }

        __global__ void kkt_kernel(const NodeObstacleView c,
                                   const real_t *const      k,
                                   real_t *const            projected,
                                   real_t *const            complementarity,
                                   real_t *const            penetration) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    n[3] = {c.normals[0][i], c.normals[1][i], c.normals[2][i]};
                real_t          ci;
                c.mu[i]            = ::node_obstacle_kkt(n, &projected[node * 3], c.gap[i], k[node * 3], &ci);
                complementarity[i] = ci;
                penetration[i]     = -c.gap[i];
            }
        }

        template <typename V>
        __global__ void curvature_kernel(const NodeObstacleView c,
                                         const real_t *const      r,
                                         const real_t *const      k,
                                         const int                curvature,
                                         const real_t             sign,
                                         const count_t *const     rowptr,
                                         const idx_t *const       colidx,
                                         V *const                 values) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    n[3] = {c.normals[0][i], c.normals[1][i], c.normals[2][i]};
                const real_t    C[6] = {c.curvature[0][i], c.curvature[1][i], c.curvature[2][i], c.curvature[3][i], c.curvature[4][i], c.curvature[5][i]};
                // The node's diagonal block.
                V *B = nullptr;
                for (count_t kk = rowptr[node]; kk < rowptr[node + 1]; ++kk) {
                    if (colidx[kk] == node) B = &values[kk * 9];
                }
                if (B) ::node_obstacle_curvature(n, C, &r[node * 3], c.gap[i], k[node * 3], curvature, sign, B);
            }
        }

        __global__ void curvature_blocks_kernel(const NodeObstacleView c,
                                                const real_t *const    r,
                                                const real_t *const    k,
                                                const int              curvature,
                                                real_t *const          blocks) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                const real_t    n[3] = {c.normals[0][i], c.normals[1][i], c.normals[2][i]};
                const real_t    C[6] = {c.curvature[0][i], c.curvature[1][i], c.curvature[2][i], c.curvature[3][i], c.curvature[4][i], c.curvature[5][i]};
                real_t          B[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                ::node_obstacle_curvature(n, C, &r[node * 3], c.gap[i], k[node * 3], curvature, 1, B);
                ::node_obstacle_pack_sym(B, &blocks[6 * i]);
            }
        }

        __global__ void apply_blocks_kernel(const NodeObstacleView c, const real_t *const blocks, const real_t *const in, real_t *const out) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                ::node_obstacle_packed_sym_apply(&blocks[6 * i], &in[node * 3], &out[node * 3]);
            }
        }

        __global__ void extra_kernel(const NodeObstacleView c,
                                     const real_t *const    blocks,
                                     const real_t           shift,
                                     const real_t *const    reference,
                                     const real_t *const    in,
                                     real_t *const          out) {
            SFEM_NODE_LOOP(node, c.n_nodes) {
                real_t              o[3] = {0, 0, 0};
                const real_t *const x    = &in[node * 3];
                const ptrdiff_t     i    = c.frame_of[node];
                if (blocks && i >= 0) ::node_obstacle_packed_sym_apply(&blocks[6 * i], x, o);
                if (shift != 0)
                    for (int d = 0; d < 3; ++d) o[d] += shift * (reference[node * 3 + d] * x[d]);
                for (int d = 0; d < 3; ++d) out[node * 3 + d] = o[d];
            }
        }

        __global__ void blocks_diagonal_kernel(const NodeObstacleView c, const real_t *const blocks, real_t *const d) {
            SFEM_NODE_LOOP(i, c.size) {
                const ptrdiff_t node = c.idx[i];
                d[node * 3] += blocks[6 * i];
                d[node * 3 + 1] += blocks[6 * i + 3];
                d[node * 3 + 2] += blocks[6 * i + 5];
            }
        }

        __global__ void reaction_kernel(const NodeObstacleView c, real_t *const out) {
            SFEM_NODE_LOOP(i, c.size) {
                for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = -c.mu[i] * c.normals[d][i];
            }
        }

        __global__ void gap_field_kernel(const NodeObstacleView c, real_t *const out) {
            SFEM_NODE_LOOP(i, c.size) {
                for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = c.gap[i] * c.normals[d][i];
            }
        }

        __global__ void project_velocity_kernel(const NodeObstacleView c, real_t *const v, const real_t dt) {
            SFEM_NODE_LOOP(i, c.size) {
                if (c.gap[i] < 0) {
                    for (int d = 0; d < 3; ++d) v[(ptrdiff_t)c.idx[i] * 3 + d] += c.gap[i] * c.normals[d][i] / dt;
                }
            }
        }
#undef SFEM_NODE_LOOP
    }  // namespace

    void d_node_obstacle_frames(const NodeObstacleView &c, const int dir) {
        frames_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, dir);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_to_frame(const NodeObstacleView &c, const real_t *const s, const real_t *const in, real_t *const out) {
        to_frame_kernel<<<blocks_for(c.n_nodes), block_size, 0, (cudaStream_t)c.stream>>>(c, s, in, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_from_frame(const NodeObstacleView &c, const real_t *const s, const real_t *const in, real_t *const out) {
        from_frame_kernel<<<blocks_for(c.n_nodes), block_size, 0, (cudaStream_t)c.stream>>>(c, s, in, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_upper_bound(const NodeObstacleView &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub) {
        fill_kernel<<<blocks_for(ndofs), block_size, 0, (cudaStream_t)c.stream>>>(ndofs, unbounded, ub);
        upper_bound_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, ub);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_scaling(const NodeObstacleView &c,
                                 const ptrdiff_t         ndofs,
                                 const real_t *const     ub,
                                 real_t *const           inverse_scale,
                                 real_t *const           scale,
                                 real_t *const           scaled_bound) {
        if (c.size > 0) node_mean_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, inverse_scale);
        scaling_kernel<<<blocks_for(ndofs), block_size, 0, (cudaStream_t)c.stream>>>(ndofs, ub, inverse_scale, scale, scaled_bound);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_offset_gap(const ptrdiff_t      size,
                                    const real_t         shift,
                                    const real_t         cutoff,
                                    const real_t         release,
                                    const real_t *const  p0,
                                    real_t *const        gap,
                                    real_t *const *const normals,
                                    void *const          stream) {
        offset_gap_kernel<<<blocks_for(size), block_size, 0, (cudaStream_t)stream>>>(
                size, shift, cutoff, release, p0, gap, normals[0], normals[1], normals[2]);
        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    void d_node_obstacle_kkt(const NodeObstacleView &c,
                             const real_t *const       k,
                             real_t *const             projected,
                             real_t *const             complementarity,
                             real_t *const             penetration) {
        kkt_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, k, projected, complementarity, penetration);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_curvature_bsr(const NodeObstacleView &c,
                                   const real_t *const       r,
                                   const real_t *const       k,
                                   const int                 curvature,
                                   const real_t              sign,
                                   const count_t *const      rowptr,
                                   const idx_t *const        colidx,
                                   const smesh::PrimitiveType values_type,
                                   void *const               values) {
        with_tangent_values(values_type, values, [&](auto *const v) {
            curvature_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(
                    c, r, k, curvature, sign, rowptr, colidx, v);
            SFEM_DEBUG_SYNCHRONIZE(c.stream);
            return SFEM_SUCCESS;
        });
    }

    void d_node_obstacle_curvature_blocks(const NodeObstacleView &c,
                                          const real_t *const     r,
                                          const real_t *const     k,
                                          const int               curvature,
                                          real_t *const           blocks) {
        curvature_blocks_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, r, k, curvature, blocks);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_apply_blocks(const NodeObstacleView &c, const real_t *const blocks, const real_t *const in, real_t *const out) {
        apply_blocks_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, blocks, in, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_extra(const NodeObstacleView &c,
                               const real_t *const     blocks,
                               const real_t            shift,
                               const real_t *const     reference,
                               const real_t *const     in,
                               real_t *const           out) {
        extra_kernel<<<blocks_for(c.n_nodes), block_size, 0, (cudaStream_t)c.stream>>>(c, blocks, shift, reference, in, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_blocks_diagonal(const NodeObstacleView &c, const real_t *const blocks, real_t *const d) {
        blocks_diagonal_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, blocks, d);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_reaction(const NodeObstacleView &c, real_t *const out) {
        reaction_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_gap_field(const NodeObstacleView &c, real_t *const out) {
        gap_field_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, out);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    void d_node_obstacle_project_velocity(const NodeObstacleView &c, real_t *const v, const real_t dt) {
        project_velocity_kernel<<<blocks_for(c.size), block_size, 0, (cudaStream_t)c.stream>>>(c, v, dt);
        SFEM_DEBUG_SYNCHRONIZE(c.stream);
    }

    real_t d_max_or_zero(const ptrdiff_t n, const real_t *const x, void *const stream) { return d_max(n, x, real_t(0), stream); }

}  // namespace sfem
