#include "sfem_NodeObstacleView.hpp"

#include <algorithm>

#include "sfem_BSR.hpp"
#include "sfem_node_obstacle_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_NodeObstacleView_cuda.hpp"
#endif

namespace sfem {

    namespace {
        void normal_at(const NodeObstacleView &c, const ptrdiff_t i, real_t *const n) {
            for (int d = 0; d < 3; ++d) n[d] = c.normals[d][i];
        }
    }  // namespace

    void node_obstacle_frames(const NodeObstacleView &c, const int dir) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_frames(c, dir);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            ::node_obstacle_frame(c.normals[0][i], c.normals[1][i], c.normals[2][i], dir, &c.frames[9 * i]);
        }
    }

    void node_obstacle_to_frame(const NodeObstacleView &c, real_t *const v) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_to_frame(c, v);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            ::node_obstacle_to_frame(&c.frames[9 * i], &v[(ptrdiff_t)c.idx[i] * 3]);
        }
    }

    void node_obstacle_from_frame(const NodeObstacleView &c, real_t *const v) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_from_frame(c, v);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            ::node_obstacle_from_frame(&c.frames[9 * i], &v[(ptrdiff_t)c.idx[i] * 3]);
        }
    }

    void node_obstacle_upper_bound(const NodeObstacleView &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_upper_bound(c, ndofs, unbounded, ub);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            ub[i] = unbounded;
        }
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            ub[(ptrdiff_t)c.idx[i] * 3] = c.gap[i];
        }
    }

    void node_obstacle_kkt(const NodeObstacleView &c,
                           const real_t *const     k,
                           real_t *const           projected,
                           real_t *const           complementarity,
                           real_t *const           penetration) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_kkt(c, k, projected, complementarity, penetration);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            const ptrdiff_t node = c.idx[i];
            real_t          n[3];
            normal_at(c, i, n);
            c.mu[i]        = ::node_obstacle_kkt(n, &projected[node * 3], c.gap[i], k[node * 3], &complementarity[i]);
            penetration[i] = -c.gap[i];
        }
    }

    void node_obstacle_curvature_bsr(const NodeObstacleView &c,
                                     const real_t *const     r,
                                     const real_t *const     k,
                                     const bool              capped,
                                     const real_t            sign,
                                     const count_t *const    rowptr,
                                     const idx_t *const      colidx,
                                     real_t *const           values) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_curvature_bsr(c, r, k, capped, sign, rowptr, colidx, values);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            const ptrdiff_t node = c.idx[i];
            real_t          n[3];
            normal_at(c, i, n);
            const real_t C[6] = {c.curvature[0][i], c.curvature[1][i], c.curvature[2][i], c.curvature[3][i], c.curvature[4][i], c.curvature[5][i]};
            real_t *const B   = bsr_diagonal_block(node, rowptr, colidx, 3, values);
            if (B) ::node_obstacle_curvature(n, C, &r[node * 3], c.gap[i], k[node * 3], capped, sign, B);
        }
    }

    void node_obstacle_curvature_blocks(const NodeObstacleView &c,
                                        const real_t *const     r,
                                        const real_t *const     k,
                                        const bool              capped,
                                        real_t *const           blocks) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_curvature_blocks(c, r, k, capped, blocks);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            const ptrdiff_t node = c.idx[i];
            real_t          n[3];
            normal_at(c, i, n);
            const real_t C[6] = {c.curvature[0][i], c.curvature[1][i], c.curvature[2][i], c.curvature[3][i], c.curvature[4][i], c.curvature[5][i]};
            real_t       B[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            ::node_obstacle_curvature(n, C, &r[node * 3], c.gap[i], k[node * 3], capped, 1, B);
            ::node_obstacle_pack_sym(B, &blocks[6 * i]);
        }
    }

    void node_obstacle_reaction(const NodeObstacleView &c, real_t *const out) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_reaction(c, out);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = -c.mu[i] * c.normals[d][i];
        }
    }

    void node_obstacle_gap_field(const NodeObstacleView &c, real_t *const out) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_gap_field(c, out);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            for (int d = 0; d < 3; ++d) out[(ptrdiff_t)c.idx[i] * 3 + d] = c.gap[i] * c.normals[d][i];
        }
    }

    void node_obstacle_project_velocity(const NodeObstacleView &c, real_t *const v, const real_t dt) {
#ifdef SFEM_ENABLE_CUDA
        if (c.es == EXECUTION_SPACE_DEVICE) return d_node_obstacle_project_velocity(c, v, dt);
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < c.size; ++i) {
            if (c.gap[i] < 0) {
                for (int d = 0; d < 3; ++d) v[(ptrdiff_t)c.idx[i] * 3 + d] += c.gap[i] * c.normals[d][i] / dt;
            }
        }
    }

    real_t max_or_zero(const ExecutionSpace es, const ptrdiff_t n, const real_t *const x) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return d_max_or_zero(n, x);
#else
        (void)es;
#endif
        real_t m = 0;
#pragma omp parallel for reduction(max : m)
        for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, x[i]);
        return m;
    }

}  // namespace sfem
