#include "sfem_UnionObstacle.hpp"

#include <algorithm>
#include <cstdio>

#include "sfem_union_obstacle_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_UnionObstacle_cuda.hpp"
#endif

namespace sfem {

    namespace {
        // The k-th row of a buffer of rows of n entries.
        void rows_of(const SharedBuffer<real_t> &b, const ptrdiff_t n, const int count, real_t **const out) {
            for (int c = 0; c < count; ++c) out[c] = b->data() + c * n;
        }

        int surface_sampling_unsupported() {
            fprintf(stderr, "UnionObstacle: samples at nodes only (the node_sdf back-end)\n");
            return SFEM_FAILURE;
        }
    }  // namespace

    UnionObstacle::UnionObstacle(const std::vector<std::shared_ptr<Obstacle>> &parts, const ExecutionContext &context)
        : parts_(parts), context_(context) {}

    UnionObstacle::~UnionObstacle() = default;

    void UnionObstacle::ensure(const ptrdiff_t n) {
        if (n == n_) return;
        const ExecutionSpace es = context_.space();
        gap_            = create_buffer<real_t>(std::max<ptrdiff_t>(n, 1), es);
        normals_        = create_buffer<real_t>(3 * std::max<ptrdiff_t>(n, 1), es);
        part_gap_       = create_buffer<real_t>(std::max<ptrdiff_t>(n, 1), es);
        part_normals_   = create_buffer<real_t>(3 * std::max<ptrdiff_t>(n, 1), es);
        part_curvature_ = create_buffer<real_t>(6 * std::max<ptrdiff_t>(n, 1), es);
        which_          = create_buffer<idx_t>(std::max<ptrdiff_t>(n, 1), es);
        n_              = n;
    }

    int UnionObstacle::sample_nodes(const ptrdiff_t                   nnodes,
                                    const idx_t *const SFEM_RESTRICT  idx,
                                    geom_t **const SFEM_RESTRICT      points,
                                    const real_t *const SFEM_RESTRICT disp,
                                    real_t *const SFEM_RESTRICT       gap,
                                    real_t **const SFEM_RESTRICT      normals) {
        ensure(nnodes);
        real_t *g = gap ? gap : gap_->data();
        real_t *n[3];
        if (normals) {
            for (int d = 0; d < 3; ++d) n[d] = normals[d];
        } else {
            rows_of(normals_, nnodes, 3, n);
        }
        real_t *nk[3];
        rows_of(part_normals_, nnodes, 3, nk);
        real_t *const gk    = part_gap_->data();
        idx_t *const  which = which_->data();

        int err = parts_[0]->sample_nodes(nnodes, idx, points, disp, g, n);
#ifdef SFEM_ENABLE_CUDA
        if (context_.space() == EXECUTION_SPACE_DEVICE) {
            d_union_obstacle_reset(nnodes, which, context_.stream());
            for (int k = 1; k < (int)parts_.size(); ++k) {
                err |= parts_[k]->sample_nodes(nnodes, idx, points, disp, gk, nk);
                d_union_obstacle_select(nnodes, k, gk, nk, g, n, which, context_.stream());
            }
            return err;
        }
#endif
        std::fill(which, which + nnodes, idx_t(0));
        for (int k = 1; k < (int)parts_.size(); ++k) {
            err |= parts_[k]->sample_nodes(nnodes, idx, points, disp, gk, nk);
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                ::union_obstacle_select(k, gk[i], nk[0][i], nk[1][i], nk[2][i], &g[i], &n[0][i], &n[1][i], &n[2][i], &which[i]);
            }
        }
        return err;
    }

    int UnionObstacle::sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                              const idx_t *const SFEM_RESTRICT  idx,
                                              geom_t **const SFEM_RESTRICT      points,
                                              const real_t *const SFEM_RESTRICT disp,
                                              real_t **const SFEM_RESTRICT      curvature) {
        // Which part constrains each node at these points.
        int err = sample_nodes(nnodes, idx, points, disp, nullptr, nullptr);

        real_t *ck[6];
        rows_of(part_curvature_, nnodes, 6, ck);
        const idx_t *const which = which_->data();

        err |= parts_[0]->sample_nodes_curvature(nnodes, idx, points, disp, curvature);
        for (int k = 1; k < (int)parts_.size(); ++k) {
            err |= parts_[k]->sample_nodes_curvature(nnodes, idx, points, disp, ck);
#ifdef SFEM_ENABLE_CUDA
            if (context_.space() == EXECUTION_SPACE_DEVICE) {
                d_union_obstacle_select_curvature(nnodes, k, which, ck, curvature, context_.stream());
                continue;
            }
#endif
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                if (which[i] == k) {
                    for (int c = 0; c < 6; ++c) curvature[c][i] = ck[c][i];
                }
            }
        }
        return err;
    }

    int UnionObstacle::sample(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const, real_t *const) {
        return surface_sampling_unsupported();
    }

    int UnionObstacle::sample_normals(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const) {
        return surface_sampling_unsupported();
    }

    int UnionObstacle::sample_value(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) {
        return surface_sampling_unsupported();
    }

    int UnionObstacle::sample_normals_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const) {
        return surface_sampling_unsupported();
    }

    int UnionObstacle::sample_value_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) {
        return surface_sampling_unsupported();
    }

    int UnionObstacle::sample_weights_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) {
        return surface_sampling_unsupported();
    }

}  // namespace sfem
