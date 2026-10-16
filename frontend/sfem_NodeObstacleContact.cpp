#include "sfem_NodeObstacleContact.hpp"

#include <algorithm>
#include <cmath>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_aliases.hpp"
#include "sfem_node_obstacle_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_NodeObstacleContact_cuda.hpp"
#endif

namespace sfem {

#ifdef SFEM_ENABLE_CUDA
    namespace {
        // normals and curvature are flat: rows of n entries.
        NodeObstacleDevice device_view(const ptrdiff_t             n,
                                       const SharedBuffer<idx_t>  &idx,
                                       const SharedBuffer<real_t> &normals,
                                       const SharedBuffer<real_t> &curvature,
                                       const SharedBuffer<real_t> &gap,
                                       const SharedBuffer<real_t> &frames,
                                       const SharedBuffer<real_t> &mu) {
            NodeObstacleDevice c;
            c.size = n;
            c.idx  = idx->data();
            for (int d = 0; d < 3; ++d) c.normals[d] = normals->data() + d * n;
            for (int k = 0; k < 6; ++k) c.curvature[k] = curvature ? curvature->data() + k * n : nullptr;
            c.gap    = gap->data();
            c.frames = frames ? frames->data() : nullptr;
            c.mu     = mu ? mu->data() : nullptr;
            return c;
        }
    }  // namespace
#endif

    NodeObstacleContact::NodeObstacleContact(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                                             const std::shared_ptr<Communicator>        &comm,
                                             const SharedBuffer<idx_t>                  &nodes,
                                             const SharedBuffer<geom_t *>               &points,
                                             const ptrdiff_t                             ndofs,
                                             const int                                   dir,
                                             const ExecutionSpace                        es)
        : nodes_(nodes), idx_(nodes), points_(points), es_(es), ndofs_(ndofs), dir_(dir) {
        obstacle_ = SDFObstacle::create(sdf, comm, points->extent(1), points->data()[2], es);

        for (int d = 0; d < 3; ++d) xyz_[d] = points->data()[d];
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            idx_ = smesh::to_device(nodes);
            for (int d = 0; d < 3; ++d) {
                point_rows_.push_back(smesh::to_device(Buffer<geom_t>::wrap(points->extent(1), points->data()[d])));
                xyz_[d] = point_rows_.back()->data();
            }
        }
#endif

        const ptrdiff_t n = nodes->size();
        normals_          = create_buffer<real_t>(3 * n, es);
        curvature_        = create_buffer<real_t>(6 * n, es);
        gap_              = create_buffer<real_t>(n, es);
        mu_               = create_buffer<real_t>(n, es);
        frames_           = create_buffer<real_t>(9 * n, es);
        ahead_            = create_buffer<real_t>(ndofs, es);
        ahead_gap_        = create_buffer<real_t>(n, es);
        ahead_normals_    = create_buffer<real_t>(3 * n, es);
        complementarity_  = create_buffer<real_t>(n, es);
        penetration_      = create_buffer<real_t>(n, es);
    }

    void NodeObstacleContact::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::update");

        real_t *nn[3];
        rows(normals_, 3, nn);
        obstacle_->sample_nodes(size(), idx_->data(), xyz_, x, gap_->data(), nn);

#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_frames(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), dir_);
            return;
        }
#endif

        real_t *const Q = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            node_obstacle_frame(nn[0][i], nn[1][i], nn[2][i], dir_, &Q[9 * i]);
        }
    }

    void NodeObstacleContact::to_frame(real_t *const v) const {
#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_to_frame(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), v);
            return;
        }
#endif
        const idx_t *const  idx = nodes_->data();
        const real_t *const Q   = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            node_obstacle_to_frame(&Q[9 * i], &v[(ptrdiff_t)idx[i] * 3]);
        }
    }

    void NodeObstacleContact::from_frame(real_t *const v) const {
#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_from_frame(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), v);
            return;
        }
#endif
        const idx_t *const  idx = nodes_->data();
        const real_t *const Q   = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            node_obstacle_from_frame(&Q[9 * i], &v[(ptrdiff_t)idx[i] * 3]);
        }
    }

    void NodeObstacleContact::upper_bound(const real_t unbounded, real_t *const ub) const {
#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_upper_bound(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), ndofs_, unbounded, ub);
            return;
        }
#endif
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            ub[i] = unbounded;
        }
        const idx_t *const  idx = nodes_->data();
        const real_t *const gap = gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            ub[(ptrdiff_t)idx[i] * 3] = gap[i];
        }
    }

    real_t NodeObstacleContact::kkt_residual(const real_t *const r,
                                             const real_t *const k,  // per dof
                                             real_t *const       projected,
                                             real_t *const       penetration) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::kkt_residual");

#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            auto blas = sfem::blas<real_t>(es_);
            blas->copy(ndofs_, r, projected);
            d_node_obstacle_kkt(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_),
                                k,
                                projected,
                                complementarity_->data(),
                                penetration_->data());
            *penetration = d_max_or_zero(size(), penetration_->data());
            const real_t c = blas->dot(size(), complementarity_->data(), complementarity_->data());
            return std::sqrt(blas->dot(ndofs_, projected, projected) + c);
        }
#endif

        const idx_t *const  idx = nodes_->data();
        real_t             *n[3];
        rows(normals_, 3, n);
        const real_t *const gap = gap_->data();
        real_t *const       mus = mu_->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            projected[i] = r[i];
        }

        real_t deepest       = 0;
        real_t complementary = 0;
#pragma omp parallel for reduction(max : deepest) reduction(+ : complementary)
        for (ptrdiff_t i = 0; i < size(); ++i) {
            const ptrdiff_t node  = idx[i];
            const real_t    ni[3] = {n[0][i], n[1][i], n[2][i]};
            real_t          c;
            mus[i] = node_obstacle_kkt(ni, &projected[node * 3], gap[i], k[node * 3], &c);
            complementary += c * c;
            deepest = std::max(deepest, -gap[i]);
        }

        *penetration = deepest;
        real_t norm  = complementary;
#pragma omp parallel for reduction(+ : norm)
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            norm += projected[i] * projected[i];
        }
        return std::sqrt(norm);
    }

    void NodeObstacleContact::add_curvature(const real_t *const  x,
                                            const real_t *const  r,
                                            const real_t *const  k,
                                            const bool           capped,
                                            const real_t         sign,
                                            const count_t *const rowptr,
                                            const idx_t *const   colidx,
                                            real_t *const        values) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::add_curvature");

        real_t *Crows[6];
        rows(curvature_, 6, Crows);
        obstacle_->sample_nodes_curvature(size(), idx_->data(), xyz_, x, Crows);

#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_curvature(
                    device_view(size(), idx_, normals_, curvature_, gap_, frames_, mu_), r, k, capped, sign, rowptr, colidx, values);
            return;
        }
#endif

        const idx_t *const  idx = nodes_->data();
        real_t             *n[3];
        rows(normals_, 3, n);
        real_t *const *const C   = Crows;
        const real_t *const  gap = gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            const ptrdiff_t node  = idx[i];
            const real_t    ni[3] = {n[0][i], n[1][i], n[2][i]};
            const real_t    Ci[6] = {C[0][i], C[1][i], C[2][i], C[3][i], C[4][i], C[5][i]};
            node_obstacle_curvature(ni, Ci, &r[node * 3], gap[i], k[node * 3], capped, sign, bsr_diagonal_block(node, rowptr, colidx, 3, values));
        }
    }

    void NodeObstacleContact::reaction(real_t *const out) const {
        sfem::blas<real_t>(es_)->zeros(ndofs_, out);
#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_reaction(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), out);
            return;
        }
#endif
        const idx_t *const  idx = nodes_->data();
        real_t             *n[3];
        rows(normals_, 3, n);
        const real_t *const mus = mu_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            for (int d = 0; d < 3; ++d) {
                out[(ptrdiff_t)idx[i] * 3 + d] = -mus[i] * n[d][i];
            }
        }
    }

    void NodeObstacleContact::project_velocity(const real_t *const x, real_t *const v, const real_t dt) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::project_velocity");

        sfem::blas<real_t>(es_)->zaxpby(ndofs_, 1, x, dt, v, ahead_->data());
        real_t *an[3];
        rows(ahead_normals_, 3, an);
        obstacle_->sample_nodes(size(), idx_->data(), xyz_, ahead_->data(), ahead_gap_->data(), an);

#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_project_velocity(device_view(size(), idx_, ahead_normals_, nullptr, ahead_gap_, nullptr, nullptr), v, dt);
            return;
        }
#endif

        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = an;
        const real_t *const  gap = ahead_gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            if (gap[i] < 0) {
                real_t *const vi = &v[(ptrdiff_t)idx[i] * 3];
                for (int d = 0; d < 3; ++d) {
                    vi[d] += gap[i] * n[d][i] / dt;
                }
            }
        }
    }

    void NodeObstacleContact::gap_field(real_t *const out) const {
        sfem::blas<real_t>(es_)->zeros(ndofs_, out);
#ifdef SFEM_ENABLE_CUDA
        if (es_ == EXECUTION_SPACE_DEVICE) {
            d_node_obstacle_gap_field(device_view(size(), idx_, normals_, nullptr, gap_, frames_, mu_), out);
            return;
        }
#endif
        const idx_t *const  idx = nodes_->data();
        real_t             *n[3];
        rows(normals_, 3, n);
        const real_t *const gap = gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            for (int d = 0; d < 3; ++d) {
                out[(ptrdiff_t)idx[i] * 3 + d] = gap[i] * n[d][i];
            }
        }
    }

}  // namespace sfem
