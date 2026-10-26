#include "sfem_NodeObstacleContact.hpp"

#ifdef SFEM_ENABLE_SRESAMPLE
#include "sfem_SDFObstacle.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>

#include "sfem_blas_factory.hpp"
#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif
#include "sfem_NodeObstacleView.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    // normals and curvature are flat: rows of n entries.
    NodeObstacleView NodeObstacleContact::view(const SharedBuffer<real_t> &normals, const SharedBuffer<real_t> &gap) const {
        NodeObstacleView c;
        c.es       = context_.space();
        c.stream   = context_.stream();
        c.size = size();
        c.idx  = idx_->data();
        c.n_nodes  = ndofs_ / 3;
        c.frame_of = frame_of_->data();
        for (int d = 0; d < 3; ++d) c.normals[d] = normals->data() + d * size();
        for (int k = 0; k < 6; ++k) c.curvature[k] = curvature_->data() + k * size();
        c.gap    = gap->data();
        c.frames = frames_->data();
        c.mu     = mu_->data();
        return c;
    }

    NodeObstacleContact::NodeObstacleContact(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                                             const std::shared_ptr<Communicator>        &comm,
                                             const SharedBuffer<idx_t>                  &nodes,
                                             const SharedBuffer<geom_t *>               &points,
                                             const ptrdiff_t                             ndofs,
                                             const ptrdiff_t                             owned_dofs,
                                             const int                                   dir,
                                             const ExecutionContext                     &context)
        : nodes_(nodes),
          idx_(nodes),
          points_(points),
          context_(context),
          comm_(comm),
          ndofs_(ndofs),
          owned_dofs_(owned_dofs),
          dir_(dir) {
        const ExecutionSpace es = context.space();
#ifdef SFEM_ENABLE_SRESAMPLE
        obstacle_               = SDFObstacle::create(sdf, comm, points->extent(1), points->data()[2], context);
#else
        SFEM_UNUSED(sdf);
        SFEM_ERROR("NodeObstacleContact: SDF obstacles need sfem built with SFEM_ENABLE_SRESAMPLE\n");
#endif

        for (int d = 0; d < 3; ++d) xyz_[d] = points->data()[d];
        frame_of_ = create_host_buffer<idx_t>(ndofs / 3);
        std::fill(frame_of_->data(), frame_of_->data() + ndofs / 3, idx_t(-1));
        for (ptrdiff_t i = 0; i < (ptrdiff_t)nodes->size(); ++i) frame_of_->data()[nodes->data()[i]] = (idx_t)i;
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            idx_      = smesh::to_device(nodes);
            frame_of_ = smesh::to_device(frame_of_);
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
        offset(gap_, normals_);
        node_obstacle_frames(view(normals_, gap_), dir_);
    }

    void NodeObstacleContact::offset(const SharedBuffer<real_t> &gap, const SharedBuffer<real_t> &normals) const {
        real_t *nn[3];
        rows(normals, 3, nn);
        node_obstacle_offset_gap(context_, size(), shift_, cutoff_, 1 - release_, p0_ ? p0_->data() : nullptr, gap->data(), nn);
    }

    void NodeObstacleContact::capture_penetration(const real_t *const x) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::capture_penetration");

        // The penetration of x with everything admitted so far dropped: what remains to release.
        p0_ = nullptr;
        update(x);
        // Once, on the host: p0 = max(0, -gap); unreachable gaps are large and positive, giving 0.
        auto gaps = gap_;
#ifdef SFEM_ENABLE_CUDA
        if (gaps->mem_space() == MEMORY_SPACE_DEVICE) gaps = smesh::to_host(gaps);
#endif
        auto p0 = create_host_buffer<real_t>(size());
        for (ptrdiff_t i = 0; i < size(); ++i) {
            p0->data()[i] = std::max(real_t(0), -gaps->data()[i]);
        }
#ifdef SFEM_ENABLE_CUDA
        if (context_.space() == EXECUTION_SPACE_DEVICE) p0 = smesh::to_device(p0);
#endif
        p0_ = p0;
        update(x);
    }

    void NodeObstacleContact::to_frame(const real_t *const s, const real_t *const in, real_t *const out) const {
        node_obstacle_to_frame(view(normals_, gap_), s, in, out);
    }

    void NodeObstacleContact::from_frame(const real_t *const s, const real_t *const in, real_t *const out) const {
        node_obstacle_from_frame(view(normals_, gap_), s, in, out);
    }

    void NodeObstacleContact::to_frame(const Team &team, const real_t *const s, const real_t *const in, real_t *const out) const {
        assert(team.begin % 3 == 0 && team.end % 3 == 0);
        node_obstacle_to_frame(view(normals_, gap_), s, in, out, team.begin / 3, team.end / 3);
    }

    void NodeObstacleContact::from_frame(const Team &team, const real_t *const s, const real_t *const in, real_t *const out) const {
        assert(team.begin % 3 == 0 && team.end % 3 == 0);
        node_obstacle_from_frame(view(normals_, gap_), s, in, out, team.begin / 3, team.end / 3);
    }

    void NodeObstacleContact::upper_bound(const real_t unbounded, real_t *const ub) const {
        node_obstacle_upper_bound(view(normals_, gap_), ndofs_, unbounded, ub);
    }

    real_t NodeObstacleContact::kkt_residual(const real_t *const r,
                                             const real_t *const k,  // per dof
                                             real_t *const       projected,
                                             real_t *const       penetration) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::kkt_residual");

        auto blas = sfem::blas<real_t>(context_);
        blas->copy(ndofs_, r, projected);
        node_obstacle_kkt(view(normals_, gap_), k, projected, complementarity_->data(), penetration_->data());
        // Summed over the ranks in one reduction, the penetration with them: a non-finite one on any
        // rank reaches every rank through the sum.
        real_t local[2] = {blas->dot(owned_dofs_, projected, projected) +
                                   blas->dot(size(), complementarity_->data(), complementarity_->data()),
                           max_or_zero(context_, size(), penetration_->data())};
        if (comm_->size() > 1) {
            comm_->sum(&local[0], 1, smesh::TypeToEnum<real_t>::value());
            local[1] = comm_->max(local[1]);
        }
        *penetration = local[1];
        return std::sqrt(local[0]);
    }

    void NodeObstacleContact::add_curvature(const real_t *const  x,
                                            const real_t *const  r,
                                            const real_t *const  k,
                                            const bool           capped,
                                            const real_t         sign,
                                            const count_t *const       rowptr,
                                            const idx_t *const         colidx,
                                            const smesh::PrimitiveType values_type,
                                            void *const                values) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::add_curvature");

        real_t *Crows[6];
        rows(curvature_, 6, Crows);
        obstacle_->sample_nodes_curvature(size(), idx_->data(), xyz_, x, Crows);
        node_obstacle_curvature_bsr(view(normals_, gap_), r, k, capped, sign, rowptr, colidx, values_type, values);
    }

    void NodeObstacleContact::reaction(real_t *const out) const {
        sfem::blas<real_t>(context_)->zeros(ndofs_, out);
        node_obstacle_reaction(view(normals_, gap_), out);
    }

    void NodeObstacleContact::project_velocity(const real_t *const x, real_t *const v, const real_t dt) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::project_velocity");

        sfem::blas<real_t>(context_)->zaxpby(ndofs_, 1, x, dt, v, ahead_->data());
        real_t *an[3];
        rows(ahead_normals_, 3, an);
        obstacle_->sample_nodes(size(), idx_->data(), xyz_, ahead_->data(), ahead_gap_->data(), an);
        offset(ahead_gap_, ahead_normals_);
        node_obstacle_project_velocity(view(ahead_normals_, ahead_gap_), v, dt);
    }

    real_t NodeObstacleContact::min_gap() const {
        auto gaps = gap_;
#ifdef SFEM_ENABLE_CUDA
        if (gaps->mem_space() == MEMORY_SPACE_DEVICE) gaps = smesh::to_host(gaps);
#endif
        const real_t *const gap = gaps->data();
        real_t              low = std::numeric_limits<real_t>::max();
#pragma omp parallel for reduction(min : low)
        for (ptrdiff_t i = 0; i < size(); ++i) low = std::min(low, gap[i]);
        return comm_->size() > 1 ? -comm_->max(-low) : low;
    }

    void NodeObstacleContact::gap_field(real_t *const out) const {
        sfem::blas<real_t>(context_)->zeros(ndofs_, out);
        node_obstacle_gap_field(view(normals_, gap_), out);
    }

}  // namespace sfem
