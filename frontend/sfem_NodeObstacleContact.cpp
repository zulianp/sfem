#include "sfem_NodeObstacleContact.hpp"

#include <algorithm>
#include <cmath>

#include "sfem_BSR.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    NodeObstacleContact::NodeObstacleContact(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                                             const std::shared_ptr<Communicator>        &comm,
                                             const SharedBuffer<idx_t>                  &nodes,
                                             const SharedBuffer<geom_t *>               &points,
                                             const ptrdiff_t                             ndofs,
                                             const int                                   dir)
        : nodes_(nodes), points_(points), ndofs_(ndofs), dir_(dir) {
        obstacle_ = SDFObstacle::create(sdf, comm, points->extent(1), points->data()[2], EXECUTION_SPACE_HOST);

        const ptrdiff_t n = nodes->size();
        normals_          = smesh::create_host_buffer<real_t>(3, n);
        curvature_        = smesh::create_host_buffer<real_t>(6, n);
        gap_              = smesh::create_host_buffer<real_t>(n);
        mu_               = smesh::create_host_buffer<real_t>(n);
        frames_           = smesh::create_host_buffer<real_t>(9 * n);
        ahead_            = smesh::create_host_buffer<real_t>(ndofs);
        ahead_gap_        = smesh::create_host_buffer<real_t>(n);
        ahead_normals_    = smesh::create_host_buffer<real_t>(3, n);
    }

    void NodeObstacleContact::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::update");

        obstacle_->sample_nodes(size(), nodes_->data(), points_->data(), x, gap_->data(), normals_->data());

        // t1 is n crossed with the axis least aligned with it, t2 = n x t1. A node outside the
        // grid has no normal and an unreachable gap; its frame is that of the contact direction.
        real_t *const *const nn = normals_->data();
        real_t *const        Q  = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            real_t n[3] = {nn[0][i], nn[1][i], nn[2][i]};
            if (n[0] == 0 && n[1] == 0 && n[2] == 0) n[dir_] = 1;
            int axis = 0;
            for (int d = 1; d < 3; ++d) {
                if (std::abs(n[d]) < std::abs(n[axis])) axis = d;
            }
            real_t e[3]        = {0, 0, 0};
            e[axis]            = 1;
            real_t       t1[3] = {n[1] * e[2] - n[2] * e[1], n[2] * e[0] - n[0] * e[2], n[0] * e[1] - n[1] * e[0]};
            const real_t len   = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
            for (int d = 0; d < 3; ++d) t1[d] /= len;
            const real_t t2[3] = {n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]};

            real_t *const q = &Q[9 * i];
            for (int d = 0; d < 3; ++d) {
                q[d]     = n[d];
                q[3 + d] = t1[d];
                q[6 + d] = t2[d];
            }
        }
    }

    void NodeObstacleContact::to_frame(real_t *const v) const {
        const idx_t *const  idx = nodes_->data();
        const real_t *const Q   = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            real_t *const       vi   = &v[(ptrdiff_t)idx[i] * 3];
            const real_t *const q    = &Q[9 * i];
            const real_t        w[3] = {vi[0], vi[1], vi[2]};
            for (int c = 0; c < 3; ++c) {
                vi[c] = q[3 * c] * w[0] + q[3 * c + 1] * w[1] + q[3 * c + 2] * w[2];
            }
        }
    }

    void NodeObstacleContact::from_frame(real_t *const v) const {
        const idx_t *const  idx = nodes_->data();
        const real_t *const Q   = frames_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            real_t *const       vi   = &v[(ptrdiff_t)idx[i] * 3];
            const real_t *const q    = &Q[9 * i];
            const real_t        w[3] = {vi[0], vi[1], vi[2]};
            for (int d = 0; d < 3; ++d) {
                vi[d] = q[d] * w[0] + q[3 + d] * w[1] + q[6 + d] * w[2];
            }
        }
    }

    void NodeObstacleContact::upper_bound(const real_t unbounded, real_t *const ub) const {
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

        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = normals_->data();
        const real_t *const  gap = gap_->data();
        real_t *const        mus = mu_->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            projected[i] = r[i];
        }

        real_t deepest       = 0;
        real_t complementary = 0;
#pragma omp parallel for reduction(max : deepest) reduction(+ : complementary)
        for (ptrdiff_t i = 0; i < size(); ++i) {
            const ptrdiff_t node = idx[i];
            real_t *const   ri   = &projected[node * 3];
            const real_t    mu   = std::max(real_t(0), -(n[0][i] * ri[0] + n[1][i] * ri[1] + n[2][i] * ri[2]));
            for (int d = 0; d < 3; ++d) {
                ri[d] += mu * n[d][i];
            }
            mus[i]         = mu;
            const real_t c = std::min(mu, k[node * 3] * std::max(gap[i], real_t(0)));
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

        obstacle_->sample_nodes_curvature(size(), nodes_->data(), points_->data(), x, curvature_->data());

        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = normals_->data();
        real_t *const *const C   = curvature_->data();
        const real_t *const  gap = gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            const ptrdiff_t     node = idx[i];
            const real_t *const ri   = &r[node * 3];
            const real_t        mu   = std::max(real_t(0), -(n[0][i] * ri[0] + n[1][i] * ri[1] + n[2][i] * ri[2]));
            if (mu == 0 || gap[i] > mu / k[node * 3]) continue;

            const real_t H[9] = {C[0][i], C[1][i], C[2][i], C[1][i], C[3][i], C[4][i], C[2][i], C[4][i], C[5][i]};

            real_t lowest = 0;
            for (int a = 0; a < 3; ++a) {
                real_t off = 0;
                for (int b = 0; b < 3; ++b) {
                    if (b != a) off += std::abs(H[a * 3 + b]);
                }
                lowest = std::min(lowest, mu * (H[a * 3 + a] - off));
            }
            const real_t scale = capped && lowest < -k[node * 3] / 2 ? k[node * 3] / (2 * -lowest) : 1;

            real_t *const B = bsr_diagonal_block(node, rowptr, colidx, 3, values);
            for (int c = 0; c < 9; ++c) {
                B[c] += sign * scale * mu * H[c];
            }
        }
    }

    void NodeObstacleContact::reaction(real_t *const out) const {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            out[i] = 0;
        }
        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = normals_->data();
        const real_t *const  mus = mu_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            for (int d = 0; d < 3; ++d) {
                out[(ptrdiff_t)idx[i] * 3 + d] = -mus[i] * n[d][i];
            }
        }
    }

    void NodeObstacleContact::project_velocity(const real_t *const x, real_t *const v, const real_t dt) {
        SFEM_TRACE_SCOPE("NodeObstacleContact::project_velocity");

        real_t *const ahead = ahead_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            ahead[i] = x[i] + dt * v[i];
        }
        obstacle_->sample_nodes(size(), nodes_->data(), points_->data(), ahead, ahead_gap_->data(), ahead_normals_->data());

        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = ahead_normals_->data();
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
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs_; ++i) {
            out[i] = 0;
        }
        const idx_t *const   idx = nodes_->data();
        real_t *const *const n   = normals_->data();
        const real_t *const  gap = gap_->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size(); ++i) {
            for (int d = 0; d < 3; ++d) {
                out[(ptrdiff_t)idx[i] * 3 + d] = gap[i] * n[d][i];
            }
        }
    }

}  // namespace sfem
