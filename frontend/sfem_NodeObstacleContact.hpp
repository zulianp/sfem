#ifndef SFEM_NODE_OBSTACLE_CONTACT_HPP
#define SFEM_NODE_OBSTACLE_CONTACT_HPP

#include <memory>

#include "sfem_ContactStabilisedNewmarkScheme.hpp"
#include "sfem_SDFObstacle.hpp"
#include "sfem_aliases.hpp"
#include "smesh_grid.hpp"

namespace sfem {

    // Frictionless contact of a set of mesh nodes with a rigid obstacle, given by its SDF grid
    // (positive inside) and sampled by the library at the displaced nodes (SDFObstacle::sample_nodes,
    // node_interpolate.hpp): the grid's interpolating cubic B-spline, whose gradient and curvature
    // are continuous. Node i keeps out of it: gap_i = -sdf(X_i + u_i) / |grad sdf| >= 0; outside the
    // grid there is no obstacle.
    // A step dx changes the gap by -n_i . dx_i to first order, n_i the unit gradient there, so the
    // linearised step is the QP with the constraints n_i . dx_i <= gap_i. In each contact node's
    // frame [n_i, t1_i, t2_i] that is an upper bound on one component, which a box-constrained QP
    // solver takes as it is.
    //
    // It is the contact of a ContactStabilisedNewmarkScheme: the reaction of the last KKT measure
    // and the velocity projection. Host only, as SDFObstacle::sample_nodes is.
    class NodeObstacleContact final : public ContactStabilisedNewmarkScheme::Contact {
    public:
        // `dir` is the axis whose frame a node outside the grid gets.
        NodeObstacleContact(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                            const std::shared_ptr<Communicator>        &comm,
                            const SharedBuffer<idx_t>                  &nodes,
                            const SharedBuffer<geom_t *>               &points,
                            const ptrdiff_t                             ndofs,
                            const int                                   dir);

        ptrdiff_t                   size() const { return nodes_->size(); }
        const SharedBuffer<idx_t>  &nodes() const { return nodes_; }
        const SharedBuffer<real_t> &gap() const { return gap_; }

        // Samples the gap, the normals and the frames with the nodes displaced by x.
        void update(const real_t *const x);

        // v <- Q^T v at the contact nodes: into their frames.
        void to_frame(real_t *const v) const;

        // v <- Q v at the contact nodes: back from their frames.
        void from_frame(real_t *const v) const;

        // The step's upper bound in the frames: the gap on each contact node's normal component,
        // `unbounded` everywhere else.
        void upper_bound(const real_t unbounded, real_t *const ub) const;

        // First-order optimality of the step. At the solution the residual r is the obstacle's
        // reaction mu_i n_i at the nodes it touches: along the normal, pushing out (mu_i >= 0),
        // and zero where the node is clear of the obstacle (min(mu_i, gap_i) = 0).
        //
        // Writes r less the admissible reaction, mu_i = max(0, -n_i . r_i), to `projected` and
        // returns sqrt(|projected|^2 + sum_i min(mu_i, k_i gap_i)^2) over the nodes clear of the
        // obstacle: stationarity and complementarity, both forces, with k_i the stiffness of the
        // node's equation. Penetration is a length, max_i -gap_i, and goes to *penetration.
        // Keeps mu_i for `reaction`.
        real_t kkt_residual(const real_t *const r,
                            const real_t *const k,  // per dof
                            real_t *const       projected,
                            real_t *const       penetration);

        // The constraints' share of the Newton tangent at the nodes displaced by x. The Lagrangian
        // E - sum_i mu_i gap_i adds -mu_i hess(gap_i), and on a curved obstacle that term is what
        // keeps Newton quadratic for a body sliding over it. At a node on the surface -hess(gap_i)
        // is the library's curvature (H - n m^T - m n^T) / |grad sdf|, m = H n, the B-spline's
        // Hessian H corrected for its gradient not being exactly unit (for a sphere it is
        // -(I - n n^T) / R). mu_i is the reaction at a node in contact, read from the residual r as
        // in kkt_residual. Adds each node's 3x3 block to the diagonal block of the BSR tangent.
        //
        // Away from the solution that estimate can be large, and on a convex obstacle the term is
        // negative across it, enough to make the tangent indefinite. With `capped`, each node's
        // share is scaled so that its most negative part (bounded by row sums) stays within half
        // of the node's inertia stiffness k_i = c0 m_i. `sign` -1 takes back an earlier call.
        void add_curvature(const real_t *const  x,
                           const real_t *const  r,
                           const real_t *const  k,
                           const bool           capped,
                           const real_t         sign,
                           const count_t *const rowptr,
                           const idx_t *const   colidx,
                           real_t *const        values);

        // -mu_i n_i at the contact nodes, with mu_i of the last kkt_residual, zero elsewhere: the
        // residual's part that is the obstacle's reaction.
        void reaction(real_t *const out) const override;

        // The velocity correction of the contact-stabilised Newmark method (Deuflhard, Krause and
        // Ertel 2008, in the form of Klapproth 2012, ZIB-Report 12-18, eq. 12f): with a lumped mass
        // the L2 projection of x + dt v onto the admissible set is pointwise, so every contact node
        // whose next position x_i + dt v_i would be inside the obstacle is moved to its surface
        // along the normal, and v_i is what reaches that point. Its normal velocity into the
        // obstacle is gone; nothing else changes.
        void project_velocity(const real_t *const x, real_t *const v, const real_t dt) override;

        // gap_i n_i at the contact nodes, zero elsewhere: the vector to the obstacle's surface.
        void gap_field(real_t *const out) const;

    private:
        std::shared_ptr<SDFObstacle> obstacle_;
        SharedBuffer<idx_t>          nodes_;
        SharedBuffer<geom_t *>       points_;
        ptrdiff_t                    ndofs_;
        int                          dir_;
        SharedBuffer<real_t *>       normals_;
        SharedBuffer<real_t *>       curvature_;
        SharedBuffer<real_t>         gap_;
        SharedBuffer<real_t>         mu_;
        SharedBuffer<real_t>         frames_;
        SharedBuffer<real_t>         ahead_;
        SharedBuffer<real_t>         ahead_gap_;
        SharedBuffer<real_t *>       ahead_normals_;
    };

}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_CONTACT_HPP
