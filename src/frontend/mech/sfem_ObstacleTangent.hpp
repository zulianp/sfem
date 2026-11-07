#ifndef SFEM_OBSTACLE_TANGENT_HPP
#define SFEM_OBSTACLE_TANGENT_HPP

#include <memory>
#include <string>

#include "sfem_InexactOptions.hpp"
#include "sfem_Operator.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_base.hpp"
#include "sfem_defs.hpp"

namespace sfem {

    class Function;
    class NodeObstacleContact;

    /// The tangent K of NodeObstacleStep's QP: the material's Hessian with its constrained rows the
    /// identity, the obstacle's curvature at the contact nodes, and a diagonal shift. The step
    /// applies it between the contact frames and reads its diagonal; nothing else.
    ///
    /// "BSR" assembles all three into one BSRTangent. "INEXACT" and "MF" apply them one after
    /// the other: the material through the function's stored tangent (Function::inexact_update, in
    /// the format `inexact` names) or its matrix-free product, the curvature as a 3x3 block per
    /// contact node, the shift as a scaled copy of a diagonal.
    class ObstacleTangent {
    public:
        struct Options {
            /// "BSR", "INEXACT" or "MF" (op_type).
            std::string    format{op_type::BSR};
            /// What INEXACT stores.
            InexactOptions inexact;
            /// BSR only: bytes per stored value, sizeof(real_t) or 4 (BSRTangent::create).
            int storage_bytes{(int)sizeof(real_t)};
        };

        static std::shared_ptr<ObstacleTangent> create(const std::shared_ptr<Function>            &f,
                                                       const std::shared_ptr<NodeObstacleContact> &contact,
                                                       const Options                              &options);
        virtual ~ObstacleTangent() = default;

        /// The material's tangent at x, without curvature or shift.
        virtual int assemble(const real_t *const x) = 0;
        /// The obstacle's curvature at x for the reactions of r (NodeObstacleContact::add_curvature),
        /// as much of it as `curvature` takes (NodeObstacleCurvature), in place of any the tangent
        /// carries.
        virtual int set_curvature(const real_t *const x, const real_t *const r, const real_t *const k, const int curvature) = 0;
        /// K_dd += factor |reference_d|, cumulative until the next assemble; `reference` is the
        /// same at every call between two assembles.
        virtual int shift_diagonal(const real_t factor, const real_t *const reference) = 0;
        /// The scalar diagonal of K as it stands.
        virtual int diagonal(real_t *const out) = 0;
        /// A bound on the largest eigenvalue of S K S, S = diag(scale); 0 when the tangent has
        /// none to offer (its caller estimates it).
        virtual real_t max_eig(const real_t *const scale) const = 0;
        /// y = K x, y overwritten. Taken again after each assemble.
        virtual std::shared_ptr<Operator<real_t>> op() const = 0;
        /// op() as a product a caller's region can run (host BSR), or null.
        virtual std::shared_ptr<TeamOperator<real_t>> team() const { return nullptr; }
        /// Whether team() is there once assembled: the product runs on a region's threads.
        virtual bool team_capable() const { return false; }
    };

}  // namespace sfem

#endif  // SFEM_OBSTACLE_TANGENT_HPP
