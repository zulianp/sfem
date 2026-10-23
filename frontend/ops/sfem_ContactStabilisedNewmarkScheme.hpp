#pragma once

#include <memory>

#include "sfem_NewmarkScheme.hpp"
#include "sfem_TimeScheme.hpp"

namespace sfem {

    /// Contact-stabilised Newmark (Deuflhard, Krause and Ertel 2008, in the form of Klapproth
    /// 2012, ZIB-Report 12-18, eqs. 12d-12f), as a `TimeScheme` a material can be handed.
    ///
    /// Plain Newmark carries the obstacle's reaction forward in the acceleration and chatters at
    /// contact. This method differs from it in what it carries out of a step, and only there:
    ///
    /// - The contact force is implicit only: the acceleration carried into the next predictor is
    ///   the material's, -M^{-1} F(u), without the obstacle's reaction.
    /// - The velocity is projected so that the next predicted position x + dt v is admissible.
    ///   With a lumped mass the L2 projection is pointwise, so it is the contact's to make.
    ///
    /// Within a step it is Newmark: the shift, the history, the predictor and the inertia are
    /// those of the `NewmarkScheme` it composes, which is the one copy of that arithmetic. It
    /// claims that scheme, so the same Newmark cannot also be handed to a material and add its
    /// inertia a second time.
    ///
    /// Newton starts from `start`, u_n + dt v_n, rather than from the predictor: the stabilised
    /// velocity makes it admissible, while the predictor's dt^2/4 a, whose material acceleration
    /// at a contact node grows as the mesh is refined, can start Newton deep inside the obstacle.
    class ContactStabilisedNewmarkScheme final : public TimeScheme {
    public:
        /// What the method needs from a contact condition.
        class Contact {
        public:
            virtual ~Contact() = default;

            /// The constraint's share of the step's residual at the state of the last converged
            /// solve: at a KKT point the residual R(x) is this. Zero away from the contact.
            virtual void reaction(real_t *const out) const = 0;

            /// Replace v so that x + dt v is admissible: the L2 projection of x + dt v onto the
            /// admissible set in the lumped mass's norm, and v what reaches it.
            virtual void project_velocity(const real_t *const x, real_t *const v, const real_t dt) = 0;
        };

        /// `newmark` must be initialised; its execution space is this scheme's.
        ContactStabilisedNewmarkScheme(const std::shared_ptr<NewmarkScheme> &newmark,
                                       const std::shared_ptr<Contact>       &contact);
        ~ContactStabilisedNewmarkScheme() override;

        real_t        shift() const override;
        const real_t *history() const override;
        real_t        weight(const char *name) const override;
        void          begin_step(real_t t, real_t dt) override;

        /// Newmark's advance, then the material acceleration and the projected velocity.
        void advance(const real_t *const x) override;

        void reconstruct(const real_t *const x,
                         real_t *const       velocity,
                         real_t *const       acceleration) const override;
        bool has_acceleration() const override { return true; }

        std::shared_ptr<Op> inertia_op() const override;

        /// Newton's start for the step opened by the last `begin_step`: u_n + dt v_n.
        void start(real_t *const x) const override;

        /// The projected velocity and the material acceleration.
        void carried_state(real_t *const velocity, real_t *const acceleration) const override;

        const std::shared_ptr<NewmarkScheme> &newmark() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
