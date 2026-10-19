#ifndef SFEM_CONTACT_COMMON_HPP
#define SFEM_CONTACT_COMMON_HPP

#include <cmath>
#include <functional>
#include <memory>

#include "sfem_API.hpp"
#include "sfem_Merit.hpp"
#include "sfem_Observers.hpp"

namespace sfem {

        // The KKT residual with the penetration counted as a force, k_max times it. Where the
        // obstacle's normal turns quickly (a crease, a corner) the full step can cycle between two
        // contact states; a step must decrease this.
        class KKTMerit final : public Merit {
        public:
            using At    = std::function<real_t(const real_t *const, const real_t *const)>;
            using Value = std::function<real_t(const real_t *const)>;

            KKTMerit(const At                            &at,
                     const Value                         &value,
                     const SharedBuffer<real_t>          &residual,
                     const std::shared_ptr<BLAS<real_t>> &blas,
                     const ptrdiff_t                      ndofs,
                     const ExecutionSpace                 es)
                : at_(at), value_(value), residual_(residual), blas_(blas), ndofs_(ndofs), trial_(create_buffer<real_t>(ndofs, es)) {}

            const char *name() const override { return "kkt"; }

            int evaluate(const real_t *const x,
                         const real_t *const h,
                         const int           nsteps,
                         const real_t *const steps,
                         real_t *const       out) override {
                for (int i = 0; i < nsteps; ++i) {
                    blas_->copy(ndofs_, x, trial_->data());
                    blas_->axpy(ndofs_, steps[i], h, trial_->data());
                    out[i] = value_(trial_->data());
                    last_  = steps[i];
                    known_ = std::isfinite(out[i]);
                }
                return SFEM_SUCCESS;
            }

            // The residual of the last trial, which `value` left in the step's trial residual.
            bool residual_at(const real_t step, real_t *const r) const override {
                if (!known_ || step != last_) return false;
                blas_->copy(ndofs_, residual_->data(), r);
                return true;
            }

            real_t at(const real_t *const x, const real_t *const r) override { return at_(x, r); }

            // A decrease in proportion to the merit itself: m(s) <= (1 - c1 s) m0.
            real_t slope(const real_t *const, const real_t *const, const real_t *const, const real_t m0) override { return -m0; }

            bool batched() const override { return false; }

        private:
            At                            at_;
            Value                         value_;
            SharedBuffer<real_t>          residual_;
            std::shared_ptr<BLAS<real_t>> blas_;
            ptrdiff_t                     ndofs_;
            SharedBuffer<real_t>          trial_;
            real_t                        last_{0};
            bool                          known_{false};
        };

    /// Columns reaction_x, reaction_y, reaction_z and touched: the resultant of the obstacle's
    /// physical force on the body -- half the step's reaction (the reaction of the obstacle back-ends
    /// is in the step's scaling, where the material force enters both ends of a trapezoidal step and
    /// the contact force once) -- and whether a node is on the obstacle (the smallest gap at most
    /// `touch_gap`; the reaction of a body in flight is the residual's rounding, not zero), read
    /// before the step is committed. `reaction` writes the step's reaction (ndofs, in `es`),
    /// `min_gap` returns the smallest gap of the contact nodes.
    std::shared_ptr<Probe> create_contact_probe(const std::function<void(real_t *const)> &reaction,
                                                const std::function<real_t()>             &min_gap,
                                                const ptrdiff_t                           ndofs,
                                                const ExecutionSpace                      es,
                                                const real_t                              touch_gap);

}  // namespace sfem

#endif  // SFEM_CONTACT_COMMON_HPP
