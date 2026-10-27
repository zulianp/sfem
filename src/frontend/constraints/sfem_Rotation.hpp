#ifndef SFEM_ROTATION_HPP
#define SFEM_ROTATION_HPP

#include <memory>
#include <vector>

#include "sfem_Constraint.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_LoadProfile.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    /// A rigid rotation of the nodes of a sideset about the line through `center` along one
    /// coordinate axis, by `angle * profile(t)` radians. The rotation acts in the plane of the
    /// two other axes, in increasing order (y-z about x, x-z about y, x-y about z), carrying the
    /// first into the second. Host only: the prescribed values move with time.
    class Rotation final : public Constraint {
    public:
        static std::shared_ptr<Rotation> create(const std::shared_ptr<FunctionSpace>        &space,
                                                const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                const int                                    axis,
                                                const real_t                                 center[3],
                                                const real_t                                 angle,
                                                const LoadProfile                           &profile);
        ~Rotation() override;

        /// Prescribes the rotation at time t.
        int    set_time(const real_t t);
        real_t angle_at(const real_t t) const;

        /// Prescribes `angle` directly, whatever the profile says.
        int set_angle(const real_t angle);
        /// The angle prescribed last.
        real_t angle() const;

        /// The moment about the axis exerted by `force` on the rotated nodes, with the lever
        /// arms in the configuration displaced by `x`: sum_n (p_a0 f_a1 - p_a1 f_a0). With the
        /// residual of the equilibrium as `force`, it is the torque the rotation applies, signed
        /// as the angle.
        real_t torque(const real_t *const x, const real_t *const force) const;

        int apply(real_t *const x) override;
        int apply_value(const real_t value, real_t *const x) override;
        int gradient(const real_t *const x, real_t *const g) override;
        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;
        int mask(mask_t *mask) override;
        int value(const real_t *const x, real_t *const out) override;
        int value_steps(const real_t       *x,
                        const real_t       *h,
                        const int           nsteps,
                        const real_t *const steps,
                        real_t *const       out) override;
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;
        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;
        int hessian_bsr_typed(const real_t *const          x,
                              const count_t *const         rowptr,
                              const idx_t *const           colidx,
                              const smesh::PrimitiveType values_type,
                              void *const                  values) override;

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                             const bool                            as_zero) const override;
        std::shared_ptr<Constraint> lor() const override;

    private:
        Rotation();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_ROTATION_HPP
