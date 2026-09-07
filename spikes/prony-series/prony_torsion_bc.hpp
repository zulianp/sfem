#pragma once

// A time-scheduled torsion boundary condition that can be released.
//
// sfem::Rotate<Axis0, Axis1> already prescribes a rigid rotation of a sideset, but it is a
// template parameterised on the rotation plane and it advances by integer step out of a fixed
// number of steps. A relaxation-and-recovery experiment needs neither: it needs the angle as a
// function of physical time, one runtime-selected axis, and the ability to stop enforcing the
// condition part way through the run. This wraps the template behind those three things.

#include <cmath>
#include <functional>
#include <memory>

#include "sfem_Function.hpp"
#include "sfem_Rotate.hpp"
#include "sfem_aliases.hpp"
#include "smesh_sideset.hpp"

#include "prony_case.hpp"

namespace prony {

    struct TorsionBC {
        std::shared_ptr<sfem::Constraint> constraint;
        sfem::SharedBuffer<idx_t>         nodeset;
        /// Prescribes a twist of `angle` radians about the configured axis.
        std::function<void(real_t)>       set_angle;
        /// The two coordinate directions spanning the rotation plane, in the order the
        /// rotation carries the first into the second. The conjugate moment is
        /// sum_n (p[a0] - c[a0]) f[a1] - (p[a1] - c[a1]) f[a0], so the reported torque has
        /// the same sign as the prescribed angle.
        int                               a0{1};
        int                               a1{2};
        real_t                            center[3]{0, 0, 0};
    };

    /// Fraction of the target load applied at a given time: a linear ramp, then held. Used for
    /// torque control, where the twist is the unknown and the moment is what gets ramped.
    inline real_t load_ramp(const Torsion &t, const real_t time) {
        if (t.ramp_time <= 0) {
            return 1;
        }
        const real_t s = time / t.ramp_time;
        return s < 1 ? s : real_t(1);
    }

    /// The prescribed twist at a given time: a linear ramp then a hold, or a sine.
    inline real_t torsion_angle_at(const Torsion &t, const real_t time) {
        if (t.profile == TwistProfile::Cyclic) {
            constexpr real_t two_pi = real_t(6.283185307179586476925286766559);
            return t.angle * std::sin(two_pi * time / t.period);
        }

        if (t.ramp_time <= 0) {
            return t.angle;
        }
        const real_t s = time / t.ramp_time;
        return t.angle * (s < 1 ? s : real_t(1));
    }

    template <int Axis0, int Axis1>
    static std::shared_ptr<TorsionBC> create_torsion_bc_impl(const std::shared_ptr<sfem::FunctionSpace> &fs,
                                                             const std::shared_ptr<sfem::Sideset>       &sideset,
                                                             const Torsion                              &t) {
        // steps = 1 and angle set per call turn Rotate's step counter into a direct angle
        // setter: update(1) then evaluates the rotation at exactly `rot->angle`.
        auto rot = sfem::Rotate<Axis0, Axis1>::create(fs, sideset, 1, t.angle, sfem::EXECUTION_SPACE_HOST);
        rot->verbose    = t.verbose;
        rot->rcenter[0] = t.center[0];
        rot->rcenter[1] = t.center[1];
        rot->rcenter[2] = t.center[2];
        rot->create_constraint();

        auto ret        = std::make_shared<TorsionBC>();
        ret->constraint = rot->create_constraint();
        ret->nodeset    = rot->nodeset;
        ret->a0         = Axis0;
        ret->a1         = Axis1;
        ret->center[0]  = t.center[0];
        ret->center[1]  = t.center[1];
        ret->center[2]  = t.center[2];
        ret->set_angle  = [rot](const real_t angle) {
            rot->angle = angle;
            rot->steps = 1;
            rot->update(1);
        };
        return ret;
    }

    inline std::shared_ptr<TorsionBC> create_torsion_bc(const std::shared_ptr<sfem::FunctionSpace> &fs, const Torsion &t) {
        if (!t.enabled) {
            return nullptr;
        }

        auto sideset = sfem::Sideset::create_from_file(fs->mesh_ptr()->comm(), smesh::Path(t.sideset));
        if (!sideset) {
            SFEM_ERROR("[prony] unable to read torsion sideset %s\n", t.sideset.c_str());
            return nullptr;
        }

        switch (t.axis) {
            case 0:
                return create_torsion_bc_impl<1, 2>(fs, sideset, t);
            case 1:
                return create_torsion_bc_impl<0, 2>(fs, sideset, t);
            case 2:
                return create_torsion_bc_impl<0, 1>(fs, sideset, t);
            default:
                SFEM_ERROR("[prony] torsion axis %d is not one of 0, 1, 2\n", t.axis);
                return nullptr;
        }
    }

    /// Moment about the rotation axis that the grip exerts on the body, read off the
    /// unconstrained residual at the twisted nodes. In equilibrium the residual
    /// r = dPi/du carries exactly the constraint force at a constrained degree of freedom,
    /// so summing r x lever over the sideset gives the applied torque -- the quantity that
    /// relaxes while the twist is held, and that vanishes the instant the grip is released.
    ///
    /// The lever arm is taken in the deformed configuration, so this is the moment about the
    /// axis through `center` in the current placement of the body.
    inline real_t reaction_torque(const TorsionBC          &bc,
                                  const std::shared_ptr<sfem::Mesh> &mesh,
                                  const int                 block_size,
                                  const real_t *const       u,
                                  const real_t *const       residual) {
        auto            points = mesh->points()->data();
        const auto      nodes  = bc.nodeset->data();
        const ptrdiff_t n      = bc.nodeset->size();

        const int a0 = bc.a0;
        const int a1 = bc.a1;

        real_t torque = 0;
#pragma omp parallel for reduction(+ : torque)
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t node = nodes[i];
            const ptrdiff_t dof  = node * block_size;

            const real_t p0 = (real_t)points[a0][node] + u[dof + a0] - bc.center[a0];
            const real_t p1 = (real_t)points[a1][node] + u[dof + a1] - bc.center[a1];

            torque += p0 * residual[dof + a1] - p1 * residual[dof + a0];
        }

        return torque;
    }

}  // namespace prony
