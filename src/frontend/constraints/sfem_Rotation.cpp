#include "sfem_Rotation.hpp"

#include <cmath>

#include "sfem_Function.hpp"
#include "sfem_LinearOperatorFactory.hpp"
#include "smesh_sideset.hpp"

namespace sfem {

    class Rotation::Impl {
    public:
        std::shared_ptr<FunctionSpace>       space;
        std::shared_ptr<DirichletConditions> conditions;
        SharedBuffer<idx_t>                  nodeset;
        SharedBuffer<real_t>                 u0, u1;
        int                                  a0{1}, a1{2};
        real_t                               center[3]{0, 0, 0};
        real_t                               angle{0};
        real_t                               prescribed{0};
        LoadProfile                          profile;
    };

    Rotation::Rotation() : impl_(std::make_unique<Impl>()) {}
    Rotation::~Rotation() = default;

    std::shared_ptr<Rotation> Rotation::create(const std::shared_ptr<FunctionSpace>        &space,
                                               const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                               const int                                    axis,
                                               const real_t                                 center[3],
                                               const real_t                                 angle,
                                               const LoadProfile                           &profile) {
        if (axis < 0 || axis > 2 || space->block_size() != 3) {
            SFEM_ERROR("Rotation: the axis must be 0, 1 or 2 of a three-dimensional displacement\n");
            return nullptr;
        }

        std::shared_ptr<Rotation> ret(new Rotation());
        auto                     &impl = *ret->impl_;
        impl.space                     = space;
        impl.a0                        = axis == 0 ? 1 : 0;
        impl.a1                        = axis == 2 ? 1 : 2;
        impl.angle                     = angle;
        impl.profile                   = profile;
        for (int d = 0; d < 3; ++d) impl.center[d] = center[d];

        impl.nodeset = sidesets.size() == 1 ? smesh::create_nodeset_from_sideset(space->mesh(), sidesets[0])
                                            : smesh::create_nodeset_from_sidesets(space->mesh(), sidesets);
        impl.u0      = create_host_buffer<real_t>(impl.nodeset->size());
        impl.u1      = create_host_buffer<real_t>(impl.nodeset->size());

        // The prescribed values are these two buffers, which set_angle rewrites in place.
        DirichletConditions::Condition c0{.sidesets = sidesets, .nodeset = impl.nodeset, .values = impl.u0, .component = impl.a0};
        DirichletConditions::Condition c1{.sidesets = sidesets, .nodeset = impl.nodeset, .values = impl.u1, .component = impl.a1};
        impl.conditions = DirichletConditions::create(space, {c0, c1});
        return ret;
    }

    real_t Rotation::angle_at(const real_t t) const { return impl_->angle * impl_->profile.value(t); }

    int Rotation::set_time(const real_t t) { return set_angle(angle_at(t)); }

    real_t Rotation::angle() const { return impl_->prescribed; }

    std::shared_ptr<DirichletConditions> Rotation::conditions() const { return impl_->conditions; }

    int Rotation::set_angle(const real_t angle) {
        auto           &impl  = *impl_;
        impl.prescribed       = angle;
        auto            xyz   = impl.space->points()->data();
        const real_t    c     = std::cos(angle);
        const real_t    s     = std::sin(angle);
        const idx_t    *nodes = impl.nodeset->data();
        real_t *const   out0  = impl.u0->data();
        real_t *const   out1  = impl.u1->data();
        const geom_t   *p0    = xyz[impl.a0];
        const geom_t   *p1    = xyz[impl.a1];
        const real_t    c0    = impl.center[impl.a0];
        const real_t    c1    = impl.center[impl.a1];
        const ptrdiff_t n     = impl.nodeset->size();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t q0 = p0[nodes[i]] - c0;
            const real_t q1 = p1[nodes[i]] - c1;
            out0[i]         = c * q0 - s * q1 - q0;
            out1[i]         = s * q0 + c * q1 - q1;
        }
        return SFEM_SUCCESS;
    }

    real_t Rotation::torque(const real_t *const x, const real_t *const force) const {
        auto           &impl  = *impl_;
        auto            xyz   = impl.space->points()->data();
        const idx_t    *nodes = impl.nodeset->data();
        const ptrdiff_t n     = impl.nodeset->size();
        const int       bs    = impl.space->block_size();
        const int       a0    = impl.a0;
        const int       a1    = impl.a1;

        real_t moment = 0;
#pragma omp parallel for reduction(+ : moment)
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t node = nodes[i];
            const ptrdiff_t dof  = node * bs;
            const real_t    q0   = (real_t)xyz[a0][node] + x[dof + a0] - impl.center[a0];
            const real_t    q1   = (real_t)xyz[a1][node] + x[dof + a1] - impl.center[a1];
            moment += q0 * force[dof + a1] - q1 * force[dof + a0];
        }
        return moment;
    }

    int Rotation::apply(real_t *const x) { return impl_->conditions->apply(x); }
    int Rotation::apply_value(const real_t value, real_t *const x) { return impl_->conditions->apply_value(value, x); }
    int Rotation::gradient(const real_t *const x, real_t *const g) { return impl_->conditions->gradient(x, g); }
    int Rotation::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        return impl_->conditions->copy_constrained_dofs(src, dest);
    }
    int Rotation::mask(mask_t *mask) { return impl_->conditions->mask(mask); }
    int Rotation::value(const real_t *const x, real_t *const out) { return impl_->conditions->value(x, out); }
    int Rotation::value_steps(const real_t       *x,
                              const real_t       *h,
                              const int           nsteps,
                              const real_t *const steps,
                              real_t *const       out) {
        return impl_->conditions->value_steps(x, h, nsteps, steps, out);
    }
    int Rotation::hessian_crs(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        return impl_->conditions->hessian_crs(x, rowptr, colidx, values);
    }
    int Rotation::hessian_bsr(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        return impl_->conditions->hessian_bsr(x, rowptr, colidx, values);
    }
    int Rotation::hessian_bsr_typed(const real_t *const          x,
                                    const count_t *const         rowptr,
                                    const idx_t *const           colidx,
                                    const smesh::PrimitiveType values_type,
                                    void *const                  values) {
        return impl_->conditions->hessian_bsr_typed(x, rowptr, colidx, values_type, values);
    }
    std::shared_ptr<Constraint> Rotation::derefine(const std::shared_ptr<FunctionSpace> &coarse_space, const bool as_zero) const {
        return impl_->conditions->derefine(coarse_space, as_zero);
    }
    std::shared_ptr<Constraint> Rotation::lor() const { return impl_->conditions->lor(); }

}  // namespace sfem
