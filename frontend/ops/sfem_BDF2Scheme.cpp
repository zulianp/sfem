#include "sfem_BDF2Scheme.hpp"

#include <cstring>

#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class BDF2Scheme::Impl {
    public:
        std::shared_ptr<FunctionSpace>    space;
        std::shared_ptr<InertiaPotential> inertia;
        ExecutionSpace                    es{EXECUTION_SPACE_HOST};
        std::shared_ptr<BLAS<real_t>>     blas;

        std::shared_ptr<Buffer<real_t>> u_n;
        std::shared_ptr<Buffer<real_t>> u_nm1;
        std::shared_ptr<Buffer<real_t>> v_n;
        std::shared_ptr<Buffer<real_t>> v_nm1;
        std::shared_ptr<Buffer<real_t>> z;
        std::shared_ptr<Buffer<real_t>> u_hat;

        real_t dt{0};
        // The length of the last step taken, which the variable-step formula weighs against.
        real_t dt_n{0};
        real_t shift{0};
        real_t alpha_a{0};

        // BDF2 needs two steps of history, so the first step is taken with
        // backward Euler.  Counted rather than passed in, because which formula
        // applies is a property of the method and not of the caller's loop.
        bool started{false};

        Impl(const std::shared_ptr<FunctionSpace> &sp, const ExecutionSpace space_of)
            : space(sp), es(space_of), blas(sfem::blas<real_t>(space_of)) {}
    };

    BDF2Scheme::BDF2Scheme(const std::shared_ptr<FunctionSpace> &space, const ExecutionSpace es)
        : impl_(std::make_unique<Impl>(space, es)) {
        impl_->inertia = std::make_shared<InertiaPotential>(space, es);
    }

    BDF2Scheme::~BDF2Scheme() = default;

    void BDF2Scheme::set_density(const real_t density) { impl_->inertia->set_density(density); }
    void BDF2Scheme::set_consistent_mass(const bool consistent) { impl_->inertia->set_consistent_mass(consistent); }

    int BDF2Scheme::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("BDF2Scheme::initialize");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->u_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->u_nm1          = create_buffer<real_t>(ndofs, impl_->es);
        impl_->v_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->v_nm1          = create_buffer<real_t>(ndofs, impl_->es);
        impl_->z              = create_buffer<real_t>(ndofs, impl_->es);
        impl_->u_hat          = create_buffer<real_t>(ndofs, impl_->es);

        impl_->inertia->set_u_hat(impl_->u_hat);
        return impl_->inertia->initialize(block_names);
    }

    real_t        BDF2Scheme::shift() const { return impl_->shift; }
    const real_t *BDF2Scheme::history() const { return impl_->z->data(); }

    real_t BDF2Scheme::weight(const char *name) const {
        if (!std::strcmp(name, "dt")) return impl_->dt;
        if (!std::strcmp(name, "shift")) return impl_->shift;
        if (!std::strcmp(name, "alpha_a")) return impl_->alpha_a;
        if (!std::strcmp(name, "order")) return impl_->started ? real_t(2) : real_t(1);
        SFEM_ERROR("BDF2Scheme has no weight named \"%s\"\n", name);
        return 0;
    }

    void BDF2Scheme::begin_step(const real_t, const real_t dt) {
        SFEM_TRACE_SCOPE("BDF2Scheme::begin_step");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;
        impl_->dt             = dt;

        // Everything the solve reads is built here, from the state carried out
        // of the last step and never from the current iterate -- which is what
        // lets a line search re-assemble the residual at several trial step
        // lengths and compare merits that mean the same thing.
        if (!impl_->started) {
            // Backward Euler startup: v = (u - u_n)/dt, a = (v - v_n)/dt.
            impl_->shift   = 1 / dt;
            impl_->alpha_a = 1 / (dt * dt);

            blas->zaxpby(ndofs, 1, impl_->u_n->data(), dt, impl_->v_n->data(), impl_->u_hat->data());
            blas->zaxpby(ndofs, -impl_->shift, impl_->u_n->data(), 0, impl_->u_n->data(), impl_->z->data());
        } else {
            // The variable-step formula, w = dt / dt_n the ratio of this step to the last:
            // u' = c0 u + c1 u_n + c2 u_nm1 with c0 = (1 + 2w) / ((1 + w) dt), c1 = -(1 + w) / dt,
            // c2 = w^2 / ((1 + w) dt); at w = 1 the constant-step (3u - 4u_n + u_nm1) / (2dt). The
            // same on the velocity gives the acceleration c0^2 (u - u_hat), with
            // u_hat = -(c1 u_n + c2 u_nm1) / c0 - (c1 v_n + c2 v_nm1) / c0^2.
            const real_t w  = dt / impl_->dt_n;
            const real_t c0 = (1 + 2 * w) / ((1 + w) * dt);
            const real_t c1 = -(1 + w) / dt;
            const real_t c2 = w * w / ((1 + w) * dt);
            impl_->shift    = c0;
            impl_->alpha_a  = c0 * c0;

            blas->zaxpby(ndofs, -c1 / c0, impl_->u_n->data(), -c2 / c0, impl_->u_nm1->data(), impl_->u_hat->data());
            blas->axpy(ndofs, -c1 / (c0 * c0), impl_->v_n->data(), impl_->u_hat->data());
            blas->axpy(ndofs, -c2 / (c0 * c0), impl_->v_nm1->data(), impl_->u_hat->data());

            blas->zaxpby(ndofs, c1, impl_->u_n->data(), c2, impl_->u_nm1->data(), impl_->z->data());
        }

        impl_->inertia->set_alpha(impl_->alpha_a);
    }

    void BDF2Scheme::reconstruct(const real_t *const x,
                                 real_t *const       velocity,
                                 real_t *const       acceleration) const {
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;

        // Both are the forms `begin_step` built the step around, so neither
        // restates the method's coefficients: the velocity is the shift and the
        // history the material already reads, and the acceleration is the
        // separable half the inertia already assembles.
        blas->zaxpby(ndofs, impl_->shift, x, 1, impl_->z->data(), velocity);
        blas->zaxpby(ndofs, impl_->alpha_a, x, -impl_->alpha_a, impl_->u_hat->data(), acceleration);
    }

    void BDF2Scheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("BDF2Scheme::advance");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;

        // Rotate first, then write the new step into the vacated slots: the
        // velocity is reconstructed from `z`, which `begin_step` built from the
        // state being rotated, so reconstructing after the rotation would read
        // a history that no longer matches it.
        blas->copy(ndofs, impl_->v_n->data(), impl_->v_nm1->data());
        blas->zaxpby(ndofs, impl_->shift, x, 1, impl_->z->data(), impl_->v_n->data());

        blas->copy(ndofs, impl_->u_n->data(), impl_->u_nm1->data());
        blas->copy(ndofs, x, impl_->u_n->data());

        impl_->dt_n    = impl_->dt;
        impl_->started = true;
    }

    void BDF2Scheme::start(real_t *const x) const {
        impl_->blas->zaxpby(impl_->space->n_dofs(), 1, impl_->u_n->data(), impl_->dt, impl_->v_n->data(), x);
    }

    void BDF2Scheme::carried_state(real_t *const velocity, real_t *const acceleration) const {
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->blas->copy(ndofs, impl_->v_n->data(), velocity);
        if (!acceleration) return;
        if (!impl_->started) {
            impl_->blas->zeros(ndofs, acceleration);
            return;
        }
        // Not carried: the step's acceleration at its solution, which `advance` left in u_n.
        impl_->blas->zaxpby(ndofs, impl_->alpha_a, impl_->u_n->data(), -impl_->alpha_a, impl_->u_hat->data(), acceleration);
    }

    std::shared_ptr<Op> BDF2Scheme::inertia_op() const { return impl_->inertia; }

    std::shared_ptr<Buffer<real_t>> BDF2Scheme::state() const { return impl_->u_n; }
    std::shared_ptr<Buffer<real_t>> BDF2Scheme::velocity() const { return impl_->v_n; }

}  // namespace sfem
