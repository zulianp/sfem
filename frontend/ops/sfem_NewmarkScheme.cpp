#include "sfem_NewmarkScheme.hpp"

#include <cstring>

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class NewmarkScheme::Impl {
    public:
        std::shared_ptr<FunctionSpace>   space;
        std::shared_ptr<InertiaPotential> inertia;
        ExecutionSpace                   es{EXECUTION_SPACE_HOST};
        std::shared_ptr<BLAS<real_t>>    blas;

        std::shared_ptr<Buffer<real_t>> u_n;
        std::shared_ptr<Buffer<real_t>> v_n;
        std::shared_ptr<Buffer<real_t>> a_n;
        std::shared_ptr<Buffer<real_t>> z;
        std::shared_ptr<Buffer<real_t>> u_hat;

        real_t beta{real_t(0.25)};
        real_t gamma{real_t(0.5)};
        real_t dt{0};
        real_t shift{0};
        real_t alpha_a{0};

        Impl(const std::shared_ptr<FunctionSpace> &sp, const ExecutionSpace space_of)
            : space(sp), es(space_of), blas(sfem::blas<real_t>(space_of)) {}
    };

    NewmarkScheme::NewmarkScheme(const std::shared_ptr<FunctionSpace> &space, const ExecutionSpace es)
        : impl_(std::make_unique<Impl>(space, es)) {
        impl_->inertia = std::make_shared<InertiaPotential>(space);
    }

    NewmarkScheme::~NewmarkScheme() = default;

    void NewmarkScheme::set_beta(const real_t beta) { impl_->beta = beta; }
    void NewmarkScheme::set_gamma(const real_t gamma) { impl_->gamma = gamma; }
    void NewmarkScheme::set_density(const real_t density) { impl_->inertia->set_density(density); }

    int NewmarkScheme::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("NewmarkScheme::initialize");

        if (impl_->beta <= real_t(0)) {
            SFEM_ERROR("NewmarkScheme requires beta > 0; beta = %g\n", (double)impl_->beta);
            return SFEM_FAILURE;
        }

        // The state follows the execution space the caller runs in: a driver
        // that solves on the device holds device vectors and hands this one of
        // them in `advance`, so the scheme's own state has to live there too.
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->u_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->v_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->a_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->z              = create_buffer<real_t>(ndofs, impl_->es);
        impl_->u_hat          = create_buffer<real_t>(ndofs, impl_->es);

        // One predictor, not two: the inertia operator reads the buffer this
        // scheme writes, so the two cannot drift apart.
        impl_->inertia->set_u_hat(impl_->u_hat);
        return impl_->inertia->initialize(block_names);
    }

    real_t        NewmarkScheme::shift() const { return impl_->shift; }
    const real_t *NewmarkScheme::history() const { return impl_->z->data(); }

    real_t NewmarkScheme::weight(const char *name) const {
        if (!std::strcmp(name, "dt")) return impl_->dt;
        if (!std::strcmp(name, "beta")) return impl_->beta;
        if (!std::strcmp(name, "gamma")) return impl_->gamma;
        if (!std::strcmp(name, "shift")) return impl_->shift;
        if (!std::strcmp(name, "alpha_a")) return impl_->alpha_a;
        SFEM_ERROR("NewmarkScheme has no weight named \"%s\"\n", name);
        return 0;
    }

    void NewmarkScheme::begin_step(const real_t, const real_t dt) {
        SFEM_TRACE_SCOPE("NewmarkScheme::begin_step");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->dt             = dt;
        impl_->alpha_a        = 1 / (impl_->beta * dt * dt);
        impl_->shift          = impl_->gamma / (impl_->beta * dt);

        // Everything the solve reads is built here, from the state carried out
        // of the last step and never from the current iterate -- which is what
        // lets a line search re-assemble the residual at several trial step
        // lengths and compare merits that mean the same thing.
        //
        // Written in BLAS rather than as a loop so it runs wherever the
        // caller's vectors live.
        auto         blas    = impl_->blas;
        const real_t predict = dt * dt * (real_t(0.5) - impl_->beta);
        const real_t a_scale = dt * (1 - impl_->gamma);

        blas->zaxpby(ndofs, 1, impl_->u_n->data(), dt, impl_->v_n->data(), impl_->u_hat->data());
        blas->axpy(ndofs, predict, impl_->a_n->data(), impl_->u_hat->data());

        blas->zaxpby(ndofs, 1, impl_->v_n->data(), a_scale, impl_->a_n->data(), impl_->z->data());
        blas->axpy(ndofs, -impl_->shift, impl_->u_hat->data(), impl_->z->data());

        impl_->inertia->set_alpha(impl_->alpha_a);
    }

    void NewmarkScheme::reconstruct(const real_t *const x,
                                    real_t *const       velocity,
                                    real_t *const       acceleration) const {
        const ptrdiff_t ndofs   = impl_->space->n_dofs();
        auto            blas    = impl_->blas;
        const real_t    alpha_a = impl_->alpha_a;

        blas->zaxpby(ndofs, alpha_a, x, -alpha_a, impl_->u_hat->data(), acceleration);
        blas->zaxpby(ndofs, impl_->shift, x, 1, impl_->z->data(), velocity);
    }

    void NewmarkScheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("NewmarkScheme::advance");

        // Reconstruct into the carried state and then take the state itself.
        // `u_hat` and `z` are the step's, not the iterate's, so neither is the
        // buffer being overwritten and the order of the two is free.
        reconstruct(x, impl_->v_n->data(), impl_->a_n->data());
        impl_->blas->copy(impl_->space->n_dofs(), x, impl_->u_n->data());
    }

    std::shared_ptr<Op> NewmarkScheme::inertia_op() const { return impl_->inertia; }

    std::shared_ptr<Buffer<real_t>> NewmarkScheme::state() const { return impl_->u_n; }
    std::shared_ptr<Buffer<real_t>> NewmarkScheme::velocity() const { return impl_->v_n; }
    std::shared_ptr<Buffer<real_t>> NewmarkScheme::acceleration() const { return impl_->a_n; }

}  // namespace sfem
