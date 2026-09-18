#include "sfem_NewmarkScheme.hpp"

#include <cstring>

#include "sfem_FunctionSpace.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class NewmarkScheme::Impl {
    public:
        std::shared_ptr<FunctionSpace>            space;
        std::shared_ptr<InertiaPotential> inertia;

        std::shared_ptr<Buffer<real_t>> u_n;
        std::shared_ptr<Buffer<real_t>> v_n;
        std::shared_ptr<Buffer<real_t>> a_n;
        std::shared_ptr<Buffer<real_t>> z;

        real_t beta{real_t(0.25)};
        real_t gamma{real_t(0.5)};
        real_t dt{0};
        real_t shift{0};
        real_t alpha_a{0};

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {}
    };

    NewmarkScheme::NewmarkScheme(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {
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

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->u_n            = create_host_buffer<real_t>(ndofs);
        impl_->v_n            = create_host_buffer<real_t>(ndofs);
        impl_->a_n            = create_host_buffer<real_t>(ndofs);
        impl_->z              = create_host_buffer<real_t>(ndofs);

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
        // lets the line search re-assemble the residual at nine trial step
        // lengths and compare merits that mean the same thing.
        const real_t *const SFEM_RESTRICT u_n     = impl_->u_n->data();
        const real_t *const SFEM_RESTRICT v_n     = impl_->v_n->data();
        const real_t *const SFEM_RESTRICT a_n     = impl_->a_n->data();
        real_t *const SFEM_RESTRICT       u_hat   = impl_->inertia->u_hat()->data();
        real_t *const SFEM_RESTRICT       z       = impl_->z->data();
        const real_t                      predict = dt * dt * (real_t(0.5) - impl_->beta);
        const real_t                      a_scale = dt * (1 - impl_->gamma);
        const real_t                      shift   = impl_->shift;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const real_t u_hat_i = u_n[i] + dt * v_n[i] + predict * a_n[i];
            u_hat[i]             = u_hat_i;
            z[i]                 = v_n[i] + a_scale * a_n[i] - shift * u_hat_i;
        }

        impl_->inertia->set_alpha(impl_->alpha_a);
    }

    void NewmarkScheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("NewmarkScheme::advance");

        const ptrdiff_t                   ndofs   = impl_->space->n_dofs();
        const real_t *const SFEM_RESTRICT u_hat   = impl_->inertia->u_hat()->data();
        const real_t *const SFEM_RESTRICT z       = impl_->z->data();
        real_t *const SFEM_RESTRICT       u_n     = impl_->u_n->data();
        real_t *const SFEM_RESTRICT       v_n     = impl_->v_n->data();
        real_t *const SFEM_RESTRICT       a_n     = impl_->a_n->data();
        const real_t                      alpha_a = impl_->alpha_a;
        const real_t                      shift   = impl_->shift;

        // Reconstruct and rotate in one pass: the new velocity and acceleration
        // are read from `x`, `u_hat` and `z`, none of which is the state being
        // overwritten.
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const real_t x_i = x[i];
            a_n[i]           = alpha_a * (x_i - u_hat[i]);
            v_n[i]           = shift * x_i + z[i];
            u_n[i]           = x_i;
        }
    }

    std::shared_ptr<Op> NewmarkScheme::inertia_op() const { return impl_->inertia; }

    std::shared_ptr<Buffer<real_t>> NewmarkScheme::state() const { return impl_->u_n; }
    std::shared_ptr<Buffer<real_t>> NewmarkScheme::velocity() const { return impl_->v_n; }
    std::shared_ptr<Buffer<real_t>> NewmarkScheme::acceleration() const { return impl_->a_n; }

}  // namespace sfem
