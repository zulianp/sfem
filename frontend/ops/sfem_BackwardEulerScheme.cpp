#include "sfem_BackwardEulerScheme.hpp"

#include <cstring>

#include "sfem_FunctionSpace.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class BackwardEulerScheme::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<Buffer<real_t>> u_n;
        std::shared_ptr<Buffer<real_t>> z;
        real_t                          dt{0};
        real_t                          shift{0};

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {}
    };

    BackwardEulerScheme::BackwardEulerScheme(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    BackwardEulerScheme::~BackwardEulerScheme() = default;

    int BackwardEulerScheme::initialize() {
        SFEM_TRACE_SCOPE("BackwardEulerScheme::initialize");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->u_n            = create_host_buffer<real_t>(ndofs);
        impl_->z              = create_host_buffer<real_t>(ndofs);
        return SFEM_SUCCESS;
    }

    real_t        BackwardEulerScheme::shift() const { return impl_->shift; }
    const real_t *BackwardEulerScheme::history() const { return impl_->z->data(); }

    real_t BackwardEulerScheme::weight(const char *name) const {
        if (!std::strcmp(name, "dt")) return impl_->dt;
        if (!std::strcmp(name, "shift")) return impl_->shift;
        SFEM_ERROR("BackwardEulerScheme has no weight named \"%s\"\n", name);
        return 0;
    }

    void BackwardEulerScheme::begin_step(const real_t, const real_t dt) {
        SFEM_TRACE_SCOPE("BackwardEulerScheme::begin_step");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->dt             = dt;
        impl_->shift          = 1 / dt;

        // Built once per step from the state carried out of the last one, and
        // never from the current iterate: `Function::value_steps` re-assembles
        // the residual at every trial step length of a line search, and a merit
        // that moved under the search would not be a merit.
        const real_t *const SFEM_RESTRICT u_n   = impl_->u_n->data();
        real_t *const SFEM_RESTRICT       z     = impl_->z->data();
        const real_t                      shift = impl_->shift;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            z[i] = -shift * u_n[i];
        }
    }

    void BackwardEulerScheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("BackwardEulerScheme::advance");

        const ptrdiff_t             ndofs = impl_->space->n_dofs();
        real_t *const SFEM_RESTRICT u_n   = impl_->u_n->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            u_n[i] = x[i];
        }
    }

    std::shared_ptr<Buffer<real_t>> BackwardEulerScheme::state() const { return impl_->u_n; }

}  // namespace sfem
