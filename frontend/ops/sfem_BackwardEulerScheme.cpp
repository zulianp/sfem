#include "sfem_BackwardEulerScheme.hpp"

#include <cstring>

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class BackwardEulerScheme::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        ExecutionSpace                  es{EXECUTION_SPACE_HOST};
        std::shared_ptr<BLAS<real_t>>   blas;
        std::shared_ptr<Buffer<real_t>> u_n;
        std::shared_ptr<Buffer<real_t>> z;
        real_t                          dt{0};
        real_t                          shift{0};

        Impl(const std::shared_ptr<FunctionSpace> &sp, const ExecutionSpace space_of)
            : space(sp), es(space_of), blas(sfem::blas<real_t>(space_of)) {}
    };

    BackwardEulerScheme::BackwardEulerScheme(const std::shared_ptr<FunctionSpace> &space,
                                             const ExecutionSpace                  es)
        : impl_(std::make_unique<Impl>(space, es)) {}

    BackwardEulerScheme::~BackwardEulerScheme() = default;

    int BackwardEulerScheme::initialize() {
        SFEM_TRACE_SCOPE("BackwardEulerScheme::initialize");

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        impl_->u_n            = create_buffer<real_t>(ndofs, impl_->es);
        impl_->z              = create_buffer<real_t>(ndofs, impl_->es);
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
        // BLAS rather than a loop, so it runs wherever the caller's vectors
        // live.
        impl_->blas->copy(ndofs, impl_->u_n->data(), impl_->z->data());
        impl_->blas->scal(ndofs, -impl_->shift, impl_->z->data());
    }

    void BackwardEulerScheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("BackwardEulerScheme::advance");

        impl_->blas->copy(impl_->space->n_dofs(), x, impl_->u_n->data());
    }

    std::shared_ptr<Buffer<real_t>> BackwardEulerScheme::state() const { return impl_->u_n; }

}  // namespace sfem
