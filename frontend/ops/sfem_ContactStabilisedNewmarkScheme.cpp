#include "sfem_ContactStabilisedNewmarkScheme.hpp"

#include "sfem_API.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    class ContactStabilisedNewmarkScheme::Impl {
    public:
        std::shared_ptr<NewmarkScheme>  newmark;
        std::shared_ptr<Contact>        contact;
        std::shared_ptr<BLAS<real_t>>   blas;
        ptrdiff_t                       ndofs{0};
        std::shared_ptr<Buffer<real_t>> inv_mass;
        std::shared_ptr<Buffer<real_t>> reaction;
    };

    ContactStabilisedNewmarkScheme::ContactStabilisedNewmarkScheme(const std::shared_ptr<NewmarkScheme> &newmark,
                                                                   const std::shared_ptr<Contact>       &contact)
        : impl_(std::make_unique<Impl>()) {
        auto mass = newmark->mass();
        if (!mass) {
            SFEM_ERROR("ContactStabilisedNewmarkScheme: the NewmarkScheme must be initialised first\n");
        }

        newmark->claim(this);
        impl_->newmark = newmark;
        impl_->contact = contact;

        const ExecutionSpace es = newmark->inertia_op()->execution_space();
        impl_->blas             = sfem::blas<real_t>(es);
        impl_->ndofs            = (ptrdiff_t)mass->size();
        impl_->inv_mass         = create_buffer<real_t>(impl_->ndofs, es);
        impl_->reaction         = create_buffer<real_t>(impl_->ndofs, es);
        impl_->blas->copy(impl_->ndofs, mass->data(), impl_->inv_mass->data());
        impl_->blas->reciprocal(impl_->ndofs, 1, impl_->inv_mass->data());
    }

    ContactStabilisedNewmarkScheme::~ContactStabilisedNewmarkScheme() { impl_->newmark->release(this); }

    real_t        ContactStabilisedNewmarkScheme::shift() const { return impl_->newmark->shift(); }
    const real_t *ContactStabilisedNewmarkScheme::history() const { return impl_->newmark->history(); }
    real_t        ContactStabilisedNewmarkScheme::weight(const char *name) const { return impl_->newmark->weight(name); }
    void          ContactStabilisedNewmarkScheme::begin_step(const real_t t, const real_t dt) { impl_->newmark->begin_step(t, dt); }

    void ContactStabilisedNewmarkScheme::reconstruct(const real_t *const x,
                                                     real_t *const       velocity,
                                                     real_t *const       acceleration) const {
        impl_->newmark->reconstruct(x, velocity, acceleration);
    }

    std::shared_ptr<Op> ContactStabilisedNewmarkScheme::inertia_op() const { return impl_->newmark->inertia_op(); }

    void ContactStabilisedNewmarkScheme::advance(const real_t *const x) {
        SFEM_TRACE_SCOPE("ContactStabilisedNewmarkScheme::advance");

        auto            blas  = impl_->blas;
        const ptrdiff_t ndofs = impl_->ndofs;
        impl_->newmark->advance(x);

        // a_n = a - M^{-1} reaction: Newmark's acceleration balances the material and the
        // obstacle, the one carried forward only the material.
        real_t *const reaction = impl_->reaction->data();
        impl_->contact->reaction(reaction);
        blas->scal(ndofs, -1, reaction);
        blas->xypaz(ndofs, reaction, impl_->inv_mass->data(), 1, impl_->newmark->acceleration()->data());

        impl_->contact->project_velocity(x, impl_->newmark->velocity()->data(), impl_->newmark->weight("dt"));
    }

    void ContactStabilisedNewmarkScheme::start(real_t *const x) const {
        impl_->blas->zaxpby(impl_->ndofs,
                            1,
                            impl_->newmark->state()->data(),
                            impl_->newmark->weight("dt"),
                            impl_->newmark->velocity()->data(),
                            x);
    }

    const std::shared_ptr<NewmarkScheme> &ContactStabilisedNewmarkScheme::newmark() const { return impl_->newmark; }

}  // namespace sfem
