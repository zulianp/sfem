#include "sfem_NewmarkInertiaPotential.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_LumpedMass.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace sfem {

    class NewmarkInertiaPotential::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<Buffer<real_t>> mass;
        std::shared_ptr<Buffer<real_t>> u_hat;
        real_t                          alpha{1};
        real_t                          density{1};

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {}

        int ensure_state() const {
            if (!mass || !u_hat) {
                SFEM_ERROR("NewmarkInertiaPotential: mass and u_hat must be initialized");
                return SFEM_FAILURE;
            }

            const ptrdiff_t ndofs = space->n_dofs();
            if (mass->size() != static_cast<size_t>(ndofs) || u_hat->size() != static_cast<size_t>(ndofs)) {
                SFEM_ERROR("NewmarkInertiaPotential: incompatible mass/u_hat sizes");
                return SFEM_FAILURE;
            }

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> NewmarkInertiaPotential::create(const std::shared_ptr<FunctionSpace> &space) {
        return std::make_unique<NewmarkInertiaPotential>(space);
    }

    NewmarkInertiaPotential::NewmarkInertiaPotential(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    NewmarkInertiaPotential::~NewmarkInertiaPotential() = default;

    ptrdiff_t NewmarkInertiaPotential::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t NewmarkInertiaPotential::n_dofs_image() const { return impl_->space->n_dofs(); }

    int NewmarkInertiaPotential::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::initialize");

        const ptrdiff_t ndofs = impl_->space->n_dofs();

        if (!impl_->mass) {
            impl_->mass = create_host_buffer<real_t>(ndofs);
            LumpedMass lumped_mass(impl_->space);
            if (lumped_mass.initialize(block_names) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            if (lumped_mass.hessian_diag(nullptr, impl_->mass->data()) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            if (impl_->density != real_t(1)) {
                real_t *const SFEM_RESTRICT mass = impl_->mass->data();
                const real_t                rho  = impl_->density;
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < ndofs; ++i) {
                    mass[i] *= rho;
                }
            }
        }

        if (!impl_->u_hat) {
            impl_->u_hat = create_host_buffer<real_t>(ndofs);
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::hessian_crs(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::hessian_crs");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const count_t begin = rowptr[i];
            const count_t end   = rowptr[i + 1];
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] == i) {
                    values[k] += alpha * mass[i];
                    break;
                }
            }
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::hessian_bsr(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::hessian_bsr");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const int           bs      = impl_->space->block_size();
        const ptrdiff_t     n_nodes = impl_->space->n_dofs() / bs;
        const ptrdiff_t     bs2     = bs * bs;
        const real_t        alpha   = impl_->alpha;
        const real_t *const mass    = impl_->mass->data();

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            const count_t begin = rowptr[node];
            const count_t end   = rowptr[node + 1];
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] == node) {
                    real_t *const block = &values[k * bs2];
                    for (int d = 0; d < bs; ++d) {
                        block[d * bs + d] += alpha * mass[node * bs + d];
                    }
                    break;
                }
            }
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::hessian_diag(const real_t *const, real_t *const SFEM_RESTRICT values) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::hessian_diag");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            values[i] += alpha * mass[i];
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::gradient(const real_t *const SFEM_RESTRICT x, real_t *const SFEM_RESTRICT out) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::gradient");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();
        const real_t *const uhat  = impl_->u_hat->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            out[i] += alpha * mass[i] * (x[i] - uhat[i]);
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::apply(const real_t *const, const real_t *const SFEM_RESTRICT h, real_t *const SFEM_RESTRICT out) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::apply");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            out[i] += alpha * mass[i] * h[i];
        }

        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::value(const real_t *const SFEM_RESTRICT x, real_t *const out) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::value");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();
        const real_t *const uhat  = impl_->u_hat->data();
        real_t              acc   = 0;

#pragma omp parallel for reduction(+ : acc)
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const real_t diff = x[i] - uhat[i];
            acc += mass[i] * diff * diff;
        }

        *out += real_t(0.5) * alpha * acc;
        return SFEM_SUCCESS;
    }

    int NewmarkInertiaPotential::value_steps(const real_t *const SFEM_RESTRICT x,
                                          const real_t *const SFEM_RESTRICT h,
                                          const int                         nsteps,
                                          const real_t *const SFEM_RESTRICT steps,
                                          real_t *const SFEM_RESTRICT       out) {
        SFEM_TRACE_SCOPE("NewmarkInertiaPotential::value_steps");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();
        const real_t *const uhat  = impl_->u_hat->data();

        const real_t half_alpha = real_t(0.5) * alpha;

#pragma omp parallel
        {
            real_t *const SFEM_RESTRICT acc = (real_t *)std::calloc(nsteps, sizeof(real_t));

#pragma omp for
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                const real_t x_minus_uhat = x[i] - uhat[i];
                const real_t hi           = h[i];
                const real_t scale        = half_alpha * mass[i];

#pragma omp simd
                for (int s = 0; s < nsteps; ++s) {
                    const real_t diff = x_minus_uhat + steps[s] * hi;
                    acc[s] += scale * diff * diff;
                }
            }

            for (int s = 0; s < nsteps; ++s) {
#pragma omp atomic update
                out[s] += acc[s];
            }

            std::free(acc);
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> NewmarkInertiaPotential::clone() const {
        auto ret            = std::make_shared<NewmarkInertiaPotential>(impl_->space);
        ret->impl_->mass    = impl_->mass;
        ret->impl_->u_hat   = impl_->u_hat;
        ret->impl_->alpha   = impl_->alpha;
        ret->impl_->density = impl_->density;
        return ret;
    }

    void NewmarkInertiaPotential::set_alpha(const real_t alpha) { impl_->alpha = alpha; }

    void NewmarkInertiaPotential::set_density(const real_t density) { impl_->density = density; }

    void NewmarkInertiaPotential::set_u_hat(const std::shared_ptr<Buffer<real_t>> &u_hat) { impl_->u_hat = u_hat; }

    void NewmarkInertiaPotential::set_mass(const std::shared_ptr<Buffer<real_t>> &mass) { impl_->mass = mass; }

    void NewmarkInertiaPotential::set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &values, const int) {
        if (!strcmp(name, "u_hat")) {
            impl_->u_hat = values;
        } else if (!strcmp(name, "mass")) {
            impl_->mass = values;
        }
    }

    void NewmarkInertiaPotential::set_value_in_block(const std::string &, const std::string &var_name, const real_t value) {
        if (var_name == "alpha") {
            impl_->alpha = value;
        } else if (var_name == "density") {
            impl_->density = value;
        }
    }

    std::shared_ptr<Buffer<real_t>> NewmarkInertiaPotential::mass() const { return impl_->mass; }

    std::shared_ptr<Buffer<real_t>> NewmarkInertiaPotential::u_hat() const { return impl_->u_hat; }

}  // namespace sfem
