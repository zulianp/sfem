#include "sfem_BDF2InertiaPotential.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_LumpedMass.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>

namespace sfem {

    class BDF2InertiaPotential::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Buffer<real_t>> mass;
        std::shared_ptr<Buffer<real_t>> u_hat;
        real_t                          alpha{1};
        real_t                          density{1};

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {}

        int ensure_state() const {
            if (!mass || !u_hat) {
                SFEM_ERROR("BDF2InertiaPotential: mass and u_hat must be initialized");
                return SFEM_FAILURE;
            }

            const ptrdiff_t ndofs = space->n_dofs();
            if (mass->size() != static_cast<size_t>(ndofs) || u_hat->size() != static_cast<size_t>(ndofs)) {
                SFEM_ERROR("BDF2InertiaPotential: incompatible mass/u_hat sizes");
                return SFEM_FAILURE;
            }

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> BDF2InertiaPotential::create(const std::shared_ptr<FunctionSpace> &space) {
        return std::make_unique<BDF2InertiaPotential>(space);
    }

    BDF2InertiaPotential::BDF2InertiaPotential(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    BDF2InertiaPotential::~BDF2InertiaPotential() = default;

    ptrdiff_t BDF2InertiaPotential::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t BDF2InertiaPotential::n_dofs_image() const { return impl_->space->n_dofs(); }

    int BDF2InertiaPotential::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::initialize");

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
                const real_t                 rho  = impl_->density;
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

    int BDF2InertiaPotential::hessian_crs(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::hessian_crs");

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

    int BDF2InertiaPotential::hessian_bsr(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::hessian_bsr");

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

    int BDF2InertiaPotential::hessian_diag(const real_t *const, real_t *const SFEM_RESTRICT values) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::hessian_diag");

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

    int BDF2InertiaPotential::gradient(const real_t *const SFEM_RESTRICT x, real_t *const SFEM_RESTRICT out) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::gradient");

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

    int BDF2InertiaPotential::apply(const real_t *const,
                                    const real_t *const SFEM_RESTRICT h,
                                    real_t *const SFEM_RESTRICT       out) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::apply");

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

    int BDF2InertiaPotential::value(const real_t *const SFEM_RESTRICT x, real_t *const out) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::value");

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

    int BDF2InertiaPotential::value_steps(const real_t *const SFEM_RESTRICT x,
                                          const real_t *const SFEM_RESTRICT h,
                                          const int                          nsteps,
                                          const real_t *const SFEM_RESTRICT steps,
                                          real_t *const SFEM_RESTRICT       out) {
        SFEM_TRACE_SCOPE("BDF2InertiaPotential::value_steps");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();
        const real_t *const uhat  = impl_->u_hat->data();

        for (int s = 0; s < nsteps; ++s) {
            const real_t step = steps[s];
            real_t       acc  = 0;
#pragma omp parallel for reduction(+ : acc)
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                const real_t diff = x[i] + step * h[i] - uhat[i];
                acc += mass[i] * diff * diff;
            }

            out[s] += real_t(0.5) * alpha * acc;
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> BDF2InertiaPotential::clone() const {
        auto ret              = std::make_shared<BDF2InertiaPotential>(impl_->space);
        ret->impl_->mass      = impl_->mass;
        ret->impl_->u_hat     = impl_->u_hat;
        ret->impl_->alpha     = impl_->alpha;
        ret->impl_->density   = impl_->density;
        return ret;
    }

    void BDF2InertiaPotential::set_alpha(const real_t alpha) { impl_->alpha = alpha; }

    void BDF2InertiaPotential::set_density(const real_t density) { impl_->density = density; }

    void BDF2InertiaPotential::set_u_hat(const std::shared_ptr<Buffer<real_t>> &u_hat) { impl_->u_hat = u_hat; }

    void BDF2InertiaPotential::set_mass(const std::shared_ptr<Buffer<real_t>> &mass) { impl_->mass = mass; }

    void BDF2InertiaPotential::set_field(const char *name,
                                         const std::shared_ptr<Buffer<real_t>> &values,
                                         const int) {
        if (!strcmp(name, "u_hat")) {
            impl_->u_hat = values;
        } else if (!strcmp(name, "mass")) {
            impl_->mass = values;
        }
    }

    void BDF2InertiaPotential::set_value_in_block(const std::string &,
                                                  const std::string &var_name,
                                                  const real_t       value) {
        if (var_name == "alpha") {
            impl_->alpha = value;
        } else if (var_name == "density") {
            impl_->density = value;
        }
    }

    std::shared_ptr<Buffer<real_t>> BDF2InertiaPotential::mass() const { return impl_->mass; }

    std::shared_ptr<Buffer<real_t>> BDF2InertiaPotential::u_hat() const { return impl_->u_hat; }

}  // namespace sfem
