#include "sfem_InertiaPotential.hpp"

#include "sfem_API.hpp"
#ifdef SFEM_ENABLE_CUDA
#include "cuda/sfem_InertiaPotential_cuda.hpp"
#endif
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

    class InertiaPotential::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<Buffer<real_t>> mass;
        std::shared_ptr<Buffer<real_t>> u_hat;
        ExecutionSpace                  es{EXECUTION_SPACE_HOST};
        std::shared_ptr<BLAS<real_t>>   blas;
        real_t                          alpha{1};
        real_t                          density{1};

        //! Room for the three vectors the quadratic form needs.  Allocated once
        //! rather than per call, and in the caller's space like everything else.
        std::shared_ptr<Buffer<real_t>> offset;   // x - u_hat, or + a step
        std::shared_ptr<Buffer<real_t>> scaled;   // that, times alpha
        std::shared_ptr<Buffer<real_t>> weighted; // and times the mass

        Impl(const std::shared_ptr<FunctionSpace> &sp, const ExecutionSpace space_of)
            : space(sp), es(space_of), blas(sfem::blas<real_t>(space_of)) {}

        int ensure_state() const {
            if (!mass || !u_hat) {
                SFEM_ERROR("InertiaPotential: mass and u_hat must be initialized");
                return SFEM_FAILURE;
            }

            const ptrdiff_t ndofs = space->n_dofs();
            if (mass->size() != static_cast<size_t>(ndofs) || u_hat->size() != static_cast<size_t>(ndofs)) {
                SFEM_ERROR("InertiaPotential: incompatible mass/u_hat sizes");
                return SFEM_FAILURE;
            }

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> InertiaPotential::create(const std::shared_ptr<FunctionSpace> &space) {
        return std::make_unique<InertiaPotential>(space);
    }

    InertiaPotential::InertiaPotential(const std::shared_ptr<FunctionSpace> &space,
                                       const ExecutionSpace                  es)
        : impl_(std::make_unique<Impl>(space, es)) {}

    InertiaPotential::~InertiaPotential() = default;

    ExecutionSpace InertiaPotential::execution_space() const { return impl_->es; }

    ptrdiff_t InertiaPotential::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t InertiaPotential::n_dofs_image() const { return impl_->space->n_dofs(); }

    int InertiaPotential::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("InertiaPotential::initialize");

        const ptrdiff_t ndofs = impl_->space->n_dofs();

        if (!impl_->mass) {
            // The lumped mass is assembled on the host, because `LumpedMass` is
            // a host operator, and then moved to wherever this one runs.  It is
            // built once, so the transfer is once too.
            auto host_mass = create_host_buffer<real_t>(ndofs);
            LumpedMass lumped_mass(impl_->space);
            if (lumped_mass.initialize(block_names) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            if (lumped_mass.hessian_diag(nullptr, host_mass->data()) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            if (impl_->density != real_t(1)) {
                sfem::blas<real_t>(EXECUTION_SPACE_HOST)
                        ->scal(ndofs, impl_->density, host_mass->data());
            }

            impl_->mass = host_mass;
#ifdef SFEM_ENABLE_CUDA
            if (impl_->es == EXECUTION_SPACE_DEVICE) {
                impl_->mass = smesh::to_device(host_mass);
            }
#endif
        }

        if (!impl_->u_hat) {
            impl_->u_hat = create_buffer<real_t>(ndofs, impl_->es);
        }

        impl_->offset   = create_buffer<real_t>(ndofs, impl_->es);
        impl_->scaled   = create_buffer<real_t>(ndofs, impl_->es);
        impl_->weighted = create_buffer<real_t>(ndofs, impl_->es);

        return SFEM_SUCCESS;
    }

    int InertiaPotential::hessian_crs(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("InertiaPotential::hessian_crs");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t     ndofs = impl_->space->n_dofs();
        const real_t        alpha = impl_->alpha;
        const real_t *const mass  = impl_->mass->data();

#ifdef SFEM_ENABLE_CUDA
        if (impl_->es == EXECUTION_SPACE_DEVICE) {
            return cu_inertia_potential_hessian_crs(ndofs, rowptr, colidx, mass, alpha, values);
        }
#endif

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

    int InertiaPotential::hessian_bsr(const real_t *const,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          real_t *const SFEM_RESTRICT        values) {
        SFEM_TRACE_SCOPE("InertiaPotential::hessian_bsr");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const int           bs      = impl_->space->block_size();
        const ptrdiff_t     n_nodes = impl_->space->n_dofs() / bs;
        const ptrdiff_t     bs2     = bs * bs;
        const real_t        alpha   = impl_->alpha;
        const real_t *const mass    = impl_->mass->data();

#ifdef SFEM_ENABLE_CUDA
        if (impl_->es == EXECUTION_SPACE_DEVICE) {
            return cu_inertia_potential_hessian_bsr(n_nodes, bs, rowptr, colidx, mass, alpha, values);
        }
#endif

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

    int InertiaPotential::hessian_diag(const real_t *const, real_t *const SFEM_RESTRICT values) {
        SFEM_TRACE_SCOPE("InertiaPotential::hessian_diag");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        impl_->blas->axpy(impl_->space->n_dofs(), impl_->alpha, impl_->mass->data(), values);
        return SFEM_SUCCESS;
    }

    int InertiaPotential::gradient(const real_t *const SFEM_RESTRICT x, real_t *const SFEM_RESTRICT out) {
        SFEM_TRACE_SCOPE("InertiaPotential::gradient");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        // out += alpha * m * (x - u_hat), in three vector operations so that
        // it runs wherever the caller's vectors are.
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;
        blas->zaxpby(ndofs, impl_->alpha, x, -impl_->alpha, impl_->u_hat->data(), impl_->scaled->data());
        blas->xypaz(ndofs, impl_->mass->data(), impl_->scaled->data(), 1, out);
        return SFEM_SUCCESS;
    }

    int InertiaPotential::apply(const real_t *const, const real_t *const SFEM_RESTRICT h, real_t *const SFEM_RESTRICT out) {
        SFEM_TRACE_SCOPE("InertiaPotential::apply");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;
        blas->copy(ndofs, h, impl_->scaled->data());
        blas->scal(ndofs, impl_->alpha, impl_->scaled->data());
        blas->xypaz(ndofs, impl_->mass->data(), impl_->scaled->data(), 1, out);
        return SFEM_SUCCESS;
    }

    int InertiaPotential::value(const real_t *const SFEM_RESTRICT x, real_t *const out) {
        SFEM_TRACE_SCOPE("InertiaPotential::value");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        // 0.5 * alpha * (x - u_hat)^T M (x - u_hat), as a weighted dot.
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;
        blas->zaxpby(ndofs, 1, x, -1, impl_->u_hat->data(), impl_->offset->data());
        blas->xypaz(ndofs, impl_->mass->data(), impl_->offset->data(), 0, impl_->weighted->data());
        *out += real_t(0.5) * impl_->alpha *
                blas->dot(ndofs, impl_->offset->data(), impl_->weighted->data());
        return SFEM_SUCCESS;
    }

    int InertiaPotential::value_steps(const real_t *const SFEM_RESTRICT x,
                                          const real_t *const SFEM_RESTRICT h,
                                          const int                         nsteps,
                                          const real_t *const SFEM_RESTRICT steps,
                                          real_t *const SFEM_RESTRICT       out) {
        SFEM_TRACE_SCOPE("InertiaPotential::value_steps");

        if (impl_->ensure_state() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }

        // The merit at `x + step * h` for each step, as the same weighted dot
        // the 0-form above is, once per step.  The vector work goes through
        // BLAS so a device caller gets device reductions.
        const ptrdiff_t ndofs = impl_->space->n_dofs();
        auto            blas  = impl_->blas;
        const real_t    half  = real_t(0.5) * impl_->alpha;

        blas->zaxpby(ndofs, 1, x, -1, impl_->u_hat->data(), impl_->offset->data());
        for (int step = 0; step < nsteps; ++step) {
            blas->zaxpby(ndofs, 1, impl_->offset->data(), steps[step], h, impl_->scaled->data());
            blas->xypaz(ndofs, impl_->mass->data(), impl_->scaled->data(), 0, impl_->weighted->data());
            out[step] += half * blas->dot(ndofs, impl_->scaled->data(), impl_->weighted->data());
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> InertiaPotential::clone() const {
        auto ret            = std::make_shared<InertiaPotential>(impl_->space, impl_->es);
        ret->impl_->mass    = impl_->mass;
        ret->impl_->u_hat   = impl_->u_hat;
        ret->impl_->alpha   = impl_->alpha;
        ret->impl_->density = impl_->density;
        return ret;
    }

    void InertiaPotential::set_alpha(const real_t alpha) { impl_->alpha = alpha; }

    void InertiaPotential::set_density(const real_t density) { impl_->density = density; }

    void InertiaPotential::set_u_hat(const std::shared_ptr<Buffer<real_t>> &u_hat) { impl_->u_hat = u_hat; }

    void InertiaPotential::set_mass(const std::shared_ptr<Buffer<real_t>> &mass) { impl_->mass = mass; }

    void InertiaPotential::set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &values, const int) {
        if (!strcmp(name, "u_hat")) {
            impl_->u_hat = values;
        } else if (!strcmp(name, "mass")) {
            impl_->mass = values;
        }
    }

    void InertiaPotential::set_value_in_block(const std::string &, const std::string &var_name, const real_t value) {
        if (var_name == "alpha") {
            impl_->alpha = value;
        } else if (var_name == "density") {
            impl_->density = value;
        }
    }

    std::shared_ptr<Buffer<real_t>> InertiaPotential::mass() const { return impl_->mass; }

    std::shared_ptr<Buffer<real_t>> InertiaPotential::u_hat() const { return impl_->u_hat; }

}  // namespace sfem
