#include "sfem_NodeObstacleStep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "sfem_ContactCommon.hpp"
#include "sfem_Merit.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_mprgp.hpp"

namespace sfem {

    namespace {
        /// The buffer itself on the host, a host copy of it on the device.
        SharedBuffer<real_t> on_host(const SharedBuffer<real_t> &buffer) {
#ifdef SFEM_ENABLE_CUDA
            if (buffer->mem_space() == MEMORY_SPACE_DEVICE) return smesh::to_host(buffer);
#endif
            return buffer;
        }
    }  // namespace

    class NodeObstacleStep::Impl {
    public:
        std::shared_ptr<Function>            f;
        std::shared_ptr<NodeObstacleContact> contact;
        std::shared_ptr<NewmarkScheme>       newmark;
        NodeObstacleStepOptions              options;
        std::shared_ptr<BLAS<real_t>>        blas;
        ptrdiff_t                            ndofs{0};
        std::shared_ptr<BSRTangent>          tangent;
        std::shared_ptr<Operator<real_t>>    tangent_op;
        std::shared_ptr<MPRGP<real_t>>       qp;
        SharedBuffer<real_t>                 k, k_ref, projected, upper_bound, qp_rhs, framed, lift, diag, trial_r;
        real_t                               k_max{0};
        real_t                               tol{0};
        bool                                 first_measure{true};
        int                                  iterations{0};

        real_t kkt(const real_t *const r, real_t &penetration) {
            return contact->kkt_residual(r, k->data(), projected->data(), &penetration);
        }

        int solve_qp(const real_t *const r) {
            qp->set_max_eig(tangent->max_abs_row_sum());
            blas->zaxpby(ndofs, -1, r, 0, r, qp_rhs->data());
            // The constrained rows are the identity: their right-hand side and start are the
            // increment the step arrived with.
            f->copy_constrained_dofs(lift->data(), qp_rhs->data());
            contact->to_frame(qp_rhs->data());
            blas->copy(ndofs, lift->data(), framed->data());
            contact->to_frame(framed->data());
            const int status = qp->apply(qp_rhs->data(), framed->data());
            iterations += qp->iterations();
            return status;
        }

        /// The merit at x with residual r: the KKT residual, penetration counted as k_max of it.
        real_t merit_at(const real_t *const x, const real_t *const r) {
            contact->update(x);
            real_t       penetration = 0;
            const real_t value       = kkt(r, penetration);
            return value + k_max * penetration;
        }

        /// The merit at x, its residual evaluated here; +inf where it cannot be.
        real_t merit_value(const real_t *const x) {
            f->update(x);
            blas->zeros(ndofs, trial_r->data());
            if (f->gradient(x, trial_r->data()) != SFEM_SUCCESS) return std::numeric_limits<real_t>::infinity();
            f->set_value_to_constrained_dofs(0, trial_r->data());
            const real_t m = merit_at(x, trial_r->data());
            return std::isfinite(m) ? m : std::numeric_limits<real_t>::infinity();
        }
    };


    NodeObstacleStep::NodeObstacleStep() : impl_(std::make_unique<Impl>()) {}
    NodeObstacleStep::~NodeObstacleStep() = default;

    std::shared_ptr<NodeObstacleStep> NodeObstacleStep::create(const std::shared_ptr<Function>            &f,
                                                               const std::shared_ptr<NodeObstacleContact> &contact,
                                                               const std::shared_ptr<NewmarkScheme>       &newmark,
                                                               const NodeObstacleStepOptions              &options) {
        const ExecutionSpace es = f->execution_space();
        if (contact->execution_space() != es) {
            fprintf(stderr, "NodeObstacleStep: the contact and the function must share an execution space\n");
            return nullptr;
        }

        std::shared_ptr<NodeObstacleStep> ret(new NodeObstacleStep());
        auto                             &impl = *ret->impl_;
        impl.f                                 = f;
        impl.contact                           = contact;
        impl.newmark                           = newmark;
        impl.options                           = options;
        impl.blas                              = blas<real_t>(es);
        impl.ndofs                             = f->space()->n_dofs();
        impl.tangent                           = BSRTangent::create(f, es);
        if (!impl.tangent) return nullptr;

        for (auto *buffer : {&impl.k, &impl.k_ref, &impl.projected, &impl.upper_bound, &impl.qp_rhs, &impl.framed, &impl.lift, &impl.diag, &impl.trial_r}) {
            *buffer = create_buffer<real_t>(impl.ndofs, es);
        }

        // Each dof's stiffness, what turns a reaction into a displacement: the tangent's diagonal
        // at the reference configuration, where no element is inverted. Once: it only scales.
        {
            auto reference = create_buffer<real_t>(impl.ndofs, es);
            if (impl.tangent->assemble(reference->data()) != SFEM_SUCCESS) return nullptr;
            if (impl.tangent->diagonal(impl.k_ref->data()) != SFEM_SUCCESS) return nullptr;
        }

        // The step in the contact nodes' frames, Q^T K Q dx' = -Q^T R with dx'_n <= gap_i: the
        // constraints hold exactly at every iterate.
        // The QP is the step's, so it reads the step's state through `self`. MPRGP zeros the
        // output before every product, which the frames then turn whole: out = Q^T K Q in.
        Impl *const self   = &impl;
        auto        framed = create_buffer<real_t>(impl.ndofs, es);
        auto        op     = make_op<real_t>(
                impl.ndofs,
                impl.ndofs,
                [self, framed](const real_t *const in, real_t *const out) {
                    self->blas->copy(self->ndofs, in, framed->data());
                    self->contact->from_frame(framed->data());
                    self->tangent_op->apply(framed->data(), out);
                    self->contact->to_frame(out);
                },
                es);
        impl.qp             = create_mprgp(op, es);
        impl.qp->verbose    = false;
        impl.qp->check_each = std::numeric_limits<int>::max();
        impl.qp->set_max_it(options.qp_max_it);
        impl.qp->set_rtol(options.qp_rtol);
        impl.qp->set_upper_bound(impl.upper_bound);
        return ret;
    }

    int NodeObstacleStep::begin_step(const StepContext &context) {
        auto &impl = *impl_;
        if (impl.newmark && !context.scheme) {
            fprintf(stderr, "NodeObstacleStep: a Newmark step needs its scheme\n");
            return SFEM_FAILURE;
        }

        // The scale of each node's equation: its stiffness, plus the inertia c0 M in a dynamic step.
        impl.blas->copy(impl.ndofs, impl.k_ref->data(), impl.k->data());
        if (impl.newmark) {
            impl.blas->axpy(impl.ndofs, context.scheme->weight("alpha_a"), impl.newmark->mass()->data(), impl.k->data());
        }
        // Read on the host: once per step.
        auto                host  = on_host(impl.k);
        const real_t *const k     = host->data();
        const idx_t *const  nodes = impl.contact->nodes()->data();
        real_t              k_max = 0;
#pragma omp parallel for reduction(max : k_max)
        for (ptrdiff_t i = 0; i < impl.contact->size(); ++i) {
            k_max = std::max(k_max, k[(ptrdiff_t)nodes[i] * 3]);
        }
        impl.k_max         = k_max;
        impl.first_measure = true;
        return SFEM_SUCCESS;
    }

    StepMeasure NodeObstacleStep::measure(const real_t *const x, const real_t *const r) {
        auto &impl = *impl_;
        impl.contact->update(x);

        // With contact the residual does not vanish at the solution: where the obstacle is
        // touched it is the obstacle's reaction. The step converges on what remains, relative to
        // the forces in balance: the residual's reaction part can far exceed what remains of it,
        // and sets the rounding floor.
        StepMeasure m;
        real_t      penetration = 0;
        m.value                 = impl.kkt(r, penetration);
        // A start inside the obstacle and otherwise at rest has no force to measure against: its
        // penetration, as the merit counts it, is the force that removing it takes.
        m.reference             = std::max(m.value + impl.k_max * penetration, impl.blas->norm2(impl.ndofs, r));
        m.infeasibility         = penetration;
        m.admissible            = penetration <= impl.options.gap_tol;
        if (impl.first_measure) {
            impl.tol           = std::max(impl.options.atol, impl.options.rtol * m.reference);
            impl.first_measure = false;
        }
        return m;
    }

    int NodeObstacleStep::linearize(const real_t *const x, const real_t *const r) {
        SFEM_TRACE_SCOPE("NodeObstacleStep::linearize");

        auto &impl = *impl_;
        if (impl.tangent->assemble(x) != SFEM_SUCCESS) return SFEM_FAILURE;
        impl.tangent_op = impl.tangent->op();
        if (impl.options.curvature) {
            impl.contact->add_curvature(
                    x, r, impl.k->data(), false, 1, impl.tangent->rowptr(), impl.tangent->colidx(), impl.tangent->values());
        }
        impl.contact->upper_bound(impl.qp->infty, impl.upper_bound->data());
        impl.qp->set_atol(real_t(0.1) * impl.tol);
        return SFEM_SUCCESS;
    }

    int NodeObstacleStep::solve(const real_t *const x, const real_t *const r, real_t *const dx) {
        SFEM_TRACE_SCOPE("NodeObstacleStep::solve");

        auto &impl      = *impl_;
        auto &tangent   = *impl.tangent;
        impl.iterations = 0;
        impl.blas->copy(impl.ndofs, dx, impl.lift->data());

        int        status = impl.solve_qp(r);
        const bool capped = status != SFEM_SUCCESS && impl.qp->iterations() >= impl.options.qp_max_it;
        if (status != SFEM_SUCCESS && !(capped && impl.options.qp_accept_capped)) {
            // MPRGP stops at negative curvature. Away from the solution the residual overstates the
            // reactions and the curvature term can make the tangent indefinite: solve again with
            // it capped.
            if (impl.options.curvature) {
                impl.contact->add_curvature(x, r, impl.k->data(), false, -1, tangent.rowptr(), tangent.colidx(), tangent.values());
                impl.contact->add_curvature(x, r, impl.k->data(), true, 1, tangent.rowptr(), tangent.colidx(), tangent.values());
            }

            // The material's tangent can be indefinite too (a hyperelastic block under
            // compression, on a mesh fine enough that c0 M no longer outweighs it): shift its
            // diagonal by a growing fraction of itself until the step solves. A shifted step still
            // descends, and a stable solution needs no shift.
            real_t shift = 0;
            while ((status = impl.solve_qp(r)) != SFEM_SUCCESS) {
                if (shift >= 100) {
                    printf("  contact QP did not reach its tolerance\n");
                    break;
                }
                if (shift == 0) tangent.diagonal(impl.diag->data());
                const real_t next = shift == 0 ? real_t(1e-2) : 10 * shift;
                tangent.shift_diagonal(next - shift, impl.diag->data());
                shift = next;
                printf("  tangent shifted by %g of its diagonal\n", (double)shift);
            }
        }

        impl.blas->copy(impl.ndofs, impl.framed->data(), dx);
        impl.contact->from_frame(dx);
        // MPRGP's recurrences leave rounding in the identity rows of the constrained dofs.
        impl.f->copy_constrained_dofs(impl.lift->data(), dx);
        return status;
    }

    std::shared_ptr<Merit> NodeObstacleStep::merit() {
        Impl *const self = impl_.get();
        return std::make_shared<KKTMerit>([self](const real_t *const x, const real_t *const r) { return self->merit_at(x, r); },
                                          [self](const real_t *const x) { return self->merit_value(x); },
                                          impl_->trial_r,
                                          impl_->blas,
                                          impl_->ndofs,
                                          impl_->f->execution_space());
    }

    int NodeObstacleStep::iterations() const { return impl_->iterations; }

    void NodeObstacleStep::reaction(real_t *const out) const { impl_->contact->reaction(out); }


}  // namespace sfem
