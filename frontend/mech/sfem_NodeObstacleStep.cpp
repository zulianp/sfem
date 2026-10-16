#include "sfem_NodeObstacleStep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "sfem_Merit.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_mprgp.hpp"

namespace sfem {

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
        SharedBuffer<real_t>                 k, projected, upper_bound, qp_rhs, framed, lift, diag, trial_r;
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

    namespace {
        // The KKT residual with the penetration counted as a force, k_max times it. Where the
        // obstacle's normal turns quickly (a crease, a corner) the full step can cycle between two
        // contact states; a step must decrease this.
        class KKTMerit final : public Merit {
        public:
            using At    = std::function<real_t(const real_t *const, const real_t *const)>;
            using Value = std::function<real_t(const real_t *const)>;

            KKTMerit(const At                            &at,
                     const Value                         &value,
                     const SharedBuffer<real_t>          &residual,
                     const std::shared_ptr<BLAS<real_t>> &blas,
                     const ptrdiff_t                      ndofs)
                : at_(at), value_(value), residual_(residual), blas_(blas), ndofs_(ndofs), trial_(create_host_buffer<real_t>(ndofs)) {}

            const char *name() const override { return "kkt"; }

            int evaluate(const real_t *const x,
                         const real_t *const h,
                         const int           nsteps,
                         const real_t *const steps,
                         real_t *const       out) override {
                for (int i = 0; i < nsteps; ++i) {
                    blas_->copy(ndofs_, x, trial_->data());
                    blas_->axpy(ndofs_, steps[i], h, trial_->data());
                    out[i] = value_(trial_->data());
                    last_  = steps[i];
                    known_ = std::isfinite(out[i]);
                }
                return SFEM_SUCCESS;
            }

            // The residual of the last trial, which `value` left in the step's trial residual.
            bool residual_at(const real_t step, real_t *const r) const override {
                if (!known_ || step != last_) return false;
                blas_->copy(ndofs_, residual_->data(), r);
                return true;
            }

            real_t at(const real_t *const x, const real_t *const r) override { return at_(x, r); }

            // A decrease in proportion to the merit itself: m(s) <= (1 - c1 s) m0.
            real_t slope(const real_t *const, const real_t *const, const real_t *const, const real_t m0) override { return -m0; }

            bool batched() const override { return false; }

        private:
            At                            at_;
            Value                         value_;
            SharedBuffer<real_t>          residual_;
            std::shared_ptr<BLAS<real_t>> blas_;
            ptrdiff_t                     ndofs_;
            SharedBuffer<real_t>          trial_;
            real_t                        last_{0};
            bool                          known_{false};
        };
    }  // namespace

    NodeObstacleStep::NodeObstacleStep() : impl_(std::make_unique<Impl>()) {}
    NodeObstacleStep::~NodeObstacleStep() = default;

    std::shared_ptr<NodeObstacleStep> NodeObstacleStep::create(const std::shared_ptr<Function>            &f,
                                                               const std::shared_ptr<NodeObstacleContact> &contact,
                                                               const std::shared_ptr<NewmarkScheme>       &newmark,
                                                               const NodeObstacleStepOptions              &options) {
        if (f->execution_space() != EXECUTION_SPACE_HOST) {
            fprintf(stderr, "NodeObstacleStep: the nodal SDF obstacle is host only (SDFObstacle::sample_nodes)\n");
            return nullptr;
        }

        std::shared_ptr<NodeObstacleStep> ret(new NodeObstacleStep());
        auto                             &impl = *ret->impl_;
        impl.f                                 = f;
        impl.contact                           = contact;
        impl.newmark                           = newmark;
        impl.options                           = options;
        impl.blas                              = blas<real_t>(EXECUTION_SPACE_HOST);
        impl.ndofs                             = f->space()->n_dofs();
        impl.tangent                           = BSRTangent::create(f, EXECUTION_SPACE_HOST);
        if (!impl.tangent) return nullptr;

        for (auto *buffer : {&impl.k, &impl.projected, &impl.upper_bound, &impl.qp_rhs, &impl.framed, &impl.lift, &impl.diag, &impl.trial_r}) {
            *buffer = create_host_buffer<real_t>(impl.ndofs);
        }

        // The step in the contact nodes' frames, Q^T K Q dx' = -Q^T R with dx'_n <= gap_i: the
        // constraints hold exactly at every iterate.
        // The QP is the step's, so it reads the step's state through `self`. MPRGP zeros the
        // output before every product, which the frames then turn whole: out = Q^T K Q in.
        Impl *const self   = &impl;
        auto        framed = create_host_buffer<real_t>(impl.ndofs);
        auto        op     = make_op<real_t>(
                impl.ndofs,
                impl.ndofs,
                [self, framed](const real_t *const in, real_t *const out) {
                    self->blas->copy(self->ndofs, in, framed->data());
                    self->contact->from_frame(framed->data());
                    self->tangent_op->apply(framed->data(), out);
                    self->contact->to_frame(out);
                },
                EXECUTION_SPACE_HOST);
        impl.qp             = create_mprgp(op, EXECUTION_SPACE_HOST);
        impl.qp->verbose    = false;
        impl.qp->check_each = std::numeric_limits<int>::max();
        impl.qp->set_max_it(options.qp_max_it);
        impl.qp->set_rtol(options.qp_rtol);
        impl.qp->set_upper_bound(impl.upper_bound);
        return ret;
    }

    int NodeObstacleStep::begin_step(const StepContext &context) {
        auto &impl = *impl_;
        if (!context.scheme) {
            fprintf(stderr, "NodeObstacleStep: the obstacle needs inertia (a Newmark scheme)\n");
            return SFEM_FAILURE;
        }

        // The scale of each node's equation, c0 M: what turns a reaction into a displacement.
        impl.blas->copy(impl.ndofs, impl.newmark->mass()->data(), impl.k->data());
        impl.blas->scal(impl.ndofs, context.scheme->weight("alpha_a"), impl.k->data());
        const real_t *const k     = impl.k->data();
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
        m.reference             = std::max(m.value, impl.blas->norm2(impl.ndofs, r));
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
        impl.contact->add_curvature(
                x, r, impl.k->data(), false, 1, impl.tangent->rowptr(), impl.tangent->colidx(), impl.tangent->values());
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

        int status = impl.solve_qp(r);
        if (status != SFEM_SUCCESS) {
            // MPRGP stops at negative curvature. Away from the solution the residual overstates the
            // reactions and the curvature term can make the tangent indefinite: solve again with
            // it capped.
            impl.contact->add_curvature(x, r, impl.k->data(), false, -1, tangent.rowptr(), tangent.colidx(), tangent.values());
            impl.contact->add_curvature(x, r, impl.k->data(), true, 1, tangent.rowptr(), tangent.colidx(), tangent.values());

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
                                          impl_->ndofs);
    }

    int NodeObstacleStep::iterations() const { return impl_->iterations; }

    void NodeObstacleStep::reaction(real_t *const out) const { impl_->contact->reaction(out); }

    namespace {
        class ContactProbe final : public Probe {
        public:
            ContactProbe(const std::shared_ptr<NodeObstacleContact> &contact, const ptrdiff_t ndofs, const real_t touch_gap)
                : contact_(contact), force_(create_host_buffer<real_t>(ndofs)), touch_gap_(touch_gap) {}

            std::vector<std::string> columns() const override { return {"reaction_x", "reaction_y", "reaction_z", "touched"}; }

            int converged(const StepRecord &) override {
                const ptrdiff_t n = force_->size();
                std::fill(force_->data(), force_->data() + n, real_t(0));
                contact_->reaction(force_->data());
                const real_t *const r = force_->data();
                real_t              x = 0, y = 0, z = 0;
#pragma omp parallel for reduction(+ : x, y, z)
                for (ptrdiff_t i = 0; i < n / 3; ++i) {
                    x += r[i * 3];
                    y += r[i * 3 + 1];
                    z += r[i * 3 + 2];
                }
                resultant_[0] = x / 2;
                resultant_[1] = y / 2;
                resultant_[2] = z / 2;

                const real_t *const gap = contact_->gap()->data();
                real_t              low = std::numeric_limits<real_t>::max();
#pragma omp parallel for reduction(min : low)
                for (ptrdiff_t i = 0; i < contact_->size(); ++i) low = std::min(low, gap[i]);
                touched_ = low <= touch_gap_;
                return SFEM_SUCCESS;
            }

            int sample(const StepRecord &record, real_t *const values) override {
                const bool measured = record.step > 0;
                for (int d = 0; d < 3; ++d) values[d] = measured ? resultant_[d] : real_t(0);
                values[3] = measured && touched_ ? 1 : 0;
                return SFEM_SUCCESS;
            }

        private:
            std::shared_ptr<NodeObstacleContact> contact_;
            SharedBuffer<real_t>                 force_;
            real_t                               touch_gap_;
            real_t                               resultant_[3]{0, 0, 0};
            bool                                 touched_{false};
        };
    }  // namespace

    std::shared_ptr<Probe> create_contact_probe(const std::shared_ptr<NodeObstacleContact> &contact,
                                                const ptrdiff_t                             ndofs,
                                                const real_t                                touch_gap) {
        return std::make_shared<ContactProbe>(contact, ndofs, touch_gap);
    }

}  // namespace sfem
