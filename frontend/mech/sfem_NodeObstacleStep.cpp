#include "sfem_NodeObstacleStep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include "sfem_blas_factory.hpp"
#include "sfem_solver_factory.hpp"
#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif
#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "sfem_ContactCommon.hpp"
#include "sfem_Merit.hpp"
#include "sfem_TimeScheme.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_OwnedReductions.hpp"
#include "sfem_ParallelOperator.hpp"
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

        /// `host`'s values into `target`, wherever it lives (no allocation).
        void fill(const SharedBuffer<real_t> &target, const SharedBuffer<real_t> &host) {
#ifdef SFEM_ENABLE_CUDA
            if (target->mem_space() == MEMORY_SPACE_DEVICE) {
                copy_host_to_device(host->size() * sizeof(real_t), host->data(), target->data());
                return;
            }
#endif
            std::copy(host->data(), host->data() + host->size(), target->data());
        }
    }  // namespace

    class NodeObstacleStep::Impl {
    public:
        std::shared_ptr<Function>            f;
        std::shared_ptr<NodeObstacleContact> contact;
        std::shared_ptr<TimeScheme>          scheme;
        NodeObstacleStepOptions              options;
        std::shared_ptr<BLAS<real_t>>        blas;
        ptrdiff_t                            ndofs{0};
        std::unique_ptr<OwnedReductions>     reductions;
        std::shared_ptr<BSRTangent>          tangent;
        std::shared_ptr<Operator<real_t>>    tangent_op;
        // tangent_op on the host, whose product the QP's runs inside its own region.
        std::shared_ptr<TeamParallelOperator<real_t>> tangent_team;
        std::shared_ptr<MPRGP<real_t>>       qp;
        SharedBuffer<real_t>                 k, k_ref, projected, upper_bound, qp_rhs, framed, lift, diag, trial_r;
        // The QP's variables are y / scale, its bounds upper_bound / scale (scaled_bound), its
        // operator scale Q^T K Q scale; qp_tmp holds the unscaled product.
        SharedBuffer<real_t>                 scale, inverse_scale, scaled_bound, qp_tmp;
        real_t                               k_max{0};
        // The step length and the scheme's shift k_ref was assembled for (0: not yet).
        real_t                               k_ref_dt{0};
        real_t                               k_ref_shift{0};
        real_t                               tol{0};
        // The merit of the iterate measured last: the reference of the QP's forcing term.
        real_t                               merit{0};
        bool                                 first_measure{true};
        int                                  iterations{0};

        real_t kkt(const real_t *const r, real_t &penetration) {
            return contact->kkt_residual(r, k->data(), projected->data(), &penetration);
        }

        // scale = D^{-1/2} from the tangent as it stands, D its diagonal with each contact node's
        // three entries replaced by their mean (the trace, which the frames keep); the QP's bounds.
        int update_scaling() {
            if (tangent->diagonal(diag->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            auto           d     = on_host(diag);
            auto           u     = on_host(upper_bound);
            real_t *const  dd    = d->data();
            const idx_t   *nodes = contact->nodes()->data();
            for (ptrdiff_t i = 0; i < contact->size(); ++i) {
                const ptrdiff_t node = nodes[i];
                const real_t    mean = (dd[node * 3] + dd[node * 3 + 1] + dd[node * 3 + 2]) / 3;
                for (int c = 0; c < 3; ++c) dd[node * 3 + c] = mean;
            }
            auto   s = create_host_buffer<real_t>(ndofs), inv = create_host_buffer<real_t>(ndofs), b = create_host_buffer<real_t>(ndofs);
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                const real_t root = dd[i] > 0 ? std::sqrt(dd[i]) : real_t(1);
                s->data()[i]      = 1 / root;
                inv->data()[i]    = root;
                b->data()[i]      = u->data()[i] * root;
            }
            fill(scale, s);
            fill(inverse_scale, inv);
            fill(scaled_bound, b);
            qp->set_upper_bound(scaled_bound);
            // Converged as the unscaled QP is, on the gradient of y = S z: S^{-1} times z's.
            qp->set_norm_weight(inverse_scale);
            qp->set_atol(std::max(real_t(0.1) * tol, options.qp_rtol * merit));
            return SFEM_SUCCESS;
        }

        int solve_qp(const real_t *const r) {
            if (update_scaling() != SFEM_SUCCESS) return SFEM_FAILURE;
            // Gershgorin's bound for S K S: the scale is one per contact node, so it commutes with
            // the node's frame, and S Q^T K Q S = Q^T (S K S) Q has the same eigenvalues.
            qp->set_max_eig(tangent->max_abs_row_sum(scale->data()));
            blas->zaxpby(ndofs, -1, r, 0, r, qp_rhs->data());
            // The constrained rows are the identity: their right-hand side and start are the
            // increment the step arrived with.
            f->copy_constrained_dofs(lift->data(), qp_rhs->data());
            contact->to_frame(scale->data(), qp_rhs->data(), qp_rhs->data());
            contact->to_frame(inverse_scale->data(), lift->data(), framed->data());
            // framed is the scaled step in the frames (solve() takes it back).
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
                                                               const std::shared_ptr<TimeScheme>          &scheme,
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
        impl.scheme                            = scheme;
        impl.options                           = options;
        impl.blas                              = blas<real_t>(es);
        impl.ndofs                             = f->space()->n_dofs();
        impl.reductions                        = std::make_unique<OwnedReductions>(f->space(), es);
        // Its values are changed in place after `op` (the curvature, the shifts), which a copy
        // taken there would miss: stored in real_t or, assembled so, in float.
        if (options.storage_bytes != (int)sizeof(real_t) && options.storage_bytes != 4) {
            SFEM_ERROR("NodeObstacleStep: storage_bytes %d (4 or %d)\n", options.storage_bytes, (int)sizeof(real_t));
            return nullptr;
        }
        impl.tangent = BSRTangent::create(f, es, options.storage_bytes);
        if (!impl.tangent) return nullptr;

        for (auto *buffer : {&impl.k, &impl.k_ref, &impl.projected, &impl.upper_bound, &impl.qp_rhs, &impl.framed, &impl.lift, &impl.diag, &impl.trial_r, &impl.qp_tmp,
                             &impl.scale, &impl.inverse_scale, &impl.scaled_bound}) {
            *buffer = create_buffer<real_t>(impl.ndofs, es);
        }

        // The step in the contact nodes' frames, Q^T K Q dx' = -Q^T R with dx'_n <= gap_i: the
        // constraints hold exactly at every iterate.
        // The QP is the step's, so it reads the step's state through `self`. Each sweep writes its
        // output whole (the tangent's product overwrites, tangent_op), so MPRGP need not zero it:
        // out = Q^T K Q in, three sweeps.
        Impl *const self   = &impl;
        auto        framed = create_buffer<real_t>(impl.ndofs, es);
        // Under MPI the owned rows, as the tangent's product computes them.
        const ptrdiff_t                           owned = f->space()->n_dofs_owned();
        std::shared_ptr<ParallelOperator<real_t>> op;
        if (es == EXECUTION_SPACE_HOST) {
            // One region: each thread takes its nodes into the frames, then, once every thread's
            // are in, its rows of the product and back out of the frames. It reads `in` only at
            // its own nodes before the barrier (input_in_share).
            op = make_team_op<real_t>(f->space()->mesh_ptr()->comm(),
                                      owned,
                                      owned,
                                      impl.ndofs,
                                      impl.ndofs,
                                      true,
                                      [self, framed](const Team &team, const real_t *const in, real_t *const out) {
                                          real_t *const tmp = self->qp_tmp->data();
                                          self->contact->from_frame(team, self->scale->data(), in, framed->data());
                                          team.barrier();
                                          self->tangent_team->team_apply(team, framed->data(), tmp);
                                          self->contact->to_frame(team, self->scale->data(), tmp, out);
                                      });
        } else {
            op = make_parallel_op<real_t>(
                    f->space()->mesh_ptr()->comm(),
                    owned,
                    owned,
                    impl.ndofs,
                    impl.ndofs,
                    [self, framed](const real_t *const in, real_t *const out) {
                        // scale Q^T K Q scale: out = scale .* (Q^T K Q (scale .* in)).
                        real_t *const tmp = self->qp_tmp->data();
                        self->contact->from_frame(self->scale->data(), in, framed->data());
                        self->tangent_op->apply(framed->data(), tmp);
                        self->contact->to_frame(self->scale->data(), tmp, out);
                    },
                    es);
        }
        impl.qp             = create_mprgp<real_t>(op, es);
        impl.qp->set_op_overwrites(true);
        impl.qp->verbose    = false;
        impl.qp->check_each = std::numeric_limits<int>::max();
        impl.qp->set_max_it(options.qp_max_it);
        // The QP's tolerance is the step's forcing term (update_scaling), not relative to its start.
        impl.qp->set_rtol(0);
        return ret;
    }

    int NodeObstacleStep::begin_step(const StepContext &context) {
        auto &impl = *impl_;
        if (impl.scheme && !context.scheme) {
            fprintf(stderr, "NodeObstacleStep: a dynamic step needs its scheme\n");
            return SFEM_FAILURE;
        }

        // The scale of each node's equation, what turns a reaction into a displacement: the
        // tangent's diagonal at the reference configuration, where no element is inverted, with
        // the step's inertia c0 M in it in a dynamic step (the material holds the scheme). It
        // only scales, so it is assembled again only when the step's length or the scheme's shift
        // changes: a material with a history is as stiff as its relaxation over the step leaves it,
        // glassy before its first step, and the inertia follows the shift (BDF2's with the ratio
        // of the step to the last, and its backward-Euler start).
        const real_t shift = context.scheme ? context.scheme->shift() : real_t(0);
        if (context.dt != impl.k_ref_dt || shift != impl.k_ref_shift) {
            auto reference = create_buffer<real_t>(impl.ndofs, impl.tangent->execution_space());
            if (impl.tangent->assemble(reference->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            if (impl.tangent->diagonal(impl.k_ref->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            impl.k_ref_dt    = context.dt;
            impl.k_ref_shift = shift;
        }
        impl.blas->copy(impl.ndofs, impl.k_ref->data(), impl.k->data());
        // Read on the host: once per step.
        auto                host  = on_host(impl.k);
        const real_t *const k     = host->data();
        const idx_t *const  nodes = impl.contact->nodes()->data();
        real_t              k_max = 0;
#pragma omp parallel for reduction(max : k_max)
        for (ptrdiff_t i = 0; i < impl.contact->size(); ++i) {
            k_max = std::max(k_max, k[(ptrdiff_t)nodes[i] * 3]);
        }
        impl.k_max         = impl.reductions->max(k_max);
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
        m.reference             = std::max(m.value + impl.k_max * penetration, impl.reductions->norm2(r));
        m.infeasibility         = penetration;
        impl.merit              = m.value + impl.k_max * penetration;
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
        // y = K x: the QP's product writes tmp whole.
        impl.tangent_op = impl.tangent->op(0);
        if (impl.tangent->execution_space() == EXECUTION_SPACE_HOST) {
            impl.tangent_team = std::dynamic_pointer_cast<TeamParallelOperator<real_t>>(impl.tangent_op);
            if (!impl.tangent_team) SFEM_ERROR("NodeObstacleStep: the host tangent has no product for a region of its caller\n");
        }
        if (impl.options.curvature) {
            impl.contact->add_curvature(
                    x, r, impl.k->data(), false, 1, impl.tangent->rowptr(), impl.tangent->colidx(), impl.tangent->values_type(), impl.tangent->values());
        }
        impl.contact->upper_bound(impl.qp->infty, impl.upper_bound->data());
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
                impl.contact->add_curvature(x, r, impl.k->data(), false, -1, tangent.rowptr(), tangent.colidx(), tangent.values_type(), tangent.values());
                impl.contact->add_curvature(x, r, impl.k->data(), true, 1, tangent.rowptr(), tangent.colidx(), tangent.values_type(), tangent.values());
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

        impl.contact->from_frame(impl.scale->data(), impl.framed->data(), dx);
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
