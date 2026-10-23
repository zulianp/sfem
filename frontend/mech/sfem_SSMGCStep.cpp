#include "sfem_SSMGCStep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_ContactCommon.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_NodeObstacleView.hpp"
#include "sfem_ssmgc.hpp"

namespace sfem {

    class SSMGCStep::Impl {
    public:
        std::shared_ptr<Function>          f;
        std::shared_ptr<ContactConditions> contact;
        std::shared_ptr<NewmarkScheme>     newmark;
        SSMGCStepOptions                   options;
        std::shared_ptr<BLAS<real_t>>      blas;
        ptrdiff_t                          ndofs{0};
        std::shared_ptr<SSMGC<real_t>>     ssmgc;
        real_t                             built_alpha{-1};
        SharedBuffer<real_t>               k, b, y, trial_r, stationary, gap, mu, complementarity, penetration;
        SharedBuffer<idx_t>                idx;  // the contact nodes in the execution space
        SharedBuffer<real_t>               curvature, blocks;  // options.curvature only
        real_t                             k_max{0};
        real_t                             tol{0};
        bool                               first_measure{true};
        int                                iterations{0};

        /// The contact nodes as last updated, with the gap and multipliers of the last kkt (and the
        /// curvature with `curvature`).
        NodeObstacleView view() const {
            NodeObstacleView c;
            c.es   = f->execution_space();
            c.size = contact->n_constrained_dofs();
            c.idx  = idx->data();
            real_t *n[3];
            contact->normal_rows(n);
            for (int d = 0; d < 3; ++d) c.normals[d] = n[d];
            if (curvature) {
                for (int k = 0; k < 6; ++k) c.curvature[k] = curvature->data() + k * c.size;
            }
            c.gap = gap->data();
            c.mu  = mu->data();
            return c;
        }

        /// The KKT residual at the state the contact was last updated to, residual r; the
        /// multipliers it implies are kept for the reaction.
        real_t kkt(const real_t *const r, real_t &deepest) {
            const ptrdiff_t n = contact->n_constrained_dofs();
            blas->zeros(n, gap->data());
            contact->signed_distance(gap->data());
            // The stationarity residual: r with the admissible reaction removed at the contact nodes.
            // Formed, not expanded: |r|^2 + 2 mu rn + mu^2 cancels the contact forces squared and keeps
            // their round-off, which at a reaction of tens is above the Newton tolerance.
            // A penetration is judged once, as a length (StepMeasure::infeasibility): SSMGC is a
            // penalty method and leaves one at its own tolerance.
            blas->copy(ndofs, r, stationary->data());
            node_obstacle_kkt(view(), k->data(), stationary->data(), complementarity->data(), penetration->data());
            deepest        = max_or_zero(f->execution_space(), n, penetration->data());
            const real_t c = blas->dot(n, complementarity->data(), complementarity->data());
            return std::sqrt(blas->dot(ndofs, stationary->data(), stationary->data()) + c);
        }

        /// The tangent's curvature blocks mu_i C_i (packed, symmetric) at x for the reactions of r, given
        /// to SSMGC; with `capped` scaled as the node-SDF step caps them (node_obstacle_curvature).
        int set_curvature(const real_t *const x, const real_t *const r, const bool capped) {
            SFEM_TRACE_SCOPE("SSMGCStep::set_curvature");
            const ptrdiff_t n = contact->n_constrained_dofs();
            if (!blocks) {
                curvature = create_buffer<real_t>(6 * n, f->execution_space());
                blocks    = create_buffer<real_t>(6 * n, f->execution_space());
            }
            if (contact->update(x) != SFEM_SUCCESS) return SFEM_FAILURE;
            blas->zeros(n, gap->data());
            if (contact->signed_distance(gap->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            real_t *C[6];
            for (int c = 0; c < 6; ++c) C[c] = curvature->data() + c * n;
            if (contact->curvature(C) != SFEM_SUCCESS) return SFEM_FAILURE;
            node_obstacle_curvature_blocks(view(), r, k->data(), capped, blocks->data());
            ssmgc->set_curvature(blocks);
            return SFEM_SUCCESS;
        }

        real_t merit_at(const real_t *const x, const real_t *const r) {
            contact->update(x);
            real_t       penetration = 0;
            const real_t value       = kkt(r, penetration);
            return value + k_max * penetration;
        }

        real_t merit_value(const real_t *const x) {
            f->update(x);
            blas->zeros(ndofs, trial_r->data());
            if (f->gradient(x, trial_r->data()) != SFEM_SUCCESS) return std::numeric_limits<real_t>::infinity();
            f->set_value_to_constrained_dofs(0, trial_r->data());
            const real_t value = merit_at(x, trial_r->data());
            return std::isfinite(value) ? value : std::numeric_limits<real_t>::infinity();
        }
    };

    SSMGCStep::SSMGCStep() : impl_(std::make_unique<Impl>()) {}
    SSMGCStep::~SSMGCStep() = default;

    std::shared_ptr<SSMGCStep> SSMGCStep::create(const std::shared_ptr<Function>          &f,
                                                 const std::shared_ptr<ContactConditions> &contact,
                                                 const std::shared_ptr<NewmarkScheme>     &newmark,
                                                 const SSMGCStepOptions                   &options) {
        const ExecutionSpace es = f->execution_space();
        if (!f->is_linear()) {
            fprintf(stderr, "SSMGCStep: SSMGC coarsens the material, which must be linear\n");
            return nullptr;
        }
        if (!f->space()->micro_elements()) {
            fprintf(stderr, "SSMGCStep: SSMGC needs a semi-structured mesh with the micro-element basis\n");
            return nullptr;
        }
        std::shared_ptr<SSMGCStep> ret(new SSMGCStep());
        auto                      &impl = *ret->impl_;
        impl.f                          = f;
        impl.contact                    = contact;
        impl.newmark                    = newmark;
        impl.options                    = options;
        impl.blas                       = blas<real_t>(es);
        impl.ndofs                      = f->space()->n_dofs();
        for (auto *buffer : {&impl.k, &impl.b, &impl.y, &impl.trial_r, &impl.stationary}) *buffer = create_buffer<real_t>(impl.ndofs, es);
        for (auto *buffer : {&impl.gap, &impl.mu, &impl.complementarity, &impl.penetration}) {
            *buffer = create_buffer<real_t>(contact->n_constrained_dofs(), es);
        }
        impl.idx = contact->node_mapping();
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) impl.idx = smesh::to_device(impl.idx);
#endif
        if (contact->init() != SFEM_SUCCESS) return nullptr;
        return ret;
    }

    int SSMGCStep::begin_step(const StepContext &context) {
        auto &impl = *impl_;
        if (!context.scheme) {
            fprintf(stderr, "SSMGCStep: the obstacle needs inertia (a Newmark scheme)\n");
            return SFEM_FAILURE;
        }
        const real_t alpha = context.scheme->weight("alpha_a");
        impl.blas->copy(impl.ndofs, impl.newmark->mass()->data(), impl.k->data());
        impl.blas->scal(impl.ndofs, alpha, impl.k->data());
        auto k_host = impl.k;
#ifdef SFEM_ENABLE_CUDA
        if (k_host->mem_space() == MEMORY_SPACE_DEVICE) k_host = smesh::to_host(k_host);
#endif
        const idx_t *const  idx   = impl.contact->node_mapping()->data();
        const real_t *const k     = k_host->data();
        real_t              k_max = 0;
#pragma omp parallel for reduction(max : k_max)
        for (ptrdiff_t i = 0; i < impl.contact->n_constrained_dofs(); ++i) k_max = std::max(k_max, k[(ptrdiff_t)idx[i] * 3]);
        impl.k_max = k_max;

        // The coarse operators are assembled with the scheme's alpha: a new step size is a new
        // hierarchy.
        if (!impl.ssmgc || alpha != impl.built_alpha) {
            impl.ssmgc = create_ssmgc(impl.f, impl.contact, impl.options.input);
            if (!impl.ssmgc) return SFEM_FAILURE;
            impl.built_alpha = alpha;
        }
        impl.first_measure = true;
        return SFEM_SUCCESS;
    }

    StepMeasure SSMGCStep::measure(const real_t *const x, const real_t *const r) {
        auto &impl = *impl_;
        impl.contact->update(x);
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

    int SSMGCStep::linearize(const real_t *const x, const real_t *const r) {
        auto &impl = *impl_;
        if (!impl.options.curvature) {
            impl.curvature = nullptr;
            impl.blocks    = nullptr;
            impl.ssmgc->set_curvature(nullptr);
            return SFEM_SUCCESS;
        }
        return impl.set_curvature(x, r, false);
    }

    int SSMGCStep::solve(const real_t *const x, const real_t *const r, real_t *const dx) {
        SFEM_TRACE_SCOPE("SSMGCStep::solve");
        auto           &impl = *impl_;
        const ptrdiff_t n    = impl.ndofs;
        const auto      run  = [&]() {
            // The material is linear: K (x + dx) = K x - r, the obstacle's constraint on x + dx;
            // with the curvature C, (K + C)(x + dx) = (K + C) x - r.
            impl.blas->zeros(n, impl.b->data());
            if (impl.f->apply(x, x, impl.b->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            if (impl.blocks) impl.ssmgc->apply_curvature(x, impl.b->data());
            impl.blas->axpy(n, -1, r, impl.b->data());
            // The constrained rows are the identity with the values the step moves them to.
            impl.blas->copy(n, x, impl.y->data());
            impl.blas->axpy(n, 1, dx, impl.y->data());
            impl.f->copy_constrained_dofs(impl.y->data(), impl.b->data());

            if (impl.ssmgc->update(impl.y->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            // The residual to a tenth of Newton's tolerance, as a residual: SSMGC's relative test measures
            // against the apply's start, which the multipliers it carries over make small from the
            // second Newton iteration on.
            impl.ssmgc->set_residual_atol(impl.tol / 10);
            const int status = impl.ssmgc->apply(impl.b->data(), impl.y->data());
            impl.iterations += impl.ssmgc->iterations();
            return status;
        };

        impl.iterations = 0;
        int status      = run();
        if (status != SFEM_SUCCESS && impl.blocks) {
            // Away from the solution the residual overstates the reactions and the curvature term can
            // make the tangent indefinite: solve again with it capped, as the node-SDF step does.
            printf("  ssmgc: solving again with the curvature capped\n");
            if (impl.set_curvature(x, r, true) != SFEM_SUCCESS) return SFEM_FAILURE;
            status = run();
        }

        impl.blas->zaxpby(n, 1, impl.y->data(), -1, x, dx);
        return status;
    }

    std::shared_ptr<Merit> SSMGCStep::merit() {
        Impl *const self = impl_.get();
        return std::make_shared<KKTMerit>([self](const real_t *const x, const real_t *const r) { return self->merit_at(x, r); },
                                          [self](const real_t *const x) { return self->merit_value(x); },
                                          impl_->trial_r,
                                          impl_->blas,
                                          impl_->ndofs,
                                          impl_->f->execution_space());
    }

    int SSMGCStep::iterations() const { return impl_->iterations; }

    void SSMGCStep::set_options(const SSMGCStepOptions &options) { impl_->options = options; }

    int SSMGCStep::gap_field(const real_t *const x, real_t *const out) {
        auto &impl = *impl_;
        impl.blas->zeros(impl.ndofs, out);
        if (impl.contact->update(x) != SFEM_SUCCESS) return SFEM_FAILURE;
        impl.blas->zeros(impl.contact->n_constrained_dofs(), impl.gap->data());
        if (impl.contact->signed_distance(impl.gap->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        node_obstacle_gap_field(impl.view(), out);
        return SFEM_SUCCESS;
    }

    void SSMGCStep::reaction(real_t *const out) const {
        impl_->blas->zeros(impl_->ndofs, out);
        node_obstacle_reaction(impl_->view(), out);
    }

    void SSMGCStep::project_velocity(const real_t *const x, real_t *const v, const real_t dt) {
        auto           &impl = *impl_;
        const ptrdiff_t n    = impl.ndofs;
        impl.blas->copy(n, x, impl.y->data());
        impl.blas->axpy(n, dt, v, impl.y->data());
        impl.contact->update(impl.y->data());
        impl.blas->zeros(impl.contact->n_constrained_dofs(), impl.gap->data());
        impl.contact->signed_distance(impl.gap->data());
        // n points towards the obstacle: a negative gap went past the surface by -gap along n.
        node_obstacle_project_velocity(impl.view(), v, dt);
    }

    real_t SSMGCStep::min_gap() const {
        auto gaps = impl_->gap;
#ifdef SFEM_ENABLE_CUDA
        if (gaps->mem_space() == MEMORY_SPACE_DEVICE) gaps = smesh::to_host(gaps);
#endif
        const real_t *const g   = gaps->data();
        real_t              low = std::numeric_limits<real_t>::max();
#pragma omp parallel for reduction(min : low)
        for (ptrdiff_t i = 0; i < impl_->contact->n_constrained_dofs(); ++i) low = std::min(low, g[i]);
        return low;
    }

}  // namespace sfem
