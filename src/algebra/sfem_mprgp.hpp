#ifndef SFEM_MPRGP_HPP
#define SFEM_MPRGP_HPP

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ParallelOperator.hpp"

#include "sfem_PowerMethod.hpp"

#include "sfem_BLAS_openmp.hpp"
#include "sfem_mprgp_control.hpp"
#include "sfem_MPRGPKernels_openmp.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "smesh_device_communicator.hpp"
#endif

namespace sfem {
    // From Active set expansion strategies in MPRGP algorithm, Kruzik et al. 2020
    template <typename T>
    class MPRGP final : public MatrixFreeLinearSolver<T> {
    public:
        /// Built in the execution context, on op.
        static std::shared_ptr<MPRGP<T>> create(const std::shared_ptr<Operator<T>> &op, const ExecutionContext &context);
        enum class Expansion { ORIGINAL = 0, PROJECTED_CG = 1 };

        ~MPRGP() override { release_workspace(); }

        Expansion expansion_type_{Expansion::ORIGINAL};
        T rtol{1e-10};
        T atol{1e-16};
        T gamma{1};  // gamma > 0
        T eps{1e-14};
        T infty{1e15};
        T eigen_solver_tol{1e-1};
        int max_it{10000};
        int check_each{10};
        ptrdiff_t n_dofs{SFEM_PTRDIFF_INVALID};
        bool verbose{true};
        T max_eig_{0};

        int iterations_{0};
        int iterations() const override { return iterations_; }

        SharedBuffer<T> upper_bound_;
        SharedBuffer<T> lower_bound_;
        // Convergence on the projected gradient weighted entry by entry, |w g_P| (and rtol relative
        // to |w g_0|): with variables z = S^{-1} y, w = S^{-1} measures the gradient in y. The
        // steps are the iteration's own; only the test changes. Null: unweighted.
        SharedBuffer<T> norm_weight_;

        std::function<void(const T* const, T* const)> apply_op;
        // The operator in team form (TeamParallelOperator, on the host), if it has one: then runs
        // of CG steps take one parallel region (team_cg_steps).
        std::shared_ptr<TeamParallelOperator<T>> team_op_;
        // Whether apply_op writes y = A x whole rather than adding to y: then y is not zeroed
        // before each product.
        bool op_overwrites_{false};
        // Runs of CG steps in the execution space without returning to the host between them,
        // where the passes take them (MPRGPKernels::cg_run): one rank, or the ranks of one node whose
        // sums the device completes (NCCL), and a product that writes A p whole. Off, the host
        // takes each step's decision. run_state_ is the loop of this solve.
        bool                  device_runs_{true};
        std::shared_ptr<void> run_state_;
        int                   device_run_count_{0};

        std::shared_ptr<BLAS<T>> blas;
        MPRGPKernels<T> impl;
        // Under MPI (a ParallelOperator of more than one rank): the vectors are the owned prefix
        // of space-sized buffers, n_dofs the owned length, n_alloc the buffers'; every reduction
        // spans the ranks, so every branch is taken alike on all of them.
        std::shared_ptr<Communicator> comm_;
        ptrdiff_t n_alloc_{-1};
        // On the device, with every rank on one node and the communicator's device communication
        // enabled (smesh::Communicator::enable_device): the passes complete their sums there, by
        // NCCL on the stream, and the iteration returns to the host once, as on one rank.
        // Otherwise empty, and the sums are completed on the host from the record.
        MPRGPReduce<T> device_reduce_;

        MPRGPReduce<T> make_device_reduce() const {
#ifdef SFEM_ENABLE_CUDA
            auto device = comm_ && this->execution_space() == EXECUTION_SPACE_DEVICE ? comm_->device() : nullptr;
            if (device && device->whole()) {
                // On the stream the pass launched on (a graph run's, or the default).
                return [device](T* const values, const int count, const bool min_last, void* const stream) {
                    if (min_last) {
                        device->sum_and_min_last(values, count, smesh::TypeToEnum<T>::value(), stream);
                    } else {
                        device->sum(values, count, smesh::TypeToEnum<T>::value(), stream);
                    }
                };
            }
#endif
            return {};
        }

        // Whether the sums are completed on the host, from the record.
        bool on_host() const { return comm_ && !device_reduce_; }

        T sum(const T local) const { return comm_ ? comm_->sum(local) : local; }
        T dot(const T* const x, const T* const y) const { return sum(this->blas->dot(n_dofs, x, y)); }
        T norm2(const T* const x) const { return std::sqrt(dot(x, x)); }
        ptrdiff_t allocation_size() const { return n_alloc_ > 0 ? n_alloc_ : n_dofs; }
        std::shared_ptr<PowerMethod<T>> power_method;

        ExecutionContext execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_expansion_type(const Expansion et) { expansion_type_ = et; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            // On a stream of its own the product follows the solver's work and the solver follows
            // it, in the device runs' graph too.
            this->apply_op = [this, op](const T* const x, T* const y) {
                order_after(op->execution_context(), this->blas->context());
                op->apply(x, y);
                order_after(this->blas->context(), op->execution_context());
            };
            team_op_ = std::dynamic_pointer_cast<TeamParallelOperator<T>>(op);
            n_dofs = op->rows();
            comm_.reset();
            n_alloc_ = -1;
            if (auto parallel = std::dynamic_pointer_cast<ParallelOperator<T>>(op)) {
                if (parallel->comm()->size() > 1) {
                    comm_    = parallel->comm();
                    n_alloc_ = std::max(parallel->row_allocation_size(), parallel->col_allocation_size());
                }
            }
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& /*op*/) override {
            // Ignoring op!
        }

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }
        void set_verbose(const bool val) { verbose = val; }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        void set_upper_bound(const SharedBuffer<T>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const SharedBuffer<T>& lb) { lower_bound_ = lb; }
        void set_norm_weight(const SharedBuffer<T>& w) { norm_weight_ = w; }
        void set_op_overwrites(const bool val) { op_overwrites_ = val; }
        void set_device_runs(const bool val) { device_runs_ = val; }
        /// The runs of CG steps the last solve took in the execution space.
        int device_run_count() const { return device_run_count_; }

        /// y <- A x.
        void product(const T* const x, T* const y) {
            if (!op_overwrites_) this->blas->zeros(n_dofs, y);
            apply_op(x, y);
        }

        void ensure_power_method() {
            if (!power_method) {
                auto blas_impl = blas;
                power_method = std::make_shared<PowerMethod<T>>();
                auto comm = comm_;
                power_method->norm2 = [blas_impl, comm](const std::ptrdiff_t n, const T* const x) {
                    const T local = blas_impl->dot(n, x, x);
                    return std::sqrt(comm ? comm->sum(local) : local);
                };
                power_method->scal = [blas_impl](const std::ptrdiff_t n, const T alpha, T* const x) { blas_impl->scal(n, alpha, x); };
                power_method->zeros = [blas_impl](const std::size_t n, T* const x) { blas_impl->zeros(n, x); };
            }
        }

        const T* lb() const { return lower_bound_ ? lower_bound_->data() : nullptr; }
        const T* ub() const { return upper_bound_ ? upper_bound_->data() : nullptr; }

        void project(T* const x) { impl.project(n_dofs, lb(), ub(), x); }

        /// Sums the ranks' partial reductions in place, one message for all of them.
        void sum(T* const values, const int count) const {
            if (comm_) comm_->sum(values, count, smesh::TypeToEnum<T>::value());
        }
        T min(const T local) const { return comm_ ? -comm_->max(-local) : local; }

        /// At the iterate a step reaches: Ad.gf for the next direction, and the norms that choose
        /// the next step and test convergence.
        struct Gradients {
            T Adgf, norm_gf, norm_gc, norm_gp;
        };

        // The iteration's record (MPRGPControl) in the execution space, its copy on the host, and
        // the constants of its decisions.
        T*              control_{nullptr};
        MPRGPControl<T> record_{};
        MPRGPParams<T>  params_{};

        // The solve's vectors (gradient, direction, its product, free or chopped gradient, the
        // passes' work) and control_, kept from one solve to the next: a Newton step solves a QP
        // of the same size, and an allocation per solve cost a device synchronisation each. They
        // are allocated again when the size or the BLAS changes, and zeroed at each solve, as a
        // fresh allocation was.
        std::shared_ptr<BLAS<T>> workspace_blas_;
        ptrdiff_t                workspace_n_{-1};
        ptrdiff_t                workspace_work_{-1};
        T*                       workspace_[5]{};

        void release_workspace() {
            if (workspace_blas_) {
                for (auto*& w : workspace_) {
                    if (w) workspace_blas_->destroy(w);
                    w = nullptr;
                }
                if (control_) workspace_blas_->destroy(control_);
            }
            control_ = nullptr;
            workspace_blas_.reset();
            workspace_n_ = workspace_work_ = -1;
        }

        void ensure_workspace() {
            const ptrdiff_t n  = allocation_size();
            const ptrdiff_t nw = impl.work_size;
            if (workspace_blas_ != this->blas || workspace_n_ != n || workspace_work_ != nw) {
                release_workspace();
                workspace_blas_ = this->blas;
                for (int k = 0; k < 4; ++k) workspace_[k] = this->blas->allocate(n);
                workspace_[4]   = nw > 0 ? this->blas->allocate(nw) : nullptr;
                control_        = this->blas->allocate(mprgp_control_size<T>());
                workspace_n_    = n;
                workspace_work_ = nw;
                return;
            }
            for (int k = 0; k < 4; ++k) this->blas->zeros(n, workspace_[k]);
            if (workspace_[4]) this->blas->zeros(nw, workspace_[4]);
            this->blas->zeros(mprgp_control_size<T>(), control_);
        }

        /// The curvature pass along d (Ad = A d), then the decision. On one rank both are the
        /// pass's; across ranks its sums are completed here and the same decision taken on the host
        /// (with device_reduce_ both stay the pass's).
        void curvature(const T* const x, const T* const g, const T* const d, const T* const Ad, const MPRGPDecision decision,
                       T* const work) {
            impl.curvature(
                    n_dofs, lb(), ub(), x, g, d, Ad, infty, on_host() ? MPRGPDecision::NONE : decision, params_, control_, work, device_reduce_);
            if (on_host()) {
                impl.read_control(control_, &record_);
                T v[4] = {record_.dAd, record_.dd, record_.gd, record_.alpha_feas};
                // The three sums and the step bound in one message.
                comm_->sum_and_min_last(v, 4, smesh::TypeToEnum<T>::value());
                record_.dAd        = v[0];
                record_.dd         = v[1];
                record_.gd         = v[2];
                record_.alpha_feas = v[3];
                mprgp_decide(record_, decision, params_);
                impl.write_control(record_, control_);
            }
        }

        /// x -= alpha d, g -= alpha Ad (along d only where the record's move is a step; neither
        /// without d), gf <- the free gradient; completed across ranks like curvature(). With `cg`
        /// the next direction's beta.
        void step(const T* const d, const T* const Ad, T* const x, T* const g, T* const gf, const bool cg, T* const work) {
            impl.step(n_dofs, lb(), ub(), d, Ad, x, g, gf, eps, norm_weight_ ? norm_weight_->data() : nullptr, on_host() ? false : cg,
                      control_, work, device_reduce_);
            if (on_host() && (!d || record_.move == T(MPRGPMove::STEP))) {
                impl.read_control(control_, &record_);
                T v[4] = {record_.Adgf, record_.gf2, record_.gc2, record_.gp2};
                sum(v, 4);
                record_.Adgf = v[0];
                record_.gf2  = v[1];
                record_.gc2  = v[2];
                record_.gp2  = v[3];
                if (cg) {
                    mprgp_after_cg_step(record_);
                    impl.write_control(record_, control_);
                }
            }
        }

        /// The record on the host; on one rank the read is the iteration's one return to it.
        const MPRGPControl<T>& record() {
            if (!on_host()) impl.read_control(control_, &record_);
            return record_;
        }

        static Gradients gradients(const MPRGPControl<T>& c) {
            return {c.Adgf, std::sqrt(c.gf2), std::sqrt(c.gc2), std::sqrt(c.gp2)};
        }

        bool good() const {
            assert(apply_op);
            assert(lower_bound_ || upper_bound_);
            return blas->good() && impl.good() && apply_op && (lower_bound_ || upper_bound_);
        }

        void default_init() {
            blas = OpenMPBLAS<T>::create();
            OpenMPMPRGPKernels<T>::build_mprgp(impl);
        }

        /// To the bound along p, then a projected step along the free gradient there (or, projected
        /// CG, the CG step past the bound, projected); p restarts from the free gradient. The
        /// record holds the step to the bound (alpha) and the CG step (alpha_cg).
        Gradients expansion_step(const MPRGPControl<T>& c, const T* const b, T* const p, const T* const Ap, T* const g,
                                 T* const x, T* const work) {
            const T alpha_feas = c.alpha, alpha_cg = c.alpha_cg, alpha_bar = params_.alpha_bar;
            // A flat direction (alpha_cg infty) has no CG step past the bound to project.
            if (alpha_cg < infty && expansion_type_ == Expansion::PROJECTED_CG) {
                this->blas->axpby(n_dofs, -alpha_feas, p, 1, x);
                this->blas->axpby(n_dofs, -alpha_cg, p, 1, x);
                this->project(x);
            } else {
                impl.expand(n_dofs, lb(), ub(), alpha_feas, alpha_bar, p, Ap, x, g);
            }
            this->gradient(x, b, g);
            this->step(nullptr, nullptr, x, g, p, false, work);
            return gradients(record());
        }

        /// A step along the chopped gradient gc; p restarts from the free gradient. False where gc
        /// sees no curvature (mprgp_no_curvature) and no bound stops it.
        bool proportioning_step(T* const p, T* const Agc, T* const g, T* const gc, T* const x, T* const work,
                                Gradients* const grads) {
            impl.chopped_gradient(n_dofs, lb(), ub(), x, g, gc, eps);

            product(gc, Agc);
            curvature(x, g, gc, Agc, MPRGPDecision::PROPORTIONING, work);
            this->step(gc, Agc, x, g, p, false, work);
            const MPRGPControl<T>& c = record();
            if (c.move == T(MPRGPMove::UNBOUNDED)) return false;
            *grads = gradients(c);
            return true;
        }

        void gradient(const T* const x, const T* const b, T* const g) {
            product(x, g);
            this->blas->axpby(n_dofs, -1, b, 1, g);
        }

        /// CG steps in one parallel region, as many in a row as the loop of apply() takes: its CG
        /// branch and the test after it, thread by thread over the operator's Team. Each pass takes
        /// the calling thread's chunks, and every thread, after a barrier, combines all of them in
        /// index order and takes the same decisions; under MPI thread 0, which has no rows,
        /// completes the sums over the ranks. Three barriers a step on one rank: in the operator,
        /// after the curvature and after the step; under MPI the operator's halo and each sum over
        /// the ranks take one more. The chunks' sums alternate between two halves of `work`: a
        /// thread still reading one half is never overwritten.
        MPRGPExit team_cg_steps(T* const     x,
                                T* const     g,
                                T* const     p,
                                T* const     Ap,
                                T* const     gf,
                                T* const     work,
                                const T      norm_gp0,
                                Gradients*   grads,
                                int*         count_cg_steps,
                                bool*        converged) {
            using Pass                 = OpenMPMPRGPKernels<T>;
            const T* const w           = norm_weight_ ? norm_weight_->data() : nullptr;
            T* const       part[2]     = {work, work + 4 * blas_max_chunk_count};
            const T        no_curv[4]  = {0, 0, 0, infty};
            const T        no_step[4]  = {0, 0, 0, 0};
            const auto     type        = smesh::TypeToEnum<T>::value();
            T              by_ranks[4] = {0, 0, 0, 0};
            MPRGPExit      exit        = MPRGPExit::PROPORTIONING;
#pragma omp parallel
            {
                const Team      team = team_op_->team();
                MPRGPControl<T> c    = record_;
                Gradients       gr   = *grads;
                int             it = iterations_, steps = *count_cg_steps;
                bool            done = false;
                MPRGPExit       why  = MPRGPExit::PROPORTIONING;
                // The ranks' sums, by thread 0, then for every thread.
                auto over_ranks = [&](T (&v)[4], const bool min_last) {
                    if (!comm_) return;
                    if (team.tid == 0) {
                        if (min_last) {
                            comm_->sum_and_min_last(v, 4, type);
                        } else {
                            comm_->sum(v, 4, type);
                        }
                        for (int k = 0; k < 4; k++) by_ranks[k] = v[k];
                    }
                    team.barrier();
                    for (int k = 0; k < 4; k++) v[k] = by_ranks[k];
                };
                with_mprgp_bounds(lb(), ub(), [&](const auto bounds) {
                    while (gr.norm_gc <= this->gamma * gr.norm_gf) {
                        if (!op_overwrites_) {
                            for (ptrdiff_t i = team.begin; i < team.end; i++) Ap[i] = 0;
                        }
                        if (!team_op_->input_in_share()) team.barrier();
                        team_op_->team_apply(team, p, Ap);

                        T v[4];
                        Pass::team_sums(team, no_curv, part[0], [&](const ptrdiff_t b, const ptrdiff_t e, T (&s)[4]) {
                            Pass::curvature_sums(bounds, b, e, x, g, p, Ap, infty, s);
                        });
                        team.barrier();
                        Pass::combine(team.chunks, part[0], no_curv, true, v);
                        over_ranks(v, true);
                        c.dAd        = v[0];
                        c.dd         = v[1];
                        c.gd         = v[2];
                        c.alpha_feas = v[3];
                        mprgp_decide(c, MPRGPDecision::CG, params_);
                        if (c.move != T(MPRGPMove::STEP)) {
                            why = MPRGPExit::NOT_A_STEP;
                            break;
                        }

                        Pass::team_sums(team, no_step, part[1], [&](const ptrdiff_t b, const ptrdiff_t e, T (&s)[4]) {
                            Pass::step_sums(bounds, b, e, p, Ap, c.alpha, x, g, gf, eps, w, s);
                        });
                        team.barrier();
                        Pass::combine(team.chunks, part[1], no_step, false, v);
                        over_ranks(v, false);
                        c.Adgf = v[0];
                        c.gf2  = v[1];
                        c.gc2  = v[2];
                        c.gp2  = v[3];
                        mprgp_after_cg_step(c);
                        Pass::direction(team.begin, team.end, gf, c.beta, p);
                        gr = gradients(c);
                        steps++;

                        const T rel_norm_gp = gr.norm_gp / norm_gp0;
                        done                = gr.norm_gp < atol || rel_norm_gp < rtol;
                        if (team.tid == 0) monitor(it, gr.norm_gp, rel_norm_gp);
                        if (++it >= max_it || done) {
                            why = MPRGPExit::DONE;
                            break;
                        }
                    }
                });
                if (team.tid == 0) {
                    record_         = c;
                    *grads          = gr;
                    iterations_     = it;
                    *count_cg_steps = steps;
                    *converged      = done;
                    exit            = why;
                }
            }
            impl.write_control(record_, control_);
            return exit;
        }

        /// The CG steps in a row in the execution space (MPRGPKernels::cg_run), as team_cg_steps takes
        /// them on the host; false where they cannot be taken there, nothing done.
        bool device_cg_steps(T* const x, T* const g, T* const p, T* const Ap, T* const gf, T* const work, const T norm_gp0,
                             Gradients* grads, int* count_cg_steps, bool* converged, MPRGPExit* exit) {
            // Across ranks only where the device completes the sums (one node, NCCL).
            if (!device_runs_ || !impl.cg_run || (comm_ && !device_reduce_) || !op_overwrites_) return false;
            MPRGPRun<T> r;
            r.n        = n_dofs;
            r.lb       = lb();
            r.ub       = ub();
            r.x        = x;
            r.g        = g;
            r.p        = p;
            r.Ap       = Ap;
            r.gf       = gf;
            r.eps      = eps;
            r.w        = norm_weight_ ? norm_weight_->data() : nullptr;
            r.infty    = infty;
            r.params   = params_;
            r.control  = control_;
            r.work     = work;
            r.gamma    = this->gamma;
            r.atol     = atol;
            r.rtol     = rtol;
            r.norm_gp0 = norm_gp0;
            r.max_it   = max_it;
            r.product  = apply_op;
            r.reduce   = device_reduce_;
            r.stream                 = this->blas->context().stream();
            const MPRGPRunResult run = impl.cg_run(r, iterations_, run_state_);
            if (!run.ran) return false;
            SFEM_TRACE_SCOPE("MPRGP::device_run");  // counts the runs taken on the device

            device_run_count_++;
            iterations_ = run.iterations;
            *count_cg_steps += run.steps;
            *grads     = gradients(record());
            *converged = run.converged;
            *exit      = run.exit;
            if (run.exit == MPRGPExit::DONE) monitor(iterations_, grads->norm_gp, grads->norm_gp / norm_gp0);
            return true;
        }

        void monitor(const int iter, const T residual, const T rel_residual) {
            if (iter == max_it || iter % check_each == 0 || residual < atol ||
                rel_residual < rtol) {
                std::cout << iter << ": " << residual << " " << rel_residual << "\n";
            }
        }

        void set_max_eig(const T max_eig) { max_eig_ = max_eig; }

        int apply(const T* const b, T* const x) override {
            assert(good());
            if (!good()) {
                fprintf(stderr, "[Error] MPRGP needs to be properly initialized!\n");
                return SFEM_FAILURE;
            }

            bool converged = false;

            ensure_workspace();
            T* g        = workspace_[0];  // Gradient
            T* p        = workspace_[1];
            T* Ap_or_Ag = workspace_[2];
            T* gf_or_gc = workspace_[3];  // Free Gradient or Chopped Gradient
            T* work     = workspace_[4];

            T alpha_bar = 1;
            if (expansion_type_ == Expansion::ORIGINAL) {
                if (max_eig_ == 0) {
                    this->ensure_power_method();
                    this->blas->values(n_dofs, 1, p);
                    max_eig_ = power_method->max_eigen_value(
                            apply_op, 10000, this->eigen_solver_tol, n_dofs, p, g);

                    printf("[MPRGP] max_eig: %g\n", max_eig_);
                }
                alpha_bar = 1.95 / max_eig_;
            } else {
                printf("Skipping power method!\n");
            }

            params_        = {max_eig_, alpha_bar, infty, std::numeric_limits<T>::epsilon()};
            record_        = MPRGPControl<T>{};
            device_reduce_ = make_device_reduce();
            // The run loop takes this solve's operands: built at its first run.
            run_state_.reset();
            device_run_count_ = 0;

            this->project(x);  // Make iterate feasible
            this->gradient(x, b, g);

            // |w g| (Ap_or_Ag, zero from the allocation, as scratch).
            if (norm_weight_) this->blas->xypaz(n_dofs, norm_weight_->data(), g, 0, Ap_or_Ag);
            T norm_g = norm2(norm_weight_ ? Ap_or_Ag : g);
            this->monitor(0, norm_g, 1);
            T norm_gp0 = norm_g;

            // The first direction: the free gradient at the feasible start.
            this->step(nullptr, nullptr, x, g, p, false, work);
            Gradients grads = gradients(record());

            int count_cg_steps = 0;
            int count_expansion_steps = 0;
            int count_proportioning_steps = 0;

            iterations_ = 0;

            while (!converged) {
                if (grads.norm_gc <= this->gamma * grads.norm_gf) {
                    if (team_op_) {
                        // The CG steps in a row in one region; back here at the first that is no
                        // step, for the expansion below.
                        const MPRGPExit exit = team_cg_steps(x, g, p, Ap_or_Ag, gf_or_gc, work, norm_gp0, &grads, &count_cg_steps, &converged);
                        if (exit == MPRGPExit::DONE) break;
                        if (exit == MPRGPExit::PROPORTIONING) continue;
                    } else if (MPRGPExit exit = MPRGPExit::NOT_A_STEP;
                               device_cg_steps(x, g, p, Ap_or_Ag, gf_or_gc, work, norm_gp0, &grads, &count_cg_steps, &converged, &exit)) {
                        if (exit == MPRGPExit::DONE) break;
                        if (exit == MPRGPExit::PROPORTIONING) continue;
                    } else {
                        product(p, Ap_or_Ag);
                        // The decision, the CG step and the next direction p <- gf - beta p, the
                        // last two only where the move is a step; then the record, read once.
                        curvature(x, g, p, Ap_or_Ag, MPRGPDecision::CG, work);
                        this->step(p, Ap_or_Ag, x, g, gf_or_gc, true, work);
                        impl.update_direction(n_dofs, gf_or_gc, p, control_);
                    }
                    const MPRGPControl<T>& c = record();

                    // Without positive curvature the energy falls along -p until a bound stops
                    // it: the expansion step, at the bound. A free body's rigid motions are such
                    // directions, held only by the bounds.
                    if (c.move == T(MPRGPMove::UNBOUNDED)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the search direction (%g)\n", (double)c.dAd);
                        break;
                    }

                    if (c.move == T(MPRGPMove::STEP)) {
                        grads = gradients(c);
                        count_cg_steps++;
                    } else {
                        // apply_op inside!
                        grads = this->expansion_step(c, b, p, Ap_or_Ag, g, x, work);
                        count_expansion_steps++;
                    }
                } else {
                    // apply_op inside!
                    if (!this->proportioning_step(p, Ap_or_Ag, g, gf_or_gc, x, work, &grads)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the chopped gradient\n");
                        break;
                    }

                    count_proportioning_steps++;
                }

                // Check for convergence
                const T norm_gp = grads.norm_gp;
                const T rel_norm_gp = norm_gp / norm_gp0;
                converged = norm_gp < atol || rel_norm_gp < rtol;

                monitor(iterations_, norm_gp, rel_norm_gp);
                if (++iterations_ >= max_it) {
                    break;
                }
            }

            if (verbose) {
                printf("#cg_steps\t\t%d\n", count_cg_steps);
                printf("#expansion_steps\t%d\n", count_expansion_steps);
                printf("#proportioning_steps\t%d\n", count_proportioning_steps);
            }

            run_state_.reset();
            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }
    };
    template <typename T>
    std::shared_ptr<MPRGP<T>> create_mprgp(const std::shared_ptr<Operator<T>> &op, const ExecutionContext &context) {
        return MPRGP<T>::create(op, context);
    }

}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
