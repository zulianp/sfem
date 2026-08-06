#ifndef SFEM_PARALLEL_CG_HPP
#define SFEM_PARALLEL_CG_HPP

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
#include "sfem_cg.hpp"
#include "sfem_openmp_blas.hpp"
#include "sfem_openmp_cg_impl.hpp"

namespace sfem {

    /// Parallel conjugate gradient for @ref ParallelOperator.
    ///
    /// Buffer contract for @ref apply:
    /// - @p x must provide @c apply_op->col_allocation_size() entries (owned + ghosts + aura).
    ///   CG updates the owned prefix in place; ghost/aura slots are used as apply scratch
    ///   (in-place gather). There is no extra solution workspace.
    /// - @p b must provide at least @c apply_op->rows() entries (owned RHS). Only the owned
    ///   prefix is read by CG. If you also call @c Function::apply_constraints on @p b with
    ///   sideset-derived nodesets, allocate @c row_allocation_size() — nodesets may index ghosts.
    ///
    /// Math (dots / axpy / copies) runs on owned length @c rows(). Workspace vectors
    /// (@c r, @c p, @c Ap, optional @c z) are allocated to
    /// @c max(row_allocation_size(), col_allocation_size()).
    template <typename T>
    class ParallelConjugateGradient final : public MatrixFreeLinearSolver<T> {
    public:
        std::shared_ptr<ParallelOperator<T>> apply_op;
        std::shared_ptr<Operator<T>>         preconditioner_op;
        std::shared_ptr<Communicator>        comm_;
        std::shared_ptr<BLAS<T>>             blas;
        CG_Tpl<T>                            impl;

        std::function<void(T*)> interceptor;

        T              rtol{1e-10};
        T              atol{1e-16};
        int            max_it{10000};
        int            check_each{1};
        ptrdiff_t      n_dofs{SFEM_PTRDIFF_INVALID};
        ptrdiff_t      work_allocation_{-1};
        int            iterations_{0};
        bool           verbose{true};
        bool           apply_overwrites_output{false};
        bool           fusion{true};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }
        void set_verbose(const bool val) { verbose = val; }
        void set_apply_overwrites_output(const bool val) { apply_overwrites_output = val; }
        void set_fusion(const bool val) { fusion = val; }
        /// \deprecated Use set_fusion
        void set_host_fusion(const bool val) { set_fusion(val); }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            auto parallel_op = std::dynamic_pointer_cast<ParallelOperator<T>>(op);
            if (!parallel_op) {
                SFEM_ERROR("ParallelConjugateGradient::set_op requires a ParallelOperator\n");
                return;
            }
            set_op(parallel_op);
        }

        void set_op(const std::shared_ptr<ParallelOperator<T>>& op) {
            apply_op         = op;
            n_dofs           = op->rows();
            work_allocation_ = std::max(op->row_allocation_size(), op->col_allocation_size());
            comm_            = op->comm();
            assert(comm_);
            assert(work_allocation_ >= n_dofs);
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) override {
            assert(execution_space() == (enum ExecutionSpace)diag->mem_space());
            set_op(op);

            if (preconditioner_op) {
                auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner_op);
                if (shiftable) {
                    return shiftable->shift(diag);
                } else {
                    SFEM_ERROR(
                            "Tried to call shift on object that is not subclass of "
                            "ShiftableOperator!\n");
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          op,
                                  const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                  const SharedBuffer<T>&                       diag) override {
            assert(execution_space() == (enum ExecutionSpace)diag->mem_space());
            set_op(op);

            if (preconditioner_op) {
                auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner_op);

                if (shiftable) {
                    return shiftable->shift(sbv, diag);
                } else {
                    SFEM_ERROR(
                            "Tried to call shift on object that is not subclass of "
                            "ShiftableOperator!\n");
                    assert(false);
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override { preconditioner_op = op; }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) { preconditioner_op = in; }

        void default_init() {
            blas             = make_openmp_blas<T>();
            OpenMP_CG<T>::build(impl);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        bool good() const { return blas && blas->good() && apply_op && comm_ && (!fusion || impl.good()); }

        void monitor(const int iter, const T residual, const T relative_residual, const T alpha) {
            if (!verbose) return;
            if (!comm_ || comm_->rank() != 0) return;

            if (iter == max_it || iter == 0 || iter % check_each == 0 || relative_residual < rtol) {
                std::cout << iter << ": residual abs: " << residual << ", rel: " << relative_residual << " (rtol = " << rtol
                          << ", atol = " << atol << ", alpha = " << alpha << ")\n";
            }
        }

        /// Solve @c Op x = b. @p x must be @c col_allocation_size(); @p b is owned-length.
        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            SFEM_TRACE_SCOPE("ParallelConjugateGradient::apply");

            if (preconditioner_op) {
                return aux_apply_precond(n, b, x);
            } else {
                return aux_apply_basic(n, b, x);
            }
        }

        int apply(const T* const b, T* const x) override {
            assert(n_dofs >= 0);
            if (this->n_dofs < 0) {
                std::cerr << "Error uninitiaized n_dofs. Set set_n_dofs to set the number of dofs\n";
                return 1;
            }

            return apply(this->n_dofs, b, x);
        }

        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        ~ParallelConjugateGradient() { destroy_workspace(); }

    private:
        T*        work_r_{nullptr};
        T*        work_z_{nullptr};
        T*        work_p_{nullptr};
        T*        work_Ap_{nullptr};
        ptrdiff_t work_n_{-1};

        T reduce(const T local) const {
            assert(comm_);
            return comm_->sum(local);
        }

        bool use_fusion() const { return fusion && impl.good(); }

        ptrdiff_t allocation_size() const {
            return (work_allocation_ > 0) ? work_allocation_ : n_dofs;
        }

        // Zero full allocation (owned + ghosts/aura) before apply into y.
        void ensure_apply_buffer(T* const y) const {
            if (!apply_overwrites_output) {
                blas->zeros(allocation_size(), y);
            }
        }

        void destroy_workspace() {
            if (!blas) {
                work_r_ = work_z_ = work_p_ = work_Ap_ = nullptr;
                work_n_                                = -1;
                return;
            }
            if (work_r_) {
                blas->destroy(work_r_);
                work_r_ = nullptr;
            }
            if (work_z_) {
                blas->destroy(work_z_);
                work_z_ = nullptr;
            }
            if (work_p_) {
                blas->destroy(work_p_);
                work_p_ = nullptr;
            }
            if (work_Ap_) {
                blas->destroy(work_Ap_);
                work_Ap_ = nullptr;
            }
            work_n_ = -1;
        }

        void ensure_workspace(const ptrdiff_t n_owned, const bool need_z) {
            const ptrdiff_t n_alloc = (work_allocation_ > 0) ? work_allocation_ : n_owned;
            assert(n_alloc >= n_owned);

            if (work_n_ == n_alloc && work_r_ && work_p_ && work_Ap_ && (!need_z || work_z_)) {
                return;
            }

            destroy_workspace();
            work_r_  = blas->allocate(n_alloc);
            work_p_  = blas->allocate(n_alloc);
            work_Ap_ = blas->allocate(n_alloc);
            blas->zeros(n_alloc, work_r_);
            blas->zeros(n_alloc, work_p_);
            blas->zeros(n_alloc, work_Ap_);
            if (need_z) {
                work_z_ = blas->allocate(n_alloc);
                blas->zeros(n_alloc, work_z_);
            }
            work_n_ = n_alloc;
        }

        T update_x_r(const ptrdiff_t n,
                     const T         alpha,
                     const T* const  p,
                     const T* const  Ap,
                     T* const        x,
                     T* const        r,
                     const bool      with_rtr) const {
            if (use_fusion()) {
                if (with_rtr) {
                    return reduce(impl.update_x_r_and_rtr(n, alpha, p, Ap, x, r));
                }
                impl.update_x_r(n, alpha, p, Ap, x, r);
                return T(0);
            }

            blas->axpby(n, alpha, p, T(1), x);
            blas->axpby(n, -alpha, Ap, T(1), r);
            return with_rtr ? reduce(blas->dot(n, r, r)) : T(0);
        }

        void update_p(const ptrdiff_t n, const T beta, const T* const z, T* const p) const {
            if (use_fusion()) {
                impl.update_p(n, beta, z, p);
            } else {
                blas->axpby(n, T(1), z, beta, p);
            }
        }

        int aux_apply_basic(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                assert(0);
                return SFEM_FAILURE;
            }

            ensure_workspace(n, false);
            T* r  = work_r_;
            T* p  = work_p_;
            T* Ap = work_Ap_;

            ensure_apply_buffer(r);
            apply_op->apply(x, r);
            blas->axpby(n, T(1), b, T(-1), r);

            const T rtr0    = reduce(blas->dot(n, r, r));
            const T r_norm0 = sqrt(rtr0);
            monitor(0, r_norm0, 1, 0);

            T rtr = rtr0;
            assert(rtr0 == rtr0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            blas->copy(n, r, p);

            int info = SFEM_FAILURE;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                ensure_apply_buffer(Ap);
                apply_op->apply(p, Ap);

                const T ptAp  = reduce(blas->dot(n, p, Ap));
                const T alpha = rtr / ptAp;

                assert(ptAp == ptAp);
                assert(alpha == alpha);
                assert(rtr != 0);
                assert(rtr == rtr);

                const T rtr_new = update_x_r(n, alpha, p, Ap, x, r, true);
                const T beta    = rtr_new / rtr;
                rtr             = rtr_new;
                update_p(n, beta, r, p);

                T r_norm = sqrt(rtr_new);
                assert(r_norm == r_norm);

                monitor(iterations_ + 1, r_norm, r_norm / r_norm0, alpha);
                if (r_norm < atol || rtr_new == 0 || r_norm / r_norm0 < rtol) {
                    info = SFEM_SUCCESS;
                    break;
                }

                if (interceptor) {
                    interceptor(x);
                }
            }

            return info;
        }

        int aux_apply_precond(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return SFEM_FAILURE;
            }

            ensure_workspace(n, true);
            T* r  = work_r_;
            T* z  = work_z_;
            T* p  = work_p_;
            T* Ap = work_Ap_;

            ensure_apply_buffer(r);
            apply_op->apply(x, r);
            blas->axpby(n, T(1), b, T(-1), r);

            const T rtr0    = reduce(blas->dot(n, r, r));
            const T r_norm0 = sqrt(rtr0);
            monitor(0, r_norm0, 1, 0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            ensure_apply_buffer(z);
            preconditioner_op->apply(r, z);
            blas->copy(n, z, p);

            ensure_apply_buffer(Ap);
            apply_op->apply(p, Ap);

            const T rtz0 = reduce(blas->dot(n, r, z));
            T       rtz  = rtz0;

            if (rtz == 0) {
                return SFEM_SUCCESS;
            }

            {
                const T ptAp  = reduce(blas->dot(n, p, Ap));
                assert(ptAp != 0);
                const T alpha = rtz / ptAp;
                update_x_r(n, alpha, p, Ap, x, r, false);
            }

            int info = SFEM_FAILURE;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                ensure_apply_buffer(z);
                preconditioner_op->apply(r, z);

                const T rtz_new = reduce(blas->dot(n, r, z));

                assert(rtz != 0);
                const T beta = rtz_new / rtz;
                rtz          = rtz_new;

                update_p(n, beta, z, p);

                ensure_apply_buffer(Ap);
                apply_op->apply(p, Ap);

                const T ptAp  = reduce(blas->dot(n, p, Ap));
                const T alpha = rtz / ptAp;

                update_x_r(n, alpha, p, Ap, x, r, false);

                auto anorm = sqrt(rtz);
                auto rnorm = anorm / sqrt(rtz0);

                monitor(iterations_ + 1, anorm, rnorm, alpha);
                if (anorm < atol || rnorm < rtol) {
                    info = SFEM_SUCCESS;
                    break;
                }

                if (interceptor) {
                    interceptor(x);
                }
            }

            return info;
        }
    };

    template <typename T>
    std::shared_ptr<ParallelConjugateGradient<T>> h_parallel_cg() {
        auto cg = std::make_shared<ParallelConjugateGradient<T>>();
        cg->default_init();
        return cg;
    }

}  // namespace sfem

#endif  // SFEM_PARALLEL_CG_HPP
