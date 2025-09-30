#ifndef SFEM_CG_HPP
#define SFEM_CG_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_openmp_blas.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
// Must check:
// https://www.dcs.warwick.ac.uk/pmbs/pmbs14/PMBS14/Workshop_Schedule_files/8-CUDAHPCG.pdf
namespace sfem {

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T>& diagonal_scaling, const ExecutionSpace es);

    template <typename T>
    class ConjugateGradient final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::shared_ptr<Operator<T>> apply_op;
        std::shared_ptr<Operator<T>> preconditioner_op;

        BLAS_Tpl<T> blas;

        std::function<void(T*)> interceptor;

        // Solver parameters
        T              rtol{1e-10};
        T              atol{1e-16};
        int            max_it{10000};
        int            check_each{1};
        ptrdiff_t      n_dofs{SFEM_PTRDIFF_INVALID};
        int            iterations_{0};
        bool           verbose{true};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        void set_atol(const T val) { atol = val; }

        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            apply_op = op;
            n_dofs   = op->rows();
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) override {
            assert(execution_space() == (enum ExecutionSpace)diag->mem_space());
            auto J         = op + sfem::diag_op(diag, execution_space());
            this->apply_op = J;
            n_dofs         = op->rows();

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
                                  const SharedBuffer<T>&            diag) override {
            assert(execution_space() == (enum ExecutionSpace)diag->mem_space());
            this->apply_op = op + sfem::create_sparse_block_vector_mult(op->rows(), sbv, diag);

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
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        bool good() const { return blas.good() && apply_op; }

        void monitor(const int iter, const T residual, const T relative_residual, const T alpha) {
            if (!verbose) return;

            if (iter == max_it || iter == 0 || iter % check_each == 0 || relative_residual < rtol) {
                std::cout << iter << ": residual abs: " << residual << ", rel: " << relative_residual << " (rtol = " << rtol
                          << ", atol = " << atol << ", alpha = " << alpha << ")\n";
            }
        }

        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            SFEM_TRACE_SCOPE("ConjugateGradient::apply");

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

    private:
        int aux_apply_basic(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                assert(0);
                return SFEM_FAILURE;
            }

            T* r = blas.allocate(n);

            apply_op->apply(x, r);

            blas.axpby(n, 1, b, -1, r);

            const T rtr0    = blas.dot(n, r, r);
            const T r_norm0 = sqrt(rtr0);
            monitor(0, r_norm0, 1, 0);

            T rtr = rtr0;
            assert(rtr0 == rtr0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            T* p  = blas.allocate(n);
            T* Ap = blas.allocate(n);

            blas.copy(n, r, p);

            int info = SFEM_FAILURE;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                blas.zeros(n, Ap);
                apply_op->apply(p, Ap);

                const T ptAp  = blas.dot(n, p, Ap);

                assert(ptAp != 0);
                assert(std::isfinite(ptAp) && ptAp != 0);
                assert(std::isfinite(rtr) && rtr != 0);

                const T alpha = rtr / ptAp;


                assert(std::isfinite(alpha));
                
                assert(ptAp == ptAp);
                assert(alpha == alpha);

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);

                assert(rtr != 0);
                assert(rtr == rtr);

                const T rtr_new = blas.dot(n, r, r);
                const T beta    = rtr_new / rtr;
                rtr             = rtr_new;
                blas.axpby(n, 1, r, beta, p);

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

            // clean-up
            blas.destroy(r);
            blas.destroy(p);
            blas.destroy(Ap);
            return info;
        }

        int aux_apply_precond(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return SFEM_FAILURE;
            }

            T* r = blas.allocate(n);

            apply_op->apply(x, r);
            blas.axpby(n, 1, b, -1, r);

            const T rtr0    = blas.dot(n, r, r);
            const T r_norm0 = sqrt(rtr0);
            monitor(0, r_norm0, 1, 0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            T* z  = blas.allocate(n);
            T* Mz = blas.allocate(n);
            T* p  = blas.allocate(n);
            T* Ap = blas.allocate(n);

            preconditioner_op->apply(r, z);
            blas.copy(n, z, p);

            blas.zeros(n, Ap);
            apply_op->apply(p, Ap);

            const T rtz0 = blas.dot(n, r, z);
            T       rtz  = rtz0;

            if (rtz == 0) {
                return SFEM_SUCCESS;
            }

            {
                const T ptAp = blas.dot(n, p, Ap);

                assert(ptAp != 0);
                const T alpha = rtz / ptAp;

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);
            }

            int info = SFEM_FAILURE;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                blas.zeros(n, z);
                preconditioner_op->apply(r, z);

                const T rtz_new = blas.dot(n, r, z);

                assert(rtz != 0);
                const T beta = rtz_new / rtz;
                rtz          = rtz_new;

                blas.axpby(n, 1, z, beta, p);

                blas.zeros(n, Ap);
                apply_op->apply(p, Ap);

                const T ptAp  = blas.dot(n, p, Ap);
                const T alpha = rtz / ptAp;

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);

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

            // clean-up
            blas.destroy(r);
            blas.destroy(p);
            blas.destroy(Ap);

            blas.destroy(z);
            blas.destroy(Mz);
            return info;
        }
    };

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> h_cg() {
        auto cg = std::make_shared<ConjugateGradient<T>>();
        cg->default_init();
        return cg;
    }
}  // namespace sfem

#endif  // SFEM_CG_HPP
