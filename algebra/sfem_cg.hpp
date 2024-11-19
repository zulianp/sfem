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
#include "sfem_openmp_blas.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
// Must check:
// https://www.dcs.warwick.ac.uk/pmbs/pmbs14/PMBS14/Workshop_Schedule_files/8-CUDAHPCG.pdf
namespace sfem {

    template <typename T>
    class ConjugateGradient final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;
        BLAS_Tpl<T> blas;

        // Solver parameters
        T rtol{1e-10};
        T atol{1e-16};
        int max_it{10000};
        int check_each{100};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        void set_atol(const T val) { atol = val; }

        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            n_dofs = op->rows();
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            preconditioner_op = in;
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        bool good() const {
            return blas.good() && apply_op;
        }

        void monitor(const int iter, const T residual, const T relative_residual) {
            if (!verbose) return;

            if (iter == max_it || iter == 0 || iter % check_each == 0 || relative_residual < rtol) {
                std::cout << iter << ": residual abs: " << residual
                          << ", rel: " << relative_residual << " (rtol = " << rtol
                          << ", atol = " << atol << ")\n";
            }
        }

        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            if (preconditioner_op) {
                return aux_apply_precond(n, b, x);
            } else {
                return aux_apply_basic(n, b, x);
            }
        }

        int apply(const T* const b, T* const x) override {
            assert(n_dofs >= 0);
            if (this->n_dofs < 0) {
                std::cerr
                        << "Error uninitiaized n_dofs. Set set_n_dofs to set the number of dofs\n";
                return 1;
            }

            return apply(this->n_dofs, b, x);
        }

        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

    private:
        int aux_apply_basic(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                assert(0);
                return -1;
            }

            T* r = blas.allocate(n);

            apply_op(x, r);

            blas.axpby(n, 1, b, -1, r);

            const T rtr0 = blas.dot(n, r, r);
            const T r_norm0 = sqrt(rtr0);
            monitor(0, r_norm0, 1);

            T rtr = rtr0;

            if (rtr0 == 0) {
                return 0;
            }

            T* p = blas.allocate(n);
            T* Ap = blas.allocate(n);

            blas.copy(n, r, p);

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                blas.zeros(n, Ap);
                apply_op(p, Ap);

                const T ptAp = blas.dot(n, p, Ap);
                const T alpha = rtr / ptAp;

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);

                assert(rtr != 0);

                const T rtr_new = blas.dot(n, r, r);
                const T beta = rtr_new / rtr;
                rtr = rtr_new;
                blas.axpby(n, 1, r, beta, p);

                T r_norm = sqrt(rtr_new);

                monitor(k + 1, r_norm, r_norm / r_norm0);
                if (r_norm < atol || rtr_new == 0 || r_norm / r_norm0 < rtol) {
                    info = 0;
                    break;
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
                return -1;
            }

            T* r = blas.allocate(n);

            apply_op(x, r);
            blas.axpby(n, 1, b, -1, r);

            const T rtr0 = blas.dot(n, r, r);
            T rtr = rtr0;

            monitor(0, sqrt(rtr), 1);

            // if (sqrt(rtr) < rtol) {
            //     blas.destroy(r);
            //     return 0;
            // }

            T* z = blas.allocate(n);
            T* Mz = blas.allocate(n);
            T* p = blas.allocate(n);
            T* Ap = blas.allocate(n);

            preconditioner_op(r, z);
            blas.copy(n, z, p);

            blas.zeros(n, Ap);
            apply_op(p, Ap);

            T rtz = blas.dot(n, r, z);

            {
                const T ptAp = blas.dot(n, p, Ap);

                assert(ptAp != 0);
                const T alpha = rtr / ptAp;

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);
            }

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                blas.zeros(n, z);
                preconditioner_op(r, z);

                const T rtz_new = blas.dot(n, r, z);

                assert(rtz != 0);
                const T beta = rtz_new / rtz;
                rtz = rtz_new;

                blas.axpby(n, 1, z, beta, p);

                blas.zeros(n, Ap);
                apply_op(p, Ap);

                const T ptAp = blas.dot(n, p, Ap);
                const T alpha = rtz / ptAp;

                blas.axpby(n, alpha, p, 1, x);
                blas.axpby(n, -alpha, Ap, 1, r);

                auto anorm = sqrt(rtz);
                auto rnorm = anorm / sqrt(rtr0);

                monitor(k + 1, anorm, rnorm);
                if (anorm < atol || rnorm < rtol) {
                    info = 0;
                    break;
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
