#ifndef SFEM_BCGS_HPP
#define SFEM_BCGS_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
namespace sfem {
    template <typename T>
    class BiCGStab final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> left_preconditioner_op;
        std::function<void(const T* const, T* const)> right_preconditioner_op;
        BLAS_Tpl<T> blas;

        ptrdiff_t n_dofs{-1};
        int iterations_{0};

        bool verbose{true};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->right_preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            // left_preconditioner_op = in;
            right_preconditioner_op = in;
        }

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};

        void set_atol(const T val) { tol = val; }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        bool good() const {
            assert(apply_op);
            return blas.good() && apply_op;
        }

        void monitor(const int iter, const T residual) {
            if (verbose && (iter == max_it || iter % 100 == 0 || residual < tol)) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            if (left_preconditioner_op || right_preconditioner_op) {
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
                return -1;
            }

            T* r0 = blas.allocate(n);

            // Residual
            apply_op(x, r0);
            blas.axpby(n, 1, b, -1, r0);

            T rho = blas.dot(n, r0, r0);

            if (sqrt(rho) < tol) {
                blas.destroy(r0);
                return 0;
            }

            T* r = blas.allocate(n);
            T* p = blas.allocate(n);
            T* v = blas.allocate(n);
            T* h = blas.allocate(n);
            T* s = blas.allocate(n);
            T* t = blas.allocate(n);

            blas.copy(n, r0, r);
            blas.copy(n, r0, p);

            int info = -1;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                blas.zeros(n, v);
                apply_op(p, v);

                const T ptv = blas.dot(n, r0, v);
                const T alpha = rho / ptv;

                blas.zaxpby(n, 1, x, alpha, p, h);
                blas.zaxpby(n, 1, r, -alpha, v, s);

                const T sts = blas.dot(n, s, s);

                if (sqrt(sts) < tol) {
                    monitor(iterations_, sqrt(sts));
                    blas.copy(n, h, x);
                    info = 0;
                    break;
                }

                blas.zeros(n, t);
                apply_op(s, t);

                const T tts = blas.dot(n, t, s);
                const T ttt = blas.dot(n, t, t);
                const T omega = tts / ttt;

                blas.zaxpby(n, 1, h, omega, s, x);
                blas.zaxpby(n, 1, s, -omega, t, r);

                const T rtr = blas.dot(n, r, r);

                monitor(iterations_, sqrt(rtr));
                if (sqrt(rtr) < tol) {
                    info = 0;
                    break;
                }

                const T rho_new = blas.dot(n, r0, r);
                const T beta = (rho_new / rho) * (alpha / omega);
                rho = rho_new;

                blas.axpby(n, 1, r, beta, p);
                blas.axpby(n, -omega * beta, v, 1, p);
            }

            // clean-up
            blas.destroy(r0);
            blas.destroy(r);
            blas.destroy(p);
            blas.destroy(v);
            blas.destroy(h);
            blas.destroy(s);
            blas.destroy(t);
            return info;
        }

        int aux_apply_precond(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return -1;
            }

            T* r0 = blas.allocate(n);

            // Residual
            apply_op(x, r0);
            blas.axpby(n, 1, b, -1, r0);

            T rho = blas.dot(n, r0, r0);

            if (sqrt(rho) < tol) {
                blas.destroy(r0);
                return 0;
            }

            T* r = blas.allocate(n);
            T* p = blas.allocate(n);
            T* v = blas.allocate(n);
            T* h = blas.allocate(n);
            T* s = blas.allocate(n);
            T* t = blas.allocate(n);

            blas.copy(n, r0, r);
            blas.copy(n, r0, p);

            int info = -1;
            for (int iterations_ = 0; iterations_ < max_it; iterations_++) {
                auto y = t;  // reuse t as a temp for y
                blas.zeros(n, y);
                right_preconditioner_op(p, y);

                blas.zeros(n, y);
                apply_op(y, v);

                const T ptv = blas.dot(n, r0, v);
                const T alpha = rho / ptv;

                blas.zaxpby(n, 1, x, alpha, y, h);
                blas.zaxpby(n, 1, r, -alpha, v, s);

                const T sts = blas.dot(n, s, s);

                if (sqrt(sts) < tol) {
                    monitor(iterations_, sqrt(sts));
                    blas.copy(n, h, x);
                    info = 0;
                    break;
                }

                auto z = x;  // reuse x as a temp for z

                blas.zeros(n, z);
                right_preconditioner_op(s, z);

                blas.zeros(n, t);
                apply_op(z, t);

                const T tts = blas.dot(n, t, s);
                const T ttt = blas.dot(n, t, t);
                const T omega = tts / ttt;

                blas.zaxpby(n, 1, h, omega, z, x);
                blas.zaxpby(n, 1, s, -omega, t, r);

                const T rtr = blas.dot(n, r, r);

                monitor(iterations_, sqrt(rtr));
                if (sqrt(rtr) < tol) {
                    info = 0;
                    break;
                }

                const T rho_new = blas.dot(n, r0, r);
                const T beta = (rho_new / rho) * (alpha / omega);
                rho = rho_new;

                blas.axpby(n, 1, r, beta, p);
                blas.axpby(n, -omega * beta, v, 1, p);
            }

            // clean-up
            blas.destroy(r0);
            blas.destroy(r);
            blas.destroy(p);
            blas.destroy(v);
            blas.destroy(h);
            blas.destroy(s);
            blas.destroy(t);
            return info;
        }
    };

    template <typename T>
    std::shared_ptr<BiCGStab<T>> h_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        cg->default_init();
        return cg;
    }
}  // namespace sfem

#endif  // SFEM_BCGS_HPP
