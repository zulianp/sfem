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

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
// Must check: https://www.dcs.warwick.ac.uk/pmbs/pmbs14/PMBS14/Workshop_Schedule_files/8-CUDAHPCG.pdf
namespace sfem {

    template <typename T>
    class ConjugateGradient final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(T*)> destroy;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};
        int check_each{1};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};

        void set_verbose(const bool val)
        {
            verbose = val;
        }


        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            preconditioner_op = in;
        }

        void default_init() {
            allocate = [](const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](T* a) { free(a); };

            copy = [](const ptrdiff_t n, const T* const src, T* const dest) {
                std::memcpy(dest, src, n * sizeof(T));
            };

            dot = [](const ptrdiff_t n, const T* const l, const T* const r) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += l[i] * r[i];
                }

                return ret;
            };

            axpby =
                [](const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] = alpha * x[i] + beta * y[i];
                    }
                };

            zeros = [](const std::size_t n, T* const x) {
                memset(x, 0, n*sizeof(T));
            };
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(dot);
            assert(axpby);
            assert(apply_op);

            return allocate && destroy && copy && dot && axpby && apply_op;
        }

        void monitor(const int iter, const T residual) {
            if(!verbose) return;

            if (iter == max_it || iter == 0 || iter % check_each == 0 || residual < tol) {
                std::cout << iter << ": " << residual << "\n";
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

            T* r = allocate(n);

            apply_op(x, r);

            axpby(n, 1, b, -1, r);

            T rtr = dot(n, r, r);

            if (sqrt(rtr) < tol) {
                destroy(r);
                return 0;
            }

            monitor(0, sqrt(rtr));

            T* p = allocate(n);
            T* Ap = allocate(n);

            copy(n, r, p);

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                zeros(n, Ap);
                apply_op(p, Ap);

                const T ptAp = dot(n, p, Ap);
                const T alpha = rtr / ptAp;

                axpby(n, alpha, p, 1, x);
                axpby(n, -alpha, Ap, 1, r);

                const T rtr_new = dot(n, r, r);

                monitor(k+1, sqrt(rtr_new));

                if (sqrt(rtr_new) < tol) {
                    info = 0;
                    break;
                }

                const T beta = rtr_new / rtr;
                rtr = rtr_new;
                axpby(n, 1, r, beta, p);
            }

            // clean-up
            destroy(r);
            destroy(p);
            destroy(Ap);
            return info;
        }

        int aux_apply_precond(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return -1;
            }

            T* r = allocate(n);

            apply_op(x, r);
            axpby(n, 1, b, -1, r);

            T rtr = dot(n, r, r);

            monitor(0, sqrt(rtr));

            if (sqrt(rtr) < tol) {
                destroy(r);
                return 0;
            }

            T* z = allocate(n);
            T* Mz = allocate(n);
            T* p = allocate(n);
            T* Ap = allocate(n);

            preconditioner_op(r, z);
            copy(n, z, p);

            zeros(n, Ap);
            apply_op(p, Ap);

            T rtz = dot(n, r, z);
            
            {
                const T ptAp = dot(n, p, Ap);
                
                assert(ptAp != 0);
                const T alpha = rtr / ptAp;

                axpby(n, alpha, p, 1, x);
                axpby(n, -alpha, Ap, 1, r);
            }

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                zeros(n, z);
                preconditioner_op(r, z);

                const T rtz_new = dot(n, r, z);

                assert(rtz != 0);
                const T beta = rtz_new / rtz;
                rtz = rtz_new;

                axpby(n, 1, z, beta, p);

                zeros(n, Ap);
                apply_op(p, Ap);

                const T ptAp = dot(n, p, Ap);
                const T alpha = rtz / ptAp;

                axpby(n, alpha, p, 1, x);
                axpby(n, -alpha, Ap, 1, r);

                monitor(k+1, sqrt(rtz));

                if (sqrt(rtz) < tol) {
                    info = 0;
                    break;
                }
            }

            // clean-up
            destroy(r);
            destroy(p);
            destroy(Ap);

            destroy(z);
            destroy(Mz);
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
