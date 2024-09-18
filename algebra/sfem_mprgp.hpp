#ifndef SFEM_MPRGP_HPP
#define SFEM_MPRGP_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {

    template <typename T>
    class MPRGP final : public MatrixFreeLinearSolver<T> {
    public:
        std::function<void(const T* const, T* const)> apply_op;

        T rtol{1e-10};
        T atol{1e-16};
        int max_it{10000};
        int check_each{100};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;

        // MPRGP() : eps_eig_est_(1e-1), power_method_max_it_(10) {}

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        void set_op(const std::shared_ptr<Operator<T>>& op) override {}
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {}
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }

        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }

        void project(T* const x) {
            // TODO
            assert(0);
        }

        void project_gradient(const T* const x, const T* const g, T* p) const {
            // TODO
            assert(0);

            // {
            //     auto d_lb = const_local_view_device(lb);
            //     auto d_ub = const_local_view_device(ub);
            //     auto d_x = const_local_view_device(x);
            //     auto d_g = const_local_view_device(g);
            //     auto d_fi = local_view_device(fi);

            //     parallel_for(
            //         local_range_device(fi), UTOPIA_LAMBDA(const SizeType i) {
            //             // read all
            //             const Scalar li = d_lb.get(i);
            //             const Scalar ui = d_ub.get(i);
            //             const Scalar xi = d_x.get(i);
            //             const Scalar gi = d_g.get(i);

            //             d_fi.set(i, (li < xi && xi < ui) ? gi : Scalar(0.0));
            //         });
            // }
        }

        T compute_alpha(const T* const x, const T* const p) const {
            assert(0);
            // assert(!empty(help_f1));
            // assert(!empty(help_f2));

            // {
            //     auto d_lb = const_local_view_device(lb);
            //     auto d_ub = const_local_view_device(ub);
            //     auto d_x = const_local_view_device(x);
            //     auto d_p = const_local_view_device(p);

            //     auto h1 = local_view_device(help_f1);
            //     auto h2 = local_view_device(help_f2);

            //     parallel_for(
            //         local_range_device(x), UTOPIA_LAMBDA(const SizeType i) {
            //             // read all for quantities
            //             const T li = d_lb.get(i);
            //             const T ui = d_ub.get(i);
            //             const T xi = d_x.get(i);
            //             const T pi = d_p.get(i);

            //             // write both helpers
            //             h1.set(i, (pi > 0) ? ((xi - li) / pi) : T(1e15));
            //             h2.set(i, (pi < 0) ? ((xi - ui) / pi) : T(1e15));
            //         });
            // }

            // return multi_min(help_f1, help_f2);

            return -1;
        }

        // void add_beta(const Vector& x,
        //                   const Vector& g,
        //                   const Vector& lb,
        //                   const Vector& ub,
        //                   Vector& beta) const {
        //     assert(!empty(beta));

        //     {
        //         auto d_lb = const_local_view_device(lb);
        //         auto d_ub = const_local_view_device(ub);
        //         auto d_x = const_local_view_device(x);
        //         auto d_g = const_local_view_device(g);
        //         auto d_beta = local_view_device(beta);

        //         parallel_for(
        //                 local_range_device(beta), UTOPIA_LAMBDA(const SizeType i) {
        //                     const T li = d_lb.get(i);
        //                     const T ui = d_ub.get(i);
        //                     const T xi = d_x.get(i);
        //                     const T gi = d_g.get(i);

        //                     const T val =
        //                             (device::abs(li - xi) < 1e-14)
        //                                     ? device::min(0.0, gi)
        //                                     : ((device::abs(ui - xi) < 1e-14) ? device::max(0.0,
        //                                     gi)
        //                                                                       : 0.0);

        //                     d_beta.set(i, val);
        //                 });
        //     }
        // }

        int apply(const T* const b, T* const x) override {
            T pAp, beta_beta, fi_fi, gp_dot, g_betta, beta_Abeta;
            T alpha_cg, alpha_f, beta_sc;
            T gnorm = -1;

            int it = 0;
            bool converged = false;

            // T* Ax = allocate(n_dofs);
            // T* g = allocate(n_dofs);
            // T* p = allocate(n_dofs);

            // this->project(x);
            // this->apply_op(x, g);
            // this->axpby(n, 1, b, -1, g);
            // this->project_gradient(x, g, p);
            // this->add_beta(x, g);

            //             gp = fi + beta;
            //             p = fi;

            //             dots(beta, beta, beta_beta, fi, fi, fi_fi);

            //             while (!converged) {
            //                 if (beta_beta <= (gamma * gamma * fi_fi)) {
            //                     A.apply(p, Ap);

            //                     dots(p, Ap, pAp, g, p, gp_dot);

            //                     // detecting negative curvature
            //                     if (pAp <= 0.0) {
            //                         return true;
            //                     }

            //                     alpha_cg = gp_dot / pAp;
            //                     alpha_f = compute_alpha(x, p, *lb, *ub, help_f1, help_f2);

            //                     if (hardik_variant_) {
            //                         x -= alpha_cg * p;

            //                         if (alpha_cg <= alpha_f) {
            //                             g -= alpha_cg * Ap;

            //                             this->project_gradient(x, g, *lb, *ub, fi);
            //                             beta_sc = dot(fi, Ap) / pAp;
            //                             p = fi - beta_sc * p;

            //                         } else {
            //                             this->project(*lb, *ub, x);
            //                             A.apply(x, g);
            //                             g -= rhs;
            //                             this->project_gradient(x, g, *lb, *ub, p);
            //                         }

            //                     } else {
            //                         y = x - alpha_cg * p;

            //                         if (alpha_cg <= alpha_f) {
            //                             x = y;
            //                             g = g - alpha_cg * Ap;
            //                             this->project_gradient(x, g, *lb, *ub, fi);
            //                             beta_sc = dot(fi, Ap) / pAp;
            //                             p = fi - beta_sc * p;
            //                         } else {
            //                             x = x - alpha_f * p;
            //                             g = g - alpha_f * Ap;
            //                             this->project_gradient(x, g, *lb, *ub, fi);

            //                             help_f1 = x - (alpha_bar * fi);
            //                             this->project(help_f1, *lb, *ub, x);

            //                             A.apply(x, Ax);
            //                             g = Ax - rhs;
            //                             this->project_gradient(x, g, *lb, *ub, p);
            //                         }
            //                     }
            //                 } else {
            //                     A.apply(beta, Abeta);

            //                     dots(g, beta, g_betta, beta, Abeta, beta_Abeta);
            //                     // detecting negative curvature
            //                     if (beta_Abeta <= 0.0) {
            //                         if (this->verbose()) {
            //                             PrintInfo::print_iter_status(it, {gnorm});
            //                         }

            //                         return true;
            //                     }

            //                     alpha_cg = g_betta / beta_Abeta;
            //                     x = x - alpha_cg * beta;
            //                     g = g - alpha_cg * Abeta;

            //                     this->project_gradient(x, g, *lb, *ub, p);
            //                 }

            //                 this->project_gradient(x, g, *lb, *ub, fi);
            //                 this->add_beta(x, g, *lb, *ub, beta);

            //                 gp = fi + beta;

            //                 dots(beta, beta, beta_beta, fi, fi, fi_fi, gp, gp, gnorm);

            //                 gnorm = std::sqrt(gnorm);
            //                 it++;

            //                 if (this->verbose()) {
            //                     PrintInfo::print_iter_status(it, {gnorm});
            //                 }

            //                 converged = this->check_convergence(it, gnorm, 1, 1);
            //             }

            //             UTOPIA_TRACE_REGION_END("MPRGP::solve(...)");
            //             return converged;
            //         }
            
            return SFEM_SUCCESS;
        }

        //         void set_eig_comp_tol(const T &eps_eig_est) { eps_eig_est_ = eps_eig_est; }

        //         void add_beta(const Vector &x, const Vector &g, const Vector &lb, const
        //         Vector &ub, Vector &beta) const {
        //             assert(!empty(beta));

        //             {
        //                 auto d_lb = const_local_view_device(lb);
        //                 auto d_ub = const_local_view_device(ub);
        //                 auto d_x = const_local_view_device(x);
        //                 auto d_g = const_local_view_device(g);
        //                 auto d_beta = local_view_device(beta);

        //                 parallel_for(
        //                     local_range_device(beta), UTOPIA_LAMBDA(const SizeType i) {
        //                         const T li = d_lb.get(i);
        //                         const T ui = d_ub.get(i);
        //                         const T xi = d_x.get(i);
        //                         const T gi = d_g.get(i);

        //                         const T val = (device::abs(li - xi) < 1e-14)
        //                                                ? device::min(0.0, gi)
        //                                                : ((device::abs(ui - xi) < 1e-14) ?
        //                                                device::max(0.0, gi) : 0.0);

        //                         d_beta.set(i, val);
        //                     });
        //             }
        //         }

        //     private:
        //         T power_method(const Operator<Vector> &A) {
        //             // Super simple power method to estimate the biggest eigenvalue
        //             assert(!empty(help_f2));
        //             help_f2.set(1.0);

        //             SizeType it = 0;
        //             bool converged = false;
        //             T gnorm, lambda = 0.0, lambda_old;

        //             while (!converged) {
        //                 help_f1 = help_f2;
        //                 A.apply(help_f1, help_f2);
        //                 help_f2 = T(1.0 / T(norm2(help_f2))) * help_f2;

        //                 lambda_old = lambda;

        //                 A.apply(help_f2, help_f1);
        //                 lambda = dot(help_f2, help_f1);

        //                 fi = help_f2 - help_f1;
        //                 gnorm = norm2(fi);

        //                 converged = ((gnorm < eps_eig_est_) || (std::abs(lambda_old - lambda)
        //                 < eps_eig_est_) ||
        //                              it > power_method_max_it_)
        //                                 ? true
        //                                 : false;

        //                 it = it + 1;
        //             }

        //             if (this->verbose() && mpi_world_rank() == 0)
        //                 sfem::out() << "Power method converged in " << it << " iterations.
        //                 Largest eig: " << lambda << "  \n";

        //             return lambda;
        //         }

        //     public:
        //         void init_memory(const Layout &layout) override {
        //             OperatorBasedQPSolver<Matrix, Vector>::init_memory(layout);

        //             fi.zeros(layout);
        //             beta.zeros(layout);
        //             gp.zeros(layout);
        //             p.zeros(layout);
        //             y.zeros(layout);
        //             Ap.zeros(layout);
        //             Abeta.zeros(layout);
        //             Ax.zeros(layout);
        //             g.zeros(layout);
        //             help_f1.zeros(layout);
        //             help_f2.zeros(layout);

        //             initialized_ = true;
        //             layout_ = layout;
        //         }

        //         void set_preconditioner(const std::shared_ptr<Preconditioner<Vector> >
        //         &precond) override {
        //             precond_ = precond;
        //         }

        //     private:
        //         Vector fi, beta, gp, p, y, Ap, Abeta, Ax, g, help_f1, help_f2;

        //         T eps_eig_est_;
        //         SizeType power_method_max_it_;

        //         bool initialized_{false};
        //         Layout layout_;

        //         std::shared_ptr<Preconditioner<Vector> > precond_;
        //         bool hardik_variant_{false};
    };
}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
