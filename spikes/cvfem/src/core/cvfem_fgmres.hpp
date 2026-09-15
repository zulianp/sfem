#pragma once

// Flexible GMRES with restart.
//
// Needed because the V-cycle stops being a fixed linear operator the moment its levels are
// smoothed with a Krylov method rather than a stationary iteration: the number of inner
// iterations, and therefore the map from residual to correction, varies between
// applications. BiCGStab assumes a constant preconditioner and breaks silently when that
// assumption fails -- it does not diverge loudly, it stagnates -- so the outer solver has to
// become flexible at the same time as the smoother becomes Krylov. The two changes are one
// change.
//
// Flexibility is exactly the difference from ordinary GMRES: the preconditioned vectors
// z_j = M_j^{-1} v_j are stored alongside the Krylov basis, and the update is built from
// those rather than from a single preconditioner applied at the end. That costs a second
// vector per iteration, which is why the restart length matters.
//
// Lives here rather than in algebra/ because this is a spike; if it earns its place it
// belongs next to sfem_bcgs.hpp with the rest of the solvers.
//
// Every vector operation goes through SFEM's OpenMP BLAS, the layer sfem_cg.hpp uses, and the
// basis is allocated once and kept across restarts and solves. The loops here were serial and
// allocated two vectors per iteration: on the FDA nozzle at 893,924 dof the orthogonalisation
// alone was ~280 s of a 713 s solve on one core while the other 71 waited. The BLAS dot is the
// deterministic fixed-chunk sum, so the solve stays reproducible across thread counts.

#include "sfem_Operator.hpp"
#include "sfem_openmp_blas.hpp"
#include "smesh_env.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace sfem {

    template <typename T>
    class FGMRES final {
    public:
        explicit FGMRES(const std::shared_ptr<Operator<T>> &op) : op_(op), blas_(make_openmp_blas<T>()) {}
        ~FGMRES() { release(); }
        FGMRES(const FGMRES &)            = delete;
        FGMRES &operator=(const FGMRES &) = delete;

        void set_max_it(const int v) { max_it_ = v; }
        void set_rtol(const T v) { rtol_ = v; }
        void set_atol(const T v) { atol_ = v; }
        void set_restart(const int v) { restart_ = v; }
        // Growth factor at which the solve is abandoned; 0 disables the test. See apply()
        // for why this is measured on the true residual at a restart rather than on the
        // Arnoldi estimate.
        void set_dtol(const T v) { dtol_ = v; }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>> &p) { prec_ = p; }
        int  iterations() const { return iterations_; }
        // Why the last solve stopped. A count that ended on max_it is a floor, not a result.
        bool has_diverged() const { return diverged_; }
        bool hit_max_it() const { return iterations_ >= max_it_; }

        bool verbose{true};

        int apply(const T *const b, T *const x) {
            const ptrdiff_t n = op_->rows();
            const int       m = restart_;
            if (n != n_) {
                release();
                n_ = n;
            }

            T *const                    r = vec(work_, 0);
            T *const                    w = vec(work_, 1);
            std::vector<std::vector<T>> H;
            std::vector<T>              cs((size_t)m, 0), sn((size_t)m, 0), g((size_t)m + 1, 0);

            iterations_ = 0;
            diverged_   = false;

            // Best iterate seen, so a solve abandoned as divergent hands back the best point
            // it reached rather than the diverging one. GMRES only writes x at the end of a
            // restart cycle, so without this the caller receives precisely the worst iterate.
            // One extra vector against the 2*(restart+1) the Krylov basis already holds.
            T *x_best    = nullptr;
            T  beta_best = std::numeric_limits<T>::max();

            T bnorm = blas_->norm2(n, b);
            if (bnorm == T(0)) bnorm = T(1);

            // Divergence is growth relative to where THIS solve started, not relative to the
            // right-hand side. The two agree whenever x0 is zero and disagree badly when it is
            // not -- and the caller here reuses its correction vector across Newton steps, so
            // x0 is the previous step's correction. Once the Jacobian is exact the residual
            // collapses by orders of magnitude between steps while that stale x0 does not, and
            // the old test fired before a single iteration: on the Re=100 cavity at 470,596
            // dof it reported "diverged after 0 iterations" on a right-hand side of 4.2e-09
            // that it had not touched, and the continuation abandoned a converged stage.
            T beta0 = T(-1);

            while (iterations_ < max_it_) {
                // r = b - A x. Operator::apply accumulates here, so the target is cleared
                // first; this is the convention a stationary smoother also relies on.
                blas_->zeros((size_t)n, r);
                op_->apply(x, r);
                blas_->axpby(n, T(1), b, T(-1), r);

                const T beta = blas_->norm2(n, r);

                // Divergence is tested here, on the true residual, and not on the Arnoldi
                // estimate below: within a restart cycle GMRES minimises over the Krylov
                // space, so the estimate is monotonically non-increasing by construction and
                // can never report growth. Growth is only visible across restarts -- and with
                // a flexible (varying) preconditioner the estimate can drift from the true
                // residual in any case, so the recomputed one is the honest measure.
                if (beta0 < T(0)) beta0 = beta > T(0) ? beta : bnorm;
                if (!std::isfinite(beta) || (dtol_ > T(0) && beta > dtol_ * beta0)) {
                    if (x_best) blas_->copy(n, x_best, x);
                    diverged_ = true;
                    break;
                }
                if (dtol_ > T(0) && beta < beta_best) {
                    beta_best = beta;
                    x_best    = vec(work_, 2);
                    blas_->copy(n, x, x_best);
                }

                if (beta / bnorm < rtol_ || beta < atol_) break;

                blas_->zaxpby(n, T(1) / beta, r, T(0), r, vec(V_, 0));
                H.clear();
                std::fill(g.begin(), g.end(), T(0));
                g[0] = beta;

                int j = 0;
                for (; j < m && iterations_ < max_it_; ++j) {
                    const T *const vj = V_[(size_t)j];
                    // The flexible step: keep this application's preconditioned vector,
                    // because the next application may not be the same operator.
                    T *const zj = vec(Z_, (size_t)j);
                    if (prec_) {
                        blas_->zeros((size_t)n, zj);
                        prec_->apply(vj, zj);
                    } else {
                        blas_->copy(n, vj, zj);
                    }

                    blas_->zeros((size_t)n, w);
                    op_->apply(zj, w);
                    ++iterations_;

                    // Modified Gram-Schmidt, as before; each projection is one parallel dot
                    // and one parallel axpy over the full vector.
                    std::vector<T> h((size_t)j + 2, T(0));
                    for (int i = 0; i <= j; ++i) {
                        const T *const vi = V_[(size_t)i];
                        const T        d  = blas_->dot(n, w, vi);
                        h[(size_t)i]      = d;
                        blas_->axpy(n, -d, vi, w);
                    }
                    const T hn       = blas_->norm2(n, w);
                    h[(size_t)j + 1] = hn;

                    T *const vnext = vec(V_, (size_t)j + 1);
                    if (hn > T(1e-300))
                        blas_->zaxpby(n, T(1) / hn, w, T(0), w, vnext);
                    else
                        blas_->zeros((size_t)n, vnext);

                    for (int i = 0; i < j; ++i) {
                        const T t        = cs[(size_t)i] * h[(size_t)i] + sn[(size_t)i] * h[(size_t)i + 1];
                        h[(size_t)i + 1] = -sn[(size_t)i] * h[(size_t)i] + cs[(size_t)i] * h[(size_t)i + 1];
                        h[(size_t)i]     = t;
                    }
                    const T d = std::sqrt(h[(size_t)j] * h[(size_t)j] + h[(size_t)j + 1] * h[(size_t)j + 1]);
                    cs[(size_t)j] = (d > T(0)) ? h[(size_t)j] / d : T(1);
                    sn[(size_t)j] = (d > T(0)) ? h[(size_t)j + 1] / d : T(0);
                    h[(size_t)j]     = d;
                    h[(size_t)j + 1] = T(0);
                    g[(size_t)j + 1] = -sn[(size_t)j] * g[(size_t)j];
                    g[(size_t)j]     = cs[(size_t)j] * g[(size_t)j];
                    H.push_back(std::move(h));

                    const T resid = std::fabs(g[(size_t)j + 1]);
                    // Print interval, settable. On a very large problem each iteration is
                    // minutes of work, so a fixed stride of 50 means a job can run its whole
                    // wall-clock allocation without emitting a single convergence line.
                    static const int print_every =
                            std::max(1, smesh::Env::read<int>("SFEM_LIN_PRINT_EVERY", 50));
                    if (verbose && (iterations_ % print_every == 0))
                        std::printf("%d: residual abs: %g, rel: %g\n", iterations_, (double)resid,
                                    (double)(resid / bnorm));

                    if (!std::isfinite(resid)) {
                        // Arnoldi breakdown: the least-squares solve below would be garbage.
                        diverged_ = true;
                        break;
                    }

                    if (resid / bnorm < rtol_ || resid < atol_) {
                        ++j;
                        break;
                    }
                }

                if (diverged_) {
                    if (x_best) blas_->copy(n, x_best, x);
                    break;
                }

                // Back-substitute the least-squares problem and form the update from the
                // stored preconditioned vectors.
                std::vector<T> y((size_t)j, T(0));
                for (int i = j - 1; i >= 0; --i) {
                    T s = g[(size_t)i];
                    for (int k = i + 1; k < j; ++k) s -= H[(size_t)k][(size_t)i] * y[(size_t)k];
                    y[(size_t)i] = (H[(size_t)i][(size_t)i] != T(0)) ? s / H[(size_t)i][(size_t)i] : T(0);
                }
                for (int i = 0; i < j; ++i) blas_->axpy(n, y[(size_t)i], Z_[(size_t)i], x);

                if (j < m) break;  // inner loop converged rather than exhausting the restart
            }
            return diverged_ ? 1 : 0;
        }

    private:
        // Slot k of a vector pool, allocated on first use and kept. A solve that converges in
        // fewer iterations than the restart never touches the columns it does not reach, and
        // the pages of the ones it does are first written by the parallel BLAS loops.
        T *vec(std::vector<T *> &pool, const size_t k) {
            if (pool.size() <= k) pool.resize(k + 1, nullptr);
            if (!pool[k]) pool[k] = blas_->allocate((size_t)n_);
            return pool[k];
        }

        void release() {
            for (auto *pool : {&V_, &Z_, &work_}) {
                for (T *p : *pool)
                    if (p) blas_->destroy(p);
                pool->clear();
            }
        }

        std::shared_ptr<Operator<T>> op_, prec_;
        std::shared_ptr<BLAS<T>>     blas_;
        std::vector<T *>             V_, Z_;
        std::vector<T *>             work_;  // r, w, x_best
        ptrdiff_t                    n_{0};
        int                          max_it_{1000};
        int                          restart_{30};
        int                          iterations_{0};
        T                            rtol_{1e-8};
        T                            atol_{1e-14};
        T                            dtol_{0};
        bool                         diverged_{false};
    };

}  // namespace sfem
