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

#include "cvfem_parallel.hpp"

#include "sfem_Operator.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_openmp_blas.hpp"
#include "smesh_env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace sfem {

    template <typename T>
    class FGMRES final {
    public:
        // The communicator comes from the operator rather than from the caller.
        //
        // Multigrid already asks its operators the same question the same way
        // (sfem_Multigrid.hpp: reduce_norm2, level_owned, verbose_rank0), so this reuses that
        // convention instead of adding a second way for a solver to learn it is distributed.
        // An operator with no parallel face leaves the domain empty, every reduction below
        // short-circuits, and the serial path is untouched.
        //
        // rows() is the OWNED dof count and col_allocation_size() the local one, which is the
        // distinction the workspace sizing below depends on.
        explicit FGMRES(const std::shared_ptr<Operator<T>> &op) : op_(op), blas_(make_openmp_blas<T>()) {
            if (auto pop = std::dynamic_pointer_cast<ParallelOperator<T>>(op_)) {
                domain_.comm    = pop->comm();
                domain_.n_owned = pop->rows();
                domain_.n_local = pop->col_allocation_size();
            }
        }
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

        // The two reductions the Krylov method needs, collective when there is anything to
        // reduce with.
        //
        // norm2 is NOT reduced directly: it returns the square root of a sum of squares, so
        // summing norms across ranks would be wrong. The square is reduced and the root taken
        // afterwards.
        //
        // Both fall through to the original BLAS call when not distributed -- the same call,
        // not an equivalent one -- so a serial solve produces the identical bits it did
        // before, which the byte-compared verification matrix requires.
        T pnorm2(const ptrdiff_t n, const T *const v) {
            if (!domain_.distributed()) return blas_->norm2(n, v);
            return std::sqrt(cvfem::sum(domain_, blas_->dot(n, v, v)));
        }

        T pdot(const ptrdiff_t n, const T *const a, const T *const c) {
            const T d = blas_->dot(n, a, c);
            return domain_.distributed() ? cvfem::sum(domain_, d) : d;
        }

        int apply(const T *const b, T *const x) {
            const ptrdiff_t n = op_->rows();
            const int       m = restart_;
            if (n != n_) {
                release();
                n_ = n;
            }
            // Arithmetic runs to n = rows(), the OWNED range, but the vectors have to be
            // ALLOCATED larger: a distributed matrix-free apply writes the ghost and aura
            // slots of its input, past the owned range it reads. Sizing the basis by rows()
            // alone hands the operator a buffer it runs off the end of.
            alloc_ = domain_.distributed() ? std::max(domain_.n_owned, domain_.n_local) : n;

            // SFEM_FGMRES_DIAG=1: what this solver believes about its own parallelism, and
            // whether the Krylov basis it builds is actually orthogonal.
            //
            // Every reduction here routes through domain_, and domain_ is populated only if
            // the constructor's dynamic_pointer_cast to ParallelOperator succeeded. If an
            // operator reaches the solver with that face erased, the cast fails silently,
            // distributed() is false, and every dot and norm below is rank-local while the
            // source still reads as collective. The symptom is not a wrong answer but a
            // degraded iteration count, which is indistinguishable by eye from a weaker
            // preconditioner -- so it has to be reported rather than inferred.
            if (smesh::Env::read<int>("SFEM_FGMRES_DIAG", 0) && !diag_announced_) {
                diag_announced_ = true;
                const int rank  = domain_.rank();
                if (domain_.is_root())
                    std::printf("fgmres: distributed %d  size %d  n(owned) %td  n_local %td  alloc %td\n",
                                domain_.distributed() ? 1 : 0, domain_.size(), (ptrdiff_t)n,
                                domain_.n_local, alloc_);
                (void)rank;
            }

            T *const                    r = vec(work_, 0);
            T *const                    w = vec(work_, 1);
            std::vector<std::vector<T>> H;
            std::vector<T>              cs((size_t)m, 0), sn((size_t)m, 0), g((size_t)m + 1, 0);

            iterations_ = 0;
            diverged_   = false;
            orth_worst_ = T(0);

            // Best iterate seen, so a solve abandoned as divergent hands back the best point
            // it reached rather than the diverging one. GMRES only writes x at the end of a
            // restart cycle, so without this the caller receives precisely the worst iterate.
            // One extra vector against the 2*(restart+1) the Krylov basis already holds.
            T *x_best    = nullptr;
            T  beta_best = std::numeric_limits<T>::max();

            T bnorm = pnorm2(n, b);
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

                // Collective, and not only so the number printed is right: beta drives the
                // convergence test, the divergence test and the best-iterate copy below. A
                // per-rank beta means the ranks take DIFFERENT branches, and ranks that
                // continue enter a collective the ranks that stopped never reach -- a hang
                // rather than a wrong answer.
                const T beta = pnorm2(n, r);

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
                        // Without the reduction the basis is not orthogonal, and FGMRES then
                        // STAGNATES rather than failing: it keeps iterating against a basis
                        // that does not span what it thinks it spans.
                        const T        d  = pdot(n, w, vi);
                        h[(size_t)i]      = d;
                        blas_->axpy(n, -d, vi, w);
                    }
                    const T hn       = pnorm2(n, w);
                    h[(size_t)j + 1] = hn;

                    T *const vnext = vec(V_, (size_t)j + 1);
                    if (hn > T(1e-300))
                        blas_->zaxpby(n, T(1) / hn, w, T(0), w, vnext);
                    else
                        blas_->zeros((size_t)n, vnext);

                    // SFEM_FGMRES_DIAG=1: the orthogonality defect of the basis just
                    // extended, max_i |<v_i, v_{j+1}>| over the vectors already built.
                    //
                    // This is the measurement that separates a reduction which is absent
                    // from one which is merely wrong. The diagnostic above reports what the
                    // solver believes about its own parallelism; this reports whether that
                    // belief produced an orthogonal basis. Modified Gram-Schmidt has just
                    // projected v_{j+1} against every earlier vector, so on a correct
                    // reduction these inner products are at rounding level; a rank-local
                    // dot leaves them O(1) while the solve still converges, slowly, which
                    // is exactly the signature that reads as a weaker preconditioner.
                    if (smesh::Env::read<int>("SFEM_FGMRES_DIAG", 0)) {
                        for (int i = 0; i <= j; ++i) {
                            const T od = pdot(n, vnext, V_[(size_t)i]);
                            const T a  = od < T(0) ? -od : od;
                            if (a > orth_worst_) orth_worst_ = a;
                        }
                    }

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

            // Reported once per solve rather than per iteration: the worst it ever got is
            // the number that matters, and a per-iteration trace would bury it.
            if (smesh::Env::read<int>("SFEM_FGMRES_DIAG", 0) && domain_.is_root())
                std::printf("fgmres: orth_worst %.3e  its %d\n", (double)orth_worst_, iterations_);

            return diverged_ ? 1 : 0;
        }

    private:
        // Slot k of a vector pool, allocated on first use and kept. A solve that converges in
        // fewer iterations than the restart never touches the columns it does not reach, and
        // the pages of the ones it does are first written by the parallel BLAS loops.
        T *vec(std::vector<T *> &pool, const size_t k) {
            if (pool.size() <= k) pool.resize(k + 1, nullptr);
            // alloc_ rather than n_: see the note in apply(). Equal at one rank.
            if (!pool[k]) pool[k] = blas_->allocate((size_t)(alloc_ > n_ ? alloc_ : n_));
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
        cvfem::Domain                domain_;
        bool                         diag_announced_{false};
        T                            orth_worst_{0};
        ptrdiff_t                    n_{0};
        ptrdiff_t                    alloc_{0};
        int                          max_it_{1000};
        int                          restart_{30};
        int                          iterations_{0};
        T                            rtol_{1e-8};
        T                            atol_{1e-14};
        T                            dtol_{0};
        bool                         diverged_{false};
    };

}  // namespace sfem
