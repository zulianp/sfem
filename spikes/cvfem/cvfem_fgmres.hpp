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

#include "sfem_Operator.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace sfem {

    template <typename T>
    class FGMRES final {
    public:
        explicit FGMRES(const std::shared_ptr<Operator<T>> &op) : op_(op) {}

        void set_max_it(const int v) { max_it_ = v; }
        void set_rtol(const T v) { rtol_ = v; }
        void set_atol(const T v) { atol_ = v; }
        void set_restart(const int v) { restart_ = v; }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>> &p) { prec_ = p; }
        int  iterations() const { return iterations_; }

        bool verbose{true};

        int apply(const T *const b, T *const x) {
            const ptrdiff_t n = op_->rows();
            const int       m = restart_;

            std::vector<T>              r((size_t)n), w((size_t)n);
            std::vector<std::vector<T>> V, Z, H;
            std::vector<T>              cs((size_t)m, 0), sn((size_t)m, 0), g((size_t)m + 1, 0);

            iterations_ = 0;

            T bnorm = 0;
            for (ptrdiff_t i = 0; i < n; ++i) bnorm += b[i] * b[i];
            bnorm = std::sqrt(bnorm);
            if (bnorm == T(0)) bnorm = T(1);

            while (iterations_ < max_it_) {
                // r = b - A x. Operator::apply accumulates here, so the target is cleared
                // first; this is the convention a stationary smoother also relies on.
                std::fill(r.begin(), r.end(), T(0));
                op_->apply(x, r.data());
                for (ptrdiff_t i = 0; i < n; ++i) r[(size_t)i] = b[i] - r[(size_t)i];

                T beta = 0;
                for (ptrdiff_t i = 0; i < n; ++i) beta += r[(size_t)i] * r[(size_t)i];
                beta = std::sqrt(beta);

                if (beta / bnorm < rtol_ || beta < atol_) break;

                V.assign(1, r);
                for (ptrdiff_t i = 0; i < n; ++i) V[0][(size_t)i] /= beta;
                Z.clear();
                H.clear();
                std::fill(g.begin(), g.end(), T(0));
                g[0] = beta;

                int j = 0;
                for (; j < m && iterations_ < max_it_; ++j) {
                    // The flexible step: keep this application's preconditioned vector,
                    // because the next application may not be the same operator.
                    std::vector<T> z((size_t)n, T(0));
                    if (prec_)
                        prec_->apply(V[(size_t)j].data(), z.data());
                    else
                        z = V[(size_t)j];
                    Z.push_back(z);

                    std::fill(w.begin(), w.end(), T(0));
                    op_->apply(z.data(), w.data());
                    ++iterations_;

                    std::vector<T> h((size_t)j + 2, T(0));
                    for (int i = 0; i <= j; ++i) {
                        T d = 0;
                        for (ptrdiff_t k = 0; k < n; ++k) d += w[(size_t)k] * V[(size_t)i][(size_t)k];
                        h[(size_t)i] = d;
                        for (ptrdiff_t k = 0; k < n; ++k) w[(size_t)k] -= d * V[(size_t)i][(size_t)k];
                    }
                    T hn = 0;
                    for (ptrdiff_t k = 0; k < n; ++k) hn += w[(size_t)k] * w[(size_t)k];
                    hn                = std::sqrt(hn);
                    h[(size_t)j + 1]  = hn;

                    std::vector<T> vnext((size_t)n, T(0));
                    if (hn > T(1e-300))
                        for (ptrdiff_t k = 0; k < n; ++k) vnext[(size_t)k] = w[(size_t)k] / hn;
                    V.push_back(std::move(vnext));

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
                    if (verbose && (iterations_ % 50 == 0))
                        std::printf("%d: residual abs: %g, rel: %g\n", iterations_, (double)resid,
                                    (double)(resid / bnorm));

                    if (resid / bnorm < rtol_ || resid < atol_) {
                        ++j;
                        break;
                    }
                }

                // Back-substitute the least-squares problem and form the update from the
                // stored preconditioned vectors.
                std::vector<T> y((size_t)j, T(0));
                for (int i = j - 1; i >= 0; --i) {
                    T s = g[(size_t)i];
                    for (int k = i + 1; k < j; ++k) s -= H[(size_t)k][(size_t)i] * y[(size_t)k];
                    y[(size_t)i] = (H[(size_t)i][(size_t)i] != T(0)) ? s / H[(size_t)i][(size_t)i] : T(0);
                }
                for (int i = 0; i < j; ++i)
                    for (ptrdiff_t k = 0; k < n; ++k) x[k] += y[(size_t)i] * Z[(size_t)i][(size_t)k];

                if (j < m) break;  // inner loop converged rather than exhausting the restart
            }
            return 0;
        }

    private:
        std::shared_ptr<Operator<T>> op_, prec_;
        int                          max_it_{1000};
        int                          restart_{30};
        int                          iterations_{0};
        T                            rtol_{1e-8};
        T                            atol_{1e-14};
    };

}  // namespace sfem
