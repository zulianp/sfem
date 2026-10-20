#ifndef SFEM_MPRGP_IMPL_HPP
#define SFEM_MPRGP_IMPL_HPP

#include <cassert>
#include <cstddef>
#include <functional>

#include "sfem_mprgp_bounds.hpp"

namespace sfem {

    /// The vector passes of an MPRGP iteration, each one sweep over the dofs. lb and ub may be
    /// null, not both. Reductions arrive in `out` on the host, over this rank's dofs; `work` is a
    /// buffer of work_size() entries in the execution space.
    template <typename T>
    struct MPRGP_Tpl {
        /// x <- P(x)
        std::function<void(const ptrdiff_t n, const T* const lb, const T* const ub, T* const x)> project;

        /// gc <- the chopped gradient at (x, g)
        std::function<void(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T* const  x,
                           const T* const  g,
                           T* const        gc,
                           const T         eps)>
                chopped_gradient;

        /// Along d, with Ad = A d: out = {d.Ad, d.d, g.d, the longest feasible step x - alpha d
        /// (infty when no bound limits it)}.
        std::function<void(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T* const  x,
                           const T* const  g,
                           const T* const  d,
                           const T* const  Ad,
                           const T         infty,
                           T* const        out,
                           T* const        work)>
                curvature;

        /// x -= alpha d and g -= alpha Ad (neither when d is null), then gf <- the free gradient at
        /// (x, g); out = {Ad.gf (0 without d), |gf|^2, |gc|^2, |gf + gc|^2}, gc the chopped gradient.
        std::function<void(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T         alpha,
                           const T* const  d,
                           const T* const  Ad,
                           T* const        x,
                           T* const        g,
                           T* const        gf,
                           const T         eps,
                           T* const        out,
                           T* const        work)>
                step;

        /// The expansion step: x <- P(x - alpha d - alpha_bar gf), gf the free gradient at
        /// (x - alpha d, g - alpha Ad). g is left as it was.
        std::function<void(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T         alpha,
                           const T         alpha_bar,
                           const T* const  d,
                           const T* const  Ad,
                           T* const        x,
                           const T* const  g)>
                expand;

        /// Entries of the work buffer curvature and step take.
        ptrdiff_t work_size{0};

        bool good() const {
            assert(project);
            assert(chopped_gradient);
            assert(curvature);
            assert(step);
            assert(expand);
            return project && chopped_gradient && curvature && step && expand;
        }
    };

    template <typename T>
    struct OpenMP_MPRGP {
        static void project(const ptrdiff_t n, const T* const lb, const T* const ub, T* const x) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] = b.project(i, x[i]);
                }
            });
        }

        static void chopped_gradient(const ptrdiff_t n,
                                     const T* const  lb,
                                     const T* const  ub,
                                     const T* const  x,
                                     const T* const  g,
                                     T* const        gc,
                                     const T         eps) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    gc[i] = b.chopped(i, x[i], g[i], eps);
                }
            });
        }

        static void curvature(const ptrdiff_t n,
                              const T* const  lb,
                              const T* const  ub,
                              const T* const  x,
                              const T* const  g,
                              const T* const  d,
                              const T* const  Ad,
                              const T         infty,
                              T* const        out,
                              T* const) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                T dAd = 0, dd = 0, gd = 0, alpha = infty;
#pragma omp parallel for reduction(+ : dAd, dd, gd) reduction(min : alpha)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T di = d[i];
                    dAd += di * Ad[i];
                    dd += di * di;
                    gd += g[i] * di;
                    alpha = b.min(alpha, b.max_alpha(i, x[i], di, infty));
                }
                out[0] = dAd;
                out[1] = dd;
                out[2] = gd;
                out[3] = alpha;
            });
        }

        static void step(const ptrdiff_t n,
                         const T* const  lb,
                         const T* const  ub,
                         const T         alpha,
                         const T* const  d,
                         const T* const  Ad,
                         T* const        x,
                         T* const        g,
                         T* const        gf,
                         const T         eps,
                         T* const        out,
                         T* const) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                T Adgf = 0, gf2 = 0, gc2 = 0, gp2 = 0;
                if (d) {
#pragma omp parallel for reduction(+ : Adgf, gf2, gc2, gp2)
                    for (ptrdiff_t i = 0; i < n; i++) {
                        const T xi = x[i] - alpha * d[i];
                        const T gi = g[i] - alpha * Ad[i];
                        const T f  = b.free(i, xi, gi);
                        const T c  = b.chopped(i, xi, gi, eps);
                        x[i]       = xi;
                        g[i]       = gi;
                        gf[i]      = f;
                        Adgf += Ad[i] * f;
                        gf2 += f * f;
                        gc2 += c * c;
                        gp2 += (f + c) * (f + c);
                    }
                } else {
#pragma omp parallel for reduction(+ : gf2, gc2, gp2)
                    for (ptrdiff_t i = 0; i < n; i++) {
                        const T f = b.free(i, x[i], g[i]);
                        const T c = b.chopped(i, x[i], g[i], eps);
                        gf[i]     = f;
                        gf2 += f * f;
                        gc2 += c * c;
                        gp2 += (f + c) * (f + c);
                    }
                }
                out[0] = Adgf;
                out[1] = gf2;
                out[2] = gc2;
                out[3] = gp2;
            });
        }

        static void expand(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T         alpha,
                           const T         alpha_bar,
                           const T* const  d,
                           const T* const  Ad,
                           T* const        x,
                           const T* const  g) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T xi = x[i] - alpha * d[i];
                    const T gi = g[i] - alpha * Ad[i];
                    x[i]       = b.project(i, xi - alpha_bar * b.free(i, xi, gi));
                }
            });
        }

        static void build_mprgp(struct MPRGP_Tpl<T>& tpl) {
            tpl.project          = project;
            tpl.chopped_gradient = chopped_gradient;
            tpl.curvature        = curvature;
            tpl.step             = step;
            tpl.expand           = expand;
            tpl.work_size        = 0;
        }
    };

}  // namespace sfem

#endif  // SFEM_MPRGP_IMPL_HPP
