#ifndef SFEM_MPRGP_IMPL_HPP
#define SFEM_MPRGP_IMPL_HPP

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>

#include "sfem_fixed_chunks.hpp"
#include "sfem_mprgp_bounds.hpp"
#include "sfem_mprgp_control.hpp"
#include "sfem_tpl_blas.hpp"

namespace sfem {

    /// Completes a pass's reductions over the ranks in the execution space, in place on `count`
    /// values: all summed, or with `min_last` the last the minimum. Empty where the sums are
    /// completed afterwards, from the record, or there is one rank.
    template <typename T>
    using MPRGPReduce = std::function<void(T* const values, const int count, const bool min_last)>;

    /// The vector passes of an MPRGP iteration, each one sweep over the dofs. lb and ub may be
    /// null, not both. Reductions, over this rank's dofs, and the decisions taken from them arrive
    /// in `control`, an MPRGPControl of mprgp_control_size() entries in the execution space; the
    /// passes after them read their scalars from there. `work` is a buffer of work_size() entries
    /// in the execution space.
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

        /// Along d, with Ad = A d: the record's d.Ad, d.d, g.d and the longest feasible step
        /// x - alpha d (infty when no bound limits it), completed over the ranks by `reduce`; then
        /// the decision, which takes final sums (MPRGP_DECIDE_NONE where they are summed over the
        /// ranks afterwards).
        std::function<void(const ptrdiff_t           n,
                           const T* const            lb,
                           const T* const            ub,
                           const T* const            x,
                           const T* const            g,
                           const T* const            d,
                           const T* const            Ad,
                           const T                   infty,
                           const MPRGPDecision       decision,
                           const MPRGPParams<T>&     params,
                           T* const                  control,
                           T* const                  work,
                           const MPRGPReduce<T>&     reduce)>
                curvature;

        /// x -= alpha d and g -= alpha Ad with the record's alpha, only where its move is a step
        /// (always, with nothing to move, when d is null); then gf <- the free gradient at (x, g)
        /// and the record's Ad.gf (0 without d), |gf|^2, |gc|^2, |w (gf + gc)|^2, gc the chopped
        /// gradient, w the entrywise weight (1 where null), completed over the ranks by `reduce`.
        /// With `cg` (final sums) also the next direction's beta.
        std::function<void(const ptrdiff_t n,
                           const T* const  lb,
                           const T* const  ub,
                           const T* const  d,
                           const T* const  Ad,
                           T* const        x,
                           T* const        g,
                           T* const        gf,
                           const T         eps,
                           const T* const  w,
                           const bool      cg,
                           T* const        control,
                           T* const        work,
                           const MPRGPReduce<T>& reduce)>
                step;

        /// p <- gf - beta p with the record's beta, only where its move is a step.
        std::function<void(const ptrdiff_t n, const T* const gf, T* const p, const T* const control)> update_direction;

        /// The record to and from the host.
        std::function<void(const T* const control, MPRGPControl<T>* const host)>  read_control;
        std::function<void(const MPRGPControl<T>& host, T* const control)>         write_control;

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
            assert(update_direction);
            assert(read_control);
            assert(write_control);
            return project && chopped_gradient && curvature && step && expand && update_direction && read_control &&
                   write_control;
        }
    };

    // Defined in sfem_openmp_blas.hpp, which only an OpenMP translation unit may include; the table
    // below is built in one (sfem_mprgp.hpp), the CUDA backend only shares MPRGP_Tpl.
    template <typename T>
    std::shared_ptr<BLAS<T>> make_openmp_blas();

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

        static MPRGPControl<T>& record(T* const control) { return *reinterpret_cast<MPRGPControl<T>*>(control); }

        /// A pass's four partial sums over the chunks of n fixed by its length (sfem_fixed_chunks.hpp),
        /// pass(b, e, v) adding entries [b, e) to v, which starts from `init`; each chunk's into
        /// `part`, then combined in index order: summed, with `min_last` the last the minimum. The
        /// result does not depend on the number of threads.
        template <typename F>
        static void chunked(const ptrdiff_t n, const T (&init)[4], const bool min_last, T* const part, T (&out)[4], F pass) {
            const ptrdiff_t nc = blas_chunk_count(n);
            assert(part || nc == 1);
            for (int k = 0; k < 4; k++) out[k] = init[k];
            if (nc == 1) {
                pass(ptrdiff_t(0), n, out);
                return;
            }
#pragma omp parallel for schedule(static)
            for (ptrdiff_t c = 0; c < nc; c++) {
                ptrdiff_t b, e;
                blas_chunk_bounds(n, nc, c, &b, &e);
                T v[4] = {init[0], init[1], init[2], init[3]};
                pass(b, e, v);
                for (int k = 0; k < 4; k++) part[4 * c + k] = v[k];
            }
            for (ptrdiff_t c = 0; c < nc; c++) {
                for (int k = 0; k < 3; k++) out[k] += part[4 * c + k];
                const T last = part[4 * c + 3];
                out[3]       = min_last ? (last < out[3] ? last : out[3]) : out[3] + last;
            }
        }

        static void curvature(const ptrdiff_t       n,
                              const T* const        lb,
                              const T* const        ub,
                              const T* const        x,
                              const T* const        g,
                              const T* const        d,
                              const T* const        Ad,
                              const T               infty,
                              const MPRGPDecision   decision,
                              const MPRGPParams<T>& params,
                              T* const              control,
                              T* const              work,
                              const MPRGPReduce<T>& reduce) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                T v[4];
                chunked(n, {0, 0, 0, infty}, true, work, v, [&](const ptrdiff_t begin, const ptrdiff_t end, T (&s)[4]) {
                    T dAd = s[0], dd = s[1], gd = s[2], alpha = s[3];
                    for (ptrdiff_t i = begin; i < end; i++) {
                        const T di = d[i];
                        dAd += di * Ad[i];
                        dd += di * di;
                        gd += g[i] * di;
                        alpha = b.min(alpha, b.max_alpha(i, x[i], di, infty));
                    }
                    s[0] = dAd, s[1] = dd, s[2] = gd, s[3] = alpha;
                });
                T dAd = v[0], dd = v[1], gd = v[2], alpha = v[3];
                if (reduce) {
                    // The three sums and the step bound in one message.
                    T v[4] = {dAd, dd, gd, alpha};
                    reduce(v, 4, true);
                    dAd   = v[0];
                    dd    = v[1];
                    gd    = v[2];
                    alpha = v[3];
                }
                MPRGPControl<T>& c = record(control);
                c.dAd              = dAd;
                c.dd               = dd;
                c.gd               = gd;
                c.alpha_feas       = alpha;
                mprgp_decide(c, decision, params);
            });
        }

        static void step(const ptrdiff_t n,
                         const T* const  lb,
                         const T* const  ub,
                         const T* const  d,
                         const T* const  Ad,
                         T* const        x,
                         T* const        g,
                         T* const        gf,
                         const T         eps,
                         const T* const  w,
                         const bool      cg,
                         T* const        control,
                         T* const        work,
                         const MPRGPReduce<T>& reduce) {
            MPRGPControl<T>& c = record(control);
            if (d && c.move != T(MPRGP_MOVE_STEP)) return;
            const T alpha = c.alpha;
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                T v[4];
                if (d) {
                    chunked(n, {0, 0, 0, 0}, false, work, v, [&](const ptrdiff_t begin, const ptrdiff_t end, T (&s)[4]) {
                        T Adgf = s[0], gf2 = s[1], gc2 = s[2], gp2 = s[3];
                        for (ptrdiff_t i = begin; i < end; i++) {
                            const T xi = x[i] - alpha * d[i];
                            const T gi = g[i] - alpha * Ad[i];
                            const T f  = b.free(i, xi, gi);
                            const T c  = b.chopped(i, xi, gi, eps);
                            const T p  = w ? w[i] * (f + c) : f + c;
                            x[i]       = xi;
                            g[i]       = gi;
                            gf[i]      = f;
                            Adgf += Ad[i] * f;
                            gf2 += f * f;
                            gc2 += c * c;
                            gp2 += p * p;
                        }
                        s[0] = Adgf, s[1] = gf2, s[2] = gc2, s[3] = gp2;
                    });
                } else {
                    chunked(n, {0, 0, 0, 0}, false, work, v, [&](const ptrdiff_t begin, const ptrdiff_t end, T (&s)[4]) {
                        T gf2 = s[1], gc2 = s[2], gp2 = s[3];
                        for (ptrdiff_t i = begin; i < end; i++) {
                            const T f = b.free(i, x[i], g[i]);
                            const T c = b.chopped(i, x[i], g[i], eps);
                            const T p = w ? w[i] * (f + c) : f + c;
                            gf[i]     = f;
                            gf2 += f * f;
                            gc2 += c * c;
                            gp2 += p * p;
                        }
                        s[1] = gf2, s[2] = gc2, s[3] = gp2;
                    });
                }
                T Adgf = v[0], gf2 = v[1], gc2 = v[2], gp2 = v[3];
                if (reduce) {
                    T v[4] = {Adgf, gf2, gc2, gp2};
                    reduce(v, 4, false);
                    Adgf = v[0];
                    gf2  = v[1];
                    gc2  = v[2];
                    gp2  = v[3];
                }
                c.Adgf = Adgf;
                c.gf2  = gf2;
                c.gc2  = gc2;
                c.gp2  = gp2;
                if (cg) mprgp_after_cg_step(c);
            });
        }


        static void read_control(const T* const control, MPRGPControl<T>* const host) {
            std::memcpy(host, control, sizeof(MPRGPControl<T>));
        }

        static void write_control(const MPRGPControl<T>& host, T* const control) {
            std::memcpy(control, &host, sizeof(MPRGPControl<T>));
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
            // Through the BLAS, as every other update of p: the same rounding as theirs.
            tpl.update_direction = [blas = make_openmp_blas<T>()](const ptrdiff_t n, const T* const gf, T* const p, const T* const control) {
                const MPRGPControl<T>& c = *reinterpret_cast<const MPRGPControl<T>*>(control);
                if (c.move == T(MPRGP_MOVE_STEP)) blas->axpby(n, 1, gf, -c.beta, p);
            };
            tpl.read_control     = read_control;
            tpl.write_control    = write_control;
            // The passes' partial sums, four per chunk.
            tpl.work_size        = 4 * blas_max_chunk_count;
        }
    };

}  // namespace sfem

#endif  // SFEM_MPRGP_IMPL_HPP
