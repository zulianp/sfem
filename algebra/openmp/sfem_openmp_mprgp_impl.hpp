#ifndef SFEM_MPRGP_IMPL_HPP
#define SFEM_MPRGP_IMPL_HPP

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>

#include "sfem_Team.hpp"
#include "sfem_fixed_chunks.hpp"
#include "sfem_mprgp_bounds.hpp"
#include "sfem_mprgp_control.hpp"
#include "sfem_tpl_blas.hpp"

namespace sfem {

    /// Completes a pass's reductions over the ranks in the execution space, in place on `count`
    /// values: all summed, or with `min_last` the last the minimum; on the device ordered on
    /// `stream`, the one the pass launched on (null on the host, or the legacy default stream).
    /// Empty where the sums are completed afterwards, from the record, or there is one rank.
    template <typename T>
    using MPRGPReduce = std::function<void(T* const values, const int count, const bool min_last, void* const stream)>;

    /// A run of MPRGP's CG steps (MPRGP::apply's CG branch and the test after it), taken where the
    /// passes run: the operands, the operator's product (Ap <- A p, written whole), and the
    /// constants of the decisions and of the stopping test.
    template <typename T>
    struct MPRGPRun {
        ptrdiff_t      n{0};
        const T       *lb{nullptr}, *ub{nullptr};
        T             *x{nullptr}, *g{nullptr}, *p{nullptr}, *Ap{nullptr}, *gf{nullptr};
        T              eps{0};
        const T       *w{nullptr};
        T              infty{0};
        MPRGPParams<T> params{};
        T             *control{nullptr}, *work{nullptr};
        T              gamma{1}, atol{0}, rtol{0}, norm_gp0{1};
        int            max_it{0};
        std::function<void(const T *const, T *const)> product;
        /// Across the ranks of one node: the passes' sums completed on the device (empty on one
        /// rank), on the run's stream, so the run needs no host across the ranks either.
        MPRGPReduce<T> reduce;
        /// The solver's stream (its context's): the run is captured from it and launched on it,
        /// so it is ordered with the solver's other work.
        void* stream{nullptr};
    };

    /// How a run ended: before a proportioning step (the chopped gradient grew), at a CG decision
    /// that is not a step (its record and A p stand, for the expansion), or with the solve over
    /// (converged, or max_it). `ran` false: the run could not be taken there, and nothing was done.
    struct MPRGPRunResult {
        enum Exit { PROPORTIONING = 1, NOT_A_STEP = 2, DONE = 3 };
        bool ran{false};
        Exit exit{DONE};
        int  iterations{0};
        int  steps{0};
        bool converged{false};
    };

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

        /// The CG steps of a run, from `iterations` done, without returning to the host between
        /// them; null where the passes have no such loop (on the host the operator's Team runs
        /// them, MPRGP::team_cg_steps). `state` is the loop's, kept between the runs of a solve
        /// with the same operands, and reset for another.
        std::function<MPRGPRunResult(const MPRGPRun<T> &run, int iterations, std::shared_ptr<void> &state)> cg_run;

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

        // The passes MPRGP's iteration in one region and its pass-by-pass route share are out of
        // line (SFEM_NOINLINE): both execute one body, so they agree to the bit.
        /// The curvature pass's sums over the entries [begin, end), added to s: d.Ad, d.d, g.d and,
        /// the last, the longest feasible step x - alpha d (a minimum, from infty).
        template <typename B>
        SFEM_NOINLINE static void curvature_sums(const B              b,
                                   const ptrdiff_t      begin,
                                   const ptrdiff_t      end,
                                   const T* const       x,
                                   const T* const       g,
                                   const T* const       d,
                                   const T* const       Ad,
                                   const T              infty,
                                   T (&s)[4]) {
            T dAd = s[0], dd = s[1], gd = s[2], alpha = s[3];
            for (ptrdiff_t i = begin; i < end; i++) {
                const T di = d[i];
                dAd += di * Ad[i];
                dd += di * di;
                gd += g[i] * di;
                alpha = b.min(alpha, b.max_alpha(i, x[i], di, infty));
            }
            s[0] = dAd, s[1] = dd, s[2] = gd, s[3] = alpha;
        }

        /// The step's update of x, g and gf over the entries [begin, end) (along d, unless null) and
        /// its sums, added to s: Ad.gf (without d none), |gf|^2, |gc|^2, |w (gf + gc)|^2.
        template <typename B>
        SFEM_NOINLINE static void step_sums(const B              b,
                              const ptrdiff_t      begin,
                              const ptrdiff_t      end,
                              const T* const       d,
                              const T* const       Ad,
                              const T              alpha,
                              T* const             x,
                              T* const             g,
                              T* const             gf,
                              const T              eps,
                              const T* const       w,
                              T (&s)[4]) {
            if (d) {
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
            } else {
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
            }
        }

        /// p <- gf - beta p over the entries [begin, end), as the BLAS's axpby(1, gf, -beta, p).
        SFEM_NOINLINE static void direction(const ptrdiff_t begin, const ptrdiff_t end, const T* const gf, const T beta, T* const p) {
            const T alpha = 1, scale = -beta;
            for (ptrdiff_t i = begin; i < end; i++) {
                p[i] = alpha * gf[i] + scale * p[i];
            }
        }

        /// The calling thread's chunks of `team`, each pass(b, e, v) from `init` into its four
        /// entries of `part`.
        template <typename F>
        static void team_sums(const Team& team, const T (&init)[4], T* const part, F pass) {
            for (ptrdiff_t c = team.first_chunk; c < team.end_chunk; c++) {
                ptrdiff_t b, e;
                team.chunk_entries(c, &b, &e);
                T v[4] = {init[0], init[1], init[2], init[3]};
                pass(b, e, v);
                for (int k = 0; k < 4; k++) part[4 * c + k] = v[k];
            }
        }

        /// The chunks' sums in `part` combined in index order from `init`: summed, with `min_last`
        /// the last the minimum.
        static void combine(const ptrdiff_t chunks, const T* const part, const T (&init)[4], const bool min_last, T (&out)[4]) {
            for (int k = 0; k < 4; k++) out[k] = init[k];
            for (ptrdiff_t c = 0; c < chunks; c++) {
                for (int k = 0; k < 3; k++) out[k] += part[4 * c + k];
                const T last = part[4 * c + 3];
                out[3]       = min_last ? (last < out[3] ? last : out[3]) : out[3] + last;
            }
        }

        /// A pass's four sums over the chunks of n a Team splits (sfem_Team.hpp), in a region of
        /// its own, then combined: the result does not depend on the number of threads, and is the
        /// one a region of the caller's takes (MPRGP's iteration in one region).
        template <typename F>
        static void chunked(const ptrdiff_t n, const T (&init)[4], const bool min_last, T* const part, T (&out)[4], F pass) {
            assert(part);
            if (team_chunk_count(n) == 1) {
                team_sums(Team::of(n, 0, 1), init, part, pass);
            } else {
#pragma omp parallel
                team_sums(Team::current(n), init, part, pass);
            }
            combine(team_chunk_count(n), part, init, min_last, out);
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
                    curvature_sums(b, begin, end, x, g, d, Ad, infty, s);
                });
                T dAd = v[0], dd = v[1], gd = v[2], alpha = v[3];
                if (reduce) {
                    // The three sums and the step bound in one message.
                    T v[4] = {dAd, dd, gd, alpha};
                    reduce(v, 4, true, nullptr);
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
                chunked(n, {0, 0, 0, 0}, false, work, v, [&](const ptrdiff_t begin, const ptrdiff_t end, T (&s)[4]) {
                    step_sums(b, begin, end, d, Ad, alpha, x, g, gf, eps, w, s);
                });
                T Adgf = v[0], gf2 = v[1], gc2 = v[2], gp2 = v[3];
                if (reduce) {
                    T v[4] = {Adgf, gf2, gc2, gp2};
                    reduce(v, 4, false, nullptr);
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
            // The BLAS's axpby(1, gf, -beta, p), expression and all, as every other update of p: the
            // same rounding as theirs. Thread by thread over a Team's shares, as the iteration in
            // one region takes it.
            tpl.update_direction = [](const ptrdiff_t n, const T* const gf, T* const p, const T* const control) {
                const MPRGPControl<T>& c = *reinterpret_cast<const MPRGPControl<T>*>(control);
                if (c.move != T(MPRGP_MOVE_STEP)) return;
#pragma omp parallel
                {
                    const Team team = Team::current(n);
                    direction(team.begin, team.end, gf, c.beta, p);
                }
            };
            tpl.read_control     = read_control;
            tpl.write_control    = write_control;
            // The passes' sums, four per chunk, twice: the curvature's and the step's, which the
            // iteration in one region keeps apart (MPRGP).
            tpl.work_size        = 8 * blas_max_chunk_count;
        }
    };

}  // namespace sfem

#endif  // SFEM_MPRGP_IMPL_HPP
