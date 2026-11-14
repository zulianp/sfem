#ifndef SFEM_OPENMP_CG_IMPL_HPP
#define SFEM_OPENMP_CG_IMPL_HPP

#include <cassert>
#include <cstddef>
#include <functional>

#include "sfem_base.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    /// Backend strategy for fused CG vector updates (OpenMP / CUDA).
    /// A preconditioned CG's iterations in the execution space (ConjugateGradientKernels::run_precond):
    /// ConjugateGradient::run_precond's loop, from its first z = M r on, with the scalars (r.z, p.Ap,
    /// alpha, beta) kept where the vectors are, so the host does not wait for each dot. The same
    /// operations in the same order: x is the loop's, bit for bit.
    template <typename T>
    struct CGDeviceRun {
        ptrdiff_t n{0};
        T        *x{nullptr}, *r{nullptr}, *z{nullptr}, *p{nullptr}, *Ap{nullptr};
        T         rtz{0}, rtz0{0}, atol{0}, rtol{0};
        int       max_it{0};
        /// z = M r and Ap = A p, as the loop applies them (an accumulating operator's output zeroed).
        std::function<void(const T *const, T *const)> precondition, product;
        /// a . b into the word `result` where the vectors are, as the loop's dots compute it.
        std::function<void(const T *const, const T *const, T *const)> dot;
        /// The solver's stream: the run is ordered on it.
        void *stream{nullptr};
    };

    /// How a run ended. `ran` false: it could not be taken there, and nothing was done.
    struct CGDeviceRunResult {
        bool ran{false};
        int  iterations{0};
        bool converged{false};
    };

    template <typename T>
    struct ConjugateGradientKernels {
        /// x ← x + alpha p;  r ← r − alpha Ap;  return (r, r)
        std::function<T(const ptrdiff_t n,
                        const T         alpha,
                        const T* const  p,
                        const T* const  Ap,
                        T* const        x,
                        T* const        r)>
                update_x_r_and_rtr;

        /// x ← x + alpha p;  r ← r − alpha Ap  (no residual reduction / no scalar return)
        std::function<void(const ptrdiff_t n,
                           const T         alpha,
                           const T* const  p,
                           const T* const  Ap,
                           T* const        x,
                           T* const        r)>
                update_x_r;

        /// p ← z + beta p
        std::function<void(const ptrdiff_t n, const T beta, const T* const z, T* const p)> update_p;

        /// A backend's own run of the preconditioned loop (CGDeviceRun); empty where it has none.
        std::function<CGDeviceRunResult(const CGDeviceRun<T>& run)> run_precond;

        bool good() const {
            assert(update_x_r_and_rtr);
            assert(update_x_r);
            assert(update_p);
            return update_x_r_and_rtr && update_x_r && update_p;
        }
    };

    template <typename T>
    struct OpenMPConjugateGradientKernels {
        static T update_x_r_and_rtr(const ptrdiff_t n,
                                    const T         alpha,
                                    const T* const  p,
                                    const T* const  Ap,
                                    T* const        x,
                                    T* const        r) {
            T rtr = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : rtr)
#endif
            for (ptrdiff_t i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                const T ri = r[i] - alpha * Ap[i];
                r[i]       = ri;
                rtr += ri * ri;
            }
            return rtr;
        }

        static void update_x_r(const ptrdiff_t n,
                               const T         alpha,
                               const T* const  p,
                               const T* const  Ap,
                               T* const        x,
                               T* const        r) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
        }

        static void update_p(const ptrdiff_t n, const T beta, const T* const z, T* const p) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }
        }

        static void build(struct ConjugateGradientKernels<T>& tpl) {
            tpl.update_x_r_and_rtr = update_x_r_and_rtr;
            tpl.update_x_r         = update_x_r;
            tpl.update_p           = update_p;
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_CG_IMPL_HPP
