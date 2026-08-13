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
    template <typename T>
    struct CG_Tpl {
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

        bool good() const {
            assert(update_x_r_and_rtr);
            assert(update_x_r);
            assert(update_p);
            return update_x_r_and_rtr && update_x_r && update_p;
        }
    };

    template <typename T>
    struct OpenMP_CG {
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

        static void build(struct CG_Tpl<T>& tpl) {
            tpl.update_x_r_and_rtr = update_x_r_and_rtr;
            tpl.update_x_r         = update_x_r;
            tpl.update_p           = update_p;
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_CG_IMPL_HPP
