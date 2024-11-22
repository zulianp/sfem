#ifndef SFEM_ShiftedPenalty_IMPL_HPP
#define SFEM_ShiftedPenalty_IMPL_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <cstddef>

namespace sfem {

    template <typename T>
    struct ShiftedPenalty_Tpl {
        std::function<T(const ptrdiff_t, const T* const, T* const)> sq_norm_ramp_p;
        std::function<T(const ptrdiff_t, const T* const, T* const)> sq_norm_ramp_m;
        std::function<void(const ptrdiff_t n, const T, const T* const, const T* const,
                           const T* const, T* const)>
                ramp_p;

        std::function<void(const ptrdiff_t n, const T, const T* const, const T* const,
                           const T* const, T* const)>
                ramp_m;

        std::function<void(const ptrdiff_t, const T, const T* const, const T* const, T* const)>
                update_lagr_p;

        std::function<void(const ptrdiff_t, const T, const T* const, const T* const, T* const)>
                update_lagr_m;

        std::function<void(const ptrdiff_t, T* const, const T, const T* const, const T* const,
                       const T* const, const T* const, T* result)>
                calc_r_pen;

        std::function<void(const ptrdiff_t, const T* const, const T, const T* const, const T* const,
                           const T* const, const T* const, T* const)>
                calc_J_pen;

        bool good() const {
            return ramp_p && ramp_m && update_lagr_p && update_lagr_m && calc_r_pen && calc_J_pen;
        }
    };

    template <typename T>
    struct OpenMP_ShiftedPenalty {
        static void build_shifted_penalty(struct ShiftedPenalty_Tpl<T>& tpl) {
            tpl.sq_norm_ramp_p = [](const ptrdiff_t n, const T* const x, T* const ub) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T diff = std::max(T(0), x[i] - ub[i]);
                    ret += diff * diff;
                }
                return ret;
            };

            tpl.sq_norm_ramp_m = [](const ptrdiff_t n, const T* const x, T* const lb) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T diff = std::min(T(0), x[i] - lb[i]);
                    ret += diff * diff;
                }
                return ret;
            };

            // Adds to negative gradient (i.e., residual)
            tpl.ramp_p = [](const ptrdiff_t n,
                            const T penalty_param,
                            const T* const x,
                            const T* const ub,
                            const T* const lagr_ub,
                            T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    out[i] -= penalty_param *
                              std::max(T(0), x[i] - ub[i] + lagr_ub[i] / penalty_param);
                }
            };

            tpl.ramp_m = [](const ptrdiff_t n,
                            const T penalty_param,
                            const T* const x,
                            const T* const lb,
                            const T* const lagr_lb,
                            T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    out[i] -= penalty_param *
                              std::min(T(0), x[i] - lb[i] + lagr_lb[i] / penalty_param);
                }
            };

            tpl.update_lagr_p = [](const ptrdiff_t n,
                                   const T penalty_param,
                                   const T* const x,
                                   const T* const ub,
                                   T* const lagr_ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    lagr_ub[i] = std::max(T(0), lagr_ub[i] + penalty_param * (x[i] - ub[i]));
                }
            };

            tpl.update_lagr_m = [](const ptrdiff_t n,
                                   const T penalty_param,
                                   const T* const x,
                                   const T* const lb,
                                   T* const lagr_lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    lagr_lb[i] = std::min(T(0), lagr_lb[i] + penalty_param * (x[i] - lb[i]));
                }
            };

            auto ramp_m = tpl.ramp_m;
            auto ramp_p = tpl.ramp_p;
            tpl.calc_r_pen = [ramp_m, ramp_p](const ptrdiff_t n,
                                              T* const x,
                                              const T penalty_param,
                                              const T* const lb,
                                              const T* const ub,
                                              const T* const lagr_lb,
                                              const T* const lagr_ub,
                                              T* result) {
                // Ramp negative and positive parts
                if (lb) ramp_m(n, penalty_param, x, lb, lagr_lb, result);
                if (ub) ramp_p(n, penalty_param, x, ub, lagr_ub, result);
            };

            tpl.calc_J_pen = [](const ptrdiff_t n,
                                const T* const x,
                                const T penalty_param,
                                const T* const lb,
                                const T* const ub,
                                const T* const lagr_lb,
                                const T* const lagr_ub,
                                T* const result) {
                if (lb) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        result[i] +=
                                ((x[i] - lb[i] + lagr_lb[i] / penalty_param) <= 0) * penalty_param;
                    }
                }

                if (ub) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        result[i] +=
                                ((x[i] - ub[i] + lagr_ub[i] / penalty_param) >= 0) * penalty_param;
                    }
                }
            };
        }
    };

}  // namespace sfem

#endif  // SFEM_ShiftedPenalty_IMPL_HPP
