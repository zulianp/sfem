#ifndef SFEM_ShiftedPenalty_IMPL_HPP
#define SFEM_ShiftedPenalty_IMPL_HPP

#include "sfem_aliases.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>

namespace sfem {

    template <typename T>
    struct ShiftedPenaltyKernels {
        std::function<T(const ptrdiff_t, const T* const, T* const)>                                               sq_norm_ramp_p;
        std::function<T(const ptrdiff_t, const T* const, T* const)>                                               sq_norm_ramp_m;
        // The penalty of row i is penalty_param * weights[i] (weights null: penalty_param).
        std::function<void(const ptrdiff_t n, const T, const T* const, const T* const, const T* const, const T* const, T* const)>
                ramp_p;

        std::function<void(const ptrdiff_t n, const T, const T* const, const T* const, const T* const, const T* const, T* const)>
                ramp_m;

        std::function<void(const ptrdiff_t, const T, const T* const, const T* const, const T* const, T* const)> update_lagr_p;

        std::function<void(const ptrdiff_t, const T, const T* const, const T* const, const T* const, T* const)> update_lagr_m;

        std::function<void(const ptrdiff_t,
                           T* const,
                           const T,
                           const T* const,
                           const T* const,
                           const T* const,
                           const T* const,
                           const T* const,
                           T* result)>
                calc_r_pen;

        std::function<void(const ptrdiff_t,
                           const T* const,
                           const T,
                           const T* const,
                           const T* const,
                           const T* const,
                           const T* const,
                           const T* const,
                           T* const)>
                calc_J_pen;

        // The augmented-Lagrangian violation: sum_i max(x_i - ub_i, -lagr_ub_i / rho_i)^2 (and the
        // lower bound's mirror), rho_i = penalty_param * weights[i]: the penetration of a row without
        // force, the opening of a row that carries one.
        std::function<T(const ptrdiff_t, const T, const T* const, const T* const, const T* const, const T* const)> sq_norm_kkt_p;
        std::function<T(const ptrdiff_t, const T, const T* const, const T* const, const T* const, const T* const)> sq_norm_kkt_m;

        // out[i * block_size + k] = scale[i] * blocks[i * block_size + k]
        std::function<void(const ptrdiff_t, const int, const T* const, const T* const, T* const)> scale_blocks;

        bool good() const {
            return sq_norm_ramp_p && sq_norm_ramp_m && ramp_p && ramp_m && update_lagr_p && update_lagr_m && calc_r_pen &&
                   calc_J_pen && scale_blocks && sq_norm_kkt_p && sq_norm_kkt_m;
        }
    };

    template <typename T>
    struct OpenMPShiftedPenaltyKernels {
        static void build(struct ShiftedPenaltyKernels<T>& tpl) {
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
                            const T         penalty_param,
                            const T* const  weights,
                            const T* const  x,
                            const T* const  ub,
                            const T* const  lagr_ub,
                            T* const        out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    out[i] -= std::max(T(0), rho * (x[i] - ub[i]) + lagr_ub[i]);
                }
            };

            tpl.ramp_m = [](const ptrdiff_t n,
                            const T         penalty_param,
                            const T* const  weights,
                            const T* const  x,
                            const T* const  lb,
                            const T* const  lagr_lb,
                            T* const        out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    out[i] -= std::min(T(0), rho * (x[i] - lb[i]) + lagr_lb[i]);
                }
            };

            tpl.update_lagr_p = [](const ptrdiff_t n,
                                   const T         penalty_param,
                                   const T* const  weights,
                                   const T* const  x,
                                   const T* const  ub,
                                   T* const        lagr_ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    lagr_ub[i]  = std::max(T(0), lagr_ub[i] + rho * (x[i] - ub[i]));
                }
            };

            tpl.update_lagr_m = [](const ptrdiff_t n,
                                   const T         penalty_param,
                                   const T* const  weights,
                                   const T* const  x,
                                   const T* const  lb,
                                   T* const        lagr_lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    lagr_lb[i]  = std::min(T(0), lagr_lb[i] + rho * (x[i] - lb[i]));
                }
            };

            auto ramp_m    = tpl.ramp_m;
            auto ramp_p    = tpl.ramp_p;
            tpl.calc_r_pen = [ramp_m, ramp_p](const ptrdiff_t n,
                                              T* const        x,
                                              const T         penalty_param,
                                              const T* const  weights,
                                              const T* const  lb,
                                              const T* const  ub,
                                              const T* const  lagr_lb,
                                              const T* const  lagr_ub,
                                              T*              result) {
                SFEM_TRACE_SCOPE("OpenMPShiftedPenaltyKernels::calc_r_pen");

                // Ramp negative and positive parts
                if (lb) ramp_m(n, penalty_param, weights, x, lb, lagr_lb, result);
                if (ub) ramp_p(n, penalty_param, weights, x, ub, lagr_ub, result);
            };

            tpl.calc_J_pen = [](const ptrdiff_t n,
                                const T* const  x,
                                const T         penalty_param,
                                const T* const  weights,
                                const T* const  lb,
                                const T* const  ub,
                                const T* const  lagr_lb,
                                const T* const  lagr_ub,
                                T* const        result) {
                SFEM_TRACE_SCOPE("OpenMPShiftedPenaltyKernels::calc_J_pen");

                if (lb) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        const T rho = weights ? penalty_param * weights[i] : penalty_param;
                        result[i] += ((rho * (x[i] - lb[i]) + lagr_lb[i]) <= 0) * rho;
                    }
                }

                if (ub) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        const T rho = weights ? penalty_param * weights[i] : penalty_param;
                        result[i] += ((rho * (x[i] - ub[i]) + lagr_ub[i]) >= 0) * rho;
                    }
                }
            };

            tpl.sq_norm_kkt_p = [](const ptrdiff_t n,
                                   const T         penalty_param,
                                   const T* const  weights,
                                   const T* const  x,
                                   const T* const  ub,
                                   const T* const  lagr_ub) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    const T v   = std::max(x[i] - ub[i], -lagr_ub[i] / rho);
                    ret += v * v;
                }
                return ret;
            };

            tpl.sq_norm_kkt_m = [](const ptrdiff_t n,
                                   const T         penalty_param,
                                   const T* const  weights,
                                   const T* const  x,
                                   const T* const  lb,
                                   const T* const  lagr_lb) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T rho = weights ? penalty_param * weights[i] : penalty_param;
                    const T v   = std::min(x[i] - lb[i], -lagr_lb[i] / rho);
                    ret += v * v;
                }
                return ret;
            };

            tpl.scale_blocks = [](const ptrdiff_t n_blocks, const int block_size, const T* const scale, const T* const blocks, T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    for (int k = 0; k < block_size; k++) {
                        out[i * block_size + k] = scale[i] * blocks[i * block_size + k];
                    }
                }
            };

            assert(tpl.good());
        }
    };

}  // namespace sfem

#endif  // SFEM_ShiftedPenalty_IMPL_HPP
