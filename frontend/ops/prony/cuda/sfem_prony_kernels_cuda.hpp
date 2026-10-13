#pragma once

#include <cstddef>

#include "sfem_HistoryStorage.hpp"
#include "sfem_base.hpp"

namespace sfem {

    /// The device versions of the Prony pointwise loops (`sfem_prony_kernels.hpp`).
    ///
    /// Same arithmetic, same layout -- [tensor][component][qp][element], one
    /// contiguous run per (tensor, component, point) -- and the same codecs, so
    /// a history written by one is read by the other bit for bit in the store.
    /// `alpha` and `b` are host arrays of the step's weights; they travel to the
    /// device in the launch.  At most `cu_prony_max_terms()` terms.  Everything
    /// runs on `stream`, the operator's, where the stresses these loops read
    /// were written; a launch failure is returned, not deferred.

    int cu_prony_max_terms();

    /// out = sum_i alpha_i H_i - b_sum * S_n.
    int cu_prony_frozen_history(const HistoryStorage storage,
                                const int            n_terms,
                                const ptrdiff_t      rows,
                                const ptrdiff_t      n,
                                const real_t *const  alpha,
                                const real_t         b_sum,
                                const void *const    H,
                                const scaling_t *const H_scales,
                                const real_t *const  S_n,
                                real_t *const        out,
                                void *const          stream);

    /// H_i <- alpha_i H_i + b_i (S_new - S_n), re-encoding H_i with a fresh scale.
    int cu_prony_advance(const HistoryStorage storage,
                         const int            n_terms,
                         const ptrdiff_t      rows,
                         const ptrdiff_t      n,
                         const real_t *const  alpha,
                         const real_t *const  b,
                         const real_t *const  S_n,
                         const real_t *const  S_new,
                         void *const          H,
                         scaling_t *const     H_scales,
                         void *const          stream);

    /// Zero `bytes` bytes and set `n_scales` scales to one.
    int cu_prony_zero(const ptrdiff_t bytes,
                      void *const     data,
                      const ptrdiff_t n_scales,
                      scaling_t *const scales,
                      void *const     stream);

}  // namespace sfem
