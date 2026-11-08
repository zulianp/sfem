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
    /// device in the launch.  The relaxed stresses are real_t or float, by
    /// `stress_bytes`.  At most `cu_prony_max_terms()` terms.  Everything
    /// runs on `stream`, the operator's, where the stresses these loops read
    /// were written; a launch failure is returned, not deferred.

    int cu_prony_max_terms();

    /// out = sum_i alpha_i H_i - b_sum * S_n.  `beta` (host, per term, null for none) and `ref`
    /// (device, the stress type): the remainder form of `prony::frozen_history`.
    int cu_prony_frozen_history(const HistoryStorage storage,
                                const int            n_terms,
                                const ptrdiff_t      rows,
                                const ptrdiff_t      n,
                                const real_t *const  alpha,
                                const real_t         b_sum,
                                const void *const    H,
                                const scaling_t *const H_scales,
                                const int            stress_bytes,
                                const void *const    S_n,
                                real_t *const        out,
                                const prony::Means  &means,
                                const real_t *const  beta,
                                const void *const    ref,
                                void *const          stream);

    /// H_i <- alpha_i H_i + b_i (S_new - S_n), re-encoding H_i with a fresh scale; with `beta`
    /// and `ref`, the remainder form of `prony::advance`.
    int cu_prony_advance(const HistoryStorage storage,
                         const int            n_terms,
                         const ptrdiff_t      rows,
                         const ptrdiff_t      n,
                         const real_t *const  alpha,
                         const real_t *const  b,
                         const int            stress_bytes,
                         const void *const    S_n,
                         const void *const    S_new,
                         void *const          H,
                         scaling_t *const     H_scales,
                         const prony::Means  &means,
                         const real_t *const  beta,
                         const void *const    ref,
                         void *const          stream);

    /// Per term, prony::squared_norms into the host array `out` (n_terms), through the device
    /// array `scratch` (n_terms); synchronises `stream`.
    int cu_prony_squared_norms(const HistoryStorage   storage,
                               const int              n_terms,
                               const ptrdiff_t        rows,
                               const ptrdiff_t        n,
                               const int              n_qp,
                               const int              n_diagonal,
                               const void *const      H,
                               const scaling_t *const H_scales,
                               const prony::Means    &means,
                               const real_t *const    beta,
                               const int              stress_bytes,
                               const void *const      S_n,
                               const void *const      ref,
                               real_t *const          scratch,
                               real_t *const          out,
                               void *const            stream);

    /// prony::squared_increment into the host scalar *out, through the device scalar *scratch.
    int cu_prony_squared_increment(const ptrdiff_t   rows,
                                   const ptrdiff_t   n,
                                   const int         n_qp,
                                   const int         n_diagonal,
                                   const int         stress_bytes,
                                   const void *const S_new,
                                   const void *const S_old,
                                   real_t *const     scratch,
                                   real_t *const     out,
                                   void *const       stream);

    /// out = (float) in: a relaxed stress into a float32 slot.
    int cu_prony_narrow(const ptrdiff_t n, const real_t *const in, float *const out, void *const stream);

    /// Copy `bytes` bytes and `n_scales` scales.
    int cu_prony_copy(const ptrdiff_t        bytes,
                      const void *const      src,
                      void *const            dst,
                      const ptrdiff_t        n_scales,
                      const scaling_t *const src_scales,
                      scaling_t *const       dst_scales,
                      void *const            stream);

    /// Zero `bytes` bytes and set `n_scales` scales to one.
    int cu_prony_zero(const ptrdiff_t bytes,
                      void *const     data,
                      const ptrdiff_t n_scales,
                      scaling_t *const scales,
                      void *const     stream);

}  // namespace sfem
