#pragma once

#include <cstddef>
#include <memory>

#include "sfem_HistoryStorage.hpp"
#include "sfem_PackedHistory.hpp"
#include "sfem_aliases.hpp"
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
    /// and `ref`, the remainder form of `prony::advance`; with `stochastic`, an integer format's
    /// codes rounded stochastically from `seed` (HistoryRounding), as on the host.
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
                         const int            stochastic,
                         const uint64_t       seed,
                         void *const          stream);

    // The packed store (HistoryStorage::QUANT_BLOCK, sfem_PackedHistory.hpp) on the device.  A
    // write of one tensor is H <- alpha H + increment, the increment b (S_new - S_n), or its
    // remainder form with beta and R; it takes three steps, because the new code area's size is
    // known only once every block's width is:
    //   cu_prony_packed_max    the largest |new value|, to the host (the grid's step follows it)
    //   cu_prony_packed_plan   every block's least code and width into base and width, and its
    //                          size, in 8-byte units, into `sizes`
    //   cu_prony_packed_pack   the codes, as the plan rounded them, into a zeroed code area at
    //                          the offsets the host summed from the sizes
    // `old` is the tensor before the write, read by all three; its codes and headers stay until
    // the caller swaps in the new ones.  Stochastic rounding (`stochastic`, `seed`, the values
    // numbered from k0) draws prony::uniform's numbers, as the host's write does.

    /// A zeroed code area of `bytes` bytes, from the stream-ordered pool and returned to it in
    /// `stream` when released: a write makes one every step, and cudaMalloc/cudaFree would
    /// synchronise the device each time.
    std::shared_ptr<Buffer<uint8_t>> cu_prony_packed_codes(const std::size_t bytes, void *const stream);

    /// The weights of one tensor's write.
    struct PackedUpdate {
        real_t alpha{1};
        real_t b{0};
        real_t beta{0};
    };

    int cu_prony_packed_max(const prony::PackedTensor &old,
                            const PackedUpdate         u,
                            const int                  stress_bytes,
                            const void *const          S_n,
                            const void *const          S_new,
                            const void *const          ref,
                            real_t *const              scratch,
                            real_t *const              host_max,
                            void *const                stream);

    int cu_prony_packed_plan(const prony::PackedTensor &old,
                             const PackedUpdate         u,
                             const int                  stress_bytes,
                             const void *const          S_n,
                             const void *const          S_new,
                             const void *const          ref,
                             const real_t               step,
                             const int                  stochastic,
                             const uint64_t             seed,
                             const uint64_t             k0,
                             int32_t *const             base,
                             uint8_t *const             width,
                             uint32_t *const            sizes,
                             void *const                stream);

    int cu_prony_packed_pack(const prony::PackedTensor &old,
                             const PackedUpdate         u,
                             const int                  stress_bytes,
                             const void *const          S_n,
                             const void *const          S_new,
                             const void *const          ref,
                             const real_t               step,
                             const int                  stochastic,
                             const uint64_t             seed,
                             const uint64_t             k0,
                             const int32_t *const       base,
                             const uint8_t *const       width,
                             const uint32_t *const      offset,
                             uint8_t *const             codes,
                             void *const                stream);

    /// prony::frozen_history_packed: out = -b_sum S_n + sum_i alpha_i (H_i + beta_i (S_n - R)),
    /// `H` a host array of n_terms tensors (none: S_hist = -b_sum S_n), of `rows` rows of `n`.
    int cu_prony_packed_frozen_history(const ptrdiff_t                  rows,
                                       const ptrdiff_t                  n,
                                       const int                        n_terms,
                                       const prony::PackedTensor *const H,
                                       const real_t *const              alpha,
                                       const real_t                     b_sum,
                                       const real_t *const              beta,
                                       const int                        stress_bytes,
                                       const void *const                S_n,
                                       const void *const                ref,
                                       real_t *const                    out,
                                       void *const                      stream);

    /// prony::squared_norms_packed into the host array `out` (n_terms), through the device array
    /// `scratch` (n_terms); synchronises `stream`.
    int cu_prony_packed_squared_norms(const int                        n_terms,
                                      const prony::PackedTensor *const H,
                                      const int                        n_qp,
                                      const int                        n_diagonal,
                                      const real_t *const              beta,
                                      const int                        stress_bytes,
                                      const void *const                S_n,
                                      const void *const                ref,
                                      real_t *const                    scratch,
                                      real_t *const                    out,
                                      void *const                      stream);

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
