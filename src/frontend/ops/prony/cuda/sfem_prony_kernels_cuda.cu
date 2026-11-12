// CUB's own NVTX ranges are NVTX 3, which cannot share a translation unit with the older NVTX
// SFEM's trace scopes include: they are turned off for CUB alone.
#define NVTX_DISABLE
#include <cub/device/device_scan.cuh>
#undef NVTX_DISABLE

#include "sfem_prony_kernels_cuda.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "sfem_cuda_base.hpp"
#include "sfem_cuda_HistoryCodecs.cuh"
#include "sfem_prony_buffers.hpp"

namespace sfem {

    namespace {

        constexpr int MAX_TERMS = 64;

        /// The step's weights, by value in the launch: a spectrum is tens of
        /// terms, so this is under a kilobyte of kernel parameters.
        struct Weights {
            real_t alpha[MAX_TERMS];
            real_t b[MAX_TERMS];
            real_t beta[MAX_TERMS];  ///< remainder form (sfem_prony_kernels.hpp, advance); 0: H itself
        };

        /// The update of term i at value k, the host's `advance` increment.
        template <typename S_t>
        __device__ inline real_t increment(const Weights &w,
                                           const int      i,
                                           const S_t *const SFEM_RESTRICT S_n,
                                           const S_t *const SFEM_RESTRICT S_new,
                                           const S_t *const SFEM_RESTRICT ref,
                                           const ptrdiff_t k) {
            const real_t sp = static_cast<real_t>(S_n[k]), sn = static_cast<real_t>(S_new[k]);
            const real_t be = w.beta[i];
            if (be == 0) return w.b[i] * (sn - sp);
            const real_t R = static_cast<real_t>(ref[k]);
            return (w.alpha[i] * be - w.b[i]) * (sp - R) + (w.b[i] - be) * (sn - R);
        }

        /// The scale at `scales[k]`, or one for an unscaled codec, whose store has no scales
        /// (`scales` null), as on the host (sfem_prony_kernels.hpp, scale_at).
        template <class Codec>
        __device__ inline scaling_t scale_at(const scaling_t *const SFEM_RESTRICT scales, const ptrdiff_t k) {
            if constexpr (Codec::scaled) {
                return scales[k];
            } else {
                (void)scales;
                (void)k;
                return 1;
            }
        }

        template <class F>
        int dispatch(const HistoryStorage storage, F &&f) {
            switch (storage) {
                case HistoryStorage::FULL:
                    return f(device_codecs::DeviceCast<real_t>{});
                case HistoryStorage::FLOAT32:
                    return f(device_codecs::DeviceCast<float>{});
                case HistoryStorage::FLOAT16:
                    return f(device_codecs::DeviceHalf{});
                case HistoryStorage::FLOAT16_SCALED:
                    return f(device_codecs::DeviceScaledHalf{});
                case HistoryStorage::INT16_SCALED:
                    return f(device_codecs::DeviceScaledInteger<int16_t, 32767>{});
                case HistoryStorage::INT8_SCALED:
                    return f(device_codecs::DeviceScaledInteger<int8_t, 127>{});
                case HistoryStorage::FLOAT16_MD:
                    return f(device_codecs::DeviceMeanDeviation<device_codecs::DeviceScaledHalf>{});
                case HistoryStorage::INT8_MD:
                    return f(device_codecs::DeviceMeanDeviation<device_codecs::DeviceScaledInteger<int8_t, 127>>{});
                case HistoryStorage::QUANT_GRID:
                    // Host only for now: PronyRelaxation refuses it on the device.
                    return SFEM_FAILURE;
            }
            return SFEM_FAILURE;
        }

        /// One thread per (row, element): the entries are independent.
        template <class Codec, typename S_t>
        __global__ void frozen_history_kernel(const int                                             n_terms,
                                              const ptrdiff_t                                       rows,
                                              const ptrdiff_t                                       n,
                                              const Weights                                         w,
                                              const real_t                                          b_sum,
                                              const typename Codec::storage_t *const SFEM_RESTRICT H,
                                              const scaling_t *const SFEM_RESTRICT                 H_scales,
                                              const S_t *const SFEM_RESTRICT                       S_n,
                                              real_t *const SFEM_RESTRICT                          out,
                                              const prony::Means                                    m,
                                              const S_t *const SFEM_RESTRICT                       ref) {
            const ptrdiff_t total = rows * n;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r   = k / n;
                const ptrdiff_t e   = k - r * n;
                real_t          acc = -b_sum * static_cast<real_t>(S_n[k]);
                for (int i = 0; i < n_terms; i++) {
                    real_t h = Codec::decode(H[(i * rows + r) * n + e], scale_at<Codec>(H_scales, i * n + e));
                    if constexpr (Codec::mean_deviation) {
                        h += static_cast<real_t>(m.data[(i * (rows / m.n_qp) + r / m.n_qp) * m.n_groups + e / m.elements]);
                    }
                    acc += w.alpha[i] * h;
                    if (w.beta[i] != 0) acc += w.alpha[i] * w.beta[i] * (static_cast<real_t>(S_n[k]) - static_cast<real_t>(ref[k]));
                }
                out[k] = acc;
            }
        }

        /// Threads per block of the mean-deviation update: a multiple of every group size.
        constexpr int MD_BLOCK = 128;
        static_assert(MD_BLOCK % prony::MAX_ELEMENTS_PER_MEAN == 0, "a group must not straddle a block");
        constexpr int MD_MAX_COMPONENTS = 9;

        /// H_i <- alpha_i H_i + b_i (S_new - S_n) in a mean-deviation store (S_n null: no
        /// increment), the host's store_mean_deviation (sfem_prony_kernels.hpp) per thread: one
        /// thread per element, and a group's mean summed by its first thread over the group's
        /// threads' sums, in the host's order.  Every thread of a block walks the same tiles, so
        /// the barriers are reached by all; the new means are written only after every thread
        /// of the group has read the old ones.
        template <class Codec, typename S_t>
        __global__ void advance_mean_deviation_kernel(const int                                       n_terms,
                                                      const ptrdiff_t                                 rows,
                                                      const ptrdiff_t                                 n,
                                                      const Weights                                   w,
                                                      const S_t *const SFEM_RESTRICT                 S_n,
                                                      const S_t *const SFEM_RESTRICT                 S_new,
                                                      typename Codec::storage_t *const               H,
                                                      scaling_t *const                               H_scales,
                                                      const prony::Means                              m,
                                                      const S_t *const SFEM_RESTRICT                 ref,
                                                      const bool                                      stochastic,
                                                      const uint64_t                                  seed) {
            __shared__ real_t         sum[MD_MAX_COMPONENTS][MD_BLOCK];
            __shared__ history_mean_t group_mean[MD_MAX_COMPONENTS][MD_BLOCK];

            const ptrdiff_t nc  = rows / m.n_qp;
            const int       tid = threadIdx.x;
            for (ptrdiff_t tile = blockIdx.x; tile * MD_BLOCK < n; tile += gridDim.x) {
                const ptrdiff_t e      = tile * MD_BLOCK + tid;
                const bool      active = e < n;
                const ptrdiff_t g      = e / m.elements;
                const bool      leader = active && tid % m.elements == 0;

                for (int i = 0; i < n_terms; i++) {
                    typename Codec::storage_t *const h  = &H[i * rows * n];
                    scaling_t *const                 hs = &H_scales[i * n];
                    history_mean_t *const            mi = m.data + i * nc * m.n_groups;
                    const real_t                     a  = w.alpha[i];

                    const auto z = [&](const ptrdiff_t r) {
                        const ptrdiff_t k = r * n + e;
                        const real_t    v = static_cast<real_t>(mi[(r / m.n_qp) * m.n_groups + g]) + Codec::decode(h[k], hs[e]);
                        return S_n ? a * v + increment(w, i, S_n, S_new, ref, k) : a * v;
                    };

                    if (active) {
                        for (ptrdiff_t c = 0; c < nc; c++) {
                            real_t acc = 0;
                            for (int q = 0; q < m.n_qp; q++) acc += z(c * m.n_qp + q);
                            sum[c][tid] = acc;
                        }
                    }
                    __syncthreads();

                    if (leader) {
                        const int hi = min(tid + m.elements, (int)min((ptrdiff_t)MD_BLOCK, n - tile * MD_BLOCK));
                        for (ptrdiff_t c = 0; c < nc; c++) {
                            real_t acc = 0;
                            for (int l = tid; l < hi; l++) acc += sum[c][l];
                            group_mean[c][tid / m.elements] = static_cast<history_mean_t>(acc / static_cast<real_t>((hi - tid) * m.n_qp));
                        }
                    }
                    __syncthreads();

                    scaling_t scale = 1;
                    if (active) {
                        const int gl      = tid / m.elements;
                        real_t    max_abs = 0;
                        for (ptrdiff_t r = 0; r < rows; r++) {
                            max_abs = fmax(max_abs, fabs(z(r) - static_cast<real_t>(group_mean[r / m.n_qp][gl])));
                        }
                        scale            = Codec::scale_for(max_abs);
                        const real_t inv = 1 / static_cast<real_t>(scale);
                        for (ptrdiff_t r = 0; r < rows; r++) {
                            const real_t v = z(r);
                            h[r * n + e]   = device_codecs::encode_rounded<Codec>(
                                    v - static_cast<real_t>(group_mean[r / m.n_qp][gl]), inv, stochastic, seed, (i * rows + r) * n + e);
                        }
                    }
                    __syncthreads();

                    if (active) hs[e] = scale;
                    if (leader) {
                        for (ptrdiff_t c = 0; c < nc; c++) mi[c * m.n_groups + g] = group_mean[c][tid / m.elements];
                    }
                    __syncthreads();
                }
            }
        }

        /// One thread per element: a scaled codec needs the max-abs of a whole
        /// tensor of the element before its first value is written, so the
        /// element's rows stay in one thread.  Adjacent threads read adjacent
        /// elements of every row, so the accesses coalesce.
        template <class Codec, typename S_t>
        __global__ void advance_kernel(const int                                       n_terms,
                                       const ptrdiff_t                                 rows,
                                       const ptrdiff_t                                 n,
                                       const Weights                                   w,
                                       const S_t *const SFEM_RESTRICT                 S_n,
                                       const S_t *const SFEM_RESTRICT                 S_new,
                                       typename Codec::storage_t *const SFEM_RESTRICT H,
                                       scaling_t *const SFEM_RESTRICT                 H_scales,
                                       const S_t *const SFEM_RESTRICT                 ref,
                                       const bool                                     stochastic,
                                       const uint64_t                                 seed) {
            for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
                 e += (ptrdiff_t)blockDim.x * gridDim.x) {
                for (int i = 0; i < n_terms; i++) {
                    typename Codec::storage_t *const h         = &H[i * rows * n];
                    const real_t                     a         = w.alpha[i];
                    const real_t                     old_scale = scale_at<Codec>(H_scales, i * n + e);

                    real_t inv = 1;
                    if constexpr (Codec::scaled) {
                        real_t max_abs = 0;
                        for (ptrdiff_t r = 0; r < rows; r++) {
                            const ptrdiff_t k = r * n + e;
                            const real_t    v = a * Codec::decode(h[k], old_scale) + increment(w, i, S_n, S_new, ref, k);
                            max_abs           = fmax(max_abs, fabs(v));
                        }
                        const scaling_t scale = Codec::scale_for(max_abs);
                        H_scales[i * n + e]   = scale;
                        inv                   = 1 / static_cast<real_t>(scale);
                    }

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        const ptrdiff_t k = r * n + e;
                        const real_t    v = a * Codec::decode(h[k], old_scale) + increment(w, i, S_n, S_new, ref, k);
                        h[k]              = device_codecs::encode_rounded<Codec>(v, inv, stochastic, seed, (i * rows + r) * n + e);
                    }
                }
            }
        }

        __global__ void fill_scales_kernel(const ptrdiff_t n, scaling_t *const SFEM_RESTRICT scales) {
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < n;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                scales[k] = 1;
            }
        }

        __global__ void narrow_kernel(const ptrdiff_t n, const real_t *const SFEM_RESTRICT in, float *const SFEM_RESTRICT out) {
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < n;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                out[k] = static_cast<float>(in[k]);
            }
        }

        constexpr int NORM_BLOCK = 256;

        /// The block's partial sum into *out (atomically: a probe's value, whose summation order
        /// is free).
        __device__ inline void add_block_sum(real_t s, real_t *const out) {
            __shared__ real_t part[NORM_BLOCK];
            part[threadIdx.x] = s;
            __syncthreads();
            for (int half = NORM_BLOCK / 2; half > 0; half /= 2) {
                if ((int)threadIdx.x < half) part[threadIdx.x] += part[threadIdx.x + half];
                __syncthreads();
            }
            if (threadIdx.x == 0) atomicAdd(out, part[0]);
        }

        /// Per term (blockIdx.y), the weighted sum of squares of H_i, prony::squared_norms.
        template <class Codec, typename S_t>
        __global__ void squared_norms_kernel(const ptrdiff_t                                       rows,
                                             const ptrdiff_t                                       n,
                                             const int                                             n_qp,
                                             const int                                             n_diagonal,
                                             const Weights                                         w,
                                             const typename Codec::storage_t *const SFEM_RESTRICT H,
                                             const scaling_t *const SFEM_RESTRICT                 H_scales,
                                             const prony::Means                                    m,
                                             const S_t *const SFEM_RESTRICT                       S_n,
                                             const S_t *const SFEM_RESTRICT                       ref,
                                             real_t *const SFEM_RESTRICT                          out) {
            const int       i     = blockIdx.y;
            const ptrdiff_t total = rows * n;
            real_t          s     = 0;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r = k / n;
                const ptrdiff_t e = k - r * n;
                real_t          v = Codec::decode(H[(i * rows + r) * n + e], scale_at<Codec>(H_scales, i * n + e));
                if constexpr (Codec::mean_deviation) {
                    v += static_cast<real_t>(m.data[(i * (rows / m.n_qp) + r / m.n_qp) * m.n_groups + e / m.elements]);
                }
                if (w.beta[i] != 0) v += w.beta[i] * (static_cast<real_t>(S_n[k]) - static_cast<real_t>(ref[k]));
                s += (r / n_qp < n_diagonal ? real_t(1) : real_t(2)) * v * v;
            }
            add_block_sum(s, &out[i]);
        }

        /// The weighted sum of squares of S_new - S_old, prony::squared_increment.
        template <typename S_t>
        __global__ void squared_increment_kernel(const ptrdiff_t                rows,
                                                 const ptrdiff_t                n,
                                                 const int                      n_qp,
                                                 const int                      n_diagonal,
                                                 const S_t *const SFEM_RESTRICT S_new,
                                                 const S_t *const SFEM_RESTRICT S_old,
                                                 real_t *const SFEM_RESTRICT    out) {
            const ptrdiff_t total = rows * n;
            real_t          s     = 0;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r = k / n;
                const real_t    d = static_cast<real_t>(S_new[k]) - static_cast<real_t>(S_old[k]);
                s += (r / n_qp < n_diagonal ? real_t(1) : real_t(2)) * d * d;
            }
            add_block_sum(s, out);
        }


        // The packed store (sfem_PackedHistory.hpp).

        /// The new value (r, e) of a packed tensor's write.
        template <typename S_t>
        __device__ inline real_t packed_new(const prony::PackedTensor &old,
                                            const PackedUpdate         u,
                                            const S_t *const           S_n,
                                            const S_t *const           S_new,
                                            const S_t *const           ref,
                                            const ptrdiff_t            r,
                                            const ptrdiff_t            e) {
            const ptrdiff_t k  = r * old.n + e;
            const real_t    sp = static_cast<real_t>(S_n[k]), sn = static_cast<real_t>(S_new[k]);
            real_t          inc;
            if (u.beta == 0) {
                inc = u.b * (sn - sp);
            } else {
                const real_t R = static_cast<real_t>(ref[k]);
                inc            = (u.alpha * u.beta - u.b) * (sp - R) + (u.b - u.beta) * (sn - R);
            }
            return u.alpha * old.value(r, e) + inc;
        }

        constexpr int PACK_THREADS = (int)prony::PACKED_BLOCK;

        /// A write's plan as the device computes it: the largest |new value| (as the bits of a
        /// non-negative double, which order as integers), the running maximum and the step that
        /// follow from it, and the code area's size in 8-byte units.
        struct DevicePlan {
            unsigned long long max_bits;
            real_t             max_abs;
            real_t             step;
            unsigned long long units;
        };

        /// The running maximum and the step, as QuadratureState computes them on the host.
        __global__ void packed_step_kernel(DevicePlan *const plan, const real_t running_max, const real_t tolerance) {
            const real_t m = fmax((real_t)__longlong_as_double((long long)plan->max_bits), running_max);
            plan->max_abs  = m;
            plan->step     = prony::grid_step(m, tolerance);
        }

        /// The code area's size: the last block's offset and size.
        __global__ void packed_units_kernel(DevicePlan *const     plan,
                                            const uint32_t *const offset,
                                            const uint32_t *const sizes,
                                            const ptrdiff_t       headers) {
            plan->units = headers > 0 ? (unsigned long long)offset[headers - 1] + sizes[headers - 1] : 0;
        }

        /// The largest |new value|, as the bits of a non-negative double (which order as integers).
        template <typename S_t>
        __global__ void packed_max_kernel(const prony::PackedTensor old,
                                          const PackedUpdate        u,
                                          const S_t *const          S_n,
                                          const S_t *const          S_new,
                                          const S_t *const          ref,
                                          unsigned long long *const out) {
            __shared__ double part[NORM_BLOCK];
            const ptrdiff_t   total = old.rows * old.n;
            double            m     = 0;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r = k / old.n;
                m                 = fmax(m, fabs((double)packed_new(old, u, S_n, S_new, ref, r, k - r * old.n)));
            }
            part[threadIdx.x] = m;
            __syncthreads();
            for (int half = NORM_BLOCK / 2; half > 0; half /= 2) {
                if ((int)threadIdx.x < half) part[threadIdx.x] = fmax(part[threadIdx.x], part[threadIdx.x + half]);
                __syncthreads();
            }
            if (threadIdx.x == 0) atomicMax(out, (unsigned long long)__double_as_longlong(part[0]));
        }

        /// One thread block per (row, block of elements), one thread per element: the block's
        /// least code and width, and its size in 8-byte units.
        template <typename S_t>
        __global__ void packed_plan_kernel(const prony::PackedTensor old,
                                           const PackedUpdate        u,
                                           const S_t *const          S_n,
                                           const S_t *const          S_new,
                                           const S_t *const          ref,
                                           const DevicePlan *const   plan,
                                           const bool                stochastic,
                                           const uint64_t            seed,
                                           const uint64_t            k0,
                                           int32_t *const            base,
                                           uint8_t *const            width,
                                           uint32_t *const           sizes) {
            __shared__ long long lo[PACK_THREADS], hi[PACK_THREADS];
            const ptrdiff_t      nb  = old.n_blocks();
            const real_t         inv = 1 / plan->step;
            for (ptrdiff_t h = blockIdx.x; h < old.rows * nb; h += gridDim.x) {
                const ptrdiff_t r = h / nb, e0 = (h - r * nb) * prony::PACKED_BLOCK;
                const ptrdiff_t e = e0 + threadIdx.x;
                long long       q_lo = LLONG_MAX, q_hi = LLONG_MIN;
                if (e < old.n) {
                    const long long q = prony::grid_code(packed_new(old, u, S_n, S_new, ref, r, e), inv, stochastic, seed, k0 + r * old.n + e);
                    q_lo = q_hi = q;
                }
                lo[threadIdx.x] = q_lo;
                hi[threadIdx.x] = q_hi;
                __syncthreads();
                for (int half = PACK_THREADS / 2; half > 0; half /= 2) {
                    if ((int)threadIdx.x < half) {
                        lo[threadIdx.x] = min(lo[threadIdx.x], lo[threadIdx.x + half]);
                        hi[threadIdx.x] = max(hi[threadIdx.x], hi[threadIdx.x + half]);
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0) {
                    const ptrdiff_t count = min((ptrdiff_t)prony::PACKED_BLOCK, old.n - e0);
                    // Beyond 32 bits the width saturates and the codes are wrong: refused on the
                    // host before (plan_packed) and checked by the tests; here kept in bounds.
                    const int w = min(32, smesh::compression::bit_width((uint64_t)(hi[0] - lo[0])));
                    base[h]     = (int32_t)lo[0];
                    width[h]    = (uint8_t)w;
                    sizes[h]    = (uint32_t)(prony::packed_block_bytes(count, w) / 8);
                }
                __syncthreads();
            }
        }

        /// The codes of each block, as the plan rounded them, ORed into the zeroed code area: a
        /// code of w <= 32 bits spans at most two 32-bit words, and blocks start on 8 bytes.
        template <typename S_t>
        __global__ void packed_pack_kernel(const prony::PackedTensor old,
                                           const PackedUpdate        u,
                                           const S_t *const          S_n,
                                           const S_t *const          S_new,
                                           const S_t *const          ref,
                                           const real_t              inv,
                                           const bool                stochastic,
                                           const uint64_t            seed,
                                           const uint64_t            k0,
                                           const int32_t *const      base,
                                           const uint8_t *const      width,
                                           const uint32_t *const     offset,
                                           uint8_t *const            codes) {
            const ptrdiff_t nb = old.n_blocks();
            for (ptrdiff_t h = blockIdx.x; h < old.rows * nb; h += gridDim.x) {
                const ptrdiff_t r = h / nb, e = (h - r * nb) * prony::PACKED_BLOCK + threadIdx.x;
                const int       w = width[h];
                if (w == 0 || e >= old.n) continue;
                const long long q    = prony::grid_code(packed_new(old, u, S_n, S_new, ref, r, e), inv, stochastic, seed, k0 + r * old.n + e);
                const uint64_t  code = (uint64_t)(q - base[h]);
                const uint64_t  bit  = (uint64_t)threadIdx.x * w;
                uint32_t *const word = reinterpret_cast<uint32_t *>(codes + (uint64_t)offset[h] * 8) + bit / 32;
                const int       sh   = (int)(bit % 32);
                atomicOr(word, (uint32_t)(code << sh));
                if (sh + w > 32) atomicOr(word + 1, (uint32_t)(code >> (32 - sh)));
            }
        }

        /// out = -b_sum S_n + ab (S_n - R): the frozen history's part that is not a stored code,
        /// ab = sum_i alpha_i beta_i.
        template <typename S_t>
        __global__ void packed_frozen_init_kernel(const ptrdiff_t  total,
                                                  const real_t     b_sum,
                                                  const real_t     ab,
                                                  const S_t *const S_n,
                                                  const S_t *const ref,
                                                  real_t *const    out) {
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const real_t s = static_cast<real_t>(S_n[k]);
                out[k]         = -b_sum * s + (ab != 0 ? ab * (s - static_cast<real_t>(ref[k])) : real_t(0));
            }
        }

        /// out += alpha H.
        __global__ void packed_frozen_add_kernel(const prony::PackedTensor H, const real_t alpha, real_t *const out) {
            const ptrdiff_t total = H.rows * H.n;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r = k / H.n;
                out[k] += alpha * H.value(r, k - r * H.n);
            }
        }

        /// The weighted sum of squares of H + beta (S_n - R) into *out.
        template <typename S_t>
        __global__ void packed_squared_norm_kernel(const prony::PackedTensor H,
                                                   const int                 n_qp,
                                                   const int                 n_diagonal,
                                                   const real_t              beta,
                                                   const S_t *const          S_n,
                                                   const S_t *const          ref,
                                                   real_t *const             out) {
            const ptrdiff_t total = H.rows * H.n;
            real_t          s     = 0;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r = k / H.n;
                real_t          v = H.value(r, k - r * H.n);
                if (beta != 0) v += beta * (static_cast<real_t>(S_n[k]) - static_cast<real_t>(ref[k]));
                s += (r / n_qp < n_diagonal ? real_t(1) : real_t(2)) * v * v;
            }
            add_block_sum(s, out);
        }

        /// Call `f` with a null pointer of the stress type `stress_bytes` names.
        template <class F>
        int dispatch_stress(const int stress_bytes, F &&f) {
            if (stress_bytes == (int)sizeof(real_t)) return f((const real_t *)nullptr);
            if (stress_bytes == (int)sizeof(float)) return f((const float *)nullptr);
            return SFEM_FAILURE;
        }

        /// The loops stride over the grid, so a capped grid covers any size.
        unsigned grid_for(const ptrdiff_t n, const int block) {
            const ptrdiff_t blocks = std::max(ptrdiff_t(1), (n + block - 1) / block);
            return (unsigned)std::min(blocks, ptrdiff_t(65535) * 32);
        }

        int launched() { return cudaGetLastError() == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE; }

        Weights pack(const int n_terms, const real_t *const alpha, const real_t *const b, const real_t *const beta) {
            Weights w{};
            for (int i = 0; i < n_terms; i++) {
                w.alpha[i] = alpha[i];
                w.b[i]     = b ? b[i] : real_t(0);
                w.beta[i]  = beta ? beta[i] : real_t(0);
            }
            return w;
        }

    }  // namespace

    int cu_prony_max_terms() { return MAX_TERMS; }

    int cu_prony_frozen_history(const HistoryStorage   storage,
                                const int              n_terms,
                                const ptrdiff_t        rows,
                                const ptrdiff_t        n,
                                const real_t *const    alpha,
                                const real_t           b_sum,
                                const void *const      H,
                                const scaling_t *const H_scales,
                                const int              stress_bytes,
                                const void *const      S_n,
                                real_t *const          out,
                                const prony::Means    &means,
                                const real_t *const    beta,
                                const void *const      ref,
                                void *const            stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, nullptr, beta);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            return dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                frozen_history_kernel<Codec, S_t><<<grid_for(rows * n, block), block, 0, (cudaStream_t)stream>>>(
                        n_terms, rows, n, w, b_sum, static_cast<const typename Codec::storage_t *>(H), H_scales, (const S_t *)S_n, out, means, (const S_t *)ref);
                return launched();
            });
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

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
                         void *const          stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, b, beta);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            return dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t     = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                const S_t *sp = (const S_t *)S_n, *sn = (const S_t *)S_new, *R = (const S_t *)ref;
                if constexpr (Codec::mean_deviation) {
                    if (rows / means.n_qp > MD_MAX_COMPONENTS) return SFEM_FAILURE;
                    advance_mean_deviation_kernel<Codec, S_t><<<grid_for(n, MD_BLOCK), MD_BLOCK, 0, (cudaStream_t)stream>>>(
                            n_terms, rows, n, w, sp, sn, static_cast<typename Codec::storage_t *>(H), H_scales, means, R, stochastic != 0, seed);
                } else {
                    advance_kernel<Codec, S_t><<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
                            n_terms, rows, n, w, sp, sn, static_cast<typename Codec::storage_t *>(H), H_scales, R, stochastic != 0, seed);
                }
                return launched();
            });
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

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
                               void *const            stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        if (n_terms == 0) return SFEM_SUCCESS;
        const cudaStream_t s = (cudaStream_t)stream;
        if (cudaMemsetAsync(scratch, 0, n_terms * sizeof(real_t), s) != cudaSuccess) return SFEM_FAILURE;
        Weights w{};
        for (int i = 0; i < n_terms; i++) w.beta[i] = beta ? beta[i] : real_t(0);
        const int ret = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            return dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                const dim3 grid(std::min<unsigned>(grid_for(rows * n, NORM_BLOCK), 1024u), (unsigned)n_terms);
                squared_norms_kernel<Codec, S_t><<<grid, NORM_BLOCK, 0, s>>>(rows,
                                                                             n,
                                                                             n_qp,
                                                                             n_diagonal,
                                                                             w,
                                                                             static_cast<const typename Codec::storage_t *>(H),
                                                                             H_scales,
                                                                             means,
                                                                             (const S_t *)S_n,
                                                                             (const S_t *)ref,
                                                                             scratch);
                return launched();
            });
        });
        if (ret != SFEM_SUCCESS) return ret;
        if (cudaMemcpyAsync(out, scratch, n_terms * sizeof(real_t), cudaMemcpyDeviceToHost, s) != cudaSuccess) return SFEM_FAILURE;
        return cudaStreamSynchronize(s) == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int cu_prony_squared_increment(const ptrdiff_t   rows,
                                   const ptrdiff_t   n,
                                   const int         n_qp,
                                   const int         n_diagonal,
                                   const int         stress_bytes,
                                   const void *const S_new,
                                   const void *const S_old,
                                   real_t *const     scratch,
                                   real_t *const     out,
                                   void *const       stream) {
        const cudaStream_t s = (cudaStream_t)stream;
        if (cudaMemsetAsync(scratch, 0, sizeof(real_t), s) != cudaSuccess) return SFEM_FAILURE;
        const int ret = dispatch_stress(stress_bytes, [&](auto stress) {
            using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
            squared_increment_kernel<S_t><<<std::min<unsigned>(grid_for(rows * n, NORM_BLOCK), 1024u), NORM_BLOCK, 0, s>>>(
                    rows, n, n_qp, n_diagonal, (const S_t *)S_new, (const S_t *)S_old, scratch);
            return launched();
        });
        if (ret != SFEM_SUCCESS) return ret;
        if (cudaMemcpyAsync(out, scratch, sizeof(real_t), cudaMemcpyDeviceToHost, s) != cudaSuccess) return SFEM_FAILURE;
        return cudaStreamSynchronize(s) == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int cu_prony_narrow(const ptrdiff_t n, const real_t *const in, float *const out, void *const stream) {
        if (n <= 0) return SFEM_SUCCESS;
        const int block = 128;
        narrow_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(n, in, out);
        const int ret = launched();
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    int cu_prony_copy(const ptrdiff_t        bytes,
                      const void *const      src,
                      void *const            dst,
                      const ptrdiff_t        n_scales,
                      const scaling_t *const src_scales,
                      scaling_t *const       dst_scales,
                      void *const            stream) {
        if (bytes > 0 && cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess) {
            return SFEM_FAILURE;
        }
        if (n_scales > 0 &&
            cudaMemcpyAsync(dst_scales, src_scales, n_scales * sizeof(scaling_t), cudaMemcpyDeviceToDevice, (cudaStream_t)stream) !=
                    cudaSuccess) {
            return SFEM_FAILURE;
        }
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return SFEM_SUCCESS;
    }

    int cu_prony_zero(const ptrdiff_t  bytes,
                      void *const      data,
                      const ptrdiff_t  n_scales,
                      scaling_t *const scales,
                      void *const      stream) {
        if (bytes > 0 && cudaMemsetAsync(data, 0, bytes, (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
        int ret = SFEM_SUCCESS;
        if (n_scales > 0) {
            const int block = 128;
            fill_scales_kernel<<<grid_for(n_scales, block), block, 0, (cudaStream_t)stream>>>(n_scales, scales);
            ret = launched();
        }
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    namespace {
        /// The pool of a call's work buffers, apart from the default pool where the packed
        /// store's code areas live from one step to the next: emptied after every call, so it is
        /// trimmed whole rather than around the codes. One per device, made at first use.
        cudaMemPool_t work_pool() {
            static cudaMemPool_t pools[64] = {};
            int                  device    = 0;
            if (cudaGetDevice(&device) != cudaSuccess || device >= 64) SFEM_ERROR("cu_prony: no usable device\n");
            if (!pools[device]) {
                cudaMemPoolProps props = {};
                props.allocType        = cudaMemAllocationTypePinned;
                props.location.type    = cudaMemLocationTypeDevice;
                props.location.id      = device;
                if (cudaMemPoolCreate(&pools[device], &props) != cudaSuccess) SFEM_ERROR("cu_prony: creating the work pool failed\n");
            }
            return pools[device];
        }
    }  // namespace

    void *cu_prony_pool_allocate(const std::size_t bytes, void *const stream, const bool persistent) {
        const cudaStream_t s    = (cudaStream_t)stream;
        void              *data = nullptr;
        // A zero-sized request still yields a distinct pointer to release.
        const std::size_t n  = bytes > 0 ? bytes : 1;
        const cudaError_t ok = persistent ? cudaMallocAsync(&data, n, s) : cudaMallocFromPoolAsync(&data, n, work_pool(), s);
        if (ok != cudaSuccess || cudaMemsetAsync(data, 0, n, s) != cudaSuccess) {
            SFEM_ERROR("cu_prony_pool_allocate: allocating %zu bytes failed\n", bytes);
        }
        return data;
    }

    void cu_prony_pool_release(void *const data, void *const stream) { cudaFreeAsync(data, (cudaStream_t)stream); }

    std::size_t cu_prony_work_pool_peak() {
        cuuint64_t high = 0;
        if (cudaMemPoolGetAttribute(work_pool(), cudaMemPoolAttrUsedMemHigh, &high) != cudaSuccess) return 0;
        return (std::size_t)high;
    }

    void cu_prony_pool_trim(void *const stream) {
        if (cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess || cudaMemPoolTrimTo(work_pool(), 0) != cudaSuccess) {
            SFEM_ERROR("cu_prony_pool_trim: returning the work pool's memory failed\n");
        }
    }

    int cu_prony_packed_plan(const prony::PackedTensor &old,
                             const PackedUpdate         u,
                             const int                  stress_bytes,
                             const void *const          S_n,
                             const void *const          S_new,
                             const void *const          ref,
                             const real_t               running_max,
                             const real_t               tolerance,
                             const int                  stochastic,
                             const uint64_t             seed,
                             const uint64_t             k0,
                             int32_t *const             base,
                             uint8_t *const             width,
                             uint32_t *const            offset,
                             PackedPlan *const          plan,
                             void *const                stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const cudaStream_t s       = (cudaStream_t)stream;
        const ptrdiff_t    headers = old.rows * old.n_blocks();
        auto *const        d_plan  = static_cast<DevicePlan *>(cu_prony_pool_allocate(sizeof(DevicePlan), stream, false));
        auto *const        sizes   = static_cast<uint32_t *>(cu_prony_pool_allocate(headers * sizeof(uint32_t), stream, false));
        const int          ret     = dispatch_stress(stress_bytes, [&](auto stress) {
            using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
            packed_max_kernel<S_t><<<grid_for(old.rows * old.n, NORM_BLOCK), NORM_BLOCK, 0, s>>>(
                    old, u, (const S_t *)S_n, (const S_t *)S_new, (const S_t *)ref, &d_plan->max_bits);
            packed_step_kernel<<<1, 1, 0, s>>>(d_plan, running_max, tolerance);
            packed_plan_kernel<S_t><<<grid_for(headers, 1), PACK_THREADS, 0, s>>>(
                    old, u, (const S_t *)S_n, (const S_t *)S_new, (const S_t *)ref, d_plan, stochastic != 0, seed, k0, base, width, sizes);
            return launched();
        });
        // The offsets: an exclusive sum of the sizes, on the device.
        std::size_t temp_bytes = 0;
        int         status     = ret;
        if (status == SFEM_SUCCESS &&
            cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, sizes, offset, (int)headers, s) != cudaSuccess) {
            status = SFEM_FAILURE;
        }
        if (status == SFEM_SUCCESS) {
            void *const temp = cu_prony_pool_allocate(temp_bytes, stream, false);
            if (cub::DeviceScan::ExclusiveSum(temp, temp_bytes, sizes, offset, (int)headers, s) != cudaSuccess) status = SFEM_FAILURE;
            cu_prony_pool_release(temp, stream);
        }
        if (status == SFEM_SUCCESS) {
            packed_units_kernel<<<1, 1, 0, s>>>(d_plan, offset, sizes, headers);
            status = launched();
        }
        // The one wait of a write: the host allocates the code area from its size.
        DevicePlan host{};
        if (status == SFEM_SUCCESS && (cudaMemcpyAsync(&host, d_plan, sizeof(host), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                                       cudaStreamSynchronize(s) != cudaSuccess)) {
            status = SFEM_FAILURE;
        }
        cu_prony_pool_release(sizes, stream);
        cu_prony_pool_release(d_plan, stream);
        if (status != SFEM_SUCCESS) return status;
        plan->max_abs = host.max_abs;
        plan->step    = host.step;
        plan->units   = host.units;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return SFEM_SUCCESS;
    }

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
                             void *const                stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const ptrdiff_t headers = old.rows * old.n_blocks();
        const int       ret     = dispatch_stress(stress_bytes, [&](auto stress) {
            using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
            packed_pack_kernel<S_t><<<grid_for(headers, 1), PACK_THREADS, 0, (cudaStream_t)stream>>>(old,
                                                                                                      u,
                                                                                                      (const S_t *)S_n,
                                                                                                      (const S_t *)S_new,
                                                                                                      (const S_t *)ref,
                                                                                                      1 / step,
                                                                                                      stochastic != 0,
                                                                                                      seed,
                                                                                                      k0,
                                                                                                      base,
                                                                                                      width,
                                                                                                      offset,
                                                                                                      codes);
            return launched();
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

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
                                       void *const                      stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const cudaStream_t s     = (cudaStream_t)stream;
        const int          block = 128;
        real_t             ab    = 0;
        for (int i = 0; i < n_terms; i++) ab += beta ? alpha[i] * beta[i] : real_t(0);
        const ptrdiff_t total = rows * n;
        int             ret   = dispatch_stress(stress_bytes, [&](auto stress) {
            using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
            packed_frozen_init_kernel<S_t><<<grid_for(total, block), block, 0, s>>>(total, b_sum, ab, (const S_t *)S_n, (const S_t *)ref, out);
            return launched();
        });
        for (int i = 0; i < n_terms && ret == SFEM_SUCCESS; i++) {
            packed_frozen_add_kernel<<<grid_for(total, block), block, 0, s>>>(H[i], alpha[i], out);
            ret = launched();
        }
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

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
                                      void *const                      stream) {
        const cudaStream_t s = (cudaStream_t)stream;
        if (cudaMemsetAsync(scratch, 0, n_terms * sizeof(real_t), s) != cudaSuccess) return SFEM_FAILURE;
        int ret = SFEM_SUCCESS;
        for (int i = 0; i < n_terms && ret == SFEM_SUCCESS; i++) {
            ret = dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                packed_squared_norm_kernel<S_t><<<grid_for(H[i].rows * H[i].n, NORM_BLOCK), NORM_BLOCK, 0, s>>>(
                        H[i], n_qp, n_diagonal, beta ? beta[i] : real_t(0), (const S_t *)S_n, (const S_t *)ref, &scratch[i]);
                return launched();
            });
        }
        if (ret != SFEM_SUCCESS) return ret;
        if (cudaMemcpyAsync(out, scratch, n_terms * sizeof(real_t), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaStreamSynchronize(s) != cudaSuccess) {
            return SFEM_FAILURE;
        }
        return SFEM_SUCCESS;
    }

}  // namespace sfem
