#include "sfem_prony_kernels_cuda.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sfem_cuda_base.hpp"
#include "sfem_cuda_HistoryCodecs.cuh"

namespace sfem {

    namespace {

        constexpr int MAX_TERMS = 64;

        /// The step's weights, by value in the launch: a spectrum is tens of
        /// terms, so this is under a kilobyte of kernel parameters.
        struct Weights {
            real_t alpha[MAX_TERMS];
            real_t b[MAX_TERMS];
        };

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
                                              const prony::Means                                    m) {
            const ptrdiff_t total = rows * n;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r   = k / n;
                const ptrdiff_t e   = k - r * n;
                real_t          acc = -b_sum * static_cast<real_t>(S_n[k]);
                for (int i = 0; i < n_terms; i++) {
                    real_t h = Codec::decode(H[(i * rows + r) * n + e], H_scales[i * n + e]);
                    if constexpr (Codec::mean_deviation) {
                        h += static_cast<real_t>(m.data[(i * (rows / m.n_qp) + r / m.n_qp) * m.n_groups + e / m.elements]);
                    }
                    acc += w.alpha[i] * h;
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
                                                      const prony::Means                              m) {
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
                    const real_t                     bi = w.b[i];

                    const auto z = [&](const ptrdiff_t r) {
                        const ptrdiff_t k = r * n + e;
                        const real_t    v = static_cast<real_t>(mi[(r / m.n_qp) * m.n_groups + g]) + Codec::decode(h[k], hs[e]);
                        return S_n ? a * v + bi * (static_cast<real_t>(S_new[k]) - static_cast<real_t>(S_n[k])) : a * v;
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
                            h[r * n + e]   = Codec::encode(v - static_cast<real_t>(group_mean[r / m.n_qp][gl]), inv);
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
                                       scaling_t *const SFEM_RESTRICT                 H_scales) {
            for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
                 e += (ptrdiff_t)blockDim.x * gridDim.x) {
                for (int i = 0; i < n_terms; i++) {
                    typename Codec::storage_t *const h         = &H[i * rows * n];
                    const real_t                     a         = w.alpha[i];
                    const real_t                     bi        = w.b[i];
                    const real_t                     old_scale = H_scales[i * n + e];

                    real_t inv = 1;
                    if constexpr (Codec::scaled) {
                        real_t max_abs = 0;
                        for (ptrdiff_t r = 0; r < rows; r++) {
                            const ptrdiff_t k = r * n + e;
                            const real_t    v = a * Codec::decode(h[k], old_scale) + bi * (static_cast<real_t>(S_new[k]) - static_cast<real_t>(S_n[k]));
                            max_abs           = fmax(max_abs, fabs(v));
                        }
                        const scaling_t scale = Codec::scale_for(max_abs);
                        H_scales[i * n + e]   = scale;
                        inv                   = 1 / static_cast<real_t>(scale);
                    }

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        const ptrdiff_t k = r * n + e;
                        const real_t    v = a * Codec::decode(h[k], old_scale) + bi * (static_cast<real_t>(S_new[k]) - static_cast<real_t>(S_n[k]));
                        h[k]              = Codec::encode(v, inv);
                    }
                }
            }
        }

        template <class Codec>
        __global__ void scale_kernel(const ptrdiff_t                                 rows,
                                     const ptrdiff_t                                 n,
                                     const real_t                                    factor,
                                     typename Codec::storage_t *const SFEM_RESTRICT H,
                                     scaling_t *const SFEM_RESTRICT                 H_scales) {
            for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
                 e += (ptrdiff_t)blockDim.x * gridDim.x) {
                const real_t old_scale = H_scales[e];

                real_t inv = 1;
                if constexpr (Codec::scaled) {
                    real_t max_abs = 0;
                    for (ptrdiff_t r = 0; r < rows; r++) {
                        max_abs = fmax(max_abs, fabs(factor * Codec::decode(H[r * n + e], old_scale)));
                    }
                    const scaling_t scale = Codec::scale_for(max_abs);
                    H_scales[e]           = scale;
                    inv                   = 1 / static_cast<real_t>(scale);
                }

                for (ptrdiff_t r = 0; r < rows; r++) {
                    const ptrdiff_t k = r * n + e;
                    H[k]              = Codec::encode(factor * Codec::decode(H[k], old_scale), inv);
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

        Weights pack(const int n_terms, const real_t *const alpha, const real_t *const b) {
            Weights w{};
            for (int i = 0; i < n_terms; i++) {
                w.alpha[i] = alpha[i];
                w.b[i]     = b ? b[i] : real_t(0);
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
                                void *const            stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, nullptr);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            return dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                frozen_history_kernel<Codec, S_t><<<grid_for(rows * n, block), block, 0, (cudaStream_t)stream>>>(
                        n_terms, rows, n, w, b_sum, static_cast<const typename Codec::storage_t *>(H), H_scales, (const S_t *)S_n, out, means);
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
                         void *const          stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, b);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            return dispatch_stress(stress_bytes, [&](auto stress) {
                using S_t     = std::remove_const_t<std::remove_pointer_t<decltype(stress)>>;
                const S_t *sp = (const S_t *)S_n, *sn = (const S_t *)S_new;
                if constexpr (Codec::mean_deviation) {
                    if (rows / means.n_qp > MD_MAX_COMPONENTS) return SFEM_FAILURE;
                    advance_mean_deviation_kernel<Codec, S_t><<<grid_for(n, MD_BLOCK), MD_BLOCK, 0, (cudaStream_t)stream>>>(
                            n_terms, rows, n, w, sp, sn, static_cast<typename Codec::storage_t *>(H), H_scales, means);
                } else {
                    advance_kernel<Codec, S_t><<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
                            n_terms, rows, n, w, sp, sn, static_cast<typename Codec::storage_t *>(H), H_scales);
                }
                return launched();
            });
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    int cu_prony_scale(const HistoryStorage storage,
                       const ptrdiff_t      rows,
                       const ptrdiff_t      n,
                       const real_t         factor,
                       void *const          H,
                       scaling_t *const     H_scales,
                       const prony::Means  &means,
                       void *const          stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const int block = 128;
        const int ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            if constexpr (Codec::mean_deviation) {
                // factor * H is the update with alpha = factor and no increment.
                if (rows / means.n_qp > MD_MAX_COMPONENTS) return SFEM_FAILURE;
                const Weights w = pack(1, &factor, nullptr);
                advance_mean_deviation_kernel<Codec, real_t><<<grid_for(n, MD_BLOCK), MD_BLOCK, 0, (cudaStream_t)stream>>>(
                        1, rows, n, w, nullptr, nullptr, static_cast<typename Codec::storage_t *>(H), H_scales, means);
            } else {
                scale_kernel<Codec><<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
                        rows, n, factor, static_cast<typename Codec::storage_t *>(H), H_scales);
            }
            return launched();
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
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

}  // namespace sfem
