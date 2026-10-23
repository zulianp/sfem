#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "sfem_HistoryStorage.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    /// Tensors stored at the integration points of a block of elements.
    ///
    /// `n_tensors` tensors of `n_components` each, at `n_qp` points of
    /// `n_elements` elements, in the SoA order
    ///
    ///     value(t, c, q, e) = data[((t * n_components + c) * n_qp + q) * n_elements + e]
    ///
    /// so every (tensor, component, point) is one contiguous run over the
    /// elements and a lane loop indexes it with the element alone.  The
    /// integration-point order is whatever the kernels that produce and consume
    /// the state use; the store never interprets it.
    ///
    /// Generic on purpose: the Prony history is its first user, and any other
    /// integration-point internal variable can be held the same way.
    ///
    /// The store lives in `es`, where the kernels that read and write it run.
    /// `encode` and `decode` are host conveniences for setting and inspecting a
    /// host store; the device loops go through the codecs directly.
    class QuadratureState {
    public:
        QuadratureState(const int            n_tensors,
                        const int            n_components,
                        const int            n_qp,
                        const ptrdiff_t      n_elements,
                        const HistoryStorage storage = HistoryStorage::FULL,
                        const ExecutionSpace es      = EXECUTION_SPACE_HOST);

        int            n_tensors() const { return n_tensors_; }
        int            n_components() const { return n_components_; }
        int            n_qp() const { return n_qp_; }
        ptrdiff_t      n_elements() const { return n_elements_; }
        HistoryStorage storage() const { return storage_; }
        ExecutionSpace execution_space() const { return es_; }

        /// Values per element and tensor, n_components * n_qp.
        ptrdiff_t values_per_tensor() const { return static_cast<ptrdiff_t>(n_components_) * n_qp_; }

        /// Bytes held, scales included.
        std::size_t bytes() const;

        /// Zero every value; scales go to one.  On the device, in `stream`.
        void zero(void *const stream = nullptr);

        /// Zero tensor `t` alone.
        void zero_tensor(const int t, void *const stream = nullptr);

        /// Copy tensor `from` of `src`, values and scales as stored, into
        /// tensor `to` of this store.  Both stores must have one shape per
        /// tensor and one storage format; `src` may be this store.
        void copy_tensor(const QuadratureState &src, const int from, const int to, void *const stream = nullptr);

        /// Multiply tensor `t` by `factor`, re-encoding it (and recomputing its
        /// scales) in the store's format.
        void scale_tensor(const int t, const real_t factor, void *const stream = nullptr);

        /// Encode tensor `t` from the real_t array `values` (laid out
        /// [c][q][e]), recomputing its scales.
        void encode(const int t, const real_t *const values);

        /// Decode tensor `t` into `values` (laid out [c][q][e]).
        void decode(const int t, real_t *const values) const;

        /// Typed access for the loops that fuse decoding into their arithmetic.
        /// `Codec::storage_t` must match `storage()`; `dispatch_history_codec`
        /// is how a caller guarantees it.
        template <class Codec>
        typename Codec::storage_t *data() {
            return reinterpret_cast<typename Codec::storage_t *>(data_->data());
        }

        template <class Codec>
        const typename Codec::storage_t *data() const {
            return reinterpret_cast<const typename Codec::storage_t *>(data_->data());
        }

        /// The store untyped, for code that dispatches on `storage()` itself.
        void       *raw() { return data_->data(); }
        const void *raw() const { return data_->data(); }

        /// One scale per (tensor, element): scale(t, e) = scales[t * n_elements + e].
        /// Present for every format, and all ones for the unscaled ones.
        scaling_t       *scales() { return scales_->data(); }
        const scaling_t *scales() const { return scales_->data(); }

    private:
        int                               n_tensors_;
        int                               n_components_;
        int                               n_qp_;
        ptrdiff_t                         n_elements_;
        HistoryStorage                    storage_;
        ExecutionSpace                    es_;
        std::shared_ptr<Buffer<uint8_t>>  data_;
        std::shared_ptr<Buffer<scaling_t>> scales_;
    };

}  // namespace sfem
