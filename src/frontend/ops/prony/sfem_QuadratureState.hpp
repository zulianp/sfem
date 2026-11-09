#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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
    /// A mean-deviation store (`is_mean_deviation(storage)`) also holds a mean
    /// per (tensor, component, group): a group is an element's points, or, on
    /// one-point elements, the points of `elements_per_mean` consecutive
    /// elements -- neighbours, when the mesh is ordered along a space-filling
    /// curve.  `elements_per_mean` 0 picks one element for several points and
    /// DEFAULT_ELEMENTS_PER_MEAN for one.
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
                        const HistoryStorage storage           = HistoryStorage::FULL,
                        const ExecutionSpace es                = EXECUTION_SPACE_HOST,
                        const int            elements_per_mean = 0);

        static constexpr int DEFAULT_ELEMENTS_PER_MEAN = 16;

        int            n_tensors() const { return n_tensors_; }
        int            n_components() const { return n_components_; }
        int            n_qp() const { return n_qp_; }
        ptrdiff_t      n_elements() const { return n_elements_; }
        HistoryStorage storage() const { return storage_; }
        ExecutionSpace execution_space() const { return es_; }

        /// Values per element and tensor, n_components * n_qp.
        ptrdiff_t values_per_tensor() const { return static_cast<ptrdiff_t>(n_components_) * n_qp_; }

        /// The mean-deviation layout, all of whose tensors' means start at
        /// `means_view().data`; empty for the other formats.
        prony::Means       means_view() const;
        int                elements_per_mean() const { return elements_per_mean_; }

        /// Bytes held, scales and means included.
        std::size_t bytes() const;

        /// Zero every value; scales go to one.  On the device, in `stream`.
        void zero(void *const stream = nullptr);

        /// Zero tensor `t` alone.
        void zero_tensor(const int t, void *const stream = nullptr);

        /// Copy tensor `from` of `src`, values and scales as stored, into
        /// tensor `to` of this store.  Both stores must have one shape per
        /// tensor and one storage format; `src` may be this store.
        void copy_tensor(const QuadratureState &src, const int from, const int to, void *const stream = nullptr);

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
        /// Present for the scaled formats (`is_scaled(storage)`) only; null for the others.
        scaling_t       *scales() { return scales_ ? scales_->data() : nullptr; }
        const scaling_t *scales() const { return scales_ ? scales_->data() : nullptr; }

        /// The store as it is (values, scales, means) into files `prefix`.values, .scales and
        /// .means, and back into a store of the same shape and format (a restart).
        int write_state(const std::string &prefix) const;
        int read_state(const std::string &prefix);

    private:
        int                               n_tensors_;
        int                               n_components_;
        int                               n_qp_;
        ptrdiff_t                         n_elements_;
        HistoryStorage                    storage_;
        ExecutionSpace                    es_;
        std::shared_ptr<Buffer<uint8_t>>  data_;
        //! Empty unless scaled.
        std::shared_ptr<Buffer<scaling_t>> scales_;
        int                               elements_per_mean_{1};
        ptrdiff_t                         n_groups_{0};
        //! [tensor][component][group], empty unless mean-deviation.
        std::shared_ptr<Buffer<history_mean_t>> means_;
    };

}  // namespace sfem
