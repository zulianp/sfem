#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "sfem_HistoryStorage.hpp"
#include "sfem_PackedHistory.hpp"
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
    /// A grid store (`is_grid(storage)`) codes each tensor on one power-of-two grid
    /// whose step follows `tolerance` times the tensor's running maximum; its
    /// "scales" are those two numbers per tensor (`scales()`).  A packed store
    /// (`is_packed(storage)`) codes on the same grid and packs the codes per block of
    /// elements (sfem_PackedHistory.hpp): its tensors are read through `packed()` and written
    /// through `write_packed`, and their code areas change size from one write to the next.
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
                        const int            elements_per_mean = 0,
                        const real_t         tolerance         = 0,
                        void *const          stream            = nullptr);

        static constexpr int DEFAULT_ELEMENTS_PER_MEAN = 16;

        int            n_tensors() const { return n_tensors_; }
        int            n_components() const { return n_components_; }
        int            n_qp() const { return n_qp_; }
        ptrdiff_t      n_elements() const { return n_elements_; }
        HistoryStorage storage() const { return storage_; }
        ExecutionSpace execution_space() const { return es_; }

        /// The stream the store's device work is ordered in, and its persistent buffers returned
        /// in: the new one's work waits for the old one's.
        void set_stream(void *const stream);

        /// Values per element and tensor, n_components * n_qp.
        ptrdiff_t values_per_tensor() const { return static_cast<ptrdiff_t>(n_components_) * n_qp_; }

        /// The mean-deviation layout, all of whose tensors' means start at
        /// `means_view().data`; empty for the other formats.
        prony::Means       means_view() const;
        int                elements_per_mean() const { return elements_per_mean_; }

        /// Bytes held, scales and means included.
        std::size_t bytes() const;

        /// The bytes the integer codes would take packed as a quant_block store packs its own:
        /// blocks of prony::PACKED_BLOCK consecutive elements of a row, each its header (base,
        /// width, offset) and its codes' offsets from the block's least at the width of the
        /// largest, rounded to whole 64-bit words (prony::packed_block_bytes); scales and means
        /// included. A quant_block store's own bytes(); for the other formats a host store of an
        /// integer format only, 0 otherwise.
        std::size_t packed_bytes() const;

        /// The grid format's tolerance, relative to each tensor's running maximum.
        real_t tolerance() const { return tolerance_; }

        /// Tensor `t` of a packed store, to read.
        prony::PackedTensor packed(const int t) const;

        /// Write tensor `t` of a host packed store from the values z(r, e), r a row
        /// (component and point) and e an element, on a step from the tolerance and the tensor's
        /// running maximum, rounded as `rounding` says (stochastic: from `seed`, the values
        /// numbered from `k0`).  z may read the tensor through `packed(t)`: its old codes stay
        /// until the new ones are written.
        template <class Z>
        void write_packed(const int t, Z &&z, const HistoryRounding rounding = HistoryRounding::NEAREST, const uint64_t seed = 0, const uint64_t k0 = 0);

        /// The same on a device packed store, for the Prony update H <- alpha H + increment
        /// (sfem_prony_kernels_cuda.hpp, PackedUpdate): the stresses of `stress_bytes` bytes a
        /// value, in device memory, R (`ref`) for a remainder.
        void write_packed_device(const int             t,
                                 const real_t          alpha,
                                 const real_t          b,
                                 const real_t          beta,
                                 const int             stress_bytes,
                                 const void *const     S_n,
                                 const void *const     S_new,
                                 const void *const     ref,
                                 const HistoryRounding rounding,
                                 const uint64_t        seed,
                                 const uint64_t        k0,
                                 void *const           stream);

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
        /// Present for the scaled formats (`is_scaled(storage)`); for a grid store the step and
        /// running maximum of tensor t at [2 t] and [2 t + 1]; null for the others.
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
        real_t                            tolerance_{0};
        ptrdiff_t                         n_groups_{0};
        //! [tensor][component][group], empty unless mean-deviation.
        std::shared_ptr<Buffer<history_mean_t>> means_;
        //! Per tensor of a packed store: the code area and the block headers.
        struct Packed {
            std::shared_ptr<Buffer<uint8_t>>  codes;
            std::shared_ptr<Buffer<int32_t>>  base;
            std::shared_ptr<Buffer<uint8_t>>  width;
            std::shared_ptr<Buffer<uint32_t>> offset;
        };
        std::vector<Packed> packed_;
        //! The store's stream, shared with its persistent buffers' deleters (set_stream).
        std::shared_ptr<void *> stream_;
        void                reset_packed(const int t, void *const stream);
    };

    template <class Z>
    void QuadratureState::write_packed(const int t, Z &&z, const HistoryRounding rounding, const uint64_t seed, const uint64_t k0) {
        if (!is_packed(storage_) || es_ != EXECUTION_SPACE_HOST) SFEM_ERROR("QuadratureState::write_packed: a host packed store only\n");
        const ptrdiff_t rows = values_per_tensor(), n = n_elements_;
        // The running maximum and its step: the step coarsens only when the maximum crosses a
        // power of two, so values that do not change keep their codes.
        real_t max_abs = scales_->data()[2 * t + 1];
#pragma omp parallel for collapse(2) reduction(max : max_abs)
        for (ptrdiff_t r = 0; r < rows; r++) {
            for (ptrdiff_t e = 0; e < n; e++) max_abs = std::max(max_abs, std::abs(z(r, e)));
        }
        const real_t step       = prony::grid_step(max_abs, tolerance_);
        const bool   stochastic = rounding == HistoryRounding::STOCHASTIC;
        auto         layout     = prony::plan_packed(rows, n, step, z, stochastic, seed, k0);
        auto         codes      = create_host_buffer<uint8_t>(static_cast<ptrdiff_t>(layout.bytes));
        std::memset(codes->data(), 0, layout.bytes);
        prony::pack_packed(rows, n, step, z, stochastic, seed, k0, layout, codes->data());
        Packed &p = packed_[t];
        p.codes   = codes;
        std::copy(layout.base.begin(), layout.base.end(), p.base->data());
        std::copy(layout.width.begin(), layout.width.end(), p.width->data());
        std::copy(layout.offset.begin(), layout.offset.end(), p.offset->data());
        scales_->data()[2 * t]     = static_cast<scaling_t>(step);
        scales_->data()[2 * t + 1] = prony::grid_max(max_abs);
    }

}  // namespace sfem
