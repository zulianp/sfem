#ifndef SFEM_CUDA_BSR_HPP
#define SFEM_CUDA_BSR_HPP

#include "sfem_base.hpp"

#include <memory>

namespace sfem {
    /// A non-blocking stream for device work that must run beside another stream, not behind it
    /// (on the legacy default stream, work waits on everything queued on any blocking stream, MPI's
    /// copies included, and they on it). Forked from and joined into the stream the work belongs
    /// to, so a graph capturing that stream takes this one's work in too.

    class DeviceSideStream {
    public:
        static std::shared_ptr<DeviceSideStream> create();
        ~DeviceSideStream();

        void *stream() const;
        /// This stream waits for the work queued on `from` so far (a cudaStream_t; null: the
        /// default device stream).
        void fork(void *const from);
        /// `into` waits for the work queued on this one so far.
        void join(void *const into);

    private:
        DeviceSideStream();
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Device version of `bsr_extract_diagonal` (sfem_BSR.hpp): all pointers on the device.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_extract_diagonal(const ptrdiff_t                     block_rows,
                               const int                           block_size,
                               const R* const SFEM_RESTRICT        rowptr,
                               const C* const SFEM_RESTRICT        colidx,
                               const TStorage* const SFEM_RESTRICT values,
                               T* const SFEM_RESTRICT              out,
                               void* const                         stream);

    /// y = scale_output y + A x on the device, A in BSR with square blocks, no library: a group of
    /// threads per block row for blocks up to 4x4 (d_bsr_spmv_grouped), one thread per scalar row
    /// otherwise. scale_output 0 overwrites y. On `stream` (a cudaStream_t; null: the
    /// default device stream).
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_native(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rowptr,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y,
                          void* const                         stream);

    /// d_bsr_spmv_ranges with `group` threads per block row (1, 2, 4, 8, 16 or 32; 0: the library's
    /// default), each reading whole blocks, consecutive threads consecutive blocks, their partial sums
    /// added in a fixed order.  Blocks of size 1 to 4; SFEM_FAILURE otherwise.  What
    /// d_bsr_spmv_ranges and d_bsr_spmv_native run for those block sizes.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_grouped(const ptrdiff_t                     block_rows,
                           const int                           block_size,
                           const R* const SFEM_RESTRICT        rows_begin,
                           const R* const SFEM_RESTRICT        rows_end,
                           const C* const SFEM_RESTRICT        colidx,
                           const TStorage* const SFEM_RESTRICT values,
                           const T                             scale_output,
                           const T* const SFEM_RESTRICT        x,
                           T* const SFEM_RESTRICT              y,
                           const int                           group,
                           void* const                         stream);

    /// d_bsr_spmv_native with block row i the entries [rows_begin[i], rows_end[i]) (see
    /// bsr_spmv_ranges): the whole matrix with (rowptr, rowptr + 1), or a part of each row.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_ranges(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rows_begin,
                          const R* const SFEM_RESTRICT        rows_end,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y,
                          void* const                         stream);

    /// Device version of `bsr_max_abs_row_sum`: max over the scalar rows of sum_j |A_ij|, of
    /// S A S with `scale` (on the device, null: none). On `stream` (a cudaStream_t; null: the
    /// default device stream), as the helpers below.
    template <typename R, typename C, typename TStorage, typename T>
    T d_bsr_max_abs_row_sum(const ptrdiff_t                     block_rows,
                            const int                           block_size,
                            const R* const SFEM_RESTRICT        rowptr,
                            const C* const SFEM_RESTRICT        colidx,
                            const TStorage* const SFEM_RESTRICT values,
                            const T* const SFEM_RESTRICT        scale,
                            void* const                         stream);

    /// Device version of `bsr_shift_diagonal`: A_dd += factor |reference_d|.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_shift_diagonal(const ptrdiff_t               block_rows,
                             const int                     block_size,
                             const R* const SFEM_RESTRICT  rowptr,
                             const C* const SFEM_RESTRICT  colidx,
                             const T                       factor,
                             const T* const SFEM_RESTRICT  reference,
                             TStorage* const SFEM_RESTRICT values,
                             void* const                   stream);
}  // namespace sfem

#endif  // SFEM_CUDA_BSR_HPP
