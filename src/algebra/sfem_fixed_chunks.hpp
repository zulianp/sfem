#ifndef SFEM_FIXED_CHUNKS_HPP
#define SFEM_FIXED_CHUNKS_HPP

#include <cstddef>

// The chunks of the deterministic reductions (TeamReduction: OpenMPBLAS's dot, the MPRGP passes):
// partial sums over chunks fixed by the length, combined in index order.

namespace sfem {

    // The chunk count is a function of the length alone and never of the thread count -- that is
    // what makes the result identical however many threads run it. It grows with the length so a
    // chunk stays around 8192 elements, which keeps each chunk's serial error bounded rather than
    // letting it grow with the problem. The result is partially pairwise and so more accurate
    // than a plain serial sum, which is why a deterministic single-threaded solve can converge
    // where a non-deterministic one stagnates.
    constexpr ptrdiff_t blas_max_chunk_count = 65536;

    // Below this many entries a reduction outside a solver's region is summed by the calling
    // thread, chunk by chunk (the same result): a thread team costs more than the arithmetic,
    // which is the same effect that made small multigrid levels slower on more cores.
    constexpr ptrdiff_t blas_serial_below = 4096;

    // Team chunks start at multiples of team_chunk_align entries: a whole number of blocks for
    // blocks of 1, 2, 3, 4 and 6 entries, so a thread's share is whole blocks whatever they are.
    constexpr ptrdiff_t team_chunk_align = 12;

    // The chunks of n entries a Team splits: as blas_chunk_count, but down to 48 entries a chunk,
    // since inside a region a chunk needs no team of its own and the threads need chunks to share.
    inline ptrdiff_t team_chunk_count(const ptrdiff_t n) {
        ptrdiff_t nc = n / 48;
        if (nc < 1) nc = 1;
        if (nc > 256) nc = 256;
        while (nc < blas_max_chunk_count && n / nc > 8192) nc *= 2;
        return nc;
    }

    // The entries [*b, *e) of chunk c of nc over n.
    inline void blas_chunk_bounds(const ptrdiff_t n, const ptrdiff_t nc, const ptrdiff_t c, ptrdiff_t *const b, ptrdiff_t *const e) {
        const ptrdiff_t q = n / nc, rem = n % nc;
        *b = c * q + (c < rem ? c : rem);
        *e = *b + q + (c < rem ? 1 : 0);
    }

}  // namespace sfem

#endif  // SFEM_FIXED_CHUNKS_HPP
