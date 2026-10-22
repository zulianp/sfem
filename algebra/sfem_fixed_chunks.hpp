#ifndef SFEM_FIXED_CHUNKS_HPP
#define SFEM_FIXED_CHUNKS_HPP

#include <cstddef>

// The chunks of the deterministic reductions (OpenMP_BLAS's dot, the MPRGP passes): partial sums
// over chunks fixed by the length, combined in index order.

namespace sfem {

    // Chunk count, a function of the length alone and never of the thread count -- that is
    // what makes the result identical however many threads run it.
    //
    // Below a few thousand elements the range is summed serially: a thread team costs more
    // than the arithmetic, which is the same effect that made small multigrid levels slower
    // on more cores. Above that the count grows with the length so a chunk stays around
    // 8192 elements, which keeps each chunk's serial error bounded rather than letting it
    // grow with the problem. The result is partially pairwise and so more accurate than a
    // plain serial sum, which is why a deterministic single-threaded solve can converge
    // where a non-deterministic one stagnates.
    constexpr ptrdiff_t blas_max_chunk_count = 65536;

    inline ptrdiff_t blas_chunk_count(const ptrdiff_t n) {
        if (n < 4096) return 1;
        ptrdiff_t nc = 256;
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
