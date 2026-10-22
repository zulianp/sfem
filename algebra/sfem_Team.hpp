#ifndef SFEM_TEAM_HPP
#define SFEM_TEAM_HPP

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_fixed_chunks.hpp"

namespace sfem {

    /// The calling thread's share of a parallel region over n entries: whole chunks fixed by n
    /// (team_chunk_count), each starting at a multiple of team_chunk_align, contiguous and as even
    /// as they divide among the threads, so that a sum taken chunk by chunk and combined in index
    /// order does not depend on the number of threads. Every thread of the region builds its own,
    /// from the same n. With `reserve_first` (and more than one thread) thread 0 takes none: it is
    /// left to communicate.
    struct Team {
        int       tid{0};
        int       threads{1};
        ptrdiff_t n{0};
        ptrdiff_t chunks{1};
        /// This thread's chunks [first_chunk, end_chunk) and their entries [begin, end).
        ptrdiff_t first_chunk{0}, end_chunk{0};
        ptrdiff_t begin{0}, end{0};

        /// Inside the region (one thread, outside one).
        static Team current(const ptrdiff_t n, const bool reserve_first = false) {
            int tid = 0, threads = 1;
#ifdef _OPENMP
            tid     = omp_get_thread_num();
            threads = omp_get_num_threads();
#endif
            return of(n, tid, threads, reserve_first);
        }

        /// Thread tid's of `threads`.
        static Team of(const ptrdiff_t n, const int tid, const int threads, const bool reserve_first = false) {
            Team t;
            t.tid      = tid;
            t.threads  = threads;
            t.n        = n;
            t.chunks   = team_chunk_count(n);
            const int reserved = reserve_first && threads > 1 ? 1 : 0;
            if (tid < reserved) {
                t.first_chunk = t.end_chunk = 0;
            } else {
                blas_chunk_bounds(t.chunks, threads - reserved, tid - reserved, &t.first_chunk, &t.end_chunk);
            }
            t.begin = t.first_chunk < t.end_chunk ? t.chunk_first_entry(t.first_chunk) : 0;
            t.end   = t.first_chunk < t.end_chunk ? t.chunk_first_entry(t.end_chunk) : 0;
            return t;
        }

        /// The first entry of chunk c (c = chunks: n).
        ptrdiff_t chunk_first_entry(const ptrdiff_t c) const {
            if (c == chunks) return n;
            const ptrdiff_t units = (n + team_chunk_align - 1) / team_chunk_align;
            ptrdiff_t       b, e;
            blas_chunk_bounds(units, chunks, c, &b, &e);
            return b * team_chunk_align < n ? b * team_chunk_align : n;
        }

        /// The entries [*b, *e) of chunk c.
        void chunk_entries(const ptrdiff_t c, ptrdiff_t *const b, ptrdiff_t *const e) const {
            *b = chunk_first_entry(c);
            *e = chunk_first_entry(c + 1);
        }

        void barrier() const {
#pragma omp barrier
        }
    };

}  // namespace sfem

#endif  // SFEM_TEAM_HPP
