#ifndef SFEM_TEAM_HPP
#define SFEM_TEAM_HPP

#include <cassert>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_fixed_chunks.hpp"
#include "sfem_team.h"

namespace sfem {

    /// The calling thread's share of a parallel region over n entries: whole chunks fixed by n
    /// (team_chunk_count), each starting at a multiple of team_chunk_align, contiguous and as even
    /// as they divide among the threads, so that a sum taken chunk by chunk and combined in index
    /// order does not depend on the number of threads. Every thread of the region builds its own,
    /// from the same n. With `reserve_first` (and more than one thread) thread 0 takes none: it is
    /// left to communicate. The same thread's share of anything else the region loops over
    /// (elements, packs, parts) is share(), the split a static schedule makes.
    struct Team {
        int       tid{0};
        int       threads{1};
        /// The first threads that take no share: 1 with `reserve_first`, else 0.
        int       reserved{0};
        ptrdiff_t n{0};
        ptrdiff_t chunks{1};
        /// This thread's chunks [first_chunk, end_chunk) and their entries [begin, end).
        ptrdiff_t first_chunk{0}, end_chunk{0};
        ptrdiff_t begin{0}, end{0};

        /// Inside the region (one thread, outside one). The region is the outermost: a team of a
        /// nested region would split rows its enclosing region's threads also take.
        static Team current(const ptrdiff_t n, const bool reserve_first = false) {
#ifdef _OPENMP
            assert(omp_get_active_level() <= 1);
#endif
            const sfem_team_t c = sfem_team_current(reserve_first);
            return of(n, c.tid, c.threads, reserve_first);
        }

        /// Thread tid's of `threads`.
        static Team of(const ptrdiff_t n, const int tid, const int threads, const bool reserve_first = false) {
            Team t;
            t.tid      = tid;
            t.threads  = threads;
            t.reserved = reserve_first && threads > 1 ? 1 : 0;
            t.n        = n;
            t.chunks   = team_chunk_count(n);
            t.share(t.chunks, 1, &t.first_chunk, &t.end_chunk);
            t.begin = t.first_chunk < t.end_chunk ? t.chunk_first_entry(t.first_chunk) : 0;
            t.end   = t.first_chunk < t.end_chunk ? t.chunk_first_entry(t.end_chunk) : 0;
            return t;
        }

        /// As the kernels see it.
        sfem_team_t c() const { return sfem_team_t{tid, threads, reserved}; }

        /// The threads that take shares.
        int workers() const { return threads - reserved; }

        /// The thread that speaks for the team: communicates, prints, writes back.
        bool leader() const { return tid == 0; }

        /// This thread's share [*b, *e) of m items cut at multiples of `align` (sfem_team_share);
        /// `all` counts the reserved threads in.
        void share(const ptrdiff_t m, const ptrdiff_t align, ptrdiff_t *const b, ptrdiff_t *const e, const bool all = false) const {
            const sfem_team_t t = c();
            sfem_team_share(&t, m, align, all ? 1 : 0, b, e);
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

        /// A team of one thread has no one to wait for (and may run inside another region's
        /// single, where a barrier is not allowed).
        void barrier() const {
            if (threads > 1) sfem_team_barrier();
        }
    };

}  // namespace sfem

#endif  // SFEM_TEAM_HPP
