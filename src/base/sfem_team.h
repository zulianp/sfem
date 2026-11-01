#ifndef SFEM_TEAM_H
#define SFEM_TEAM_H

#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// The threads of a parallel region as a kernel sees them: the calling thread, the region's thread
// count and how many of the first threads take no share (under MPI thread 0 keeps the messages
// moving). A kernel of the region takes its contiguous share of whatever it loops over (elements,
// packs, rows) with sfem_team_share and waits for the others with sfem_team_barrier; every thread
// of the region calls it. Plain data, so C kernels take it by pointer; sfem::Team (sfem_Team.hpp)
// is the C++ side, with the chunks of the deterministic sums.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int tid;
    int threads;
    int reserved;
} sfem_team_t;

/// The calling thread, inside a region (thread 0 of 1 outside one); with `reserve_first` and more
/// than one thread, thread 0 takes no share.
static inline sfem_team_t sfem_team_current(const int reserve_first) {
    sfem_team_t t;
#ifdef _OPENMP
    t.tid     = omp_get_thread_num();
    t.threads = omp_get_num_threads();
#else
    t.tid     = 0;
    t.threads = 1;
#endif
    t.reserved = reserve_first && t.threads > 1 ? 1 : 0;
    return t;
}

/// The calling thread's share [*b, *e) of n items cut at multiples of `align`: contiguous, the
/// first n / align % workers threads one unit more, as a static schedule assigns them, so a loop
/// over the share takes the iterations `omp for schedule(static)` would. `all` counts the reserved
/// threads in (for work after their messages are done).
static inline void sfem_team_share(const sfem_team_t *const team,
                                   const ptrdiff_t          n,
                                   const ptrdiff_t          align,
                                   const int                all,
                                   ptrdiff_t *const         b,
                                   ptrdiff_t *const         e) {
    const int       reserved = all ? 0 : team->reserved;
    const ptrdiff_t workers  = team->threads - reserved;
    const ptrdiff_t w        = team->tid - reserved;
    if (w < 0 || n <= 0) {
        *b = *e = 0;
        return;
    }
    const ptrdiff_t units = (n + align - 1) / align;
    const ptrdiff_t q = units / workers, rem = units % workers;
    const ptrdiff_t ub = w * q + (w < rem ? w : rem);
    const ptrdiff_t ue = ub + q + (w < rem ? 1 : 0);
    *b                 = ub * align < n ? ub * align : n;
    *e                 = ue * align < n ? ue * align : n;
}

/// Every thread of the region waits here.
static inline void sfem_team_barrier(void) {
#pragma omp barrier
}

#ifdef __cplusplus
}
#endif

#endif  // SFEM_TEAM_H
