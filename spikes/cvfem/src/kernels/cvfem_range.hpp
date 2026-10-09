#ifndef CVFEM_RANGE_HPP
#define CVFEM_RANGE_HPP

#include <cstddef>

// THE ONLY THING A KERNEL IS TOLD ABOUT THE THREADING.
//
// DESIGN.md: "The threading model for atomics free kernels is abstract outside the function and
// what is passed from outside is a range ... (this will allow to use other thread libraries other
// than OpenMP)". A sweep that owns its own `#pragma omp parallel` cannot be driven by anything
// else, and it also cannot be called for part of the work -- which is what a colour, a halo pass
// or a task-based scheduler needs.
//
// A plain struct of two offsets, deliberately: no iterator, no accessor, nothing that would make
// the kernels depend on a library to say "these elements".
typedef struct {
    ptrdiff_t begin;
    ptrdiff_t end;
} cvfem_range;

// The three things a launcher needs from its thread library, and nothing more. DESIGN.md allows
// src/kernels/ an OpenMP wrapper, and this is the whole of it: swapping the backend means
// changing these three functions, not the sweeps.
//
// These are deliberately NOT threads_active() from the staging layer, which reports
// omp_get_max_threads() -- the size of the team that WOULD be created. A launcher is already
// inside its parallel region and needs the size of the team that WAS.
#ifdef _OPENMP
#include <omp.h>
static inline int  cvfem_n_threads() { return omp_get_num_threads(); }
static inline int  cvfem_thread_index() { return omp_get_thread_num(); }
#define cvfem_thread_barrier() _Pragma("omp barrier")
#else
static inline int  cvfem_n_threads() { return 1; }
static inline int  cvfem_thread_index() { return 0; }
#define cvfem_thread_barrier() ((void)0)
#endif

// An equal split of [begin, end) across n_parts, with every boundary a multiple of `stride` away
// from `begin`.
//
// The alignment is not cosmetic. The sweeps step in lane groups of CVFEM_HEX8_VEC_SIZE starting
// at `begin`, and a part boundary that fell mid-group would either process a group twice or skip
// its tail. Aligning the cuts to the group lets every part clamp its last group against its own
// `end`, which is what the sweeps do, and the final part absorbs the remainder.
//
// This reproduces `#pragma omp for schedule(static)` over the same strided loop, which is what
// the sweeps used to carry, so the work per thread is unchanged.
static inline cvfem_range cvfem_range_split(const ptrdiff_t begin,
                                            const ptrdiff_t end,
                                            const ptrdiff_t stride,
                                            const int       part,
                                            const int       n_parts) {
    const ptrdiff_t n_groups   = (end - begin + stride - 1) / stride;
    const ptrdiff_t per_part   = n_groups / (ptrdiff_t)n_parts;
    const ptrdiff_t remainder  = n_groups % (ptrdiff_t)n_parts;
    const ptrdiff_t first      = (ptrdiff_t)part * per_part + (part < remainder ? part : remainder);
    const ptrdiff_t n_mine     = per_part + (part < remainder ? 1 : 0);
    cvfem_range     r;
    r.begin = begin + first * stride;
    r.end   = r.begin + n_mine * stride;
    if (r.end > end) r.end = end;
    if (r.begin > end) r.begin = r.end = end;
    return r;
}

#endif  // CVFEM_RANGE_HPP
