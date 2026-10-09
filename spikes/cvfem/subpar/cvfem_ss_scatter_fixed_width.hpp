#pragma once

// The two fixed-width SoA scatter wrappers, which nothing calls.
//
// These are `_w<N_FIELDS>` spellings of the width-templated semi-structured scatter and shared
// reduction. They were already unreferenced before the staging objects came out of the kernel
// signatures: the only caller of either template is the nodal gradient, which instantiates it at
// NG = 3 because it stopped carrying a volume weight. So they are a second name for an operation
// the tree already performs at the width it performs it.
//
// They are kept rather than deleted because the width is the interesting part. If a pass over
// the full field width appears, this is the spelling it wants, and the comment on
// sscvfem_scatter_element_soa_w records why write and read must agree on the width and therefore
// why it is a template parameter rather than an argument.
//
// This is a separate header from cvfem_sshex8_em.hpp deliberately. That one does not compile as
// written -- CVFEM_ENABLE_SUBPAR_EM is off for exactly that reason -- and putting anything
// buildable inside it would make it unbuildable too. These two compile under
// -DCVFEM_ENABLE_SUBPAR alone.

#include "frontend/ss/cvfem_sshex8_ns.hpp"

static SFEM_INLINE void sscvfem_scatter_element_soa(const int *const SFEM_RESTRICT      slot,
                                                   scalar_t *const SFEM_RESTRICT       stage,
                                                   const int                           nxe,
                                                   const ptrdiff_t                     e,
                                                   const idx_t *const SFEM_RESTRICT    lg,
                                                   const scalar_t *const SFEM_RESTRICT lacc,
                                                   scalar_t *const                     dst[N_FIELDS]) {
    sscvfem_scatter_element_soa_w<N_FIELDS>(slot, stage, nxe, e, lg, lacc, dst);
}

// THE ROW COUNT BECAME A RANGE, and this wrapper was not followed. DESIGN.md: "the threading
// model for atomics free kernels is abstract outside the function and what is passed from
// outside is a range", so the sweep lost its `n_shared` bound and gained a cvfem_range; the
// wrapper kept calling it with the old argument list and stopped compiling. It is quarantined,
// so only -DCVFEM_ENABLE_SUBPAR builds it and the break was invisible.
//
// The wrapper keeps its own signature -- n_shared, not a range -- because its whole purpose is
// to present the fixed-width reduction the way it looked before the width was a template
// parameter. It owns the parallel region for the same reason every other launcher does.
inline void sscvfem_reduce_shared_soa(const ptrdiff_t *const SFEM_RESTRICT red_idx,
                                      const ptrdiff_t *const SFEM_RESTRICT red_ptr,
                                      const idx_t *const SFEM_RESTRICT     shared_node,
                                      scalar_t *const SFEM_RESTRICT        stage,
                                      const ptrdiff_t                      n_shared,
                                      scalar_t *const                      dst[N_FIELDS]) {
#pragma omp parallel
    sscvfem_reduce_shared_soa_w<N_FIELDS>(
            cvfem_range_split(0, n_shared, 1, cvfem_thread_index(), cvfem_n_threads()),
            red_idx, red_ptr, shared_node, stage, dst);
}
