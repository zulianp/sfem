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

#include "ss/cvfem_sshex8_ns.hpp"

static SFEM_INLINE void sscvfem_scatter_element_soa(const int *const SFEM_RESTRICT      slot,
                                                   scalar_t *const SFEM_RESTRICT       stage,
                                                   const int                           nxe,
                                                   const ptrdiff_t                     e,
                                                   const idx_t *const SFEM_RESTRICT    lg,
                                                   const scalar_t *const SFEM_RESTRICT lacc,
                                                   scalar_t *const                     dst[N_FIELDS]) {
    sscvfem_scatter_element_soa_w<N_FIELDS>(slot, stage, nxe, e, lg, lacc, dst);
}

inline void sscvfem_reduce_shared_soa(const ptrdiff_t *const SFEM_RESTRICT red_idx,
                                      const ptrdiff_t *const SFEM_RESTRICT red_ptr,
                                      const idx_t *const SFEM_RESTRICT     shared_node,
                                      scalar_t *const SFEM_RESTRICT        stage,
                                      const ptrdiff_t                      n_shared,
                                      scalar_t *const                      dst[N_FIELDS]) {
    sscvfem_reduce_shared_soa_w<N_FIELDS>(red_idx, red_ptr, shared_node, stage, n_shared, dst);
}
