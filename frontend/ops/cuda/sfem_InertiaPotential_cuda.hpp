#pragma once

#include "sfem_base.hpp"

namespace sfem {

    /// Add `alpha * mass` to the diagonal of a CRS matrix, on the device.
    ///
    /// The only part of `InertiaPotential` that is not a vector operation.
    /// Everything else it does goes through `sfem::blas<real_t>(es)` and is
    /// portable by construction; finding a row's diagonal entry is an indexed
    /// search, so it needs a kernel of its own.
    int cu_inertia_potential_hessian_crs(const ptrdiff_t      ndofs,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         const real_t *const  mass,
                                         const real_t         alpha,
                                         real_t *const        values);

    /// The same for BSR, where the diagonal of the `bs x bs` block on the
    /// diagonal takes one mass entry per component.
    int cu_inertia_potential_hessian_bsr(const ptrdiff_t      n_nodes,
                                         const int            block_size,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         const real_t *const  mass,
                                         const real_t         alpha,
                                         real_t *const        values);

    /// The block-diagonal, upper-triangle-packed format.  `alpha * m * I` on
    /// each node's block, so only the packed diagonal entries are touched.
    int cu_inertia_potential_hessian_block_diag_sym(const ptrdiff_t     n_nodes,
                                                    const int           block_size,
                                                    const real_t *const mass,
                                                    const real_t        alpha,
                                                    real_t *const       values);

}  // namespace sfem
