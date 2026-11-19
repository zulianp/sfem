#ifndef LUMPED_PTPD_H
#define LUMPED_PTPD_H

#include "sfem_base.hpp"

#include <stddef.h>
#include <stdint.h>

namespace sfem {

    /// The row sums of P^T D P (the lumped Galerkin product) of the diagonal D, P in CRS.
    int lumped_galerkin_crs(const ptrdiff_t                    fine_nodes,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT   colidx,
                            const real_t *const SFEM_RESTRICT  values,
                            const real_t *const SFEM_RESTRICT  diagonal,
                            real_t *const SFEM_RESTRICT        coarse);

    /// lumped_galerkin_crs for block_size components per node.
    int lumped_galerkin_crs_vector(const ptrdiff_t                    fine_nodes,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT   colidx,
                                   const real_t *const SFEM_RESTRICT  values,
                                   const int                          block_size,
                                   const real_t *const SFEM_RESTRICT  diagonal,
                                   real_t *const SFEM_RESTRICT        coarse);

}  // namespace sfem

#endif  // LUMPED_PTPD_H
