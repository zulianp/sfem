#ifndef MG_BUILDER_H
#define MG_BUILDER_H

#include "sfem_Multigrid.hpp"
#include "sparse.h"

std::shared_ptr<sfem::Multigrid<real_t>> builder(const real_t coarsening_factor,
                                                 const mask_t *bdy_dofs,
                                                 SymmCOOMatrix *fine_mat);
#endif  // MG_BUILDER_H
