#ifndef MG_BUILDER_H
#define MG_BUILDER_H

#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_Multigrid.hpp"

std::shared_ptr<sfem::Multigrid<real_t>> builder(
        const real_t coarsening_factor, const std::shared_ptr<sfem::Buffer<mask_t>> bdy_dofs,
        std::shared_ptr<sfem::Buffer<real_t>> near_null,
        std::shared_ptr<sfem::CooSymSpMV<idx_t, real_t>> &fine_mat);
#endif  // MG_BUILDER_H
