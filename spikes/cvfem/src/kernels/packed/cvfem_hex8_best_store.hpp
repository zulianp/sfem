#ifndef CVFEM_HEX8_BEST_STORE_HPP
#define CVFEM_HEX8_BEST_STORE_HPP

// Store layout: a packed assembly whose pack-local matrix has its *owned* rows
// laid out in the global sparsity pattern. A pack's owned block is then a
// contiguous slice of the global BSR values and is flushed with one streaming
// memcpy, so every global block is written exactly once: no zero_bsr4 pass and no
// read-modify-write. Only the ghost rows still need a reduction.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/cvfem_phases.hpp"
#include "kernels/cvfem_range.hpp"
#include "kernels/packed/cvfem_hex8_pack_staging.hpp"

// Build the "store" layout. Owned rows of a pack map 1:1 onto the contiguous
// global slice [rowptr_g[owned], rowptr_g[owned + n_contiguous]), so assembling
// a pack ends in one memcpy that writes every one of those blocks exactly once.




#endif  // CVFEM_HEX8_BEST_STORE_HPP
