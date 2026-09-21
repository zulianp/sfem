#ifndef SFEM_SSMGC_KERNELS_HPP
#define SFEM_SSMGC_KERNELS_HPP

#include "sfem_base.hpp"

namespace sfem {
    /// Pack a nodal AOS diagonal into one symmetric 3x3 block per node.
    ///
    /// The device counterpart of `pack_nodal_diag_to_block_sym6` in
    /// `ssmg/sfem_ssmgc.cpp`, which that function calls when the execution space
    /// is the device.  The call and the declaration were on this branch while
    /// the definition was on no branch at all, so `SFEM_ENABLE_CUDA=ON` did not
    /// build: the header was missing and the symbol was never defined anywhere
    /// in the repository.  The host loop beside the call determines the packing
    /// exactly -- `[xx, xy, xz, yy, yz, zz]` from an AOS `(xx, yy, zz)` -- so
    /// this is that loop and nothing more.
    void pack_nodal_diag_to_block_sym6_device(const ptrdiff_t     n_nodes,
                                              const real_t *const d3,
                                              real_t *const       d6);
}

#endif
