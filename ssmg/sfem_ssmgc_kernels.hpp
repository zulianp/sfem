#ifndef SFEM_SSMGC_KERNELS_HPP
#define SFEM_SSMGC_KERNELS_HPP

#include "sfem_base.hpp"

namespace sfem {
    void pack_nodal_diag_to_block_sym6_device(const ptrdiff_t     n_nodes,
                                              const real_t *const d3,
                                              real_t *const       d6);
}

#endif
