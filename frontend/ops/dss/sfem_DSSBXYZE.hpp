#pragma once

#include "sfem_aliases.hpp"

namespace sfem {

struct DSSBXYZEBuffer {
    int                 level{0};
    int                 element_block_size{0};
    ptrdiff_t           nelements{0};
    ptrdiff_t           nvalues{0};
    SharedBuffer<real_t> values;
};

int dss_global_to_bxyze(const SharedMesh &mesh,
                        int element_block_size,
                        const real_t *const global,
                        DSSBXYZEBuffer *const out);

}  // namespace sfem
