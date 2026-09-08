#pragma once

#include <cstddef>
#include <string>

#include "sfem_aliases.hpp"
#include "sfem_defs.hpp"

namespace sfem {

    int read_state_field(const std::string &path, ptrdiff_t ndofs, real_t *out);

    int read_state_field_components(const std::string &paths, ptrdiff_t n_nodes, int block_size, real_t *out);

}  // namespace sfem
