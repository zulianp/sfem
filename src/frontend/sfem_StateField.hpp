#pragma once

#include <cstddef>
#include <string>

#include "sfem_aliases.hpp"
#include "sfem_defs.hpp"

namespace sfem {

    // Node i reads entry file_nodes[i] of the file when file_nodes is given (a mesh holding part of
    // the file's nodes, e.g. read by blocks or renumbered by packing), else entry i, and the file must
    // then hold exactly the entries read. The interleaved file holds block_size values per node.
    int read_state_field(const std::string &path, ptrdiff_t ndofs, real_t *out, int block_size = 1,
                         const idx_t *file_nodes = nullptr);

    // One file per component, read as above.
    int read_state_field_components(const std::string &paths, ptrdiff_t n_nodes, int block_size, real_t *out,
                                    const idx_t *file_nodes = nullptr);

}  // namespace sfem
