#include "sfem_StateField.hpp"

#include <cstring>
#include <sstream>
#include <vector>

#include "sfem_base.hpp"
#include "sfem_logger.hpp"
#include "smesh_buffer.hpp"
#include "smesh_path.hpp"

namespace sfem {

    int read_state_field(const std::string &path, const ptrdiff_t ndofs, real_t *const out) {
        auto values = Buffer<real_t>::from_file(smesh::Path(path));
        if (!values || static_cast<ptrdiff_t>(values->size()) != ndofs) {
            SFEM_ERROR("State field %s has %td entries; expected %td\n",
                       path.c_str(),
                       values ? static_cast<ptrdiff_t>(values->size()) : ptrdiff_t(0),
                       ndofs);
            return SFEM_FAILURE;
        }

        std::memcpy(out, values->data(), static_cast<size_t>(ndofs) * sizeof(real_t));
        return SFEM_SUCCESS;
    }

    int read_state_field_components(const std::string &paths, const ptrdiff_t n_nodes, const int block_size, real_t *const out) {
        std::vector<std::string> components;
        std::istringstream       stream(paths);
        std::string              path;
        while (std::getline(stream, path, ',')) {
            const size_t begin = path.find_first_not_of(" \t");
            const size_t end   = path.find_last_not_of(" \t");
            if (begin != std::string::npos) components.push_back(path.substr(begin, end - begin + 1));
        }

        if (static_cast<int>(components.size()) != block_size) {
            SFEM_ERROR("State component list has %zu paths; expected %d\n", components.size(), block_size);
            return SFEM_FAILURE;
        }

        for (int d = 0; d < block_size; ++d) {
            auto values = Buffer<real_t>::from_file(smesh::Path(components[d]));
            if (!values || static_cast<ptrdiff_t>(values->size()) != n_nodes) {
                SFEM_ERROR("State component %s has %td entries; expected %td\n",
                           components[d].c_str(),
                           values ? static_cast<ptrdiff_t>(values->size()) : ptrdiff_t(0),
                           n_nodes);
                return SFEM_FAILURE;
            }

            const real_t *const input = values->data();
#pragma omp parallel for
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                out[node * block_size + d] = input[node];
            }
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
