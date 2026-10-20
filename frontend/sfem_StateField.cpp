#include "sfem_StateField.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

#include "sfem_base.hpp"
#include "sfem_logger.hpp"
#include "smesh_buffer.hpp"
#include "smesh_path.hpp"

namespace sfem {

    int read_state_field(const std::string &path,
                         const ptrdiff_t    ndofs,
                         real_t *const      out,
                         const int          block_size,
                         const idx_t *const file_nodes) {
        auto            values  = Buffer<real_t>::from_file(smesh::Path(path));
        const ptrdiff_t size    = values ? static_cast<ptrdiff_t>(values->size()) : ptrdiff_t(0);
        const ptrdiff_t n_nodes = ndofs / block_size;
        ptrdiff_t       needed  = ndofs;
        if (file_nodes) {
            needed = 0;
            for (ptrdiff_t node = 0; node < n_nodes; ++node) needed = std::max(needed, ptrdiff_t(file_nodes[node]) + 1);
            needed *= block_size;
        }
        if (!values || (file_nodes ? size < needed : size != needed)) {
            SFEM_ERROR("State field %s has %td entries; expected %s%td\n", path.c_str(), size, file_nodes ? "at least " : "", needed);
            return SFEM_FAILURE;
        }

        if (!file_nodes) {
            std::memcpy(out, values->data(), static_cast<size_t>(ndofs) * sizeof(real_t));
            return SFEM_SUCCESS;
        }

        const real_t *const input = values->data();
#pragma omp parallel for
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            for (int d = 0; d < block_size; ++d) out[node * block_size + d] = input[file_nodes[node] * block_size + d];
        }
        return SFEM_SUCCESS;
    }

    int read_state_field_components(const std::string &paths,
                                    const ptrdiff_t    n_nodes,
                                    const int          block_size,
                                    real_t *const      out,
                                    const idx_t *const file_nodes) {
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
            auto            values = Buffer<real_t>::from_file(smesh::Path(components[d]));
            const ptrdiff_t size   = values ? static_cast<ptrdiff_t>(values->size()) : ptrdiff_t(0);
            ptrdiff_t       needed = n_nodes;
            if (file_nodes) {
                needed = 0;
                for (ptrdiff_t node = 0; node < n_nodes; ++node) needed = std::max(needed, ptrdiff_t(file_nodes[node]) + 1);
            }
            if (!values || (file_nodes ? size < needed : size != needed)) {
                SFEM_ERROR("State component %s has %td entries; expected %s%td\n",
                           components[d].c_str(),
                           size,
                           file_nodes ? "at least " : "",
                           needed);
                return SFEM_FAILURE;
            }

            const real_t *const input = values->data();
#pragma omp parallel for
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                out[node * block_size + d] = input[file_nodes ? file_nodes[node] : node];
            }
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
