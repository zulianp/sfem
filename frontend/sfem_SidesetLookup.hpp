#ifndef SFEM_SIDESET_LOOKUP_HPP
#define SFEM_SIDESET_LOOKUP_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "smesh_path.hpp"
#include "smesh_sideset.hpp"

namespace sfem {

    /// What a configuration's boundary conditions refer to outside themselves: sidesets by the
    /// names the configuration gave them, and the directory its relative paths are read against.
    struct SidesetLookup {
        /// One sideset per mesh block, as `Sideset::create_from_selector` gives them.
        std::map<std::string, std::vector<std::shared_ptr<smesh::Sideset>>> named;
        /// Empty: paths as written.
        std::string base_dir;

        /// The sidesets called `name`, or none.
        std::vector<std::shared_ptr<smesh::Sideset>> find(const std::string &name) const {
            auto it = named.find(name);
            return it == named.end() ? std::vector<std::shared_ptr<smesh::Sideset>>() : it->second;
        }

        std::string resolve(const std::string &path) const {
            if (base_dir.empty() || path.empty() || path[0] == '/') return path;
            return (smesh::Path(base_dir) / path).to_string();
        }
    };

}  // namespace sfem

#endif  // SFEM_SIDESET_LOOKUP_HPP
