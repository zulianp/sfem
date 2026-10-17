#include "sfem_ElasticityCase.hpp"

#ifdef SFEM_ENABLE_RYAML

#include <cstdio>
#include <fstream>
#include <list>
#include <sstream>

#include "smesh_path.hpp"

namespace sfem {

    namespace {
        /// Every key a case may hold, as dotted paths; "[]" stands for the items of a list, and a
        /// trailing ".*" hands the whole subtree to the reader that owns it.
        const char *const SCHEMA[] = {
                "execution_space",
                "mesh.path",
                "mesh.box.n",
                "mesh.box.min",
                "mesh.box.max",
                "mesh.reorder",
                "mesh.promote_to_p2",
                "mesh.refine",
                "mesh.basis",
                "mesh.packed",
                "mesh.renumber",
                "mesh.sidesets[].name",
                "mesh.sidesets[].plane.axis",
                "mesh.sidesets[].plane.at",
                "mesh.sidesets[].plane.tol",
                "mesh.sidesets[].file",
                "material.*",
                "loads.body_force.density",
                "loads.body_force.g",
                "loads.neumann.*",
                "loads.profile.*",
                "dirichlet_conditions.*",
                "constraints[].type",
                "constraints[].sideset",
                "constraints[].axis",
                "constraints[].center",
                "constraints[].angle",
                "constraints[].profile.*",
                "constraints[].release",
                "constraints[].control.torque",
                "constraints[].control.tol",
                "constraints[].control.max_it",
                "fields[].name",
                "fields[].file",
                "fields[].values_per_element",
                "fields[].inclusion.a11",
                "fields[].inclusion.radius",
                "fields[].inclusion.center",
                "fields[].profile.*",
                "initial_conditions.displacement.file",
                "initial_conditions.displacement.components",
                "initial_conditions.velocity.file",
                "initial_conditions.velocity.components",
                "initial_conditions.acceleration.file",
                "initial_conditions.acceleration.components",
                "obstacle.backend",
                "obstacle.nodes",
                "obstacle.direction",
                "obstacle.sdf.file",
                "obstacle.shape.type",
                "obstacle.shape.plane",
                "obstacle.shape.slope",
                "obstacle.shape.slope_axis",
                "obstacle.shape.step_width",
                "obstacle.shape.step_drop",
                "obstacle.shape.radius",
                "obstacle.shape.center",
                "obstacle.shape.fillet",
                "obstacle.shape.grid_cells",
                "obstacle.qp.max_it",
                "obstacle.qp.rtol",
                "regime",
                "time.dt",
                "time.t_end",
                "time.steps",
                "time.max_cutbacks",
                "dynamics.scheme",
                "dynamics.density",
                "dynamics.mass",
                "dynamics.beta",
                "dynamics.gamma",
                "solver.newton.max_it",
                "solver.newton.atol",
                "solver.newton.rtol",
                "solver.newton.divergence_factor",
                "solver.newton.merit",
                "solver.newton.verbose",
                "solver.newton.line_search.enabled",
                "solver.newton.line_search.steps",
                "solver.newton.line_search.c1",
                "solver.newton.line_search.batched",
                "solver.linear.type",
                "solver.linear.format",
                "solver.linear.preconditioner",
                "solver.linear.rtol",
                "solver.linear.atol",
                "solver.linear.max_it",
                "solver.linear.storage_bytes",
                "output.path",
                "output.export_freq",
                "output.history_csv",
                "output.fields",
                "output.quantities",
                "output.probes[].type",
                "output.probes[].at",
                "output.probes[].rates",
                "output.probes[].constraint",
        };

        bool starts_with(const std::string &s, const std::string &prefix) { return s.compare(0, prefix.size(), prefix) == 0; }

        enum class Match { NONE, PREFIX, LEAF, SUBTREE };

        Match match(const std::string &path) {
            Match best = Match::NONE;
            for (const char *const entry : SCHEMA) {
                std::string pattern(entry);
                const bool  subtree = pattern.size() > 2 && pattern.compare(pattern.size() - 2, 2, ".*") == 0;
                if (subtree) {
                    pattern.resize(pattern.size() - 2);
                    if (path == pattern || starts_with(path, pattern + ".") || starts_with(path, pattern + "[]")) {
                        return Match::SUBTREE;
                    }
                }
                if (path == pattern) {
                    best = Match::LEAF;
                } else if (best == Match::NONE && (starts_with(pattern, path + ".") || starts_with(pattern, path + "[]"))) {
                    best = Match::PREFIX;
                }
            }
            return best;
        }

        int validate(const ryml::ConstNodeRef &node, const std::string &path) {
            if (!path.empty()) {
                const Match m = match(path);
                if (m == Match::NONE) {
                    fprintf(stderr, "ElasticityCase: unknown key \"%s\"\n", path.c_str());
                    return SFEM_FAILURE;
                }
                if (m == Match::SUBTREE || m == Match::LEAF) return SFEM_SUCCESS;
            }

            if (node.is_map()) {
                for (auto child : node.children()) {
                    const auto        k   = child.key();
                    const std::string key = std::string(k.str, k.len);
                    if (validate(child, path.empty() ? key : path + "." + key) != SFEM_SUCCESS) return SFEM_FAILURE;
                }
                return SFEM_SUCCESS;
            }

            if (node.is_seq()) {
                for (auto child : node.children()) {
                    if (validate(child, path + "[]") != SFEM_SUCCESS) return SFEM_FAILURE;
                }
                return SFEM_SUCCESS;
            }

            fprintf(stderr, "ElasticityCase: \"%s\" needs a value below it\n", path.c_str());
            return SFEM_FAILURE;
        }

        std::vector<std::string> split(const std::string &s, const char sep) {
            std::vector<std::string> parts;
            std::stringstream        ss(s);
            std::string              part;
            while (std::getline(ss, part, sep)) parts.push_back(part);
            return parts;
        }

        bool is_index(const std::string &s) {
            if (s.empty()) return false;
            for (char c : s) {
                if (c < '0' || c > '9') return false;
            }
            return true;
        }

        /// "a.b.c=value", the value YAML: a scalar, or a flow list or map.
        /// `parsed` keeps the trees the values were parsed into: merge_with refers to their
        /// strings rather than copying them.
        int apply_override(ryml::Tree &tree, std::list<ryml::Tree> &parsed_values, const std::string &assignment) {
            const auto eq = assignment.find('=');
            if (eq == std::string::npos || eq == 0) {
                fprintf(stderr, "ElasticityCase: override \"%s\" is not key=value\n", assignment.c_str());
                return SFEM_FAILURE;
            }

            const std::string path  = assignment.substr(0, eq);
            const std::string value = "v: " + assignment.substr(eq + 1);

            parsed_values.push_back(ryml::parse_in_arena(ryml::to_csubstr(value)));
            ryml::Tree &parsed = parsed_values.back();

            ryml::id_type node = tree.root_id();
            for (const auto &key : split(path, '.')) {
                if (tree.is_seq(node)) {
                    if (!is_index(key) || std::stoul(key) >= (size_t)tree.num_children(node)) {
                        fprintf(stderr, "ElasticityCase: override \"%s\": no item %s\n", path.c_str(), key.c_str());
                        return SFEM_FAILURE;
                    }
                    node = tree.child(node, (ryml::id_type)std::stoul(key));
                    continue;
                }

                if (!tree.is_map(node)) {
                    if (tree.has_children(node)) tree.remove_children(node);
                    tree.set_map(node);
                }

                ryml::id_type child = tree.find_child(node, ryml::to_csubstr(key));
                if (child == ryml::NONE) {
                    child = tree.append_child(node);
                    tree.set_key(child, tree.to_arena(ryml::to_csubstr(key)));
                    tree.set_val(child, "");
                }
                node = child;
            }

            // merge_with copies the source's key along with its value; the node keeps its own.
            const bool    keyed = tree.has_key(node);
            const ryml::csubstr key = keyed ? tree.key(node) : ryml::csubstr();
            if (tree.has_children(node)) tree.remove_children(node);
            tree.merge_with(&parsed, parsed["v"].id(), node);
            if (keyed) tree.set_key(node, key);
            return SFEM_SUCCESS;
        }
    }  // namespace

    std::unique_ptr<ElasticityCase> ElasticityCase::read(const std::string &path, const std::vector<std::string> &overrides) {
        std::ifstream is(path);
        if (!is.good()) {
            fprintf(stderr, "ElasticityCase: unable to read %s\n", path.c_str());
            return nullptr;
        }
        std::stringstream contents;
        contents << is.rdbuf();

        const auto  slash    = path.find_last_of('/');
        std::string base_dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
        return parse(contents.str(), base_dir, overrides);
    }

    std::unique_ptr<ElasticityCase> ElasticityCase::parse(const std::string              &yaml,
                                                          const std::string              &base_dir,
                                                          const std::vector<std::string> &overrides) {
        std::unique_ptr<ElasticityCase> ret(new ElasticityCase());
        ret->tree_     = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        ret->base_dir_ = base_dir;

        if (!ret->tree_.rootref().is_map()) {
            if (ret->tree_.rootref().has_children()) {
                fprintf(stderr, "ElasticityCase: the document must be a map\n");
                return nullptr;
            }
            ret->tree_.rootref().set_map();
        }

        for (const auto &o : overrides) {
            if (apply_override(ret->tree_, ret->overrides_, o) != SFEM_SUCCESS) return nullptr;
        }

        if (validate(ret->tree_.crootref(), "") != SFEM_SUCCESS) return nullptr;
        return ret;
    }

    ryml::ConstNodeRef ElasticityCase::root() const { return tree_.crootref(); }
    const std::string &ElasticityCase::base_dir() const { return base_dir_; }

    std::string ElasticityCase::emit() const { return ryml::emitrs_yaml<std::string>(tree_); }

    std::string ElasticityCase::resolve(const std::string &path) const {
        if (path.empty() || path[0] == '/' || base_dir_.empty()) return path;
        return (smesh::Path(base_dir_) / path).to_string();
    }

}  // namespace sfem

#endif  // SFEM_ENABLE_RYAML
