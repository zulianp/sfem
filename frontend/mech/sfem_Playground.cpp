#include "sfem_Playground.hpp"

#ifdef SFEM_ENABLE_RYAML

#include <cmath>
#include <cstdio>
#include <fstream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BDF2Scheme.hpp"
#include "sfem_BackwardEulerScheme.hpp"
#include "sfem_ContactStabilisedNewmarkScheme.hpp"
#include "sfem_Function.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_NodeObstacleStep.hpp"
#include "sfem_ObstacleShape.hpp"
#include "sfem_Observers.hpp"
#include "sfem_Rotation.hpp"
#include "sfem_StateField.hpp"
#include "sfem_TorqueControl.hpp"
#include "sfem_UnconstrainedStep.hpp"
#include "smesh_mesh_reorder.hpp"
#include "smesh_sideset.hpp"

namespace sfem {

#ifdef SFEM_ENABLE_CUDA
    std::shared_ptr<Op> to_device(const std::shared_ptr<NeumannConditions> &nc);
#endif

    namespace {
        std::string as_string(const ryml::ConstNodeRef &node) {
            const auto v = node.val();
            return std::string(v.str, v.len);
        }

        template <typename T>
        T read(const ryml::ConstNodeRef &parent, const char *key, const T fallback) {
            if (!parent.readable() || !parent.has_child(ryml::to_csubstr(key))) return fallback;
            T value = fallback;
            parent[ryml::to_csubstr(key)].load(&value);
            return value;
        }

        std::string read_string(const ryml::ConstNodeRef &parent, const char *key, const std::string &fallback) {
            if (!parent.readable() || !parent.has_child(ryml::to_csubstr(key))) return fallback;
            return as_string(parent[ryml::to_csubstr(key)]);
        }

        template <typename T>
        std::vector<T> read_list(const ryml::ConstNodeRef &parent, const char *key, const std::vector<T> &fallback) {
            if (!parent.readable() || !parent.has_child(ryml::to_csubstr(key))) return fallback;
            std::vector<T> values;
            for (auto item : parent[ryml::to_csubstr(key)].children()) {
                T v{};
                item.load(&v);
                values.push_back(v);
            }
            return values;
        }

        bool has(const ryml::ConstNodeRef &parent, const char *key) {
            return parent.readable() && parent.has_child(ryml::to_csubstr(key));
        }

        /// The child, or an empty node that every reader here treats as absent. Indexing a
        /// missing key and then indexing below it is undefined in ryml.
        ryml::ConstNodeRef child(const ryml::ConstNodeRef &parent, const char *key) {
            return has(parent, key) ? parent[ryml::to_csubstr(key)] : ryml::ConstNodeRef();
        }

        /// The active strain of a spherical inclusion, per element: diag(a11, 1, 1) / det inside the
        /// ball, the identity outside, 9 entries per element, row-major.
        void fill_inclusion(const std::shared_ptr<Mesh> &mesh,
                            const real_t                 center[3],
                            const real_t                 radius,
                            const real_t                 a11,
                            real_t *const                Fa) {
            const ptrdiff_t nelements = mesh->n_elements();
            const int       nxe       = mesh->n_nodes_per_element(0);
            auto            elements  = mesh->elements(0)->data();
            auto            points    = mesh->points()->data();
            const real_t    r2        = radius * radius;

#pragma omp parallel for
            for (ptrdiff_t e = 0; e < nelements; ++e) {
                real_t c[3] = {0, 0, 0};
                for (int v = 0; v < nxe; ++v) {
                    for (int d = 0; d < 3; ++d) c[d] += points[d][elements[v][e]];
                }
                real_t d2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const real_t dc = c[d] / nxe - center[d];
                    d2 += dc * dc;
                }
                const real_t a = d2 <= r2 ? a11 : real_t(1);
                real_t *const F = &Fa[9 * e];
                for (int k = 0; k < 9; ++k) F[k] = 0;
                // diag(a, 1, 1) divided by its determinant a.
                F[0] = 1;
                F[4] = 1 / a;
                F[8] = 1 / a;
            }
        }

        /// Whether any condition of the list has a load profile.
        [[maybe_unused]] bool has_profile(const ryml::ConstNodeRef &conditions) {
            for (auto c : conditions.children()) {
                if (c.is_map() && c.has_child("profile")) return true;
            }
            return false;
        }

        struct Box {
            real_t lo[3]{0, 0, 0};
            real_t hi[3]{0, 0, 0};
        };

        /// The mesh's bounding box, reduced across the threads. smesh's
        /// Mesh::compute_bounding_box updates one shared min and max from every thread of a
        /// parallel loop; at 72 threads it returns a different box from run to run (a hemisphere
        /// centred at x = 0.75 instead of 1, its radius 0.1875 instead of 0.25).
        Box bounding_box(const std::shared_ptr<Mesh> &mesh) {
            const int            dim    = mesh->spatial_dimension();
            const geom_t *const *points = mesh->points()->data();
            const ptrdiff_t      n      = mesh->n_nodes();
            Box                  box;
            for (int d = 0; d < dim; ++d) {
                real_t lo = points[d][0], hi = points[d][0];
#pragma omp parallel for reduction(min : lo) reduction(max : hi)
                for (ptrdiff_t i = 0; i < n; ++i) {
                    lo = std::min(lo, (real_t)points[d][i]);
                    hi = std::max(hi, (real_t)points[d][i]);
                }
                box.lo[d] = lo;
                box.hi[d] = hi;
            }
            return box;
        }

        int axis_of(const std::string &name) {
            if (name == "x" || name == "0") return 0;
            if (name == "y" || name == "1") return 1;
            if (name == "z" || name == "2") return 2;
            return -1;
        }
    }  // namespace

    class Playground::Impl {
    public:
        ExecutionSpace                                     es{EXECUTION_SPACE_HOST};
        std::shared_ptr<Mesh>                              mesh;
        std::shared_ptr<FunctionSpace>                     fs;
        std::shared_ptr<Function>                          f;
        std::shared_ptr<Op>                                material;
        std::shared_ptr<TimeScheme>                        scheme;
        std::shared_ptr<NewtonSolver>                      newton;
        std::shared_ptr<NonlinearSolver>                   nonlinear;
        std::unique_ptr<Stepper>                           stepper;
        SharedBuffer<real_t>                               x;
        SidesetLookup                                      lookup;
        bool                                               verbose{true};
        bool                                               packed{true};
        bool                                               renumber{true};
        std::shared_ptr<DirichletConditions>               host_dirichlet;
        std::vector<std::pair<std::shared_ptr<Constraint>, real_t>> releases;
        std::vector<std::shared_ptr<Rotation>>             rotations;
        std::shared_ptr<Rotation>                          torque_controlled;
        TorqueControlOptions                               torque_control;
        std::list<ryml::Tree>                              documents;
        std::shared_ptr<NodeObstacleContact>               contact;
        std::shared_ptr<smesh::Grid<geom_t>>               obstacle_sdf;
        std::shared_ptr<NewmarkScheme>                     newmark;
        real_t                                             extent{0};

        int build_mesh(const std::shared_ptr<Communicator> &comm, const ElasticityCase &config) {
            auto node = child(config.root(), "mesh");
            if (has(node, "path")) {
                mesh = Mesh::create_from_file(comm, smesh::Path(config.resolve(read_string(node, "path", ""))));
            } else if (has(node, "box")) {
                auto box = node["box"];
                auto n   = read_list<ptrdiff_t>(box, "n", {1, 1, 1});
                auto lo  = read_list<real_t>(box, "min", {0, 0, 0});
                auto hi  = read_list<real_t>(box, "max", {1, 1, 1});
                if (n.size() != 3 || lo.size() != 3 || hi.size() != 3) {
                    fprintf(stderr, "Playground: mesh.box needs three entries in n, min and max\n");
                    return SFEM_FAILURE;
                }
                mesh = Mesh::create_hex8_cube(comm, n[0], n[1], n[2], lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
            } else {
                fprintf(stderr, "Playground: mesh needs a path or a box\n");
                return SFEM_FAILURE;
            }
            if (!mesh) return SFEM_FAILURE;

            const auto reorder = read_string(node, "reorder", "none");
            if (reorder != "none") {
                // Sidesets and nodesets read from files name the numbering on disk: reorder only
                // meshes whose sets are given by selectors here.
                smesh::SFC sfc(reorder);
                if (sfc.reorder(*mesh) != SFEM_SUCCESS) return SFEM_FAILURE;
            }
            const int refine = read<int>(node, "refine", 0);
            if (read<bool>(node, "promote_to_p2", false)) {
                if (refine > 0) {
                    fprintf(stderr, "Playground: mesh.promote_to_p2 and mesh.refine exclude each other\n");
                    return SFEM_FAILURE;
                }
                mesh = smesh::promote_to(smesh::TET10, mesh);
            } else if (refine > 0) {
                mesh = smesh::to_semistructured(refine, mesh, true, false);
            }
            packed   = read<bool>(node, "packed", true);
            renumber = read<bool>(node, "renumber", true);

            if (!has(node, "sidesets")) return SFEM_SUCCESS;

            const Box bbox = bounding_box(mesh);
            for (auto s : node["sidesets"].children()) {
                const std::string name = read_string(s, "name", "");
                if (name.empty()) {
                    fprintf(stderr, "Playground: every sideset needs a name\n");
                    return SFEM_FAILURE;
                }

                if (has(s, "file")) {
                    lookup.named[name] = {Sideset::create_from_file(comm, smesh::Path(config.resolve(read_string(s, "file", ""))))};
                } else if (has(s, "plane")) {
                    auto        plane = s["plane"];
                    const int   axis  = axis_of(read_string(plane, "axis", ""));
                    const auto  at    = read_string(plane, "at", "");
                    if (axis < 0 || at.empty()) {
                        fprintf(stderr, "Playground: sideset %s: plane needs an axis (x, y, z) and at (min, max, a value)\n", name.c_str());
                        return SFEM_FAILURE;
                    }
                    const real_t lo     = bbox.lo[axis];
                    const real_t hi     = bbox.hi[axis];
                    const real_t target = at == "min" ? lo : at == "max" ? hi : (real_t)std::stod(at);
                    const real_t tol    = read<real_t>(plane, "tol", real_t(1e-6)) * std::max(hi - lo, real_t(1e-30));

                    lookup.named[name] = Sideset::create_from_selector(mesh, [=](const geom_t px, const geom_t py, const geom_t pz) -> bool {
                        const geom_t p[3] = {px, py, pz};
                        return std::abs(p[axis] - target) <= tol;
                    });
                } else {
                    fprintf(stderr, "Playground: sideset %s needs a plane or a file\n", name.c_str());
                    return SFEM_FAILURE;
                }

                if (lookup.named[name].empty()) {
                    fprintf(stderr, "Playground: sideset %s is empty\n", name.c_str());
                    return SFEM_FAILURE;
                }
            }
            return SFEM_SUCCESS;
        }

        /// A list of conditions, given in the case or as {file: path} to a YAML document whose `key`
        /// holds it. The document is kept: the conditions are read from its nodes.
        ryml::ConstNodeRef conditions(const ElasticityCase &config, const ryml::ConstNodeRef &node, const char *key) {
            if (!node.is_map() || !node.has_child("file")) return node;

            const std::string path = config.resolve(read_string(node, "file", ""));
            std::ifstream     is(path);
            if (!is.good()) {
                fprintf(stderr, "Playground: unable to read %s\n", path.c_str());
                return ryml::ConstNodeRef();
            }
            std::stringstream contents;
            contents << is.rdbuf();
            documents.push_back(ryml::parse_in_arena(ryml::to_csubstr(contents.str())));
            auto root = documents.back().crootref();
            if (!root.is_map() || !root.has_child(ryml::to_csubstr(key))) {
                fprintf(stderr, "Playground: %s has no %s\n", path.c_str(), key);
                return ryml::ConstNodeRef();
            }
            return root[ryml::to_csubstr(key)];
        }

        int build_function(const ElasticityCase &config) {
            auto root = config.root();

            if (packed && !renumber) {
                // Packed in the numbering on disk, for cases that bring per-node data of their own.
                auto packed_mesh = FunctionSpace::PackedMesh::create(mesh, {}, false);
                if (!packed_mesh) return SFEM_FAILURE;
                fs = FunctionSpace::create(packed_mesh, mesh->spatial_dimension());
            } else {
                fs = FunctionSpace::create(mesh, mesh->spatial_dimension());
                if (packed && fs->initialize_packed_mesh() != SFEM_SUCCESS) return SFEM_FAILURE;
            }
            f = Function::create(fs);

            if (!has(root, "material")) {
                fprintf(stderr, "Playground: the case needs a material\n");
                return SFEM_FAILURE;
            }
            material = create_op_from_yaml(fs, root["material"], es);
            if (!material) {
                fprintf(stderr, "Playground: unable to create the material\n");
                return SFEM_FAILURE;
            }
            f->add_operator(material);

            auto loads = child(root, "loads");
            if (has(loads, "body_force")) {
                auto bf = loads["body_force"];
                auto op = create_op(fs, "GeneratedBodyForce", es);
                if (!op || op->initialize() != SFEM_SUCCESS) return SFEM_FAILURE;
                const real_t density = read<real_t>(bf, "density", 1);
                auto         g       = read_list<real_t>(bf, "g", {});
                if ((int)g.size() != fs->block_size()) {
                    fprintf(stderr, "Playground: loads.body_force.g needs %d components\n", fs->block_size());
                    return SFEM_FAILURE;
                }
                for (const auto &block : mesh->blocks()) {
                    op->set_value_in_block(block->name(), "density", density);
                    for (int d = 0; d < (int)g.size(); ++d) {
                        op->set_value_in_block(block->name(), ("g" + std::to_string(d)).c_str(), g[d]);
                    }
                }
                f->add_operator(op);
            }

            lookup.base_dir = config.base_dir();
            if (has(loads, "neumann")) {
                auto list = conditions(config, loads["neumann"], "neumann_conditions");
                if (!list.readable()) return SFEM_FAILURE;
                auto neumann = NeumannConditions::create_from_yaml(fs, list, lookup);
                if (!neumann) return SFEM_FAILURE;
#ifdef SFEM_ENABLE_CUDA
                if (es == EXECUTION_SPACE_DEVICE) {
                    if (has_profile(list) || has(loads, "profile")) {
                        fprintf(stderr, "Playground: time-dependent Neumann conditions are host only\n");
                        return SFEM_FAILURE;
                    }
                    f->add_operator(to_device(neumann));
                } else
#endif
                {
                    // Every condition's own profile, times the loads' common one.
                    LoadProfile scale;
                    if (has(loads, "profile") && LoadProfile::from_yaml(loads["profile"], scale) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    f->add_operator(neumann);
                    time_dependent.push_back([neumann, scale](const real_t t) { return neumann->set_time(t, scale.value(t)); });
                }
            }

            if (has(root, "dirichlet_conditions")) {
                auto list = conditions(config, root["dirichlet_conditions"], "dirichlet_conditions");
                if (!list.readable()) return SFEM_FAILURE;
                auto dirichlet = DirichletConditions::create_from_yaml(fs, list, lookup);
                if (!dirichlet) return SFEM_FAILURE;
#ifdef SFEM_ENABLE_CUDA
                if (es == EXECUTION_SPACE_DEVICE) {
                    if (has_profile(list)) {
                        fprintf(stderr, "Playground: time-dependent Dirichlet conditions are host only\n");
                        return SFEM_FAILURE;
                    }
                    f->add_constraint(to_device(dirichlet));
                } else
#endif
                {
                    f->add_constraint(dirichlet);
                    time_dependent.push_back([dirichlet](const real_t t) { return dirichlet->set_time(t); });
                }
                host_dirichlet = dirichlet;
            }
            return SFEM_SUCCESS;
        }

        int build_constraints(const ElasticityCase &config) {
            auto root = config.root();
            if (!has(root, "constraints")) return SFEM_SUCCESS;

            for (auto c : root["constraints"].children()) {
                const auto type = read_string(c, "type", "");
                if (type != "rotation") {
                    fprintf(stderr, "Playground: unknown constraint \"%s\" (rotation)\n", type.c_str());
                    return SFEM_FAILURE;
                }
                if (es != EXECUTION_SPACE_HOST) {
                    fprintf(stderr, "Playground: rotation constraints are host only\n");
                    return SFEM_FAILURE;
                }

                const auto name     = read_string(c, "sideset", "");
                auto       sidesets = lookup.find(name);
                const int  axis     = axis_of(read_string(c, "axis", ""));
                auto       center   = read_list<real_t>(c, "center", {0, 0, 0});
                const bool torque   = has(c, "control");
                if (sidesets.empty() || axis < 0 || center.size() != 3 || has(c, "angle") == torque) {
                    fprintf(stderr,
                            "Playground: a rotation needs a named sideset, an axis (x, y, z), a center [x, y, z], and an "
                            "angle or a torque control\n");
                    return SFEM_FAILURE;
                }
                LoadProfile profile;
                if (has(c, "profile") && LoadProfile::from_yaml(c["profile"], profile) != SFEM_SUCCESS) return SFEM_FAILURE;

                auto rotation = Rotation::create(fs, sidesets, axis, center.data(), read<real_t>(c, "angle", 0), profile);
                if (!rotation) return SFEM_FAILURE;
                f->add_constraint(rotation);
                rotations.push_back(rotation);
                if (torque) {
                    // The profile scales the torque, and the control sets the angle.
                    if (torque_controlled) {
                        fprintf(stderr, "Playground: one rotation at most may be torque controlled\n");
                        return SFEM_FAILURE;
                    }
                    auto control           = c["control"];
                    torque_controlled      = rotation;
                    torque_control.torque  = read<real_t>(control, "torque", 0);
                    torque_control.tol     = read<real_t>(control, "tol", torque_control.tol);
                    torque_control.max_it  = read<int>(control, "max_it", torque_control.max_it);
                    torque_control.profile = profile;
                    if (torque_control.torque == 0) {
                        fprintf(stderr, "Playground: a torque control needs a non-zero torque\n");
                        return SFEM_FAILURE;
                    }
                    // Untwisted until the control's first solve.
                    if (rotation->set_angle(0) != SFEM_SUCCESS) return SFEM_FAILURE;
                } else {
                    time_dependent.push_back([rotation](const real_t t) { return rotation->set_time(t); });
                }
                if (has(c, "release")) releases.push_back({rotation, read<real_t>(c, "release", 0)});
            }
            return SFEM_SUCCESS;
        }

        int build_fields(const ElasticityCase &config) {
            auto root = config.root();
            if (!has(root, "fields")) return SFEM_SUCCESS;

            for (auto field : root["fields"].children()) {
                const auto name = read_string(field, "name", "");
                if (name.empty()) {
                    fprintf(stderr, "Playground: every field needs a name\n");
                    return SFEM_FAILURE;
                }

                if (has(field, "file")) {
                    const ptrdiff_t n      = read<ptrdiff_t>(field, "values_per_element", 9) * mesh->n_elements();
                    auto            values = create_host_buffer<real_t>(n);
                    if (read_state_field(config.resolve(read_string(field, "file", "")), n, values->data()) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    material->set_field(name.c_str(), values, 0);
                } else if (has(field, "inclusion")) {
                    // An active strain a11 along x inside a ball, a11(t) = 1 - (1 - a11) profile(t).
                    auto inclusion = field["inclusion"];
                    const Box bbox = bounding_box(mesh);
                    auto      lo   = bbox.lo;
                    auto      hi   = bbox.hi;
                    real_t span    = std::min(hi[0] - lo[0], std::min(hi[1] - lo[1], hi[2] - lo[2]));
                    auto center    = read_list<real_t>(inclusion, "center",
                                                     {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2});
                    const real_t radius = read<real_t>(inclusion, "radius", span / 4);
                    const real_t a11    = read<real_t>(inclusion, "a11", real_t(0.5));
                    LoadProfile  profile;
                    if (has(field, "profile") && LoadProfile::from_yaml(field["profile"], profile) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                    if (center.size() != 3) {
                        fprintf(stderr, "Playground: fields.inclusion.center needs three coordinates\n");
                        return SFEM_FAILURE;
                    }

                    auto values = create_host_buffer<real_t>(9 * mesh->n_elements());
                    auto op     = material;
                    auto m      = mesh;
                    time_dependent.push_back([=](const real_t t) {
                        const real_t c[3] = {center[0], center[1], center[2]};
                        fill_inclusion(m, c, radius, 1 - (1 - a11) * profile.value(t), values->data());
                        op->set_field(name.c_str(), values, 0);
                        return SFEM_SUCCESS;
                    });
                } else {
                    fprintf(stderr, "Playground: field %s needs a file or an inclusion\n", name.c_str());
                    return SFEM_FAILURE;
                }
            }
            return SFEM_SUCCESS;
        }

        int initial_conditions(const ElasticityCase &config) {
            auto node = child(config.root(), "initial_conditions");
            if (!node.readable()) return SFEM_SUCCESS;

            auto read_into = [&](const ryml::ConstNodeRef &entry, real_t *const out) {
                const ptrdiff_t ndofs = fs->n_dofs();
                auto            host  = create_host_buffer<real_t>(ndofs);
                int             status;
                if (has(entry, "file")) {
                    status = read_state_field(config.resolve(read_string(entry, "file", "")), ndofs, host->data());
                } else if (has(entry, "components")) {
                    std::string paths;
                    for (const auto &p : read_list<std::string>(entry, "components", {})) {
                        paths += (paths.empty() ? "" : ",") + config.resolve(p);
                    }
                    status = read_state_field_components(paths, mesh->n_nodes(), fs->block_size(), host->data());
                } else {
                    fprintf(stderr, "Playground: an initial condition needs a file or components\n");
                    return SFEM_FAILURE;
                }
                if (status != SFEM_SUCCESS) return status;
                if (auto map = fs->packed_node_map()) {
                    // The files speak the numbering on disk; packing renumbered the nodes.
                    auto       renumbered = create_host_buffer<real_t>(ndofs);
                    const int  bs         = fs->block_size();
                    const auto to_new     = map->data();
                    for (ptrdiff_t node = 0; node < (ptrdiff_t)map->size(); ++node) {
                        for (int d = 0; d < bs; ++d) renumbered->data()[to_new[node] * bs + d] = host->data()[node * bs + d];
                    }
                    host = renumbered;
                }
#ifdef SFEM_ENABLE_CUDA
                if (es == EXECUTION_SPACE_DEVICE) {
                    sfem::blas<real_t>(es)->copy(ndofs, smesh::to_device(host)->data(), out);
                    return SFEM_SUCCESS;
                }
#endif
                std::copy(host->data(), host->data() + ndofs, out);
                return SFEM_SUCCESS;
            };

            SharedBuffer<real_t> state, velocity;
            if (newmark) {
                state    = newmark->state();
                velocity = newmark->velocity();
            } else if (auto bdf2 = std::dynamic_pointer_cast<BDF2Scheme>(scheme)) {
                state    = bdf2->state();
                velocity = bdf2->velocity();
            } else if (auto be = std::dynamic_pointer_cast<BackwardEulerScheme>(scheme)) {
                state = be->state();
            }

            if (has(node, "displacement")) {
                if (read_into(node["displacement"], x->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
                if (state) sfem::blas<real_t>(es)->copy(fs->n_dofs(), x->data(), state->data());
            }
            if (has(node, "velocity")) {
                if (!velocity) {
                    fprintf(stderr, "Playground: an initial velocity needs a Newmark or BDF2 scheme\n");
                    return SFEM_FAILURE;
                }
                if (read_into(node["velocity"], velocity->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        /// The rigid obstacle and the contact of the body's nodes with it.
        int build_obstacle(const ElasticityCase &config, const std::string &regime) {
            auto node = child(config.root(), "obstacle");
            if (!node.readable()) return SFEM_SUCCESS;

            if (read_string(node, "backend", "node_sdf") != "node_sdf") {
                fprintf(stderr, "Playground: unknown obstacle backend \"%s\" (node_sdf)\n", read_string(node, "backend", "").c_str());
                return SFEM_FAILURE;
            }
            if (regime != "dynamic" || read_string(child(config.root(), "dynamics"), "scheme", "newmark") != "newmark") {
                fprintf(stderr, "Playground: the node_sdf obstacle needs regime dynamic with the newmark scheme, which it stabilises\n");
                return SFEM_FAILURE;
            }

            auto      m   = fs->mesh_ptr();
            const int dir = axis_of(read_string(node, "direction", "z"));
            if (dir < 0 || m->spatial_dimension() != 3) {
                fprintf(stderr, "Playground: obstacle.direction must be x, y or z of a three-dimensional mesh\n");
                return SFEM_FAILURE;
            }

            const Box bbox = bounding_box(m);
            real_t    lo[3], hi[3];
            for (int d = 0; d < 3; ++d) {
                lo[d]  = bbox.lo[d];
                hi[d]  = bbox.hi[d];
                extent = std::max(extent, hi[d] - lo[d]);
            }

            if (has(node, "sdf")) {
                obstacle_sdf = smesh::Grid<geom_t>::create_from_file(m->comm(), config.resolve(read_string(child(node, "sdf"), "file", "")));
            } else {
                auto          s = child(node, "shape");
                ObstacleShape shape;
                const auto    type = read_string(s, "type", "hemisphere");
                if (type == "plane") {
                    shape.type = ObstacleShape::Type::PLANE;
                } else if (type == "slope") {
                    shape.type = ObstacleShape::Type::SLOPE;
                } else if (type == "staircase") {
                    shape.type = ObstacleShape::Type::STAIRCASE;
                } else if (type == "hemisphere") {
                    shape.type = ObstacleShape::Type::HEMISPHERE;
                } else {
                    fprintf(stderr, "Playground: unknown obstacle shape \"%s\" (plane, slope, staircase, hemisphere)\n", type.c_str());
                    return SFEM_FAILURE;
                }
                shape.dir        = dir;
                shape.plane      = read<real_t>(s, "plane", shape.plane);
                shape.slope      = read<real_t>(s, "slope", shape.slope);
                shape.slope_axis = has(s, "slope_axis") ? axis_of(read_string(s, "slope_axis", "")) : -1;
                shape.step_width = read<real_t>(s, "step_width", shape.step_width);
                shape.step_drop  = read<real_t>(s, "step_drop", shape.step_drop);
                shape.radius     = read<real_t>(s, "radius", shape.radius);
                shape.fillet     = read<real_t>(s, "fillet", shape.fillet);
                shape.grid_cells = read<int>(s, "grid_cells", shape.grid_cells);
                if (has(s, "center")) {
                    auto center = read_list<real_t>(s, "center", {});
                    if (center.size() != 3) {
                        fprintf(stderr, "Playground: obstacle.shape.center needs three coordinates\n");
                        return SFEM_FAILURE;
                    }
                    shape.has_center = true;
                    std::copy(center.begin(), center.end(), shape.center);
                }
                obstacle_sdf = create_obstacle_sdf(m->comm(), shape, lo, hi);
            }
            if (!obstacle_sdf) return SFEM_FAILURE;

            // Every node of the body's surface by default: a free body can touch with any face.
            const auto          which = read_string(node, "nodes", "skin");
            SharedBuffer<idx_t> nodes;
            if (which == "skin") {
                nodes = smesh::create_nodeset_from_sidesets(m, smesh::skin_sidesets(m));
            } else {
                auto sidesets = lookup.find(which);
                if (sidesets.empty() || (packed && renumber)) {
                    fprintf(stderr, "Playground: obstacle.nodes is skin or a named sideset of an unpacked or unrenumbered mesh\n");
                    return SFEM_FAILURE;
                }
                nodes = smesh::create_nodeset_from_sidesets(m, sidesets);
            }
            contact = std::make_shared<NodeObstacleContact>(obstacle_sdf, m->comm(), nodes, m->points(), fs->n_dofs(), dir, es);
            if (contact->size() == 0) {
                fprintf(stderr, "Playground: the obstacle has no contact nodes\n");
                return SFEM_FAILURE;
            }
            printf("Contact: %td nodes\n", contact->size());
            return SFEM_SUCCESS;
        }

        int build_scheme(const ElasticityCase &config, const std::string &regime) {
            auto history = std::dynamic_pointer_cast<HistoryDependent>(material);
            if (regime == "steady" && history) {
                fprintf(stderr, "Playground: the material carries a history, which a steady solve has no time for\n");
                return SFEM_FAILURE;
            }
            if (regime != "dynamic") return SFEM_SUCCESS;

            auto        dynamics = child(config.root(), "dynamics");
            const auto  type     = read_string(dynamics, "scheme", "newmark");
            const real_t density = read<real_t>(dynamics, "density", 1);

            if (type == "newmark") {
                const real_t beta  = read<real_t>(dynamics, "beta", real_t(0.25));
                const real_t gamma = read<real_t>(dynamics, "gamma", real_t(0.5));
                // Unconditionally stable only for beta >= gamma/2 >= 1/4; the damped pairing for
                // gamma > 1/2 is beta = (1 + gamma)^2 / 4.
                if (gamma < real_t(0.5) || beta < gamma / 2) {
                    fprintf(stderr,
                            "Playground: Newmark is unconditionally stable only for beta >= gamma/2 >= 1/4, but "
                            "beta = %g and gamma = %g; use beta = (1 + gamma)^2/4 = %g, or gamma = 0.5\n",
                            (double)beta,
                            (double)gamma,
                            (double)((1 + gamma) * (1 + gamma) / 4));
                    return SFEM_FAILURE;
                }
                auto newmark = std::make_shared<NewmarkScheme>(fs, es);
                newmark->set_beta(beta);
                newmark->set_gamma(gamma);
                newmark->set_density(density);
                if (newmark->initialize() != SFEM_SUCCESS) return SFEM_FAILURE;
                scheme        = newmark;
                this->newmark = newmark;
                // Plain Newmark carries the obstacle's reaction into the next predictor and chatters.
                if (contact) scheme = std::make_shared<ContactStabilisedNewmarkScheme>(newmark, contact);
            } else if (type == "bdf2") {
                auto bdf2 = std::make_shared<BDF2Scheme>(fs, es);
                bdf2->set_density(density);
                if (bdf2->initialize() != SFEM_SUCCESS) return SFEM_FAILURE;
                scheme = bdf2;
            } else if (type == "backward_euler") {
                auto be = std::make_shared<BackwardEulerScheme>(fs, es);
                if (be->initialize() != SFEM_SUCCESS) return SFEM_FAILURE;
                scheme = be;
            } else {
                fprintf(stderr, "Playground: unknown scheme \"%s\" (newmark, bdf2, backward_euler)\n", type.c_str());
                return SFEM_FAILURE;
            }

            auto steppable = std::dynamic_pointer_cast<TimeSteppable>(material);
            if (!steppable) {
                fprintf(stderr, "Playground: material %s cannot hold a time scheme\n", material->name());
                return SFEM_FAILURE;
            }
            steppable->set_time_scheme(scheme);
            return SFEM_SUCCESS;
        }

        int build_solver(const ElasticityCase &config) {
            auto solver = child(config.root(), "solver");
            auto nn     = child(solver, "newton");
            auto ln     = child(solver, "linear");

            NewtonOptions newton_options;
            newton_options.max_it            = read<int>(nn, "max_it", newton_options.max_it);
            newton_options.atol              = read<real_t>(nn, "atol", newton_options.atol);
            newton_options.rtol              = read<real_t>(nn, "rtol", newton_options.rtol);
            newton_options.divergence_factor = read<real_t>(nn, "divergence_factor", newton_options.divergence_factor);
            newton_options.merit             = read_string(nn, "merit", newton_options.merit);
            verbose                          = read<bool>(nn, "verbose", true);
            // A linear problem is solved by its first step, which needs no merit to judge it.
            newton_options.line_search.enabled = !f->is_linear();
            if (has(nn, "line_search")) {
                auto ls                            = nn["line_search"];
                newton_options.line_search.enabled = read<bool>(ls, "enabled", newton_options.line_search.enabled);
                newton_options.line_search.steps   = read_list<real_t>(ls, "steps", newton_options.line_search.steps);
                newton_options.line_search.c1      = read<real_t>(ls, "c1", newton_options.line_search.c1);
                newton_options.line_search.batched = read<bool>(ls, "batched", newton_options.line_search.batched);
            }

            LinearSolverOptions linear;
            linear.type           = read_string(ln, "type", linear.type);
            linear.format         = read_string(ln, "format", linear.format);
            linear.preconditioner = read_string(ln, "preconditioner", linear.preconditioner);
            linear.atol           = read<real_t>(ln, "atol", linear.atol);
            linear.max_it         = read<int>(ln, "max_it", linear.max_it);
            linear.storage_bytes  = read<int>(ln, "storage_bytes", linear.storage_bytes);
            // A truncated direction needs the line search to scale it.
            linear.stop_on_negative_curvature = newton_options.line_search.enabled;
            // A linear problem is solved by one Newton step, so its linear solve goes as far as
            // the Newton tolerance; a nonlinear one stops where the Newton direction stops being
            // accurate.
            linear.rtol = read<real_t>(ln, "rtol", f->is_linear() ? newton_options.rtol : linear.rtol);

            std::shared_ptr<LinearizedStep> step;
            if (contact) {
                NodeObstacleStepOptions obstacle;
                obstacle.atol = newton_options.atol;
                obstacle.rtol = newton_options.rtol;
                // Contact is resolved to this fraction of the mesh's size.
                obstacle.gap_tol   = real_t(1e-10) * extent;
                auto qp            = child(child(config.root(), "obstacle"), "qp");
                obstacle.qp_max_it = read<int>(qp, "max_it", obstacle.qp_max_it);
                obstacle.qp_rtol   = read<real_t>(qp, "rtol", obstacle.qp_rtol);
                step               = NodeObstacleStep::create(f, contact, newmark, obstacle);
            } else {
                step = UnconstrainedStep::create(f, es, linear);
            }
            if (!step) return SFEM_FAILURE;
            newton = std::make_shared<NewtonSolver>(f, step, newton_options);
            nonlinear = newton;
            if (torque_controlled) {
                nonlinear = std::make_shared<TorqueControl>(f, newton, torque_controlled, material, torque_control);
            }
            if (verbose) {
                newton->add_monitor([](const NewtonIteration &it) {
                    if (it.iteration == 0) {
                        printf("  Newton %2d: |r| = %.6e (tol %.3e)\n", it.iteration, (double)it.measure.value, (double)it.tolerance);
                    } else {
                        printf("  Newton %2d: |r| = %.6e, %d linear%s, alpha = %g (%s)\n",
                               it.iteration,
                               (double)it.measure.value,
                               it.linear_iterations,
                               it.linear_capped ? " (max iterations)" : "",
                               (double)it.line_search.alpha,
                               to_string(it.line_search.status));
                    }
                    fflush(stdout);
                });
            }
            return SFEM_SUCCESS;
        }

        int build_stepper(const ElasticityCase &config, const std::string &regime) {
            auto           time = child(config.root(), "time");
            StepperOptions options;
            if (regime == "steady") {
                options.dt    = 1;
                options.steps = 1;
            } else {
                options.dt = read<real_t>(time, "dt", 0);
                if (!(options.dt > 0)) {
                    fprintf(stderr, "Playground: time.dt must be positive\n");
                    return SFEM_FAILURE;
                }
                if (has(time, "steps")) {
                    options.steps = read<int>(time, "steps", 1);
                } else if (has(time, "t_end")) {
                    options.steps = (int)std::llround(read<real_t>(time, "t_end", 0) / options.dt);
                } else {
                    fprintf(stderr, "Playground: time needs steps or t_end\n");
                    return SFEM_FAILURE;
                }
                // A failed step is retried with half its length: hard contact steps need it (a
                // hemisphere impact at dt 0.08, a staircase at 0.04). BDF2 is a constant-step
                // method, whose coefficients a shorter step would make wrong, so it has none.
                const bool bdf2      = std::dynamic_pointer_cast<BDF2Scheme>(scheme) != nullptr;
                options.max_cutbacks = read<int>(time, "max_cutbacks", bdf2 ? 0 : 4);
                if (bdf2 && options.max_cutbacks > 0) {
                    fprintf(stderr, "Playground: BDF2 is a constant-step method; time.max_cutbacks must be 0 with it\n");
                    return SFEM_FAILURE;
                }
            }

            stepper = std::make_unique<Stepper>(f, nonlinear, options);
            stepper->set_time_scheme(scheme);
            if (auto history = std::dynamic_pointer_cast<HistoryDependent>(material)) stepper->add_history(history);
            for (auto &set : time_dependent) stepper->add_time_dependent(set);
            for (auto &release : releases) stepper->add_release(release.first, release.second);

            auto output = child(config.root(), "output");
            if (has(output, "path")) {
                const smesh::Path path(config.resolve(read_string(output, "path", "")));
                auto              fields = std::make_shared<FieldExport>(f, path, read<int>(output, "export_freq", 1));
                for (const auto &name : read_list<std::string>(output, "fields", {})) {
                    if (name == "material_reaction") {
                        auto op   = material;
                        auto blas = sfem::blas<real_t>(es);
                        auto n    = fs->n_dofs();
                        fields->add_field(name, [op, blas, n](const StepRecord &record, real_t *const out) {
                            blas->zeros(n, out);
                            return op->gradient(record.x, out);
                        });
                    } else if ((name == "gap" || name == "contact_force") && contact) {
                        auto c    = contact;
                        auto blas = sfem::blas<real_t>(es);
                        auto n    = fs->n_dofs();
                        if (name == "gap") {
                            fields->add_field(name, [c](const StepRecord &record, real_t *const out) {
                                c->update(record.x);
                                c->gap_field(out);
                                return SFEM_SUCCESS;
                            });
                        } else {
                            // In this trapezoidal form the material force enters both ends of the
                            // step and the contact force once: the force is half the reaction.
                            fields->add_field(name, [c, blas, n](const StepRecord &record, real_t *const out) {
                                blas->zeros(n, out);
                                if (record.step == 0) return SFEM_SUCCESS;
                                c->reaction(out);
                                blas->scal(n, real_t(0.5), out);
                                return SFEM_SUCCESS;
                            });
                        }
                    } else {
                        fprintf(stderr,
                                "Playground: unknown output field \"%s\" (material_reaction; gap, contact_force with an obstacle)\n",
                                name.c_str());
                        return SFEM_FAILURE;
                    }
                }
                stepper->add_observer(fields);
                // The obstacle's grid: contour sdf/sdf.xdmf at 0.
                if (obstacle_sdf) obstacle_sdf->to_file(path / "sdf");

                if (read<bool>(output, "quantities", false)) {
                    stepper->add_observer(std::make_shared<Quantities>(f, material, host_dirichlet, path));
                }
            } else if (has(output, "fields") || has(output, "quantities")) {
                fprintf(stderr, "Playground: output.fields and output.quantities need output.path\n");
                return SFEM_FAILURE;
            }

            const std::string csv = has(output, "history_csv") ? config.resolve(read_string(output, "history_csv", "")) : "";
            // One line per step whatever the Newton verbosity, which is about its iterations.
            auto history = std::make_shared<StepHistory>(csv, true);
            if (has(output, "probes")) {
                for (auto p : output["probes"].children()) {
                    const auto type = read_string(p, "type", "");
                    if (type == "control_point") {
                        history->add_probe(create_control_point_probe(f, read_list<real_t>(p, "at", {}), read<bool>(p, "rates", scheme != nullptr)));
                    } else if (type == "torque") {
                        const int which = read<int>(p, "constraint", 0);
                        if (which < 0 || which >= (int)rotations.size()) {
                            fprintf(stderr, "Playground: a torque probe needs constraint, the index of a rotation in constraints\n");
                            return SFEM_FAILURE;
                        }
                        history->add_probe(create_torque_probe(f, rotations[which], material));
                    } else if (type == "contact" && contact) {
                        history->add_probe(create_contact_probe(contact, fs->n_dofs(), real_t(1e-6) * extent));
                    } else {
                        fprintf(stderr, "Playground: unknown probe \"%s\" (control_point, torque; contact with an obstacle)\n", type.c_str());
                        return SFEM_FAILURE;
                    }
                }
            }
            stepper->add_observer(history);
            return SFEM_SUCCESS;
        }

        std::vector<std::function<int(const real_t)>> time_dependent;
    };

    Playground::Playground() : impl_(std::make_unique<Impl>()) {}
    Playground::~Playground() = default;

    std::unique_ptr<Playground> Playground::create(const std::shared_ptr<Communicator> &comm, const ElasticityCase &config) {
        SFEM_TRACE_SCOPE("Playground::create");

        if (comm->size() > 1) {
            fprintf(stderr, "Playground: runs on one rank\n");
            return nullptr;
        }

        std::unique_ptr<Playground> ret(new Playground());
        auto                       &impl = *ret->impl_;
        auto                        root = config.root();

        const auto es = read_string(root, "execution_space", "host");
        if (es == "host") {
            impl.es = EXECUTION_SPACE_HOST;
        } else if (es == "device") {
#ifdef SFEM_ENABLE_CUDA
            impl.es = EXECUTION_SPACE_DEVICE;
#else
            fprintf(stderr, "Playground: execution_space device needs a CUDA build\n");
            return nullptr;
#endif
        } else {
            fprintf(stderr, "Playground: unknown execution_space \"%s\" (host, device)\n", es.c_str());
            return nullptr;
        }

        const auto regime = read_string(root, "regime", "steady");
        if (regime != "steady" && regime != "quasi_static" && regime != "dynamic") {
            fprintf(stderr, "Playground: unknown regime \"%s\" (steady, quasi_static, dynamic)\n", regime.c_str());
            return nullptr;
        }

        if (impl.build_mesh(comm, config) != SFEM_SUCCESS) return nullptr;
        if (impl.build_function(config) != SFEM_SUCCESS) return nullptr;
        if (impl.build_constraints(config) != SFEM_SUCCESS) return nullptr;
        if (impl.build_fields(config) != SFEM_SUCCESS) return nullptr;
        if (impl.build_obstacle(config, regime) != SFEM_SUCCESS) return nullptr;
        if (impl.build_scheme(config, regime) != SFEM_SUCCESS) return nullptr;
        if (impl.build_solver(config) != SFEM_SUCCESS) return nullptr;
        if (impl.build_stepper(config, regime) != SFEM_SUCCESS) return nullptr;

        impl.x = create_buffer<real_t>(impl.fs->n_dofs(), impl.es);
        if (impl.initial_conditions(config) != SFEM_SUCCESS) return nullptr;
        // The problem size, whatever the Newton verbosity: a timing is read against it.
        printf("Playground: %s, %s, %td elements, %td dofs, %d steps of %g\n",
               regime.c_str(),
               impl.material->name(),
               impl.mesh->n_elements(),
               impl.fs->n_dofs(),
               impl.stepper->options().steps,
               (double)impl.stepper->options().dt);
        return ret;
    }

    int Playground::run() { return impl_->stepper->run(impl_->x->data()); }

    std::shared_ptr<Function>   Playground::function() const { return impl_->f; }
    std::shared_ptr<Op>         Playground::material() const { return impl_->material; }
    std::shared_ptr<TimeScheme> Playground::scheme() const { return impl_->scheme; }
    NewtonSolver               &Playground::newton() { return *impl_->newton; }
    Stepper                    &Playground::stepper() { return *impl_->stepper; }
    SharedBuffer<real_t>        Playground::solution() const { return impl_->x; }

}  // namespace sfem

#endif  // SFEM_ENABLE_RYAML
