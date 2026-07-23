#include "sfem_GeneratedNeumann.hpp"
#include "sfem_GeneratedNeumann_c_abi.hpp"

#include "sfem_aliases.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <cstring>
#include <memory>
#include <vector>



namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = 3;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("t0", 0);
            parameters.set_value("t1", 0);
            parameters.set_value("t2", 0);
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

        struct AffineOption {
            const char *name;
            bool       *flag;
        };

        inline bool set_affine_option(const std::string &name,
                                      const bool val,
                                      const AffineOption *const options,
                                      const int n_options) {
            if (name == "ASSUME_AFFINE" || name == "assume_affine") {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = val;
                }
                return true;
            }
            bool matched = false;
            for (int i = 0; i < n_options; ++i) {
                if (name == options[i].name) {
                    *options[i].flag = val;
                    matched = true;
                }
            }
            return matched;
        }

        void material_defaults(real_t *const values) {
            values[0] = 0;
            values[1] = 0;
            values[2] = 0;
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 3;
        constexpr int N_MATERIAL_PARAMETERS = 3;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"t0", "t1", "t2"};

        bool yaml_read_real(const ryml::ConstNodeRef &node,
                            const char *const key,
                            real_t &value) {
            if (!node.has_child(key)) {
                return false;
            }
            node[key] >> value;
            return true;
        }

        bool yaml_read_parameter(const ryml::ConstNodeRef &node,
                                 const char *const key,
                                 real_t &value) {
            if (yaml_read_real(node, key, value)) {
                return true;
            }
            if (node.has_child("parameters") &&
                yaml_read_real(node["parameters"], key, value)) {
                return true;
            }
            if (node.has_child("material") &&
                yaml_read_real(node["material"], key, value)) {
                return true;
            }
            return false;
        }

        std::string yaml_read_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        void copy_material_parameters(const real_t *const src,
                                      real_t *const dst) {
            for (int i = 0; i < N_MATERIAL_PARAMETERS; ++i) {
                dst[i] = src[i];
            }
        }

        bool material_from_yaml(const ryml::ConstNodeRef &node,
                                const real_t *const base,
                                real_t *const values) {
            copy_material_parameters(base, values);
            bool changed = false;
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                changed |= yaml_read_parameter(node,
                                               MATERIAL_PARAMETER_NAMES[i],
                                               values[i]);
            }
            return changed;
        }

        void set_material(MultiDomainOp &domains,
                          const real_t *const values) {
            for (auto &entry : domains.domains()) {
                for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                    entry.second.parameters->set_value(MATERIAL_PARAMETER_NAMES[i],
                                                       values[i]);
                }
            }
        }

        void set_material_in_block(MultiDomainOp &domains,
                                   const std::string &block_name,
                                   const real_t *const values) {
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                domains.set_value_in_block(block_name,
                                           MATERIAL_PARAMETER_NAMES[i],
                                           values[i]);
            }
        }

        bool yaml_read_bool(const ryml::ConstNodeRef &node,
                            const char *const key,
                            bool &value) {
            if (!node.has_child(key)) {
                return false;
            }
            int raw = value ? 1 : 0;
            node[key] >> raw;
            value = raw != 0;
            return true;
        }

        inline void read_affine_options(const ryml::ConstNodeRef &node,
                                        const AffineOption *const options,
                                        const int n_options) {
            bool all = true;
            for (int i = 0; i < n_options; ++i) {
                all = all && *options[i].flag;
            }
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = all;
                }
            }
            for (int i = 0; i < n_options; ++i) {
                yaml_read_bool(node, options[i].name, *options[i].flag);
            }
        }
#endif  // SFEM_ENABLE_RYAML

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
            switch (dim) {
                case 2:
                    values[index++] = parameters.require_real_value("t0");
                    values[index++] = parameters.require_real_value("t1");
                    break;
                case 3:
                    values[index++] = parameters.require_real_value("t0");
                    values[index++] = parameters.require_real_value("t1");
                    values[index++] = parameters.require_real_value("t2");
                    break;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\n", dim);
                    break;
            }
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 2;
                case 3: return 3;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual block size\n", dim);
                    return 0;
            }
        }

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedNeumann: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }

#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<smesh::Sideset> sideset_from_yaml(
                const std::shared_ptr<FunctionSpace> &space,
                const ryml::ConstNodeRef             &node) {
            const bool is_sideset = node["type"].readable() && node["type"].val() == "sideset";
            const bool is_file    = node["format"].readable() && node["format"].val() == "file";
            const bool is_expr    = node["format"].readable() && node["format"].val() == "expr";

            if (!is_sideset && node.has_child("type")) {
                SFEM_ERROR("GeneratedNeumann neumann condition requires type=sideset\n");
                return nullptr;
            }

            if (is_file || node.has_child("path")) {
                if (!node.has_child("path")) {
                    SFEM_ERROR("GeneratedNeumann file sideset condition requires path\n");
                    return nullptr;
                }
                const std::string path = yaml_read_string(node["path"]);
                return smesh::Sideset::create_from_file(
                        space->mesh_ptr()->comm(), smesh::Path(path));
            }

            if (is_expr || (node.has_child("parent") && node.has_child("lfi"))) {
                if (!node["parent"].is_seq() || !node["lfi"].is_seq()) {
                    SFEM_ERROR("GeneratedNeumann expr sideset condition requires parent/lfi sequences\n");
                    return nullptr;
                }

                const ptrdiff_t size = node["parent"].num_children();
                if (node["lfi"].num_children() != size) {
                    SFEM_ERROR("GeneratedNeumann expr sideset parent/lfi length mismatch\n");
                    return nullptr;
                }

                auto parent = create_host_buffer<element_idx_t>(size);
                auto lfi    = create_host_buffer<int16_t>(size);

                ptrdiff_t parent_count = 0;
                for (auto p : node["parent"].children()) {
                    p >> parent->data()[parent_count++];
                }

                ptrdiff_t lfi_count = 0;
                for (auto p : node["lfi"].children()) {
                    p >> lfi->data()[lfi_count++];
                }

                return std::make_shared<smesh::Sideset>(
                        space->mesh_ptr()->comm(), parent, lfi);
            }

            SFEM_ERROR("GeneratedNeumann neumann condition requires format=file or format=expr\n");
            return nullptr;
        }
#endif  // SFEM_ENABLE_RYAML
    }  // namespace

    class GeneratedNeumann::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::vector<NeumannConditions::Condition> conditions;
    };

    std::unique_ptr<Op> GeneratedNeumann::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("GeneratedNeumann requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<GeneratedNeumann>(space);
        op->initialize();
        return op;
    }

    GeneratedNeumann::GeneratedNeumann(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedNeumann::~GeneratedNeumann() = default;

    ptrdiff_t GeneratedNeumann::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedNeumann::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedNeumann::flops_value() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedNeumann::memory_traffic_bytes_value() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedNeumann::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedNeumann::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedNeumann::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedNeumann::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {

                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    int GeneratedNeumann::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        return SFEM_SUCCESS;
    }

    void GeneratedNeumann::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset) {
        real_t values[MAX_PARAMETERS];
        material_defaults(values);
        add_sideset(sideset, values);
    }

    void GeneratedNeumann::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset,
                             const real_t *const parameters) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::add_sideset");
        NeumannConditions::Condition condition;
        condition.sidesets = {sideset};
        condition.values = create_host_buffer<real_t>(MAX_PARAMETERS);
        for (int i = 0; i < MAX_PARAMETERS; ++i) {
            condition.values->data()[i] = parameters[i];
        }
        condition.value = parameters[0];
        condition.component = 0;
        add_condition(condition);
    }

    void GeneratedNeumann::add_condition(const NeumannConditions::Condition &condition) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::add_condition");
        impl_->conditions.push_back(condition);
    }

    int GeneratedNeumann::gradient(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::gradient");
        if (impl_->conditions.empty()) {
            return SFEM_SUCCESS;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
            int status = SFEM_SUCCESS;
            for (const auto &condition : impl_->conditions) {
                const auto sideset = condition.sidesets.empty() ? nullptr : condition.sidesets[0];
                if (!sideset || !condition.values || sideset->block_id() != block_id) {
                    continue;
                }
                switch (domain.element_type) {
                    case smesh::TRI3: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                        status |= neumann_tri3_edgeshell2_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], FIELD_STRIDE, u_out[0], u_out[1]);
                        break;
                    }
                    case smesh::QUAD4: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                        status |= neumann_quad4_edgeshell2_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], FIELD_STRIDE, u_out[0], u_out[1]);
                        break;
                    }
                    case smesh::TET4: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_tet4_trishell3_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::TET10: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_tet10_trishell6_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::HEX8: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_hex8_quadshell4_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::HEX27: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_hex27_quadshell9_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::PROTEUS_HEX8: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::PROTEUS_HEX27: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_proteus_hex27_proteus_quadshell9_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::PROTEUS_HEX64: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_proteus_hex64_proteus_quadshell16_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::PROTEUS_HEX125: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                        status |= neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], condition.values->data()[2], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
                        break;
                    }
                    case smesh::PROTEUS_QUAD4: {
                        static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                        status |= neumann_proteus_quad4_edgeshell2_boundary_residual_sideset_soa(sideset->size(), mesh->n_nodes(), domain.block->elements()->data(), sideset->parent()->data(), sideset->lfi()->data(), points, condition.values->data()[0], condition.values->data()[1], FIELD_STRIDE, u_out[0], u_out[1]);
                        break;
                    }
                    default:
                        SFEM_ERROR("GeneratedNeumann does not support element type %d\n",
                                   domain.element_type);
                        return SFEM_FAILURE;
                }
            }
            return status;
        });
    }

    int GeneratedNeumann::apply(const real_t *const,
                      const real_t *const,
                      real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::apply");
        return SFEM_SUCCESS;
    }

    int GeneratedNeumann::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::value");
        return SFEM_SUCCESS;
    }

    int GeneratedNeumann::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::hessian_crs");
        return SFEM_SUCCESS;
    }

    void GeneratedNeumann::set_field(const char *,
                           const std::shared_ptr<Buffer<real_t>> &,
                           const int) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::set_field");
    }

    void GeneratedNeumann::set_option(const std::string &, const bool) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::set_option");
    }

    void GeneratedNeumann::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedNeumann::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedNeumann::create_from_yaml");
        auto ret = std::make_shared<GeneratedNeumann>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[MAX_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[MAX_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        material_from_yaml(node, defaults, top_values);

        const auto neumann_node =
                node.has_child("neumann_conditions") ? node["neumann_conditions"] :
                 ryml::ConstNodeRef();
        if (neumann_node.readable() && neumann_node.is_seq()) {
            for (auto condition_node : neumann_node.children()) {
                auto sideset = sideset_from_yaml(space, condition_node);
                if (!sideset) {
                    return nullptr;
                }
                real_t condition_values[MAX_PARAMETERS];
                material_from_yaml(condition_node, top_values, condition_values);
                ret->add_sideset(sideset, condition_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem
