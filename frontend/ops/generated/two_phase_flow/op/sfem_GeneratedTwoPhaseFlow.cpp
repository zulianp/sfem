#include "sfem_GeneratedTwoPhaseFlow.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"

#include <cstring>

extern "C" {
int generated_two_phase_flow_tri3_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tet4_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_tet4_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_quad4_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *, const real_t *, const real_t *, real_t *);
}

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = 26;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("porosity", 0.10000000000000001);
            parameters.set_value("S_res", 0.39000000000000001);
            parameters.set_value("P_r", 0.095000000000000001);
            parameters.set_value("m", 4.2000000000000002);
            parameters.set_value("rho_w0", 1100);
            parameters.set_value("kappa_T", 0.000455);
            parameters.set_value("p_wr", 1);
            parameters.set_value("M_c", 0.044010000000000001);
            parameters.set_value("Z", 0.42520000000000002);
            parameters.set_value("R", 8.3140000000000004e-06);
            parameters.set_value("T", 333);
            parameters.set_value("mu_w", 5.2000000000000002);
            parameters.set_value("mu_c", 1.5);
            parameters.set_value("C_kw1", 0.52000000000000002);
            parameters.set_value("C_ka1", 1.8);
            parameters.set_value("C_ka2", 0.34999999999999998);
            parameters.set_value("dt", 1);
            parameters.set_value("K_0", 86.400000000000006);
            parameters.set_value("K_1", 0);
            parameters.set_value("K_2", 0);
            parameters.set_value("K_3", 0);
            parameters.set_value("K_4", 86.400000000000006);
            parameters.set_value("K_5", 0);
            parameters.set_value("K_6", 0);
            parameters.set_value("K_7", 0);
            parameters.set_value("K_8", 86.400000000000006);
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 26;
        constexpr int N_MATERIAL_PARAMETERS = 26;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"porosity", "S_res", "P_r", "m", "rho_w0", "kappa_T", "p_wr", "M_c", "Z", "R", "T", "mu_w", "mu_c", "C_kw1", "C_ka1", "C_ka2", "dt", "K_0", "K_1", "K_2", "K_3", "K_4", "K_5", "K_6", "K_7", "K_8"};

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

        void material_defaults(real_t *const values) {
            values[0] = 0.10000000000000001;
            values[1] = 0.39000000000000001;
            values[2] = 0.095000000000000001;
            values[3] = 4.2000000000000002;
            values[4] = 1100;
            values[5] = 0.000455;
            values[6] = 1;
            values[7] = 0.044010000000000001;
            values[8] = 0.42520000000000002;
            values[9] = 8.3140000000000004e-06;
            values[10] = 333;
            values[11] = 5.2000000000000002;
            values[12] = 1.5;
            values[13] = 0.52000000000000002;
            values[14] = 1.8;
            values[15] = 0.34999999999999998;
            values[16] = 1;
            values[17] = 86.400000000000006;
            values[18] = 0;
            values[19] = 0;
            values[20] = 0;
            values[21] = 86.400000000000006;
            values[22] = 0;
            values[23] = 0;
            values[24] = 0;
            values[25] = 86.400000000000006;
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

        void read_affine_options(const ryml::ConstNodeRef &node,
                                 bool &objective,
                                 bool &gradient,
                                 bool &hessian_action) {
            bool all = objective && gradient && hessian_action;
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                objective = all;
                gradient = all;
                hessian_action = all;
            }
            yaml_read_bool(node, "ASSUME_AFFINE_OBJECTIVE", objective);
            yaml_read_bool(node, "objective_assume_affine", objective);
            yaml_read_bool(node, "ASSUME_AFFINE_GRADIENT", gradient);
            yaml_read_bool(node, "gradient_assume_affine", gradient);
            yaml_read_bool(node, "ASSUME_AFFINE_HESSIAN_ACTION", hessian_action);
            yaml_read_bool(node, "hessian_action_assume_affine", hessian_action);
            yaml_read_bool(node, "ASSUME_AFFINE_APPLY", hessian_action);
            yaml_read_bool(node, "apply_assume_affine", hessian_action);
        }
#endif  // SFEM_ENABLE_RYAML

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
            values[index++] = parameters.require_real_value("porosity");
            values[index++] = parameters.require_real_value("S_res");
            values[index++] = parameters.require_real_value("P_r");
            values[index++] = parameters.require_real_value("m");
            values[index++] = parameters.require_real_value("rho_w0");
            values[index++] = parameters.require_real_value("kappa_T");
            values[index++] = parameters.require_real_value("p_wr");
            values[index++] = parameters.require_real_value("M_c");
            values[index++] = parameters.require_real_value("Z");
            values[index++] = parameters.require_real_value("R");
            values[index++] = parameters.require_real_value("T");
            values[index++] = parameters.require_real_value("mu_w");
            values[index++] = parameters.require_real_value("mu_c");
            values[index++] = parameters.require_real_value("C_kw1");
            values[index++] = parameters.require_real_value("C_ka1");
            values[index++] = parameters.require_real_value("C_ka2");
            values[index++] = parameters.require_real_value("dt");
            for (int i = 0; i < dim * dim; ++i) {
                values[index++] =
                        parameters.require_real_value("K_" + std::to_string(i));
            }
        }
    }  // namespace

    class GeneratedTwoPhaseFlow::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
    };

    std::unique_ptr<Op> GeneratedTwoPhaseFlow::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != 2) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires block_size=2\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedTwoPhaseFlow>(space);
        op->initialize();
        return op;
    }

    GeneratedTwoPhaseFlow::GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedTwoPhaseFlow::~GeneratedTwoPhaseFlow() = default;

    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedTwoPhaseFlow::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedTwoPhaseFlow::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const x) {
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const previous,
                       const real_t *const current) {
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::gradient(const real_t *const state, real_t *const out) {
        if (!impl_->previous) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires a previous state\n");
            return SFEM_FAILURE;
        }
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
            const real_t *const parameters = storage;
            const real_t *const previous = impl_->previous;
            switch (domain.element_type) {
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, state, previous, out);
                default:
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedTwoPhaseFlow::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        const real_t *const current = state ? state : impl_->current;
        if (!current) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
            const real_t *const parameters = storage;

            switch (domain.element_type) {
                case smesh::TRI3:
                    return generated_two_phase_flow_tri3_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, direction, out);
                case smesh::TET4:
                    return generated_two_phase_flow_tet4_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, direction, out);
                case smesh::QUAD4:
                    return generated_two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, direction, out);
                case smesh::HEX8:
                    return generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, parameters, current, direction, out);
                default:
                    SFEM_ERROR("GeneratedTwoPhaseFlow does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    void GeneratedTwoPhaseFlow::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedTwoPhaseFlow supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void GeneratedTwoPhaseFlow::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void GeneratedTwoPhaseFlow::set_option(const std::string &, const bool) {}

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedTwoPhaseFlow::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        auto ret = std::make_shared<GeneratedTwoPhaseFlow>(space);

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

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML

    int GeneratedTwoPhaseFlow::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        return SFEM_FAILURE;
    }

    int GeneratedTwoPhaseFlow::value(const real_t *, real_t *const) {
        return SFEM_FAILURE;
    }
}  // namespace sfem
