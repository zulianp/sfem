#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <cstring>



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

        void read_residual_affine_options(const ryml::ConstNodeRef &node,
                                          bool &residual,
                                          bool &jacobian_action) {
            bool all = residual && jacobian_action;
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                residual = all;
                jacobian_action = all;
            }
            yaml_read_bool(node, "ASSUME_AFFINE_RESIDUAL", residual);
            yaml_read_bool(node, "residual_assume_affine", residual);
            yaml_read_bool(node, "ASSUME_AFFINE_GRADIENT", residual);
            yaml_read_bool(node, "gradient_assume_affine", residual);
            yaml_read_bool(node, "ASSUME_AFFINE_JACOBIAN_ACTION", jacobian_action);
            yaml_read_bool(node, "jacobian_action_assume_affine", jacobian_action);
            yaml_read_bool(node, "ASSUME_AFFINE_APPLY", jacobian_action);
            yaml_read_bool(node, "apply_assume_affine", jacobian_action);
        }
#endif  // SFEM_ENABLE_RYAML

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedTwoPhaseFlow: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
            switch (dim) {
                case 2:
                    values[index++] = parameters.require_real_value("C_ka1");
                    values[index++] = parameters.require_real_value("C_ka2");
                    values[index++] = parameters.require_real_value("C_kw1");
                    values[index++] = parameters.require_real_value("K_0");
                    values[index++] = parameters.require_real_value("K_1");
                    values[index++] = parameters.require_real_value("K_2");
                    values[index++] = parameters.require_real_value("K_3");
                    values[index++] = parameters.require_real_value("M_c");
                    values[index++] = parameters.require_real_value("P_r");
                    values[index++] = parameters.require_real_value("R");
                    values[index++] = parameters.require_real_value("S_res");
                    values[index++] = parameters.require_real_value("T");
                    values[index++] = parameters.require_real_value("Z");
                    values[index++] = parameters.require_real_value("dt");
                    values[index++] = parameters.require_real_value("kappa_T");
                    values[index++] = parameters.require_real_value("m");
                    values[index++] = parameters.require_real_value("mu_c");
                    values[index++] = parameters.require_real_value("mu_w");
                    values[index++] = parameters.require_real_value("p_wr");
                    values[index++] = parameters.require_real_value("porosity");
                    values[index++] = parameters.require_real_value("rho_w0");
                    break;
                case 3:
                    values[index++] = parameters.require_real_value("C_ka1");
                    values[index++] = parameters.require_real_value("C_ka2");
                    values[index++] = parameters.require_real_value("C_kw1");
                    values[index++] = parameters.require_real_value("K_0");
                    values[index++] = parameters.require_real_value("K_1");
                    values[index++] = parameters.require_real_value("K_2");
                    values[index++] = parameters.require_real_value("K_3");
                    values[index++] = parameters.require_real_value("K_4");
                    values[index++] = parameters.require_real_value("K_5");
                    values[index++] = parameters.require_real_value("K_6");
                    values[index++] = parameters.require_real_value("K_7");
                    values[index++] = parameters.require_real_value("K_8");
                    values[index++] = parameters.require_real_value("M_c");
                    values[index++] = parameters.require_real_value("P_r");
                    values[index++] = parameters.require_real_value("R");
                    values[index++] = parameters.require_real_value("S_res");
                    values[index++] = parameters.require_real_value("T");
                    values[index++] = parameters.require_real_value("Z");
                    values[index++] = parameters.require_real_value("dt");
                    values[index++] = parameters.require_real_value("kappa_T");
                    values[index++] = parameters.require_real_value("m");
                    values[index++] = parameters.require_real_value("mu_c");
                    values[index++] = parameters.require_real_value("mu_w");
                    values[index++] = parameters.require_real_value("p_wr");
                    values[index++] = parameters.require_real_value("porosity");
                    values[index++] = parameters.require_real_value("rho_w0");
                    break;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\n", dim);
                    break;
            }
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 2;
                case 3: return 2;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual block size\n", dim);
                    return 0;
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
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> GeneratedTwoPhaseFlow::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
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
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->residual_uses_affine ||
                impl_->jacobian_action_uses_affine;
        if (needs_affine_geometry) {
            for (auto &entry : impl_->domains->domains()) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
        }
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedTwoPhaseFlow::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::gradient");
        if (!impl_->previous) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires a previous state\n");
            return SFEM_FAILURE;
        }
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedTwoPhaseFlow affine residual requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
            const real_t *const previous = impl_->previous;
            switch (domain.element_type) {
                case smesh::TRI3: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = state + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = state + 1;
                    const real_t *const SFEM_RESTRICT p_w_old_data = previous + 0;
                    const real_t *const SFEM_RESTRICT p_c_old_data = previous + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->residual_uses_affine ? two_phase_flow_tri3_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_tri3_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::TET4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = state + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = state + 1;
                    const real_t *const SFEM_RESTRICT p_w_old_data = previous + 0;
                    const real_t *const SFEM_RESTRICT p_c_old_data = previous + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->residual_uses_affine ? two_phase_flow_tet4_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_tet4_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::QUAD4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = state + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = state + 1;
                    const real_t *const SFEM_RESTRICT p_w_old_data = previous + 0;
                    const real_t *const SFEM_RESTRICT p_c_old_data = previous + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->residual_uses_affine ? two_phase_flow_quad4_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_quad4_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::HEX8: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = state + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = state + 1;
                    const real_t *const SFEM_RESTRICT p_w_old_data = previous + 0;
                    const real_t *const SFEM_RESTRICT p_c_old_data = previous + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->residual_uses_affine ? two_phase_flow_hex8_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_hex8_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_old_data, p_c_old_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
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
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::apply");
        const real_t *const current = state ? state : impl_->current;
        if (!current) {
            SFEM_ERROR("GeneratedTwoPhaseFlow requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedTwoPhaseFlow affine jacobian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            switch (domain.element_type) {
                case smesh::TRI3: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = current + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = current + 1;
                    const real_t *const SFEM_RESTRICT p_w_direction_data = direction + 0;
                    const real_t *const SFEM_RESTRICT p_c_direction_data = direction + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->jacobian_action_uses_affine ? two_phase_flow_tri3_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_tri3_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::TET4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = current + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = current + 1;
                    const real_t *const SFEM_RESTRICT p_w_direction_data = direction + 0;
                    const real_t *const SFEM_RESTRICT p_c_direction_data = direction + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->jacobian_action_uses_affine ? two_phase_flow_tet4_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_tet4_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::QUAD4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = current + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = current + 1;
                    const real_t *const SFEM_RESTRICT p_w_direction_data = direction + 0;
                    const real_t *const SFEM_RESTRICT p_c_direction_data = direction + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->jacobian_action_uses_affine ? two_phase_flow_quad4_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
                case smesh::HEX8: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT p_w_data = current + 0;
                    const real_t *const SFEM_RESTRICT p_c_data = current + 1;
                    const real_t *const SFEM_RESTRICT p_w_direction_data = direction + 0;
                    const real_t *const SFEM_RESTRICT p_c_direction_data = direction + 1;
                    real_t *const SFEM_RESTRICT p_w_out = out + 0;
                    real_t *const SFEM_RESTRICT p_c_out = out + 1;
                    return impl_->jacobian_action_uses_affine ? two_phase_flow_hex8_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out) : two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], storage[2], storage[3], storage[4], storage[5], storage[6], storage[7], storage[8], storage[9], storage[10], storage[11], storage[12], storage[13], storage[14], storage[15], storage[16], storage[17], storage[18], storage[19], storage[20], storage[21], storage[22], storage[23], storage[24], storage[25], FIELD_STRIDE, p_w_data, p_c_data, FIELD_STRIDE, p_w_direction_data, p_c_direction_data, FIELD_STRIDE, p_w_out, p_c_out);
                }
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
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::set_field");
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
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void GeneratedTwoPhaseFlow::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::set_option");
        if (name == "assume_affine") {
            impl_->residual_uses_affine = val;
            impl_->jacobian_action_uses_affine = val;
        } else if (name == "residual_assume_affine" ||
                   name == "gradient_assume_affine") {
            impl_->residual_uses_affine = val;
        } else if (name == "jacobian_action_assume_affine" ||
                   name == "apply_assume_affine") {
            impl_->jacobian_action_uses_affine = val;
        }
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedTwoPhaseFlow::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::create_from_yaml");
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

        read_residual_affine_options(node,
                                     ret->impl_->residual_uses_affine,
                                     ret->impl_->jacobian_action_uses_affine);

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
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::hessian_crs");
        return SFEM_FAILURE;
    }

    int GeneratedTwoPhaseFlow::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedTwoPhaseFlow::value");
        return SFEM_FAILURE;
    }
}  // namespace sfem
