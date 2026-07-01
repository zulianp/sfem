#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_GeneratedLinearElasticity_c_abi.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>



namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
            parameters.set_value("mu", 1);
            parameters.set_value("lmbda", 1);
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 2;
        constexpr int N_MATERIAL_PARAMETERS = 2;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"mu", "lmbda"};

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
            values[0] = 1;
            values[1] = 1;
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
            SFEM_ERROR("GeneratedLinearElasticity: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }
    }  // namespace

    class GeneratedLinearElasticity::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
    };

    std::unique_ptr<Op> GeneratedLinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("GeneratedLinearElasticity requires block_size=spatial_dimension\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedLinearElasticity>(space);
        op->initialize();
        return op;
    }

    GeneratedLinearElasticity::GeneratedLinearElasticity(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedLinearElasticity::~GeneratedLinearElasticity() = default;

    ptrdiff_t GeneratedLinearElasticity::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedLinearElasticity::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedLinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
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
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int GeneratedLinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedLinearElasticity affine gradient requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            switch (domain.element_type) {
                case smesh::TRI3:
                    return impl_->gradient_uses_affine ? linear_elasticity_tri3_tri3_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_tri3_tri3_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TRI6:
                    return impl_->gradient_uses_affine ? linear_elasticity_tri6_tri6_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_tri6_tri6_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::QUAD4:
                    return impl_->gradient_uses_affine ? linear_elasticity_quad4_quad4_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TET4:
                    return impl_->gradient_uses_affine ? linear_elasticity_tet4_tet4_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet4_tet4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::TET10:
                    return impl_->gradient_uses_affine ? linear_elasticity_tet10_tet10_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX8:
                    return impl_->gradient_uses_affine ? linear_elasticity_hex8_hex8_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex8_hex8_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX27:
                    return impl_->gradient_uses_affine ? linear_elasticity_hex27_hex27_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex27_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX8:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex8_proteus_hex8_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX27:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX64:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex64_proteus_hex64_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX125:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex125_proteus_hex125_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX729:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedLinearElasticity::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedLinearElasticity affine hessian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            switch (domain.element_type) {
                case smesh::TRI3:
                    return impl_->apply_uses_affine ? linear_elasticity_tri3_tri3_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1) : linear_elasticity_tri3_tri3_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::TRI6:
                    return impl_->apply_uses_affine ? linear_elasticity_tri6_tri6_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1) : linear_elasticity_tri6_tri6_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::QUAD4:
                    return impl_->apply_uses_affine ? linear_elasticity_quad4_quad4_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1) : linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::TET4:
                    return impl_->apply_uses_affine ? linear_elasticity_tet4_tet4_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet4_tet4_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::TET10:
                    return impl_->apply_uses_affine ? linear_elasticity_tet10_tet10_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX8:
                    return impl_->apply_uses_affine ? linear_elasticity_hex8_hex8_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex8_hex8_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX27:
                    return impl_->apply_uses_affine ? linear_elasticity_hex27_hex27_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex27_hex27_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX8:
                    return impl_->apply_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex8_proteus_hex8_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX27:
                    return impl_->apply_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX64:
                    return impl_->apply_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex64_proteus_hex64_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX125:
                    return impl_->apply_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex125_proteus_hex125_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX729:
                    return impl_->apply_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedLinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedLinearElasticity affine objective requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri3_tri3_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_tri3_tri3_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri6_tri6_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_tri6_tri6_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_quad4_quad4_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet4_tet4_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_tet4_tet4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet10_tet10_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex8_hex8_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_hex8_hex8_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex27_hex27_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_hex27_hex27_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex8_proteus_hex8_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX64:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex64_proteus_hex64_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX125:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex125_proteus_hex125_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX729:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }

    int GeneratedLinearElasticity::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const ptrdiff_t nvalues = (ptrdiff_t)nsteps * nelements;
            const real_t *const *adjugate = nullptr;
            const real_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedLinearElasticity affine objective_steps requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const real_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri3_tri3_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tri3_tri3_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri6_tri6_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tri6_tri6_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_quad4_quad4_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_quad4_quad4_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet4_tet4_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tet4_tet4_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex8_hex8_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_hex8_hex8_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex27_hex27_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_hex27_hex27_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX64:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX125:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX729:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            for (int step = 0; step < nsteps; ++step) {
                real_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (ptrdiff_t element = 0; element < nelements; ++element) {
                    sum += impl_->element_values[(ptrdiff_t)step * nelements + element];
                }
                out[step] += sum;
            }
            return SFEM_SUCCESS;
        });
    }

    int GeneratedLinearElasticity::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_crs");
        return SFEM_FAILURE;
    }

    void GeneratedLinearElasticity::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::set_option");
        if (name == "assume_affine") {
            impl_->objective_uses_affine = val;
            impl_->gradient_uses_affine = val;
            impl_->apply_uses_affine = val;
        } else if (name == "objective_assume_affine") {
            impl_->objective_uses_affine = val;
        } else if (name == "gradient_assume_affine") {
            impl_->gradient_uses_affine = val;
        } else if (name == "hessian_action_assume_affine" ||
                   name == "apply_assume_affine") {
            impl_->apply_uses_affine = val;
        }
    }

    void GeneratedLinearElasticity::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedLinearElasticity::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::create_from_yaml");
        auto ret = std::make_shared<GeneratedLinearElasticity>(space);

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

        read_affine_options(node,
                            ret->impl_->objective_uses_affine,
                            ret->impl_->gradient_uses_affine,
                            ret->impl_->apply_uses_affine);

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
}  // namespace sfem
