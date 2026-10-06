#include "sfem_GeneratedPoroHyperelasticity.hpp"
#include "sfem_GeneratedPoroHyperelasticity_c_abi.hpp"

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
        constexpr int MAX_PARAMETERS = 6;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("mu", 1);
            parameters.set_value("lmbda", 1);
            parameters.set_value("alpha", 0.80000000000000004);
            parameters.set_value("storage", 0.001);
            parameters.set_value("dt", 1);
            parameters.set_value("hydraulic_conductivity", 1);
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

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 6;
        constexpr int N_MATERIAL_PARAMETERS = 6;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"mu", "lmbda", "alpha", "storage", "dt", "hydraulic_conductivity"};

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
            values[2] = 0.80000000000000004;
            values[3] = 0.001;
            values[4] = 1;
            values[5] = 1;
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

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedPoroHyperelasticity: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                if (entry.second.user_data) {
                    continue;
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             real_t *const values) {
            values[0] = parameters.require_real_value("mu");
            values[1] = parameters.require_real_value("lmbda");
            values[2] = parameters.require_real_value("alpha");
            values[3] = parameters.require_real_value("storage");
            values[4] = parameters.require_real_value("dt");
            values[5] = parameters.require_real_value("hydraulic_conductivity");
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 3;
                case 3: return 4;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated coupled block size\n", dim);
                    return 0;
            }
        }
    }  // namespace

    class GeneratedPoroHyperelasticity::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> GeneratedPoroHyperelasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("GeneratedPoroHyperelasticity requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<GeneratedPoroHyperelasticity>(space);
        op->initialize();
        return op;
    }

    GeneratedPoroHyperelasticity::GeneratedPoroHyperelasticity(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedPoroHyperelasticity::~GeneratedPoroHyperelasticity() = default;

    ptrdiff_t GeneratedPoroHyperelasticity::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedPoroHyperelasticity::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedPoroHyperelasticity::flops_value() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tri6_tri6_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tet10_tet10_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_hex27_hex27_objective_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedPoroHyperelasticity::memory_traffic_bytes_value() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tri6_tri6_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tri6_tri6_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tet10_tet10_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tet10_tet10_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_hex27_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_hex27_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedPoroHyperelasticity::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tri6_tri6_gradient_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_tri6_tri3_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tet10_tet10_gradient_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_hex27_hex27_gradient_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_hex27_hex8_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedPoroHyperelasticity::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tri6_tri6_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tri6_tri6_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_tri6_tri3_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_tri6_tri3_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tet10_tet10_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tet10_tet10_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_hex27_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_hex27_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_hex27_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_hex27_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedPoroHyperelasticity::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tri6_tri6_apply_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_tri6_tri3_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_tet10_tet10_apply_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_solid_hex27_hex27_apply_soa_diagnostics(), nelements);
                    total += sfem::codegen::KernelDiagnostics_total_flops(poro_hyperelasticity_poro_hex27_hex8_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedPoroHyperelasticity::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tri6_tri6_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tri6_tri6_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_tri6_tri3_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_tri6_tri3_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_tet10_tet10_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_tet10_tet10_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_solid_hex27_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_solid_hex27_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(poro_hyperelasticity_poro_hex27_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(poro_hyperelasticity_poro_hex27_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    int GeneratedPoroHyperelasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine ||
                impl_->residual_uses_affine ||
                impl_->jacobian_action_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
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

    int GeneratedPoroHyperelasticity::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedPoroHyperelasticity::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedPoroHyperelasticity::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::gradient");
        if (!impl_->previous) {
            SFEM_ERROR("GeneratedPoroHyperelasticity requires a previous state\n");
            return SFEM_FAILURE;
        }
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine || impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedPoroHyperelasticity affine gradient/residual requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
            const real_t *const previous = impl_->previous;
            switch (domain.element_type) {
                case smesh::TRI6: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[2] = {state + 0, state + 1};
                    const real_t *const SFEM_RESTRICT p_data = state + 2;
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    const real_t *const SFEM_RESTRICT p_old_data = previous + 2;
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    real_t *const SFEM_RESTRICT p_out = out + 2;
                    int status = impl_->gradient_uses_affine ? poro_hyperelasticity_solid_tri6_tri6_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 3, state + 0, state + 1, 3, out + 0, out + 1) : poro_hyperelasticity_solid_tri6_tri6_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, 3, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? poro_hyperelasticity_poro_tri6_tri3_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[2], storage[4], storage[5], storage[3], 3, u_data, p_data, 3, u_old_data, p_old_data, 3, u_out, p_out) : poro_hyperelasticity_poro_tri6_tri3_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 3, u_data, p_data, 3, u_old_data, p_old_data, 3, u_out, p_out);
                }
                case smesh::TET10: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 4;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT p_data = state + 3;
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT p_old_data = previous + 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    real_t *const SFEM_RESTRICT p_out = out + 3;
                    int status = impl_->gradient_uses_affine ? poro_hyperelasticity_solid_tet10_tet10_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, out + 0, out + 1, out + 2) : poro_hyperelasticity_solid_tet10_tet10_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[2], storage[4], storage[5], storage[3], 4, u_data, p_data, 4, u_old_data, p_old_data, 4, u_out, p_out) : poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 4, u_data, p_data, 4, u_old_data, p_old_data, 4, u_out, p_out);
                }
                case smesh::HEX27: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 4;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT p_data = state + 3;
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT p_old_data = previous + 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    real_t *const SFEM_RESTRICT p_out = out + 3;
                    int status = impl_->gradient_uses_affine ? poro_hyperelasticity_solid_hex27_hex27_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, out + 0, out + 1, out + 2) : poro_hyperelasticity_solid_hex27_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? poro_hyperelasticity_poro_hex27_hex8_residual_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[2], storage[4], storage[5], storage[3], 4, u_data, p_data, 4, u_old_data, p_old_data, 4, u_out, p_out) : poro_hyperelasticity_poro_hex27_hex8_residual_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 4, u_data, p_data, 4, u_old_data, p_old_data, 4, u_out, p_out);
                }
                default:
                    SFEM_ERROR("GeneratedPoroHyperelasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedPoroHyperelasticity::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::apply");
        const real_t *const current = state ? state : impl_->current;
        if (!current) {
            SFEM_ERROR("GeneratedPoroHyperelasticity requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine || impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedPoroHyperelasticity affine hessian/jacobian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);

            switch (domain.element_type) {
                case smesh::TRI6: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_direction_data[2] = {direction + 0, direction + 1};
                    const real_t *const SFEM_RESTRICT p_direction_data = direction + 2;
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    real_t *const SFEM_RESTRICT p_out = out + 2;
                    int status = impl_->apply_uses_affine ? poro_hyperelasticity_solid_tri6_tri6_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 3, state + 0, state + 1, 3, direction + 0, direction + 1, 3, out + 0, out + 1) : poro_hyperelasticity_solid_tri6_tri6_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, 3, direction + 0, direction + 1, 3, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? poro_hyperelasticity_poro_tri6_tri3_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[2], storage[4], storage[5], storage[3], 3, u_direction_data, p_direction_data, 3, u_out, p_out) : poro_hyperelasticity_poro_tri6_tri3_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 3, u_direction_data, p_direction_data, 3, u_out, p_out);
                }
                case smesh::TET10: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 4;
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    const real_t *const SFEM_RESTRICT p_direction_data = direction + 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    real_t *const SFEM_RESTRICT p_out = out + 3;
                    int status = impl_->apply_uses_affine ? poro_hyperelasticity_solid_tet10_tet10_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, direction + 0, direction + 1, direction + 2, 4, out + 0, out + 1, out + 2) : poro_hyperelasticity_solid_tet10_tet10_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, direction + 0, direction + 1, direction + 2, 4, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[2], storage[4], storage[5], storage[3], 4, u_direction_data, p_direction_data, 4, u_out, p_out) : poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 4, u_direction_data, p_direction_data, 4, u_out, p_out);
                }
                case smesh::HEX27: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 4;
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    const real_t *const SFEM_RESTRICT p_direction_data = direction + 3;
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    real_t *const SFEM_RESTRICT p_out = out + 3;
                    int status = impl_->apply_uses_affine ? poro_hyperelasticity_solid_hex27_hex27_apply_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, direction + 0, direction + 1, direction + 2, 4, out + 0, out + 1, out + 2) : poro_hyperelasticity_solid_hex27_hex27_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, 4, direction + 0, direction + 1, direction + 2, 4, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? poro_hyperelasticity_poro_hex27_hex8_jacobian_action_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[2], storage[4], storage[5], storage[3], 4, u_direction_data, p_direction_data, 4, u_out, p_out) : poro_hyperelasticity_poro_hex27_hex8_jacobian_action_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[2], storage[4], storage[5], storage[3], 4, u_direction_data, p_direction_data, 4, u_out, p_out);
                }
                default:
                    SFEM_ERROR("GeneratedPoroHyperelasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedPoroHyperelasticity::value(const real_t *state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedPoroHyperelasticity affine objective requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? poro_hyperelasticity_solid_tri6_tri6_objective_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 3, state + 0, state + 1, impl_->element_values.get()) : poro_hyperelasticity_solid_tri6_tri6_objective_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine ? poro_hyperelasticity_solid_tet10_tet10_objective_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, impl_->element_values.get()) : poro_hyperelasticity_solid_tet10_tet10_objective_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine ? poro_hyperelasticity_solid_hex27_hex27_objective_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 4, state + 0, state + 1, state + 2, impl_->element_values.get()) : poro_hyperelasticity_solid_hex27_hex27_objective_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 4, state + 0, state + 1, state + 2, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedPoroHyperelasticity does not support element type %d\n",
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
    void GeneratedPoroHyperelasticity::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedPoroHyperelasticity supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void GeneratedPoroHyperelasticity::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::set_option");
        AffineOption options[] = {
            {"ASSUME_AFFINE_OBJECTIVE", &impl_->objective_uses_affine},
            {"objective_assume_affine", &impl_->objective_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &impl_->gradient_uses_affine},
            {"gradient_assume_affine", &impl_->gradient_uses_affine},
            {"ASSUME_AFFINE_HESSIAN_ACTION", &impl_->apply_uses_affine},
            {"hessian_action_assume_affine", &impl_->apply_uses_affine},
            {"ASSUME_AFFINE_APPLY", &impl_->apply_uses_affine},
            {"apply_assume_affine", &impl_->apply_uses_affine},
            {"ASSUME_AFFINE_RESIDUAL", &impl_->residual_uses_affine},
            {"residual_assume_affine", &impl_->residual_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &impl_->residual_uses_affine},
            {"gradient_assume_affine", &impl_->residual_uses_affine},
            {"ASSUME_AFFINE_JACOBIAN_ACTION", &impl_->jacobian_action_uses_affine},
            {"jacobian_action_assume_affine", &impl_->jacobian_action_uses_affine},
            {"ASSUME_AFFINE_APPLY", &impl_->jacobian_action_uses_affine},
            {"apply_assume_affine", &impl_->jacobian_action_uses_affine},
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains &&
            cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
            SFEM_ERROR("GeneratedPoroHyperelasticity failed to cache affine geometry\n");
        }
    }

    void GeneratedPoroHyperelasticity::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedPoroHyperelasticity::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::create_from_yaml");
        auto ret = std::make_shared<GeneratedPoroHyperelasticity>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
            {"ASSUME_AFFINE_OBJECTIVE", &ret->impl_->objective_uses_affine},
            {"objective_assume_affine", &ret->impl_->objective_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &ret->impl_->gradient_uses_affine},
            {"gradient_assume_affine", &ret->impl_->gradient_uses_affine},
            {"ASSUME_AFFINE_HESSIAN_ACTION", &ret->impl_->apply_uses_affine},
            {"hessian_action_assume_affine", &ret->impl_->apply_uses_affine},
            {"ASSUME_AFFINE_APPLY", &ret->impl_->apply_uses_affine},
            {"apply_assume_affine", &ret->impl_->apply_uses_affine},
            {"ASSUME_AFFINE_RESIDUAL", &ret->impl_->residual_uses_affine},
            {"residual_assume_affine", &ret->impl_->residual_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &ret->impl_->residual_uses_affine},
            {"gradient_assume_affine", &ret->impl_->residual_uses_affine},
            {"ASSUME_AFFINE_JACOBIAN_ACTION", &ret->impl_->jacobian_action_uses_affine},
            {"jacobian_action_assume_affine", &ret->impl_->jacobian_action_uses_affine},
            {"ASSUME_AFFINE_APPLY", &ret->impl_->jacobian_action_uses_affine},
            {"apply_assume_affine", &ret->impl_->jacobian_action_uses_affine},
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

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

    int GeneratedPoroHyperelasticity::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedPoroHyperelasticity::hessian_crs");
        return SFEM_FAILURE;
    }
}  // namespace sfem
