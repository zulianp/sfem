#include "sfem_GeneratedLaplace.hpp"
#include "sfem_GeneratedLaplace_c_abi.hpp"
#include "sfem_PackedLaplacian.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <cstring>



namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = 1;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("kappa", 1);
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
            values[0] = 1;
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 1;
        constexpr int N_MATERIAL_PARAMETERS = 1;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"kappa"};

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

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedLaplace: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian;
            std::shared_ptr<smesh::FFF> metric_soa;
            std::shared_ptr<smesh::FFF> metric_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains,
                                  const bool needs_jacobian,
                                  const bool needs_metric_soa,
                                  const bool needs_metric_aos) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        entry.second.user_data);
                if (!cache) {
                    cache = std::make_shared<AffineGeometryCache>();
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                if (needs_jacobian && !cache->jacobian) {
                    cache->jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_soa && !cache->metric_soa) {
                    cache->metric_soa = smesh::FFF::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_soa) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_aos && !cache->metric_aos) {
                    cache->metric_aos = smesh::FFF::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
            switch (dim) {
                case 2:
                    values[index++] = parameters.require_real_value("kappa");
                    break;
                case 3:
                    values[index++] = parameters.require_real_value("kappa");
                    break;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\n", dim);
                    break;
            }
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 1;
                case 3: return 1;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated residual block size\n", dim);
                    return 0;
            }
        }

        bool packed_laplacian_apply_supported(const smesh::ElemType element_type) {
            switch (element_type) {
                case smesh::TET4:
                case smesh::TET10:
                case smesh::HEX8:
                    return true;
                default:
                    return false;
            }
        }

        bool can_use_packed_laplacian_apply(const FunctionSpace &space,
                                            MultiDomainOp &domains) {
            if (!space.has_packed_mesh()) {
                return false;
            }

            for (auto &entry : domains.domains()) {
                const OpDomain &domain = entry.second;
                if (!packed_laplacian_apply_supported(domain.element_type)) {
                    return false;
                }
                if (domain.parameters->require_real_value("kappa") != real_t(1)) {
                    return false;
                }
            }

            return true;
        }

    }  // namespace

    class GeneratedLaplace::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Op> packed_affine_apply;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> GeneratedLaplace::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("GeneratedLaplace requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<GeneratedLaplace>(space);
        op->initialize();
        return op;
    }

    GeneratedLaplace::GeneratedLaplace(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedLaplace::~GeneratedLaplace() = default;

    ptrdiff_t GeneratedLaplace::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedLaplace::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedLaplace::flops_value() const {
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

    size_t GeneratedLaplace::memory_traffic_bytes_value() const {
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

    double GeneratedLaplace::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tri3_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tri3_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tri6_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tri6_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_quad4_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_quad4_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_quad4_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_quad4_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tet4_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tet4_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tet10_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tet10_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_hex8_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_hex8_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_hex27_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_hex27_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex8_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex8_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex27_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex27_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex64_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex64_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex125_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex125_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex729_residual_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex729_residual_element_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedLaplace::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tri3_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tri3_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tri6_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tri6_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_quad4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_quad4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_quad4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_quad4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tet4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tet4_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tet10_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tet10_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_hex27_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_hex27_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex8_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex27_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex27_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex64_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex64_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex125_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex125_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex729_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex729_residual_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedLaplace::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tri3_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tri3_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tri6_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tri6_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_quad4_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_quad4_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tet4_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tet4_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_tet10_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_tet10_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_hex8_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_hex8_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_hex27_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_hex27_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedLaplace::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tri3_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tri3_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tri6_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tri6_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_quad4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_quad4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tet4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tet4_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_tet10_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_tet10_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_hex27_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_hex27_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    int GeneratedLaplace::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        seed_material(*impl_->domains);
        const bool needs_affine_jacobian =
                (impl_->residual_uses_affine && true) ||
                (impl_->jacobian_action_uses_affine && true);
        const bool needs_affine_metric =
                (impl_->residual_uses_affine && (false || true)) ||
                (impl_->jacobian_action_uses_affine && (false || true));
        const bool needs_affine_metric_soa =
                (impl_->residual_uses_affine && false) ||
                (impl_->jacobian_action_uses_affine && false);
        const bool needs_affine_metric_aos =
                (impl_->residual_uses_affine && true) ||
                (impl_->jacobian_action_uses_affine && true);
        if (needs_affine_jacobian || needs_affine_metric) {
            return cache_affine_geometry(impl_->space,
                                         *impl_->domains,
                                         needs_affine_jacobian,
                                         needs_affine_metric_soa,
                                         needs_affine_metric_aos);
        }
        return SFEM_SUCCESS;
    }

    int GeneratedLaplace::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedLaplace::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedLaplace::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::gradient");

        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->residual_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("GeneratedLaplace affine residual requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                if (true) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("GeneratedLaplace affine residual requires cached jacobian geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (false) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("GeneratedLaplace affine residual requires cached SoA metric geometry\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (true) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("GeneratedLaplace affine residual requires cached AoS metric geometry\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                    const real_t *const SFEM_RESTRICT u_data = state + 0;
                    real_t *const SFEM_RESTRICT u_out = out + 0;
                if (impl_->residual_uses_affine) {
                    if ((domain.element_type == smesh::TRI3) && storage[0] == real_t(1)) {
                        return laplace_residual_2d_affine_mesh_soa_aos_unit(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, u_data, u_out);
                    }
                    if (domain.element_type == smesh::TRI3) {
                        return laplace_residual_2d_affine_mesh_soa_aos(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
                    }
                    return laplace_residual_2d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
                }
                return laplace_residual_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
            }
            else if (dim == 3) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                    const real_t *const SFEM_RESTRICT u_data = state + 0;
                    real_t *const SFEM_RESTRICT u_out = out + 0;
                if (impl_->residual_uses_affine) {
                    if ((domain.element_type == smesh::TET4) && storage[0] == real_t(1)) {
                        return laplace_residual_3d_affine_mesh_soa_aos_unit(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, u_data, u_out);
                    }
                    if (domain.element_type == smesh::TET4) {
                        return laplace_residual_3d_affine_mesh_soa_aos(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
                    }
                    return laplace_residual_3d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
                }
                return laplace_residual_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], FIELD_STRIDE, u_data, FIELD_STRIDE, u_out);
            }
            SFEM_ERROR("laplace residual does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::apply");
        const real_t *const current = state ? state : impl_->current;

        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());

        if (impl_->jacobian_action_uses_affine &&
            can_use_packed_laplacian_apply(*impl_->space, *impl_->domains)) {
            if (!impl_->packed_affine_apply) {
                impl_->packed_affine_apply = std::make_shared<PackedLaplacian>(impl_->space);
                if (impl_->packed_affine_apply->initialize() != SFEM_SUCCESS) {
                    SFEM_ERROR("GeneratedLaplace failed to initialize packed affine apply backend\n");
                    return SFEM_FAILURE;
                }
            }
            return impl_->packed_affine_apply->apply(current, direction, out);
        }

        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->jacobian_action_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("GeneratedLaplace affine jacobian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                if (true) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("GeneratedLaplace affine jacobian action requires cached jacobian geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (false) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("GeneratedLaplace affine jacobian action requires cached SoA metric geometry\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (true) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("GeneratedLaplace affine jacobian action requires cached AoS metric geometry\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                    const real_t *const SFEM_RESTRICT u_direction_data = direction + 0;
                    real_t *const SFEM_RESTRICT u_out = out + 0;
                if (impl_->jacobian_action_uses_affine) {
                    if ((domain.element_type == smesh::TRI3) && storage[0] == real_t(1)) {
                        return laplace_jacobian_action_2d_affine_mesh_soa_aos_unit(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, u_direction_data, u_out);
                    }
                    if (domain.element_type == smesh::TRI3) {
                        return laplace_jacobian_action_2d_affine_mesh_soa_aos(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                    }
                    return laplace_jacobian_action_2d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                }
                if (impl_->space->has_packed_mesh()) {
                    auto packed = impl_->space->packed_mesh();
                    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                    if (packed_block >= 0) {
                        auto packed_elements = packed->elements(packed_block);
                        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                        auto n_shared_nodes = packed->n_shared(packed_block);
                        auto ghost_ptr = packed->ghost_ptr(packed_block);
                        auto ghost_idx = packed->ghost_idx(packed_block);
                        return laplace_jacobian_action_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                    }
                }
                return laplace_jacobian_action_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
            }
            else if (dim == 3) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                    const real_t *const SFEM_RESTRICT u_direction_data = direction + 0;
                    real_t *const SFEM_RESTRICT u_out = out + 0;
                if (impl_->jacobian_action_uses_affine) {
                    if ((domain.element_type == smesh::TET4) && storage[0] == real_t(1)) {
                        return laplace_jacobian_action_3d_affine_mesh_soa_aos_unit(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, u_direction_data, u_out);
                    }
                    if (domain.element_type == smesh::TET4) {
                        return laplace_jacobian_action_3d_affine_mesh_soa_aos(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric_aos, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                    }
                    return laplace_jacobian_action_3d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                }
                if (impl_->space->has_packed_mesh()) {
                    auto packed = impl_->space->packed_mesh();
                    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                    if (packed_block >= 0) {
                        auto packed_elements = packed->elements(packed_block);
                        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                        auto n_shared_nodes = packed->n_shared(packed_block);
                        auto ghost_ptr = packed->ghost_ptr(packed_block);
                        auto ghost_idx = packed->ghost_idx(packed_block);
                        return laplace_jacobian_action_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
                    }
                }
                return laplace_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], FIELD_STRIDE, u_direction_data, FIELD_STRIDE, u_out);
            }
            SFEM_ERROR("laplace jacobian_action does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    void GeneratedLaplace::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedLaplace supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void GeneratedLaplace::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void GeneratedLaplace::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::set_option");
        AffineOption options[] = {
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
        if (matched && val && impl_->domains) {
            const bool needs_affine_jacobian =
                    (impl_->residual_uses_affine && true) ||
                    (impl_->jacobian_action_uses_affine && true);
            const bool needs_affine_metric =
                    (impl_->residual_uses_affine && (false || true)) ||
                    (impl_->jacobian_action_uses_affine && (false || true));
            const bool needs_affine_metric_soa =
                    (impl_->residual_uses_affine && false) ||
                    (impl_->jacobian_action_uses_affine && false);
            const bool needs_affine_metric_aos =
                    (impl_->residual_uses_affine && true) ||
                    (impl_->jacobian_action_uses_affine && true);
            if (cache_affine_geometry(impl_->space,
                                      *impl_->domains,
                                      needs_affine_jacobian,
                                      needs_affine_metric_soa,
                                      needs_affine_metric_aos) != SFEM_SUCCESS) {
                SFEM_ERROR("GeneratedLaplace failed to cache affine geometry\n");
            }
        }
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedLaplace::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::create_from_yaml");
        auto ret = std::make_shared<GeneratedLaplace>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
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

    int GeneratedLaplace::hessian_crs(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_crs");


        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_crs_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], rowptr, colidx, values);
            }
            else if (dim == 3) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_crs_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], rowptr, colidx, values);
            }
            SFEM_ERROR("laplace hessian_crs does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::hessian_bsr(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_bsr");


        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_bsr_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], rowptr, colidx, values);
            }
            else if (dim == 3) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_bsr_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], rowptr, colidx, values);
            }
            SFEM_ERROR("laplace hessian_bsr does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::hessian_dia(const real_t *const state,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_dia");


        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);

            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_dia_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], diag_offsets, ndiag, values);
            }
            else if (dim == 3) {
                static constexpr ptrdiff_t FIELD_STRIDE = 1;
                return laplace_hessian_dia_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], diag_offsets, ndiag, values);
            }
            SFEM_ERROR("laplace hessian_dia does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::value");
        return SFEM_FAILURE;
    }
}  // namespace sfem
