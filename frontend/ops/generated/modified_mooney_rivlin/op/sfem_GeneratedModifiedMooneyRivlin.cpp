#include "sfem_GeneratedModifiedMooneyRivlin.hpp"
#include "sfem_GeneratedModifiedMooneyRivlin_c_abi.hpp"
#include "packed_thread_scratch.hpp"
#include "smesh_env.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>



namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
            parameters.set_value("c1", 1);
            parameters.set_value("c2", 1);
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
            values[1] = 1;
            values[2] = 1;
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 3;
        constexpr int N_MATERIAL_PARAMETERS = 3;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"c1", "c2", "kappa"};

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
            SFEM_ERROR("GeneratedModifiedMooneyRivlin: mesh block pointer not found in mesh.blocks()\n");
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
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            const bool needs_jacobian_aos =
                    false ||
                    false;
            for (auto &entry : domains.domains()) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if (needs_jacobian_aos) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }
    }  // namespace

    class GeneratedModifiedMooneyRivlin::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool use_packed_two_pass{false};
        std::vector<SharedBuffer<real_t>> packed_ghost_buf;
    };

    std::unique_ptr<Op> GeneratedModifiedMooneyRivlin::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin requires block_size=spatial_dimension\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedModifiedMooneyRivlin>(space);
        op->initialize();
        return op;
    }

    GeneratedModifiedMooneyRivlin::GeneratedModifiedMooneyRivlin(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedModifiedMooneyRivlin::~GeneratedModifiedMooneyRivlin() = default;

    ptrdiff_t GeneratedModifiedMooneyRivlin::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedModifiedMooneyRivlin::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedModifiedMooneyRivlin::flops_value() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_objective_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_objective_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedModifiedMooneyRivlin::memory_traffic_bytes_value() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_objective_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_objective_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedModifiedMooneyRivlin::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedModifiedMooneyRivlin::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedModifiedMooneyRivlin::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_apply_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedModifiedMooneyRivlin::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = modified_mooney_rivlin_apply_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    int GeneratedModifiedMooneyRivlin::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::initialize");
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
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if ((impl_->gradient_uses_affine && false) ||
                    (impl_->apply_uses_affine && false)) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        impl_->use_packed_two_pass = smesh::Env::read("SFEM_PACKED_TWO_PASS", false);
        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            const size_t scratch_size = (size_t)dim * (size_t)max_nodes_per_pack;
            sfem::codegen::prealloc_thread_scratch<real_t>(0, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(1, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(2, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(3, scratch_size);
            impl_->packed_ghost_buf.resize((size_t)packed->n_blocks());
            for (int b = 0; b < packed->n_blocks(); ++b) {
                const ptrdiff_t n_ghost = packed->n_ghost_entries(b);
                const ptrdiff_t n_slots = (n_ghost > 0 ? n_ghost : 1) * (ptrdiff_t)dim;
                impl_->packed_ghost_buf[b] = create_host_buffer<real_t>(n_slots);
            }
        }
        return SFEM_SUCCESS;
    }

    int GeneratedModifiedMooneyRivlin::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedModifiedMooneyRivlin affine gradient requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (false) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedModifiedMooneyRivlin affine gradient requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
            if (impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                    auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                    auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_gradient_packed_two_pass_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                        }
                        return modified_mooney_rivlin_gradient_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                    }
                    else if (dim == 3) {
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_gradient_packed_two_pass_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return modified_mooney_rivlin_gradient_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
            }
            if (!impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                    auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                    auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_gradient_packed_two_pass_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                        }
                        return modified_mooney_rivlin_gradient_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                    }
                    else if (dim == 3) {
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_gradient_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return modified_mooney_rivlin_gradient_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
            }
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->gradient_uses_affine) {
                    return modified_mooney_rivlin_gradient_2d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                }
                return modified_mooney_rivlin_gradient_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, out + 0, out + 1);
            }
            else if (dim == 3) {
                if (impl_->gradient_uses_affine) {
                    return modified_mooney_rivlin_gradient_3d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                }
                return modified_mooney_rivlin_gradient_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
            }
            SFEM_ERROR("modified_mooney_rivlin gradient does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedModifiedMooneyRivlin affine hessian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (false) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedModifiedMooneyRivlin affine hessian action requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->apply_uses_affine) {
                    if (impl_->space->has_packed_mesh()) {
                        auto packed = impl_->space->packed_mesh();
                        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                        if (packed_block >= 0) {
                            auto packed_elements = packed->elements(packed_block);
                            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                            auto n_shared_nodes = packed->n_shared(packed_block);
                            auto ghost_ptr = packed->ghost_ptr(packed_block);
                            auto ghost_idx = packed->ghost_idx(packed_block);
                            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                            if (impl_->use_packed_two_pass) {
                                return modified_mooney_rivlin_apply_packed_two_pass_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                            }
                            return modified_mooney_rivlin_apply_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                        }
                    }
                    return modified_mooney_rivlin_apply_2d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
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
                        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_apply_packed_two_pass_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                        }
                        return modified_mooney_rivlin_apply_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                    }
                }
                return modified_mooney_rivlin_apply_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
            }
            else if (dim == 3) {
                if (impl_->apply_uses_affine) {
                    if (impl_->space->has_packed_mesh()) {
                        auto packed = impl_->space->packed_mesh();
                        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                        if (packed_block >= 0) {
                            auto packed_elements = packed->elements(packed_block);
                            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                            auto n_shared_nodes = packed->n_shared(packed_block);
                            auto ghost_ptr = packed->ghost_ptr(packed_block);
                            auto ghost_idx = packed->ghost_idx(packed_block);
                            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                            if (impl_->use_packed_two_pass) {
                                return modified_mooney_rivlin_apply_packed_two_pass_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                            }
                            return modified_mooney_rivlin_apply_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                        }
                    }
                    return modified_mooney_rivlin_apply_3d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
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
                        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                        if (impl_->use_packed_two_pass) {
                            return modified_mooney_rivlin_apply_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return modified_mooney_rivlin_apply_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
                return modified_mooney_rivlin_apply_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
            }
            SFEM_ERROR("modified_mooney_rivlin apply does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedModifiedMooneyRivlin affine objective requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            int status = SFEM_FAILURE;
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->objective_uses_affine) {
                    status = modified_mooney_rivlin_objective_2d_affine_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, impl_->element_values.get());
                } else {
                    status = modified_mooney_rivlin_objective_2d_isoparametric_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, impl_->element_values.get());
                }
            }
            else if (dim == 3) {
                if (impl_->objective_uses_affine) {
                    status = modified_mooney_rivlin_objective_3d_affine_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                } else {
                    status = modified_mooney_rivlin_objective_3d_isoparametric_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                }
            }
            if (dim != 2 && dim != 3) {
                SFEM_ERROR("modified_mooney_rivlin objective does not support spatial dimension %d\n", dim);
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

    int GeneratedModifiedMooneyRivlin::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const ptrdiff_t nvalues = (ptrdiff_t)nsteps * nelements;
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedModifiedMooneyRivlin affine objective_steps requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
            if (impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        status = modified_mooney_rivlin_objective_steps_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = modified_mooney_rivlin_objective_steps_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    }
                }
            }
            if (!impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        status = modified_mooney_rivlin_objective_steps_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = modified_mooney_rivlin_objective_steps_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    }
                }
            }
            if (status == SFEM_FAILURE) {
                const int dim = mesh->spatial_dimension();
                if (dim == 2) {
                    if (impl_->objective_uses_affine) {
                        status = modified_mooney_rivlin_objective_steps_2d_affine_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    } else {
                        status = modified_mooney_rivlin_objective_steps_2d_isoparametric_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    }
                }
                else if (dim == 3) {
                    if (impl_->objective_uses_affine) {
                        status = modified_mooney_rivlin_objective_steps_3d_affine_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    } else {
                        status = modified_mooney_rivlin_objective_steps_3d_isoparametric_mesh_soa(domain.element_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    }
                }
                if (dim != 2 && dim != 3) {
                    SFEM_ERROR("modified_mooney_rivlin objective_steps does not support spatial dimension %d\n", dim);
                    return SFEM_FAILURE;
                }
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

    int GeneratedModifiedMooneyRivlin::hessian_crs(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::hessian_crs");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin::hessian_crs requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("modified_mooney_rivlin hessian_crs 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("modified_mooney_rivlin hessian_crs 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("modified_mooney_rivlin hessian_crs does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::hessian_bsr(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::hessian_bsr");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin::hessian_bsr requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                return modified_mooney_rivlin_hessian_bsr_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 2, current + 0, current + 1, rowptr, colidx, values);
            }
            else if (dim == 3) {
                return modified_mooney_rivlin_hessian_bsr_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("c1"), domain.parameters->require_real_value("c2"), domain.parameters->require_real_value("kappa"), 3, current + 0, current + 1, current + 2, rowptr, colidx, values);
            }
            SFEM_ERROR("modified_mooney_rivlin hessian_bsr does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::hessian_dia(const real_t *const x,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::hessian_dia");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin::hessian_dia requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("modified_mooney_rivlin hessian_dia 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("modified_mooney_rivlin hessian_dia 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("modified_mooney_rivlin hessian_dia does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::hessian_coo(const real_t *const x,
                            const ptrdiff_t nnz,
                            const idx_t *const rows,
                            const idx_t *const cols,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::hessian_coo");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin::hessian_coo requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("modified_mooney_rivlin hessian_coo 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("modified_mooney_rivlin hessian_coo 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("modified_mooney_rivlin hessian_coo does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedModifiedMooneyRivlin::hessian_patch(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::hessian_patch");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedModifiedMooneyRivlin::hessian_patch requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("modified_mooney_rivlin hessian_patch 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("modified_mooney_rivlin hessian_patch 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("modified_mooney_rivlin hessian_patch does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    void GeneratedModifiedMooneyRivlin::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::set_option");
        if (name == "PACKED_TWO_PASS" || name == "two_pass") {
            impl_->use_packed_two_pass = val;
            return;
        }
        AffineOption options[] = {
            {"ASSUME_AFFINE_OBJECTIVE", &impl_->objective_uses_affine},
            {"objective_assume_affine", &impl_->objective_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &impl_->gradient_uses_affine},
            {"gradient_assume_affine", &impl_->gradient_uses_affine},
            {"ASSUME_AFFINE_HESSIAN_ACTION", &impl_->apply_uses_affine},
            {"hessian_action_assume_affine", &impl_->apply_uses_affine},
            {"ASSUME_AFFINE_APPLY", &impl_->apply_uses_affine},
            {"apply_assume_affine", &impl_->apply_uses_affine},
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            if (cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
                SFEM_ERROR("GeneratedModifiedMooneyRivlin failed to cache affine geometry\n");
            }
        }
    }

    void GeneratedModifiedMooneyRivlin::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedModifiedMooneyRivlin::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedModifiedMooneyRivlin::create_from_yaml");
        auto ret = std::make_shared<GeneratedModifiedMooneyRivlin>(space);

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
}  // namespace sfem
