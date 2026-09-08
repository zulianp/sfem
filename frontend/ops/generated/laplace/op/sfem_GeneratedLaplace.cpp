#include "sfem_GeneratedLaplace.hpp"
#include "sfem_GeneratedLaplace_c_abi.hpp"
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
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
            std::shared_ptr<smesh::FFF> metric_soa;
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
                cache->metric_soa = smesh::FFF::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->metric_soa) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 1;
                case 3: return 1;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated block size\n", dim);
                    return 0;
            }
        }
    }  // namespace

    class GeneratedLaplace::Impl {
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_objective_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_objective_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_objective_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_objective_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_apply_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
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

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = laplace_apply_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    // Establish once, at setup, that this operator's dof graph is well formed:
    // rows in order, every column in range, each row sorted and duplicate free.
    // The assembly kernels assume it -- they locate an entry and write to it
    // without re-checking that it is there -- so this is where the assumption
    // is earned.
    //
    // It used to be earned per element instead: every scatter walked its
    // N_SHAPE x N_SHAPE candidates, tested each with a three-condition branch
    // and reported through std::fprintf from inside the caller's parallel
    // region.  That paid O(elements x N_SHAPE^2) on every assembly for a
    // property of the mesh and the graph together, which cannot change between
    // elements or between calls.  Here it is O(nnz), once.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    static int validate_dof_graph(const count_t *const rowptr,
                                  const idx_t *const colidx,
                                  const ptrdiff_t n_nodes,
                                  const ptrdiff_t nnz) {
        if (!rowptr || !colidx || n_nodes < 0) {
            return SFEM_FAILURE;
        }
        if (rowptr[0] != 0 || (ptrdiff_t)rowptr[n_nodes] != nnz) {
            return SFEM_FAILURE;
        }
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];
            if (end < begin || (ptrdiff_t)end > nnz) {
                return SFEM_FAILURE;
            }
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] < 0 || (ptrdiff_t)colidx[k] >= n_nodes) {
                    return SFEM_FAILURE;
                }
                if (k > begin && colidx[k] <= colidx[k - 1]) {
                    return SFEM_FAILURE;
                }
            }
        }
        return SFEM_SUCCESS;
    }

    int GeneratedLaplace::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("GeneratedLaplace::initialize: the dof graph is malformed; the assembly kernels assume it is not\n");
                return SFEM_FAILURE;
            }
        }
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
        }
        // One cache builder, shared with set_option.  This used to be a second
        // copy of the loop inlined here, and the copies drifted: the inlined
        // one never built the metric, so an operator whose affine kernels read
        // it worked when the option was set after initialize and failed when it
        // was set before.
        if (needs_affine_geometry &&
            cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
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

    int GeneratedLaplace::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            if (impl_->gradient_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLaplace affine gradient requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (false) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedLaplace affine gradient requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
                if (!cache->metric_soa) {
                    SFEM_ERROR("GeneratedLaplace affine gradient requires cached metric geometry\n");
                    return SFEM_FAILURE;
                }
                geom_metric = reinterpret_cast<const geom_t *const *>(
                        cache->metric_soa->fff_SoA()->data());
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
                    if (dim == 3) {
                        if (domain.element_type == smesh::TET4) {
                            if (impl_->use_packed_two_pass) {
                                return laplace_gradient_packed_two_pass_3d_affine_metric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                            }
                            return laplace_gradient_packed_3d_affine_metric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                        }
                        if (impl_->use_packed_two_pass) {
                            return laplace_gradient_packed_two_pass_3d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                        }
                        return laplace_gradient_packed_3d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
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
                    if (dim == 3) {
                        if (impl_->use_packed_two_pass) {
                            return laplace_gradient_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                        }
                        return laplace_gradient_packed_3d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                    }
                }
            }
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->gradient_uses_affine) {
                    if (domain.element_type == smesh::TRI3) {
                        return laplace_gradient_2d_affine_metric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                    }
                    return laplace_gradient_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                }
                return laplace_gradient_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
            }
            else if (dim == 3) {
                if (impl_->gradient_uses_affine) {
                    if (domain.element_type == smesh::TET4) {
                        return laplace_gradient_3d_affine_metric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                    }
                    return laplace_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
                }
                return laplace_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0);
            }
            SFEM_ERROR("laplace gradient does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            if (impl_->apply_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLaplace affine hessian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (false) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedLaplace affine hessian action requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
                if (!cache->metric_soa) {
                    SFEM_ERROR("GeneratedLaplace affine hessian action requires cached metric geometry\n");
                    return SFEM_FAILURE;
                }
                geom_metric = reinterpret_cast<const geom_t *const *>(
                        cache->metric_soa->fff_SoA()->data());
            }
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->apply_uses_affine) {
                    if (domain.element_type == smesh::TRI3) {
                        return laplace_apply_2d_affine_metric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                    }
                    return laplace_apply_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                }
                return laplace_apply_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
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
                            if (domain.element_type == smesh::TET4) {
                                if (impl_->use_packed_two_pass) {
                                    return laplace_apply_packed_two_pass_3d_affine_metric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                                }
                                return laplace_apply_packed_3d_affine_metric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                            }
                            if (impl_->use_packed_two_pass) {
                                return laplace_apply_packed_two_pass_3d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                            }
                            return laplace_apply_packed_3d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                        }
                    }
                    if (domain.element_type == smesh::TET4) {
                        return laplace_apply_3d_affine_metric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                    }
                    return laplace_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
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
                            return laplace_apply_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                        }
                        return laplace_apply_packed_3d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
                    }
                }
                return laplace_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0);
            }
            SFEM_ERROR("laplace apply does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::value");
        // The objective is the 0-form at one step of length zero.  `value_steps`
        // evaluates at `x + alpha * h`, so alpha = 0 leaves the increment
        // unused and `x` itself can stand in for it -- `x + 0 * x` is `x`
        // exactly in IEEE arithmetic for any finite state, and the kernel then
        // calls the same block function the objective kernel called.
        //
        // Writing it this way is what keeps the two from disagreeing.  They
        // did: this method zeroed `*out` before accumulating while
        // `value_steps` only accumulated, so the same Op answered the same
        // question two ways depending on which entry point was used.  With one
        // implementation there is nothing left to diverge.
        const real_t objective_step = 0;
        *out = 0;
        return value_steps(x, x, 1, &objective_step, out);
    }

    int GeneratedLaplace::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::value_steps");
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
            const geom_t *const *geom_metric = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLaplace affine objective_steps requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (!cache->metric_soa) {
                    SFEM_ERROR("GeneratedLaplace affine objective_steps requires cached metric geometry\n");
                    return SFEM_FAILURE;
                }
                geom_metric = reinterpret_cast<const geom_t *const *>(
                        cache->metric_soa->fff_SoA()->data());
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
                        status = laplace_objective_steps_packed_2d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, h + 0, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = laplace_objective_steps_packed_3d_affine_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, h + 0, nsteps, steps, impl_->element_values.get());
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
                        status = laplace_objective_steps_packed_2d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, h + 0, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = laplace_objective_steps_packed_3d_isoparametric_mesh_soa(domain.element_type, real_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, h + 0, nsteps, steps, impl_->element_values.get());
                    }
                }
            }
            if (status == SFEM_FAILURE) {
                const int dim = mesh->spatial_dimension();
                if (dim == 2) {
                    if (impl_->objective_uses_affine) {
                        if (domain.element_type == smesh::TRI3) {
                            status = laplace_objective_steps_2d_affine_metric_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, x + 0, 2, h + 0, nsteps, steps, impl_->element_values.get());
                        } else {
                            status = laplace_objective_steps_2d_affine_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 2, h + 0, nsteps, steps, impl_->element_values.get());
                        }
                    } else {
                        status = laplace_objective_steps_2d_isoparametric_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 2, h + 0, nsteps, steps, impl_->element_values.get());
                    }
                }
                else if (dim == 3) {
                    if (impl_->objective_uses_affine) {
                        if (domain.element_type == smesh::TET4) {
                            status = laplace_objective_steps_3d_affine_metric_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, steps, impl_->element_values.get());
                        } else {
                            status = laplace_objective_steps_3d_affine_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, steps, impl_->element_values.get());
                        }
                    } else {
                        status = laplace_objective_steps_3d_isoparametric_mesh_soa(domain.element_type, real_type, nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, steps, impl_->element_values.get());
                    }
                }
                if (dim != 2 && dim != 3) {
                    SFEM_ERROR("laplace objective_steps does not support spatial dimension %d\n", dim);
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

    int GeneratedLaplace::hessian_crs(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_crs");
        (void)x;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                return laplace_hessian_crs_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), rowptr, colidx, values);
            }
            else if (dim == 3) {
                return laplace_hessian_crs_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), rowptr, colidx, values);
            }
            SFEM_ERROR("laplace hessian_crs does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLaplace::hessian_bsr(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_bsr");
        (void)x;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                return laplace_hessian_bsr_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), rowptr, colidx, values);
            }
            else if (dim == 3) {
                return laplace_hessian_bsr_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("kappa"), rowptr, colidx, values);
            }
            SFEM_ERROR("laplace hessian_bsr does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }




    int GeneratedLaplace::hessian_block_diag_sym(const real_t *const x,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::hessian_block_diag_sym");
        (void)x;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("laplace hessian_block_diag_sym 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("laplace hessian_block_diag_sym 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("laplace hessian_block_diag_sym does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    void GeneratedLaplace::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::set_option");
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
                SFEM_ERROR("GeneratedLaplace failed to cache affine geometry\n");
            }
        }
    }

    void GeneratedLaplace::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedLaplace::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
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
