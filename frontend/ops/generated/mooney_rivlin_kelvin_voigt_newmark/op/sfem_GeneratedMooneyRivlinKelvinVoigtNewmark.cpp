#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark.hpp"
#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_c_abi.hpp"

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
        constexpr int MAX_PARAMETERS = 5;

        void seed_parameters(Parameters &parameters) {
            parameters.set_value("lmbda", 1);
            parameters.set_value("mu", 1);
            parameters.set_value("eta_s", 0.10000000000000001);
            parameters.set_value("eta_b", 0);
            parameters.set_value("newmark_velocity_alpha", 1);
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
            values[2] = 0.10000000000000001;
            values[3] = 0;
            values[4] = 1;
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 5;
        constexpr int N_MATERIAL_PARAMETERS = 5;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"lmbda", "mu", "eta_s", "eta_b", "newmark_velocity_alpha"};

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
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark: mesh block pointer not found in mesh.blocks()\n");
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
            values[0] = parameters.require_real_value("lmbda");
            values[1] = parameters.require_real_value("mu");
            values[2] = parameters.require_real_value("eta_s");
            values[3] = parameters.require_real_value("eta_b");
            values[4] = parameters.require_real_value("newmark_velocity_alpha");
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
                case 2: return 2;
                case 3: return 3;
                default:
                    SFEM_ERROR("unsupported spatial dimension %d for generated coupled block size\n", dim);
                    return 0;
            }
        }
    }  // namespace

    class GeneratedMooneyRivlinKelvinVoigtNewmark::Impl {
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

    std::unique_ptr<Op> GeneratedMooneyRivlinKelvinVoigtNewmark::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<GeneratedMooneyRivlinKelvinVoigtNewmark>(space);
        op->initialize();
        return op;
    }

    GeneratedMooneyRivlinKelvinVoigtNewmark::GeneratedMooneyRivlinKelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedMooneyRivlinKelvinVoigtNewmark::~GeneratedMooneyRivlinKelvinVoigtNewmark() = default;

    ptrdiff_t GeneratedMooneyRivlinKelvinVoigtNewmark::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedMooneyRivlinKelvinVoigtNewmark::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedMooneyRivlinKelvinVoigtNewmark::flops_value() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();

            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedMooneyRivlinKelvinVoigtNewmark::memory_traffic_bytes_value() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();

            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedMooneyRivlinKelvinVoigtNewmark::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedMooneyRivlinKelvinVoigtNewmark::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedMooneyRivlinKelvinVoigtNewmark::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
                    }
                }
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedMooneyRivlinKelvinVoigtNewmark::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            if (dim == 2) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_soa_diagnostics(domain.element_type);
                    if (diagnostics) {
                        total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    }
                }
            }
            if (dim == 3) {
                {
                    const sfem::codegen::KernelDiagnostics *const diagnostics = mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_soa_diagnostics(domain.element_type);
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

    int GeneratedMooneyRivlinKelvinVoigtNewmark::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark::initialize: the dof graph is malformed; the assembly kernels assume it is not\n");
                return SFEM_FAILURE;
            }
        }
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

    int GeneratedMooneyRivlinKelvinVoigtNewmark::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::gradient");
        if (!impl_->previous) {
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark requires a previous state\n");
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
                    SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark affine gradient/residual requires cached geometry\n");
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
                case smesh::TRI3: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {state + 0, state + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::TRI6: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {state + 0, state + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::QUAD4:
                case smesh::PROTEUS_QUAD4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {state + 0, state + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::TET4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::TET10: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::HEX8:
                case smesh::PROTEUS_HEX8: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::HEX27:
                case smesh::PROTEUS_HEX27: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::PROTEUS_HEX64: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {state + 0, state + 1, state + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->gradient_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->residual_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                default:
                    SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::apply");
        const real_t *const current = state ? state : impl_->current;
        if (!current || !impl_->previous) {
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark requires current and previous states\n");
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
                    SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark affine hessian/jacobian action requires cached geometry\n");
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
                case smesh::TRI3: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {current + 0, current + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    const real_t *const SFEM_RESTRICT u_direction_data[2] = {direction + 0, direction + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::TRI6: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {current + 0, current + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    const real_t *const SFEM_RESTRICT u_direction_data[2] = {direction + 0, direction + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::QUAD4:
                case smesh::PROTEUS_QUAD4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 2;
                    const real_t *const SFEM_RESTRICT u_data[2] = {current + 0, current + 1};
                    const real_t *const SFEM_RESTRICT u_old_data[2] = {previous + 0, previous + 1};
                    const real_t *const SFEM_RESTRICT u_direction_data[2] = {direction + 0, direction + 1};
                    real_t *const SFEM_RESTRICT u_out[2] = {out + 0, out + 1};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 2, state + 0, state + 1, 2, direction + 0, direction + 1, 2, out + 0, out + 1);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 2, u_data[0], u_data[1], 2, u_old_data[0], u_old_data[1], 2, u_direction_data[0], u_direction_data[1], 2, u_out[0], u_out[1]);
                }
                case smesh::TET4: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {current + 0, current + 1, current + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::TET10: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {current + 0, current + 1, current + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::HEX8:
                case smesh::PROTEUS_HEX8: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {current + 0, current + 1, current + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::HEX27:
                case smesh::PROTEUS_HEX27: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {current + 0, current + 1, current + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                case smesh::PROTEUS_HEX64: {
                    static constexpr ptrdiff_t FIELD_STRIDE = 3;
                    const real_t *const SFEM_RESTRICT u_data[3] = {current + 0, current + 1, current + 2};
                    const real_t *const SFEM_RESTRICT u_old_data[3] = {previous + 0, previous + 1, previous + 2};
                    const real_t *const SFEM_RESTRICT u_direction_data[3] = {direction + 0, direction + 1, direction + 2};
                    real_t *const SFEM_RESTRICT u_out[3] = {out + 0, out + 1, out + 2};
                    int status = impl_->apply_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2) : mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[0], storage[1], 3, state + 0, state + 1, state + 2, 3, direction + 0, direction + 1, direction + 2, 3, out + 0, out + 1, out + 2);
                    if (status != SFEM_SUCCESS) return status;
                    return impl_->jacobian_action_uses_affine ? mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_affine_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]) : mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, storage[3], storage[2], storage[4], 3, u_data[0], u_data[1], u_data[2], 3, u_old_data[0], u_old_data[1], u_old_data[2], 3, u_direction_data[0], u_direction_data[1], u_direction_data[2], 3, u_out[0], u_out[1], u_out[2]);
                }
                default:
                    SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }


    int GeneratedMooneyRivlinKelvinVoigtNewmark::value_steps(const real_t *state,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::value_steps");
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        const ptrdiff_t ndofs = n_dofs_domain();
        std::vector<real_t> stepped(ndofs);
        std::vector<real_t> residual(ndofs);
        for (int step = 0; step < nsteps; ++step) {
            const real_t alpha = steps[step];
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                stepped[i] = state[i] + alpha * h[i];
            }
            std::fill(residual.begin(), residual.end(), real_t(0));
            const int status = gradient(stepped.data(), residual.data());
            if (status != SFEM_SUCCESS) {
                return status;
            }
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                sum += residual[i] * residual[i];
            }
            out[step] += real_t(0.5) * sum;
        }
        return SFEM_SUCCESS;
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::value(const real_t *state, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::value");
        // One step of length zero: `state + 0 * h` is `state` exactly, so the
        // increment is unused and `state` can stand in for it.  One
        // implementation, so the two cannot disagree.
        const real_t objective_step = 0;
        *out = 0;
        return value_steps(state, state, 1, &objective_step, out);
    }

    void GeneratedMooneyRivlinKelvinVoigtNewmark::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark supports set_field(\"previous\", buffer, 0)\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void GeneratedMooneyRivlinKelvinVoigtNewmark::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::set_option");
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
            SFEM_ERROR("GeneratedMooneyRivlinKelvinVoigtNewmark failed to cache affine geometry\n");
        }
    }

    void GeneratedMooneyRivlinKelvinVoigtNewmark::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedMooneyRivlinKelvinVoigtNewmark::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::create_from_yaml");
        auto ret = std::make_shared<GeneratedMooneyRivlinKelvinVoigtNewmark>(space);

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

    int GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_crs");
        return SFEM_FAILURE;
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_bsr(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_bsr");
        return SFEM_FAILURE;
    }

    int GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_dia(const real_t *const,
                            const int *const,
                            const ptrdiff_t,
                            real_t *const) {
        SFEM_TRACE_SCOPE("GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_dia");
        return SFEM_FAILURE;
    }
}  // namespace sfem
