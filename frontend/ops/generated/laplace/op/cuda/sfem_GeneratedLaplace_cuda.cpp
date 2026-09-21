#include "sfem_GeneratedLaplace_cuda.hpp"
#include "sfem_GeneratedLaplace_cuda_c_abi.hpp"


#include "sfem_API.hpp"
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

    //! The part of the time discretisation that is not in this material's form:
    //! the inertia.  It is a *potential* here, because this operator's 0-form
    //! is one, so it reaches `value_steps` as well as `gradient` -- the energy
    //! merit and the residual have to describe the same problem, or a line
    //! search minimises something the Newton step is not solving.
    std::shared_ptr<Op> time_scheme_term(const std::shared_ptr<TimeScheme> &scheme) {
      return scheme ? scheme->inertia_op() : nullptr;
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
      SFEM_ERROR("GPUGeneratedLaplace: mesh block pointer not found in mesh.blocks()\n");
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
            mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
        if (!cache->jacobian_soa) {
          return SFEM_FAILURE;
        }
        if (needs_jacobian_aos) {
          cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
              mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
          if (!cache->jacobian_aos) {
            return SFEM_FAILURE;
          }
        }
        cache->metric_soa = smesh::FFF::create_SoA(
            mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
        if (!cache->metric_soa) {
          return SFEM_FAILURE;
        }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
      }
      return SFEM_SUCCESS;
    }

    //! Where this build's kernels read the connectivity from.
    //!
    //! One function rather than the same expression at every call site, because
    //! the host and the device differ only here: a device Op hands its kernels
    //! the block's device copy, which is what every `gpu:` Op in SFEM passes
    //! and what a `__global__` body can dereference.
    idx_t **element_connectivity(const OpDomain &domain) {
      return const_cast<idx_t **>(domain.block->device_elements_SoA()->data());
    }

    //! Where the kernels read the mesh geometry from.  The mesh's own array on
    //! the host; a device target reads smesh's device copy, because a
    //! `__global__` body cannot dereference a host pointer -- and on a Grace
    //! Hopper node it sometimes can, which is worse: the merit came out exact
    //! at one mesh size and nonsense at the next.
    const geom_t *const *element_points(const std::shared_ptr<smesh::Mesh> &mesh) {
      return const_cast<const geom_t *const *>(mesh->device_points_SoA()->data());
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

  class GPUGeneratedLaplace::Impl {
  public:
    explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    std::shared_ptr<FunctionSpace> space;
    std::shared_ptr<MultiDomainOp> domains;
    std::shared_ptr<TimeScheme> time_scheme;
    SharedBuffer<real_t> element_values;
    SharedBuffer<real_t> element_ones;
    ptrdiff_t element_capacity{0};
    SharedBuffer<real_t> step_values;
    int step_capacity{0};
    bool objective_uses_affine{false};
    bool gradient_uses_affine{false};
    bool apply_uses_affine{false};
    bool use_packed_two_pass{false};
    std::vector<SharedBuffer<real_t>> packed_ghost_buf;
  };

  std::unique_ptr<Op> GPUGeneratedLaplace::create(const std::shared_ptr<FunctionSpace> &space) {
    const ptrdiff_t expected_block_size =
        block_size_for_dim(space->mesh_ptr()->spatial_dimension());
    if (space->block_size() != expected_block_size) {
      SFEM_ERROR("GPUGeneratedLaplace requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
      return nullptr;
    }
    auto op = std::make_unique<GPUGeneratedLaplace>(space);
    op->initialize();
    return op;
  }

  GPUGeneratedLaplace::GPUGeneratedLaplace(const std::shared_ptr<FunctionSpace> &space)
    : impl_(std::make_unique<Impl>(space)) {}
  GPUGeneratedLaplace::~GPUGeneratedLaplace() = default;

  ptrdiff_t GPUGeneratedLaplace::n_dofs_domain() const { return impl_->space->n_dofs(); }
  ptrdiff_t GPUGeneratedLaplace::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GPUGeneratedLaplace::flops_value() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_objective_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_objective_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedLaplace::memory_traffic_bytes_value() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_objective_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_objective_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GPUGeneratedLaplace::flops_gradient() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_gradient_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_gradient_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedLaplace::memory_traffic_bytes_gradient() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_gradient_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_gradient_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GPUGeneratedLaplace::flops_apply() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_apply_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_apply_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedLaplace::memory_traffic_bytes_apply() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_apply_2d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_laplace_apply_3d_soa_diagnostics(domain.element_type);
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
  // NS x NS candidates, tested each with a three-condition branch
  // and reported through std::fprintf from inside the caller's parallel
  // region.  That paid O(elements x NS^2) on every assembly for a
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

  int GPUGeneratedLaplace::initialize(const std::vector<std::string> &block_names) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::initialize");
    impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
    {
      auto dof_graph = impl_->space->dof_to_dof_graph();
      if (!dof_graph ||
        validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
        SFEM_ERROR("GPUGeneratedLaplace::initialize: the dof graph is malformed; the assembly kernels assume it is not\n");
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
    impl_->element_values = create_buffer<real_t>(impl_->element_capacity, EXECUTION_SPACE_DEVICE);
    impl_->element_ones = create_buffer<real_t>(impl_->element_capacity, EXECUTION_SPACE_DEVICE);
    sfem::blas<real_t>(EXECUTION_SPACE_DEVICE)->values(impl_->element_capacity, real_t(1), impl_->element_ones->data());

    return SFEM_SUCCESS;
  }

  void GPUGeneratedLaplace::set_time_scheme(const std::shared_ptr<TimeScheme> &scheme) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::set_time_scheme");
    if (impl_->time_scheme) {
      impl_->time_scheme->release(this);
    }
    impl_->time_scheme = scheme;
    if (scheme) {
      scheme->claim(this);
    }
  }

  int GPUGeneratedLaplace::gradient(const real_t *const x, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::gradient");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->gradient(x, out) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *adjugate_aos = nullptr;
      const geom_t *determinant = nullptr;
      const geom_t *const *geom_metric = nullptr;
            if (impl_->gradient_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedLaplace affine gradient requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
        if (false) {
          if (!cache->jacobian_aos) {
            SFEM_ERROR("GPUGeneratedLaplace affine gradient requires cached AoS geometry\n");
            return SFEM_FAILURE;
          }
          adjugate_aos = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_adjugate_AoS()->data());
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_determinant()->data());
        }
        if (!cache->metric_soa) {
          SFEM_ERROR("GPUGeneratedLaplace affine gradient requires cached metric geometry\n");
          return SFEM_FAILURE;
        }
        geom_metric = reinterpret_cast<const geom_t *const *>(
            cache->metric_soa->fff_SoA()->data());
            }

      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        if (impl_->gradient_uses_affine) {
          return cu_laplace_gradient_2d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0, stream);
        }
        return cu_laplace_gradient_2d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0, stream);
      }
      else if (dim == 3) {
        if (impl_->gradient_uses_affine) {
          if (domain.element_type == smesh::TET4) {
            return cu_laplace_gradient_3d_a_met_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0, stream);
          }
          return cu_laplace_gradient_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0, stream);
        }
        return cu_laplace_gradient_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 1, out + 0, stream);
      }
      SFEM_ERROR("laplace gradient does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedLaplace::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::apply");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->apply(x, h, out) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *adjugate_aos = nullptr;
      const geom_t *determinant = nullptr;
      const geom_t *const *geom_metric = nullptr;
            if (impl_->apply_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedLaplace affine hessian action requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
        if (false) {
          if (!cache->jacobian_aos) {
            SFEM_ERROR("GPUGeneratedLaplace affine hessian action requires cached AoS geometry\n");
            return SFEM_FAILURE;
          }
          adjugate_aos = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_adjugate_AoS()->data());
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_determinant()->data());
        }
        if (!cache->metric_soa) {
          SFEM_ERROR("GPUGeneratedLaplace affine hessian action requires cached metric geometry\n");
          return SFEM_FAILURE;
        }
        geom_metric = reinterpret_cast<const geom_t *const *>(
            cache->metric_soa->fff_SoA()->data());
            }
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        if (impl_->apply_uses_affine) {
          return cu_laplace_apply_2d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0, stream);
        }
        return cu_laplace_apply_2d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0, stream);
      }
      else if (dim == 3) {
        if (impl_->apply_uses_affine) {
          if (domain.element_type == smesh::TET4) {
            return cu_laplace_apply_3d_a_met_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0, stream);
          }
          return cu_laplace_apply_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0, stream);
        }
        return cu_laplace_apply_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, h + 0, 1, out + 0, stream);
      }
      SFEM_ERROR("laplace apply does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedLaplace::value(const real_t *x, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::value");
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
    //
    // The zeroing that used to sit here is gone too, which is the other half
    // of the same disagreement: `value_steps` accumulates, every hand-written
    // `Op` accumulates (`NeumannConditions::value` ends `*out += acc`), and
    // `Function::value` does not clear `out` before its loop -- so an `Op` that
    // zeroed discarded every contribution made before it.  With two generated
    // operators in one `Function`, the answer depended on their order.
    const real_t objective_step = 0;
    return value_steps(x, x, 1, &objective_step, out);
  }

  int GPUGeneratedLaplace::value_steps(const real_t *x,
              const real_t *h,
              const int nsteps,
              const real_t *const steps,
              real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::value_steps");
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    if (nsteps <= 0) {
      return SFEM_SUCCESS;
    }
    // The scheme's potential, at every trial step the line search asks about.
    // `value` is `value_steps` at one step of length zero, so adding it here
    // covers both and cannot let them disagree.
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->value_steps(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
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
          SFEM_ERROR("GPUGeneratedLaplace affine objective_steps requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
        if (!cache->metric_soa) {
          SFEM_ERROR("GPUGeneratedLaplace affine objective_steps requires cached metric geometry\n");
          return SFEM_FAILURE;
        }
        geom_metric = reinterpret_cast<const geom_t *const *>(
            cache->metric_soa->fff_SoA()->data());
            }
      if (nsteps > impl_->step_capacity) {
        impl_->step_values = create_buffer<real_t>(nsteps, EXECUTION_SPACE_DEVICE);
        impl_->step_capacity = nsteps;
      }
      buffer_host_to_device(nsteps * sizeof(real_t), steps, impl_->step_values->data());
      if (nvalues > impl_->element_capacity) {
        impl_->element_values = create_buffer<real_t>(nvalues, EXECUTION_SPACE_DEVICE);
        impl_->element_ones = create_buffer<real_t>(nvalues, EXECUTION_SPACE_DEVICE);
        sfem::blas<real_t>(EXECUTION_SPACE_DEVICE)->values(nvalues, real_t(1), impl_->element_ones->data());
        impl_->element_capacity = nvalues;
      }
      sfem::blas<real_t>(EXECUTION_SPACE_DEVICE)->zeros(nvalues, impl_->element_values->data());
      int status = SFEM_FAILURE;

      if (status == SFEM_FAILURE) {
        const int dim = mesh->spatial_dimension();
        if (dim == 2) {
          if (impl_->objective_uses_affine) {
            status = cu_laplace_objective_steps_2d_a_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], domain.parameters->require_real_value("kappa"), 1, x + 0, 2, h + 0, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
          } else {
            status = cu_laplace_objective_steps_2d_i_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 2, h + 0, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
          }
        }
        else if (dim == 3) {
          if (impl_->objective_uses_affine) {
            if (domain.element_type == smesh::TET4) {
              status = cu_laplace_objective_steps_3d_a_met_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), geom_metric[0], geom_metric[1], geom_metric[2], geom_metric[3], geom_metric[4], geom_metric[5], domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
            } else {
              status = cu_laplace_objective_steps_3d_a_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
            }
          } else {
            status = cu_laplace_objective_steps_3d_i_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("kappa"), 1, x + 0, 3, h + 0, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
          }
        }
        if (dim != 2 && dim != 3) {
          SFEM_ERROR("laplace objective_steps does not support spatial dimension %d\n", dim);
          return SFEM_FAILURE;
        }
      }
      if (status != SFEM_SUCCESS) return status;
      sfem::device_synchronize();
      auto element_blas = sfem::blas<real_t>(EXECUTION_SPACE_DEVICE);
      for (int step = 0; step < nsteps; ++step) {
        out[step] += element_blas->dot(nelements,
            impl_->element_values->data() + (ptrdiff_t)step * nelements,
            impl_->element_ones->data());
      }
      return SFEM_SUCCESS;
    });
  }

  int GPUGeneratedLaplace::hessian_crs(const real_t *const x,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::hessian_crs");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_crs(x, rowptr, colidx, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }

    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        SFEM_ERROR("laplace hessian_crs 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        SFEM_ERROR("laplace hessian_crs 3d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      SFEM_ERROR("laplace hessian_crs does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedLaplace::hessian_bsr(const real_t *const x,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::hessian_bsr");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_bsr(x, rowptr, colidx, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }

    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        SFEM_ERROR("laplace hessian_bsr 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        SFEM_ERROR("laplace hessian_bsr 3d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      SFEM_ERROR("laplace hessian_bsr does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }




  int GPUGeneratedLaplace::hessian_block_diag_sym(const real_t *const x,
                                       real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::hessian_block_diag_sym");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_block_diag_sym(x, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }

    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
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

  void GPUGeneratedLaplace::set_option(const std::string &name, const bool val) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::set_option");
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
        SFEM_ERROR("GPUGeneratedLaplace failed to cache affine geometry\n");
      }
    }
  }

  void GPUGeneratedLaplace::set_value_in_block(const std::string &block_name,
                  const std::string &var_name,
                  const real_t value) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::set_value_in_block");
    impl_->domains->set_value_in_block(block_name, var_name, value);
  }

#ifdef SFEM_ENABLE_RYAML
  std::shared_ptr<Op> GPUGeneratedLaplace::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
    SFEM_TRACE_SCOPE("GPUGeneratedLaplace::create_from_yaml");
    auto ret = std::make_shared<GPUGeneratedLaplace>(space);

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
