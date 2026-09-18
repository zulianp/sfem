#include "sfem_GeneratedBodyForce_cuda.hpp"
#include "sfem_GeneratedBodyForce_cuda_c_abi.hpp"



#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <cstring>
#include <vector>



namespace sfem {
  namespace {
    constexpr int MAX_PARAMETERS = 4;

    void seed_parameters(Parameters &parameters) {
      parameters.set_value("density", 1);
      parameters.set_value("g0", 0);
      parameters.set_value("g1", 0);
      parameters.set_value("g2", 0);
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
      values[1] = 0;
      values[2] = 0;
      values[3] = 0;
    }

#ifdef SFEM_ENABLE_RYAML
    constexpr int N_DEFINED_MATERIAL_PARAMETERS = 4;
    constexpr int N_MATERIAL_PARAMETERS = 4;
    static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"density", "g0", "g1", "g2"};

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
      SFEM_ERROR("GPUGeneratedBodyForce: mesh block pointer not found in mesh.blocks()\n");
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
              mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
          if (!cache->jacobian) {
            return SFEM_FAILURE;
          }
        }
        if (needs_metric_soa && !cache->metric_soa) {
          cache->metric_soa = smesh::FFF::create_SoA(
              mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
          if (!cache->metric_soa) {
            return SFEM_FAILURE;
          }
        }
        if (needs_metric_aos && !cache->metric_aos) {
          cache->metric_aos = smesh::FFF::create_AoS(
              mesh, smesh::MEMORY_SPACE_DEVICE, block_id);
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
          values[index++] = parameters.require_real_value("density");
          values[index++] = parameters.require_real_value("g0");
          values[index++] = parameters.require_real_value("g1");
          break;
        case 3:
          values[index++] = parameters.require_real_value("density");
          values[index++] = parameters.require_real_value("g0");
          values[index++] = parameters.require_real_value("g1");
          values[index++] = parameters.require_real_value("g2");
          break;
        default:
          SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\n", dim);
          break;
      }
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
        case 2: return 2;
        case 3: return 3;
        default:
          SFEM_ERROR("unsupported spatial dimension %d for generated block size\n", dim);
          return 0;
      }
    }

  }  // namespace

  class GPUGeneratedBodyForce::Impl {
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

  std::unique_ptr<Op> GPUGeneratedBodyForce::create(const std::shared_ptr<FunctionSpace> &space) {
    const ptrdiff_t expected_block_size =
        block_size_for_dim(space->mesh_ptr()->spatial_dimension());
    if (space->block_size() != expected_block_size) {
      SFEM_ERROR("GPUGeneratedBodyForce requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
      return nullptr;
    }
    auto op = std::make_unique<GPUGeneratedBodyForce>(space);
    op->initialize();
    return op;
  }

  GPUGeneratedBodyForce::GPUGeneratedBodyForce(const std::shared_ptr<FunctionSpace> &space)
    : impl_(std::make_unique<Impl>(space)) {}
  GPUGeneratedBodyForce::~GPUGeneratedBodyForce() = default;

  ptrdiff_t GPUGeneratedBodyForce::n_dofs_domain() const { return impl_->space->n_dofs(); }
  ptrdiff_t GPUGeneratedBodyForce::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GPUGeneratedBodyForce::flops_value() const {
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

    size_t GPUGeneratedBodyForce::memory_traffic_bytes_value() const {
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

    double GPUGeneratedBodyForce::flops_gradient() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_residual_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_residual_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedBodyForce::memory_traffic_bytes_gradient() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_residual_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_residual_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GPUGeneratedBodyForce::flops_apply() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_jacobian_action_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_jacobian_action_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedBodyForce::memory_traffic_bytes_apply() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_jacobian_action_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_body_force_jacobian_action_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
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

  int GPUGeneratedBodyForce::initialize(const std::vector<std::string> &block_names) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::initialize");
    impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
    {
      auto dof_graph = impl_->space->dof_to_dof_graph();
      if (!dof_graph ||
        validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
        SFEM_ERROR("GPUGeneratedBodyForce::initialize: the dof graph is malformed; the assembly kernels assume it is not\n");
        return SFEM_FAILURE;
      }
    }
    seed_material(*impl_->domains);
    const bool needs_affine_jacobian =
        (impl_->residual_uses_affine && true) ||
        (impl_->jacobian_action_uses_affine && true);
    const bool needs_affine_metric =
        (impl_->residual_uses_affine && (false || false)) ||
        (impl_->jacobian_action_uses_affine && (false || false));
    const bool needs_a_met_soa =
        (impl_->residual_uses_affine && false) ||
        (impl_->jacobian_action_uses_affine && false);
    const bool needs_affine_metric_aos =
        (impl_->residual_uses_affine && false) ||
        (impl_->jacobian_action_uses_affine && false);
    if (needs_affine_jacobian || needs_affine_metric) {
      const int status = cache_affine_geometry(impl_->space,
                                                     *impl_->domains,
                                                     needs_affine_jacobian,
                                                     needs_a_met_soa,
                                                     needs_affine_metric_aos);
      if (status != SFEM_SUCCESS) return status;
    }

    return SFEM_SUCCESS;
  }

  int GPUGeneratedBodyForce::update(const real_t *const x) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::update");
    impl_->current = x;
    return SFEM_SUCCESS;
  }

  int GPUGeneratedBodyForce::update(const real_t *const previous,
                       const real_t *const current) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::update");
    impl_->previous_buffer.reset();
    impl_->previous = previous;
    impl_->current = current;
    return SFEM_SUCCESS;
  }

  int GPUGeneratedBodyForce::gradient(const real_t *const state, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::gradient");

    impl_->current = state;
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *determinant = nullptr;
      const geom_t *const *geom_metric = nullptr;
      const geom_t *geom_metric_aos = nullptr;
      if (impl_->residual_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache) {
          SFEM_ERROR("GPUGeneratedBodyForce affine residual requires cached geometry\n");
          return SFEM_FAILURE;
        }
        if (true) {
          if (!cache->jacobian) {
            SFEM_ERROR("GPUGeneratedBodyForce affine residual requires cached jacobian geometry\n");
            return SFEM_FAILURE;
          }
          adjugate = reinterpret_cast<const geom_t *const *>(
              cache->jacobian->jacobian_adjugate_SoA()->data());
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian->jacobian_determinant()->data());
        }
        if (false) {
          if (!cache->metric_soa) {
            SFEM_ERROR("GPUGeneratedBodyForce affine residual requires cached SoA metric geometry\n");
            return SFEM_FAILURE;
          }
          geom_metric = reinterpret_cast<const geom_t *const *>(
              cache->metric_soa->fff_SoA()->data());
        }
        if (false) {
          if (!cache->metric_aos) {
            SFEM_ERROR("GPUGeneratedBodyForce affine residual requires cached AoS metric geometry\n");
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
        static constexpr ptrdiff_t FIELD_STRIDE = 2;
          real_t *const RSTR u_out[2] = {out + 0, out + 1};
        if (impl_->residual_uses_affine) {
          return cu_body_force_residual_2d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), determinant, storage[0], storage[1], storage[2], FIELD_STRIDE, u_out[0], u_out[1], stream);
        }
        return cu_body_force_residual_2d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, storage[0], storage[1], storage[2], FIELD_STRIDE, u_out[0], u_out[1], stream);
      }
      else if (dim == 3) {
        static constexpr ptrdiff_t FIELD_STRIDE = 3;
          real_t *const RSTR u_out[3] = {out + 0, out + 1, out + 2};
        if (impl_->residual_uses_affine) {
          return cu_body_force_residual_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), determinant, storage[0], storage[1], storage[2], storage[3], FIELD_STRIDE, u_out[0], u_out[1], u_out[2], stream);
        }
        return cu_body_force_residual_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, storage[0], storage[1], storage[2], storage[3], FIELD_STRIDE, u_out[0], u_out[1], u_out[2], stream);
      }
      SFEM_ERROR("body_force residual does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedBodyForce::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::apply");
    const real_t *const current = state ? state : impl_->current;

    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);

    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *determinant = nullptr;
      const geom_t *const *geom_metric = nullptr;
      const geom_t *geom_metric_aos = nullptr;
      if (impl_->jacobian_action_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache) {
          SFEM_ERROR("GPUGeneratedBodyForce affine jacobian action requires cached geometry\n");
          return SFEM_FAILURE;
        }
        if (true) {
          if (!cache->jacobian) {
            SFEM_ERROR("GPUGeneratedBodyForce affine jacobian action requires cached jacobian geometry\n");
            return SFEM_FAILURE;
          }
          adjugate = reinterpret_cast<const geom_t *const *>(
              cache->jacobian->jacobian_adjugate_SoA()->data());
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian->jacobian_determinant()->data());
        }
        if (false) {
          if (!cache->metric_soa) {
            SFEM_ERROR("GPUGeneratedBodyForce affine jacobian action requires cached SoA metric geometry\n");
            return SFEM_FAILURE;
          }
          geom_metric = reinterpret_cast<const geom_t *const *>(
              cache->metric_soa->fff_SoA()->data());
        }
        if (false) {
          if (!cache->metric_aos) {
            SFEM_ERROR("GPUGeneratedBodyForce affine jacobian action requires cached AoS metric geometry\n");
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
        static constexpr ptrdiff_t FIELD_STRIDE = 2;
          real_t *const RSTR u_out[2] = {out + 0, out + 1};
        if (impl_->jacobian_action_uses_affine) {
          SFEM_ERROR("body_force jacobian_action affine 2d dispatch was not generated\n");
          return SFEM_FAILURE;
        }
        SFEM_ERROR("body_force jacobian_action isoparametric 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        static constexpr ptrdiff_t FIELD_STRIDE = 3;
          real_t *const RSTR u_out[3] = {out + 0, out + 1, out + 2};
        if (impl_->jacobian_action_uses_affine) {
          SFEM_ERROR("body_force jacobian_action affine 3d dispatch was not generated\n");
          return SFEM_FAILURE;
        }
        SFEM_ERROR("body_force jacobian_action isoparametric 3d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      SFEM_ERROR("body_force jacobian_action does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  void GPUGeneratedBodyForce::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::set_field");
    if (component != 0 || std::strcmp(name, "previous") != 0) {
      SFEM_ERROR("GPUGeneratedBodyForce supports set_field(\"previous\", buffer, 0)\n");
      return;
    }
    impl_->previous_buffer = values;
    impl_->previous = values->data();
  }

  void GPUGeneratedBodyForce::set_value_in_block(const std::string &block_name,
                  const std::string &var_name,
                  const real_t value) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::set_value_in_block");
    impl_->domains->set_value_in_block(block_name, var_name, value);
  }

  void GPUGeneratedBodyForce::set_option(const std::string &name, const bool val) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::set_option");
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
          (impl_->residual_uses_affine && (false || false)) ||
          (impl_->jacobian_action_uses_affine && (false || false));
      const bool needs_a_met_soa =
          (impl_->residual_uses_affine && false) ||
          (impl_->jacobian_action_uses_affine && false);
      const bool needs_affine_metric_aos =
          (impl_->residual_uses_affine && false) ||
          (impl_->jacobian_action_uses_affine && false);
      if (cache_affine_geometry(impl_->space,
                                      *impl_->domains,
                                      needs_affine_jacobian,
                                      needs_a_met_soa,
                                      needs_affine_metric_aos) != SFEM_SUCCESS) {
        SFEM_ERROR("GPUGeneratedBodyForce failed to cache affine geometry\n");
      }
    }
  }

#ifdef SFEM_ENABLE_RYAML
  std::shared_ptr<Op> GPUGeneratedBodyForce::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::create_from_yaml");
    auto ret = std::make_shared<GPUGeneratedBodyForce>(space);

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

  int GPUGeneratedBodyForce::hessian_crs(const real_t *const state,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::hessian_crs");
    return SFEM_FAILURE;
  }

  int GPUGeneratedBodyForce::hessian_bsr(const real_t *const state,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::hessian_bsr");
    return SFEM_FAILURE;
  }


  int GPUGeneratedBodyForce::value(const real_t *x, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedBodyForce::value");
    // `-rho * g . u`, from the load vector this operator's own gradient
    // assembles.  Linear in the state, so the identity is exact.
    const ptrdiff_t ndofs = impl_->space->n_dofs();
    std::vector<real_t> work(ndofs, 0);
    if (gradient(x, work.data()) != SFEM_SUCCESS) {
      return SFEM_FAILURE;
    }
    real_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
      acc += work[i] * x[i];
    }
    *out += acc;
    return SFEM_SUCCESS;
  }
}  // namespace sfem
