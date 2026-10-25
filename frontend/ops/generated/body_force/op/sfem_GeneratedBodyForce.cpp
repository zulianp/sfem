#include "sfem_GeneratedBodyForce.hpp"
#include "sfem_GeneratedBodyForce_c_abi.hpp"



#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_DerefinedTimeScheme.hpp"
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

    [[maybe_unused]] void material_defaults(real_t *const values) {
      values[0] = 1;
      values[1] = 0;
      values[2] = 0;
      values[3] = 0;
    }

    // derefine_op copies the parameters by name, with or without YAML.
    [[maybe_unused]] constexpr int N_DEFINED_MATERIAL_PARAMETERS = 4;
    [[maybe_unused]] constexpr int N_MATERIAL_PARAMETERS = 4;
    [[maybe_unused]] static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"density", "g0", "g1", "g2"};

#ifdef SFEM_ENABLE_RYAML

    bool yaml_read_real(const ryml::ConstNodeRef &node,
              const char *const key,
              real_t &value) {
      if (!node.has_child(key)) {
        return false;
      }
      node[key].load(&value);
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
      bool read[N_MATERIAL_PARAMETERS] = {false};
      for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
        read[i] = yaml_read_parameter(node,
                                               MATERIAL_PARAMETER_NAMES[i],
                                               values[i]);
        changed |= read[i];
      }
      (void)read;
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
      node[key].load(&raw);
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
      SFEM_ERROR("GeneratedBodyForce: mesh block pointer not found in mesh.blocks()\n");
      return 0;
    }

    struct AffineGeometryCache {
      std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian;
      std::shared_ptr<smesh::FFF> metric_soa;
      std::shared_ptr<smesh::FFF> metric_aos;
    };

    //! Whether this element leaves the operator no choice of geometry.
    //!
    //! A constant-P1 simplex publishes no isoparametric residual or action
    //! kernel, and assembly takes the affine kernel of every element that has
    //! one, so the operator reads a cached adjugate for these elements
    //! whatever the caller asked.  Every other element still caches only when
    //! asked.
    bool geometry_is_forced_affine(const smesh::ElemType element_type) {
      switch (element_type) {
        case smesh::TET4:
          return true;
        case smesh::TRI3:
          return true;
        default:
          return false;
      }
    }

    int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains,
                                  const bool requested_jacobian,
                                  const bool requested_metric_soa,
                                  const bool requested_metric_aos) {
      auto mesh = space->mesh_ptr();
      for (auto &entry : domains.domains()) {
        const bool forced = geometry_is_forced_affine(entry.second.element_type);
        const bool needs_jacobian = requested_jacobian || forced;
        const bool needs_metric_soa = requested_metric_soa || (forced && false);
        const bool needs_metric_aos = requested_metric_aos || (forced && false);
        if (!needs_jacobian && !needs_metric_soa && !needs_metric_aos) {
          continue;
        }
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
      return domain.block->elements()->data();
    }

    //! Where the kernels read the mesh geometry from.  The mesh's own array on
    //! the host; a device target reads smesh's device copy, because a
    //! `__global__` body cannot dereference a host pointer -- and on a Grace
    //! Hopper node it sometimes can, which is worse: the merit came out exact
    //! at one mesh size and nonsense at the next.
    const geom_t *const *element_points(const std::shared_ptr<smesh::Mesh> &mesh) {
      return const_cast<const geom_t *const *>(mesh->points()->data());
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


    // The graph an assembly was last handed and found well formed.
    struct CheckedGraph {
      const count_t *rowptr{nullptr};
      const idx_t *colidx{nullptr};
      ptrdiff_t n_rows{-1};
    };

  }  // namespace

  class GeneratedBodyForce::Impl {
  public:
    explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    std::shared_ptr<FunctionSpace> space;
    std::shared_ptr<MultiDomainOp> domains;

    std::shared_ptr<Buffer<real_t>> previous_buffer;
    const real_t *previous{nullptr};
    const real_t *current{nullptr};
    bool residual_uses_affine{false};
    bool jacobian_action_uses_affine{false};
    CheckedGraph crs_graph;
    CheckedGraph bsr_graph;
  };

  std::unique_ptr<Op> GeneratedBodyForce::create(const std::shared_ptr<FunctionSpace> &space) {
    const ptrdiff_t expected_block_size =
        block_size_for_dim(space->mesh_ptr()->spatial_dimension());
    if (space->block_size() != expected_block_size) {
      SFEM_ERROR("GeneratedBodyForce requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
      return nullptr;
    }
    auto op = std::make_unique<GeneratedBodyForce>(space);
    op->initialize();
    return op;
  }

  GeneratedBodyForce::GeneratedBodyForce(const std::shared_ptr<FunctionSpace> &space)
    : impl_(std::make_unique<Impl>(space)) {}
  GeneratedBodyForce::~GeneratedBodyForce() = default;

  ptrdiff_t GeneratedBodyForce::n_dofs_domain() const { return impl_->space->n_dofs(); }
  ptrdiff_t GeneratedBodyForce::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedBodyForce::flops_value() const {
    return 0;
  }

    size_t GeneratedBodyForce::memory_traffic_bytes_value() const {
    return 0;
  }

    double GeneratedBodyForce::flops_gradient() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_residual_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_residual_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GeneratedBodyForce::memory_traffic_bytes_gradient() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_residual_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_residual_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->residual_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GeneratedBodyForce::flops_apply() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_jacobian_action_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_jacobian_action_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GeneratedBodyForce::memory_traffic_bytes_apply() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 2) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_jacobian_action_2d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = body_force_jacobian_action_3d_esoa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->jacobian_action_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

  int GeneratedBodyForce::initialize(const std::vector<std::string> &block_names) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::initialize");
    impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
    // A micro-element space hands each semi-structured macro element over as its micro
    // elements, which these kernels assemble as HEX8.
    impl_->domains->set_micro_elements(impl_->space->micro_elements());
    seed_material(*impl_->domains);
    const bool needs_affine_jacobian =
        (impl_->residual_uses_affine && true) ||
        (impl_->jacobian_action_uses_affine && true);
    const bool needs_a_met_soa =
        (impl_->residual_uses_affine && false) ||
        (impl_->jacobian_action_uses_affine && false);
    const bool needs_affine_metric_aos =
        (impl_->residual_uses_affine && false) ||
        (impl_->jacobian_action_uses_affine && false);
    // Unconditionally, because the builder skips a domain that neither asked
    // for affine geometry nor is an element that forces it.  Gating the call
    // on the request alone left a constant-P1 simplex with no cached adjugate
    // and an affine kernel as its only route.
    {
      const int status = cache_affine_geometry(impl_->space,
                                                     *impl_->domains,
                                                     needs_affine_jacobian,
                                                     needs_a_met_soa,
                                                     needs_affine_metric_aos);
      if (status != SFEM_SUCCESS) return status;
    }

    return SFEM_SUCCESS;
  }

  int GeneratedBodyForce::update(const real_t *const x) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::update");
    impl_->current = x;
    return SFEM_SUCCESS;
  }

  int GeneratedBodyForce::update(const real_t *const previous,
                       const real_t *const current) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::update");
    impl_->previous_buffer.reset();
    impl_->previous = previous;
    impl_->current = current;
    return SFEM_SUCCESS;
  }

  int GeneratedBodyForce::gradient(const real_t *const state, real_t *const out) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::gradient");

    impl_->current = state;
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *determinant = nullptr;
      if (impl_->residual_uses_affine || domain.element_type == smesh::TET4 || domain.element_type == smesh::TRI3) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache) {
          SFEM_ERROR("GeneratedBodyForce affine residual requires cached geometry\n");
          return SFEM_FAILURE;
        }
        if (true) {
          if (!cache->jacobian) {
            SFEM_ERROR("GeneratedBodyForce affine residual requires cached jacobian geometry\n");
            return SFEM_FAILURE;
          }
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian->jacobian_determinant()->data());
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
        if (impl_->residual_uses_affine || domain.element_type == smesh::TRI3) {
          return body_force_residual_2d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), determinant, storage[0], storage[1], storage[2], FIELD_STRIDE, u_out[0], u_out[1]);
        }
        return body_force_residual_2d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, storage[0], storage[1], storage[2], FIELD_STRIDE, u_out[0], u_out[1]);
      }
      else if (dim == 3) {
        static constexpr ptrdiff_t FIELD_STRIDE = 3;
          real_t *const RSTR u_out[3] = {out + 0, out + 1, out + 2};
        if (impl_->residual_uses_affine || domain.element_type == smesh::TET4) {
          return body_force_residual_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), determinant, storage[0], storage[1], storage[2], storage[3], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
        }
        return body_force_residual_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, storage[0], storage[1], storage[2], storage[3], FIELD_STRIDE, u_out[0], u_out[1], u_out[2]);
      }
      SFEM_ERROR("body_force residual does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GeneratedBodyForce::apply(const real_t *const /*state*/,
                      const real_t *const /*direction*/,
                      real_t *const /*out*/) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::apply");

    auto mesh = impl_->space->mesh_ptr();

    return impl_->domains->iterate([&](const OpDomain &domain) {
      if (impl_->jacobian_action_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache) {
          SFEM_ERROR("GeneratedBodyForce affine jacobian action requires cached geometry\n");
          return SFEM_FAILURE;
        }
        if (true) {
          if (!cache->jacobian) {
            SFEM_ERROR("GeneratedBodyForce affine jacobian action requires cached jacobian geometry\n");
            return SFEM_FAILURE;
          }
        }
      }
      real_t storage[MAX_PARAMETERS];
      parameter_array(*domain.parameters,
              mesh->spatial_dimension(),
              storage);

      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        if (impl_->jacobian_action_uses_affine) {
          SFEM_ERROR("body_force jacobian_action affine 2d dispatch was not generated\n");
          return SFEM_FAILURE;
        }
        SFEM_ERROR("body_force jacobian_action isoparametric 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
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

  void GeneratedBodyForce::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::set_field");
    if (component != 0 || std::strcmp(name, "previous") != 0) {
      SFEM_ERROR("GeneratedBodyForce supports set_field(\"previous\", buffer, 0)\n");
      return;
    }
    impl_->previous_buffer = values;
    impl_->previous = values->data();
  }

  void GeneratedBodyForce::set_value_in_block(const std::string &block_name,
                  const std::string &var_name,
                  const real_t value) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::set_value_in_block");
    impl_->domains->set_value_in_block(block_name, var_name, value);
  }

  std::shared_ptr<Op> GeneratedBodyForce::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::derefine_op");
    auto ret = std::make_shared<GeneratedBodyForce>(space);
    if (ret->initialize() != SFEM_SUCCESS) return nullptr;
    for (const auto &entry : impl_->domains->domains()) {
      for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
        ret->set_value_in_block(entry.first,
                                MATERIAL_PARAMETER_NAMES[i],
                                entry.second.parameters->require_real_value(MATERIAL_PARAMETER_NAMES[i]));
      }
    }
    return ret;
  }

  void GeneratedBodyForce::set_option(const std::string &name, const bool val) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::set_option");
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
        SFEM_ERROR("GeneratedBodyForce failed to cache affine geometry\n");
      }
    }
  }

#ifdef SFEM_ENABLE_RYAML
  std::shared_ptr<Op> GeneratedBodyForce::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::create_from_yaml");
    auto ret = std::make_shared<GeneratedBodyForce>(space);

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

  int GeneratedBodyForce::hessian_crs(const real_t *const /*state*/,
              const count_t *const /*rowptr*/,
              const idx_t *const /*colidx*/,
              real_t *const /*values*/) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::hessian_crs");
    // Linear work: the Jacobian is zero.
    return SFEM_SUCCESS;
  }

  int GeneratedBodyForce::hessian_bsr(const real_t *const x,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    return hessian_bsr_typed(x, rowptr, colidx, smesh::SMESH_DEFAULT, values);
  }

  int GeneratedBodyForce::hessian_bsr_typed(const real_t *const /*state*/,
              const count_t *const /*rowptr*/,
              const idx_t *const /*colidx*/,
              const enum smesh::PrimitiveType /*values_type*/,
              void *const /*values*/) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::hessian_bsr");
    // Linear work: the Jacobian is zero.
    return SFEM_SUCCESS;
  }


  int GeneratedBodyForce::value_steps(const real_t *x,
              const real_t *h,
              const int nsteps,
              const real_t *const steps,
              real_t *const out) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::value_steps");
    // The work is linear in the state, so `g` is the same at every step and one
    // gradient serves all of them.
    const ptrdiff_t ndofs = impl_->space->n_dofs();
    std::vector<real_t> work(ndofs, 0);
    if (gradient(x, work.data()) != SFEM_SUCCESS) {
      return SFEM_FAILURE;
    }
    real_t gx = 0;
    real_t gh = 0;
#pragma omp parallel for reduction(+ : gx, gh)
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
      gx += work[i] * x[i];
      gh += work[i] * h[i];
    }
    for (int step = 0; step < nsteps; ++step) {
      out[step] += gx + steps[step] * gh;
    }
    return SFEM_SUCCESS;
  }

  int GeneratedBodyForce::value(const real_t *x, real_t *const out) {
    SFEM_TRACE_SCOPE("GeneratedBodyForce::value");
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
