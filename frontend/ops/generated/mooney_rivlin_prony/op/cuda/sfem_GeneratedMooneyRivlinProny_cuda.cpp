#include "sfem_GeneratedMooneyRivlinProny_cuda.hpp"
#include "sfem_GeneratedMooneyRivlinProny_cuda_c_abi.hpp"


#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_DerefinedTimeScheme.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"
#include "sfem_RelaxationGroups.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>



namespace sfem {
  namespace {
    void seed_parameters(Parameters &parameters) {
      parameters.set_value("C01", 0);
      parameters.set_value("C10", 1);
      parameters.set_value("K", 1);
    }

    //! The relaxation groups the material declares, per spatial dimension.
    std::vector<RelaxationGroupDeclaration> relaxation_declarations(const int dim) {
      switch (dim) {
        case 3:
          return {{"iso", "iso_history", "iso_stress", 6}};
        default:
          break;
      }
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny declares no relaxation groups in dimension %d\n", dim);
      return {};
    }

    //! The points of the rule an element's kernels integrate with, which is
    //! where its history lives.  Every kernel of an element reads one rule, so
    //! the history the energy kernels read is the one the output kernel wrote.
    int quadrature_points(const smesh::ElemType element_type) {
      switch (element_type) {
        case smesh::HEX8:
          return 8;
        case smesh::PROTEUS_HEX8:
          return 8;
        default:
          break;
      }
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny has no integration-point history for this element type\n");
      return 0;
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
      values[0] = 0;
      values[1] = 1;
      values[2] = 1;
    }

#ifdef SFEM_ENABLE_RYAML
    constexpr int N_DEFINED_MATERIAL_PARAMETERS = 3;
    constexpr int N_MATERIAL_PARAMETERS = 3;
    static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"C01", "C10", "K"};

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
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny: mesh block pointer not found in mesh.blocks()\n");
      return 0;
    }

    struct AffineGeometryCache {
      std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
      std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
        };

    //! Whether this element leaves the operator no choice of geometry.
    //!
    //! A constant-P1 simplex publishes no isoparametric kernel -- its affine
    //! kernel computes the same numbers on a constant Jacobian -- so the
    //! operator takes the affine route for it whatever the caller asked, and
    //! has to have cached the adjugate it reads.  Every other element still
    //! caches only when asked, which is what keeps an isoparametric run from
    //! paying for geometry it never reads.
    bool geometry_is_forced_affine(const smesh::ElemType element_type) {
      switch (element_type) {
        default:
          return false;
      }
    }

    //! Whether the stored-tangent path assembles this element.
    //!
    //! Its tangent is built from the cached adjugate, so the cache is built for
    //! these elements whether or not an affine option is on -- and only for
    //! these: smesh has no adjugate fill for every element (HEX27, for one),
    //! and asking for one it cannot fill aborts inside `initialize`.
    bool geometry_assembles_inexact(const smesh::ElemType element_type) {
      switch (element_type) {
        default:
          return false;
      }
    }

    int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains,
                                  const bool requested) {
      auto mesh = space->mesh_ptr();
      const bool needs_jacobian_aos =
          false ||
          false;
      for (auto &entry : domains.domains()) {
        if (!requested && !geometry_is_forced_affine(entry.second.element_type) &&
            !geometry_assembles_inexact(entry.second.element_type)) {
          continue;
        }
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

    // Establish once per graph, at the first assembly into it, that the graph is well formed:
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
    // Not at setup: `initialize` built the space's scalar dof graph for this, which a
    // matrix-free solver never uses -- on a 76.6M-dof semi-structured mesh 6.2e9 entries,
    // past count_t -- while the block assembly writes into the node graph, which it never
    // checked.  `checked` remembers the last graph that passed.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    int check_graph(CheckedGraph &checked,
                    const count_t *const rowptr,
                    const idx_t *const colidx,
                    const ptrdiff_t n_rows) {
      if (rowptr == checked.rowptr && colidx == checked.colidx && n_rows == checked.n_rows) {
        return SFEM_SUCCESS;
      }
      if (!rowptr || !colidx || n_rows < 0) {
        return SFEM_FAILURE;
      }
      std::vector<count_t> rowptr_copy(n_rows + 1);
      if (cudaMemcpy(rowptr_copy.data(), rowptr, rowptr_copy.size() * sizeof(count_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
          rowptr_copy[n_rows] < 0) {
        return SFEM_FAILURE;
      }
      std::vector<idx_t> colidx_copy(rowptr_copy[n_rows]);
      if (cudaMemcpy(colidx_copy.data(), colidx, colidx_copy.size() * sizeof(idx_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return SFEM_FAILURE;
      }
      const count_t *const h_rowptr = rowptr_copy.data();
      const idx_t *const h_colidx = colidx_copy.data();
      if (h_rowptr[0] != 0) {
        return SFEM_FAILURE;
      }
      const ptrdiff_t nnz = (ptrdiff_t)h_rowptr[n_rows];
      for (ptrdiff_t i = 0; i < n_rows; ++i) {
        const count_t begin = h_rowptr[i];
        const count_t end = h_rowptr[i + 1];
        if (end < begin || (ptrdiff_t)end > nnz) {
          return SFEM_FAILURE;
        }
        for (count_t k = begin; k < end; ++k) {
          if (h_colidx[k] < 0 || (ptrdiff_t)h_colidx[k] >= n_rows) {
            return SFEM_FAILURE;
          }
          if (k > begin && h_colidx[k] <= h_colidx[k - 1]) {
            return SFEM_FAILURE;
          }
        }
      }
      checked = CheckedGraph{rowptr, colidx, n_rows};
      return SFEM_SUCCESS;
    }
  }  // namespace

  class GPUGeneratedMooneyRivlinProny::Impl {
  public:
    explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    std::shared_ptr<FunctionSpace> space;
    std::shared_ptr<MultiDomainOp> domains;
    std::shared_ptr<TimeScheme> time_scheme;
    std::shared_ptr<RelaxationGroups> relaxation;
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
    CheckedGraph crs_graph;
    CheckedGraph bsr_graph;
  };

  std::unique_ptr<Op> GPUGeneratedMooneyRivlinProny::create(const std::shared_ptr<FunctionSpace> &space) {
    const ptrdiff_t expected_block_size =
        block_size_for_dim(space->mesh_ptr()->spatial_dimension());
    if (space->block_size() != expected_block_size) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny requires block_size=%ld\n",
                       static_cast<long>(expected_block_size));
      return nullptr;
    }
    auto op = std::make_unique<GPUGeneratedMooneyRivlinProny>(space);
    op->initialize();
    return op;
  }

  GPUGeneratedMooneyRivlinProny::GPUGeneratedMooneyRivlinProny(const std::shared_ptr<FunctionSpace> &space)
    : impl_(std::make_unique<Impl>(space)) {}
  GPUGeneratedMooneyRivlinProny::~GPUGeneratedMooneyRivlinProny() = default;

  ptrdiff_t GPUGeneratedMooneyRivlinProny::n_dofs_domain() const { return impl_->space->n_dofs(); }
  ptrdiff_t GPUGeneratedMooneyRivlinProny::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GPUGeneratedMooneyRivlinProny::flops_value() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_objective_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedMooneyRivlinProny::memory_traffic_bytes_value() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_objective_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GPUGeneratedMooneyRivlinProny::flops_gradient() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_gradient_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedMooneyRivlinProny::memory_traffic_bytes_gradient() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_gradient_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    double GPUGeneratedMooneyRivlinProny::flops_apply() const {
    double total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_apply_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

    size_t GPUGeneratedMooneyRivlinProny::memory_traffic_bytes_apply() const {
    size_t total = 0;
    if (!impl_->domains) {
      return total;
    }

    const int dim = impl_->space->mesh_ptr()->spatial_dimension();
    impl_->domains->iterate([&](const OpDomain &domain) {
      const ptrdiff_t nelements = domain.block->n_elements();
      if (dim == 3) {
        {
          const sfem::codegen::KernelDiagnostics *const diagnostics = cu_mooney_rivlin_prony_apply_3d_soa_diagnostics(domain.element_type);
          if (diagnostics) {
            total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
          }
        }
      }
      return SFEM_SUCCESS;
    });

    return total;
  }

  int GPUGeneratedMooneyRivlinProny::initialize(const std::vector<std::string> &block_names) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::initialize");
    impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
    // A micro-element space hands each semi-structured macro element over as its micro
    // elements, which these kernels assemble as HEX8.
    impl_->domains->set_micro_elements(impl_->space->micro_elements());
    // Not only what the caller asked for.  An element whose geometry is
    // constant by construction has no isoparametric kernel, so the operator
    // takes the affine route for it either way and needs the cache either way.
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
    // Unconditionally, because the loop skips a domain that neither asked for
    // affine geometry nor is an element that forces it.  Gating the call on
    // the request alone left an operator meshing a constant-P1 simplex with no
    // cached adjugate and an affine kernel as its only route to that element.
    if (cache_affine_geometry(impl_->space, *impl_->domains, needs_affine_geometry) != SFEM_SUCCESS) {
      return SFEM_FAILURE;
    }
    impl_->relaxation = std::make_shared<RelaxationGroups>(
        relaxation_declarations(impl_->space->mesh_ptr()->spatial_dimension()),
        impl_->space->n_dofs(),
        EXECUTION_SPACE_DEVICE);
    for (auto &entry : impl_->domains->domains()) {
      impl_->relaxation->add_block(entry.second.block->name(),
                                   entry.second.block->n_elements(),
                                   quadrature_points(entry.second.element_type));
    }
    impl_->element_values = create_buffer<real_t>(impl_->element_capacity, EXECUTION_SPACE_DEVICE);
    impl_->element_ones = create_buffer<real_t>(impl_->element_capacity, EXECUTION_SPACE_DEVICE);
    sfem::blas<real_t>(EXECUTION_SPACE_DEVICE)->values(impl_->element_capacity, real_t(1), impl_->element_ones->data());

    return SFEM_SUCCESS;
  }

  void GPUGeneratedMooneyRivlinProny::set_time_scheme(const std::shared_ptr<TimeScheme> &scheme) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::set_time_scheme");
    if (impl_->time_scheme) {
      impl_->time_scheme->release(this);
    }
    impl_->time_scheme = scheme;
    if (scheme) {
      scheme->claim(this);
    }
  }

  // Null until `initialize`, which every factory route calls; an operator
  // driven without it gets a message instead of a crash.
  RelaxationGroups &GPUGeneratedMooneyRivlinProny::relaxation_groups() {
    if (!impl_->relaxation) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny: call initialize() before using the relaxation\n");
    }
    impl_->relaxation->set_stream(stream);
    return *impl_->relaxation;
  }

  // The relaxed stress the step starts from is the one the last `advance`
  // evaluated at the converged state, kept by the relaxation: each stress is
  // evaluated once, and with no spectrum anywhere not at all.
  void GPUGeneratedMooneyRivlinProny::begin_step(const real_t t, const real_t dt) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::begin_step");
    relaxation_groups().begin_step(
        t, dt, [this](const real_t *const state, const int slot) { return quadrature_output(state, slot); });
  }

  void GPUGeneratedMooneyRivlinProny::advance(const real_t *const x) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::advance");
    relaxation_groups().advance(
        x, [this](const real_t *const state, const int slot) { return quadrature_output(state, slot); });
  }

  void GPUGeneratedMooneyRivlinProny::reset_history(const real_t *const x) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::reset_history");
    relaxation_groups().reset(
        x, [this](const real_t *const state, const int slot) { return quadrature_output(state, slot); });
  }

  int GPUGeneratedMooneyRivlinProny::quadrature_output(const real_t *const x, const int slot) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::quadrature_output");
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *determinant = nullptr;
      if (impl_->gradient_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine quadrature output requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
      }
      const int dim = mesh->spatial_dimension();
      if (dim == 3) {
        if (adjugate) {
          return cu_mooney_rivlin_prony_quadrature_output_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), impl_->relaxation->stress_scratch(domain.block->name(), "iso_stress", slot), 3, x + 0, x + 1, x + 2, stream);
        }
        return cu_mooney_rivlin_prony_quadrature_output_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), impl_->relaxation->stress_scratch(domain.block->name(), "iso_stress", slot), 3, x + 0, x + 1, x + 2, stream);
      }
      SFEM_ERROR("mooney_rivlin_prony quadrature output does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedMooneyRivlinProny::gradient(const real_t *const x, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::gradient");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->gradient(x, out) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine gradient requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
        if (false) {
          if (!cache->jacobian_aos) {
            SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine gradient requires cached AoS geometry\n");
            return SFEM_FAILURE;
          }
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_determinant()->data());
        }
            }

      const int dim = mesh->spatial_dimension();
      if (dim == 3) {
        if (impl_->gradient_uses_affine) {
          return cu_mooney_rivlin_prony_gradient_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2, stream);
        }
        return cu_mooney_rivlin_prony_gradient_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2, stream);
      }
      SFEM_ERROR("mooney_rivlin_prony gradient does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedMooneyRivlinProny::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::apply");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->apply(x, h, out) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const geom_t *const *adjugate = nullptr;
      const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine hessian action requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
        if (false) {
          if (!cache->jacobian_aos) {
            SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine hessian action requires cached AoS geometry\n");
            return SFEM_FAILURE;
          }
          determinant = reinterpret_cast<const geom_t *>(
              cache->jacobian_aos->jacobian_determinant()->data());
        }
            }
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        if (impl_->apply_uses_affine) {
          SFEM_ERROR("mooney_rivlin_prony affine apply 2d dispatch was not generated\n");
          return SFEM_FAILURE;
        }
        SFEM_ERROR("mooney_rivlin_prony isoparametric apply 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        if (impl_->apply_uses_affine) {
          return cu_mooney_rivlin_prony_apply_3d_a_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2, stream);
        }
        return cu_mooney_rivlin_prony_apply_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2, stream);
      }
      SFEM_ERROR("mooney_rivlin_prony apply does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedMooneyRivlinProny::value(const real_t *x, real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::value");
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

  int GPUGeneratedMooneyRivlinProny::value_steps(const real_t *x,
              const real_t *h,
              const int nsteps,
              const real_t *const steps,
              real_t *const out) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::value_steps");
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
            if (impl_->objective_uses_affine) {
        auto cache = std::static_pointer_cast<AffineGeometryCache>(
            domain.user_data);
        if (!cache || !cache->jacobian_soa) {
          SFEM_ERROR("GPUGeneratedMooneyRivlinProny affine objective_steps requires cached geometry\n");
          return SFEM_FAILURE;
        }
        adjugate = reinterpret_cast<const geom_t *const *>(
            cache->jacobian_soa->jacobian_adjugate_SoA()->data());
        determinant = reinterpret_cast<const geom_t *>(
            cache->jacobian_soa->jacobian_determinant()->data());
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
        if (dim == 3) {
          if (impl_->objective_uses_affine) {
            status = cu_mooney_rivlin_prony_objective_steps_3d_a_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
          } else {
            status = cu_mooney_rivlin_prony_objective_steps_3d_i_msoa(domain.element_type, real_type, nelements, mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, impl_->step_values->data(), impl_->element_values->data(), stream);
          }
        }
        if (dim != 3) {
          SFEM_ERROR("mooney_rivlin_prony objective_steps does not support spatial dimension %d\n", dim);
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

  int GPUGeneratedMooneyRivlinProny::hessian_crs(const real_t *const x,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::hessian_crs");
    if (check_graph(impl_->crs_graph, rowptr, colidx, impl_->space->n_dofs()) != SFEM_SUCCESS) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny::hessian_crs: the graph is malformed; the assembly kernels assume it is not\n");
      return SFEM_FAILURE;
    }
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_crs(x, rowptr, colidx, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    const real_t *const current = x;
    if (!current) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny::hessian_crs requires a current state\n");
      return SFEM_FAILURE;
    }
    auto mesh = impl_->space->mesh_ptr();
    return impl_->domains->iterate([&](const OpDomain &/*domain*/) {
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        SFEM_ERROR("mooney_rivlin_prony hessian_crs 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        SFEM_ERROR("mooney_rivlin_prony hessian_crs 3d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      SFEM_ERROR("mooney_rivlin_prony hessian_crs does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  int GPUGeneratedMooneyRivlinProny::hessian_bsr(const real_t *const x,
              const count_t *const rowptr,
              const idx_t *const colidx,
              real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::hessian_bsr");
    if (check_graph(impl_->bsr_graph, rowptr, colidx, impl_->space->n_dofs() / impl_->space->block_size()) != SFEM_SUCCESS) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny::hessian_bsr: the graph is malformed; the assembly kernels assume it is not\n");
      return SFEM_FAILURE;
    }
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_bsr(x, rowptr, colidx, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    const real_t *const current = x;
    if (!current) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny::hessian_bsr requires a current state\n");
      return SFEM_FAILURE;
    }
    auto mesh = impl_->space->mesh_ptr();
    auto points = element_points(mesh);
    return impl_->domains->iterate([&](const OpDomain &domain) {
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        SFEM_ERROR("mooney_rivlin_prony hessian_bsr 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        return cu_mooney_rivlin_prony_hessian_bsr_3d_i_msoa(domain.element_type, real_type, domain.block->n_elements(), mesh->n_nodes(), element_connectivity(domain), points, domain.parameters->require_real_value("C01"), domain.parameters->require_real_value("C10"), domain.parameters->require_real_value("K"), impl_->relaxation->gamma(domain.block->name(), "iso"), impl_->relaxation->history(domain.block->name(), "iso_history"), 3, current + 0, current + 1, current + 2, rowptr, colidx, values, stream);
      }
      SFEM_ERROR("mooney_rivlin_prony hessian_bsr does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }




  int GPUGeneratedMooneyRivlinProny::hessian_block_diag_sym(const real_t *const x,
                                       real_t *const values) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::hessian_block_diag_sym");
    if (auto term = time_scheme_term(impl_->time_scheme)) {
      if (term->hessian_block_diag_sym(x, values) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
      }
    }
    const real_t *const current = x;
    if (!current) {
      SFEM_ERROR("GPUGeneratedMooneyRivlinProny::hessian_block_diag_sym requires a current state\n");
      return SFEM_FAILURE;
    }
    auto mesh = impl_->space->mesh_ptr();
    return impl_->domains->iterate([&](const OpDomain &/*domain*/) {
      const int dim = mesh->spatial_dimension();
      if (dim == 2) {
        SFEM_ERROR("mooney_rivlin_prony hessian_block_diag_sym 2d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      else if (dim == 3) {
        SFEM_ERROR("mooney_rivlin_prony hessian_block_diag_sym 3d dispatch was not generated\n");
        return SFEM_FAILURE;
      }
      SFEM_ERROR("mooney_rivlin_prony hessian_block_diag_sym does not support spatial dimension %d\n", dim);
      return SFEM_FAILURE;
    });
  }

  void GPUGeneratedMooneyRivlinProny::set_option(const std::string &name, const bool val) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::set_option");
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
      if (cache_affine_geometry(impl_->space, *impl_->domains, true) != SFEM_SUCCESS) {
        SFEM_ERROR("GPUGeneratedMooneyRivlinProny failed to cache affine geometry\n");
      }
    }
  }

  void GPUGeneratedMooneyRivlinProny::set_value_in_block(const std::string &block_name,
                  const std::string &var_name,
                  const real_t value) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::set_value_in_block");
    impl_->domains->set_value_in_block(block_name, var_name, value);
  }

  std::shared_ptr<Op> GPUGeneratedMooneyRivlinProny::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::derefine_op");
    auto ret = std::make_shared<GPUGeneratedMooneyRivlinProny>(space);
    if (impl_->time_scheme) {
      auto coarse_scheme = derefine_time_scheme(impl_->time_scheme, space);
      if (!coarse_scheme) return nullptr;
      ret->set_time_scheme(coarse_scheme);
    }
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

#ifdef SFEM_ENABLE_RYAML
  std::shared_ptr<Op> GPUGeneratedMooneyRivlinProny::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
    SFEM_TRACE_SCOPE("GPUGeneratedMooneyRivlinProny::create_from_yaml");
    auto ret = std::make_shared<GPUGeneratedMooneyRivlinProny>(space);

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
    ret->impl_->relaxation->configure_from_yaml("", node);

    if (node.has_child("blocks")) {
      for (auto block : node["blocks"].children()) {
        if (!block.has_child("name")) {
          continue;
        }
        ret->impl_->relaxation->configure_from_yaml(yaml_read_string(block["name"]), block);

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
