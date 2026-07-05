#include "sfem_Function_incore_cuda.hpp"
#include <cstring>
#include <memory>
#include "boundary_condition.hpp"

#include <cuda_runtime_api.h>
#include "sfem_defs.hpp"
#include "smesh_mesh.hpp"

// CPU
#include "sshex8_laplacian.hpp"

// GPU
#include "cu_boundary_condition.hpp"
#include "cu_hex8_adjugate.hpp"
#include "cu_hex8_fff.hpp"
#include "cu_integrate_values.hpp"
#include "cu_kelvin_voigt_newmark.hpp"
#include "cu_laplacian.hpp"
#include "cu_linear_elasticity.hpp"
#include "cu_mask.hpp"
#include "cu_sshex8_elemental_matrix.hpp"
#include "cu_tet4_adjugate.hpp"
#include "cu_tet4_fff.hpp"

// C++ includes
#include "sfem_API.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_device_buffer.hpp"
#include "smesh_device_sideset.hpp"
#include "smesh_kernel_data.hpp"

namespace sfem {
    template <typename T>
    static std::shared_ptr<Buffer<T *>> aos_to_soa(const ptrdiff_t                   n0,
                                                   const ptrdiff_t                   n1,
                                                   const ptrdiff_t                   in_stride0,
                                                   const ptrdiff_t                   in_stride1,
                                                   const std::shared_ptr<Buffer<T>> &in) {
        auto out = sfem::create_host_buffer<T>(n0, n1);

        {
            auto d_in  = in->data();
            auto d_out = out->data();

            for (ptrdiff_t i = 0; i < n0; i++) {
                for (ptrdiff_t j = 0; j < n1; j++) {
                    d_out[i][j] = d_in[i * in_stride0 + j * in_stride1];
                }
            }
        }

        return out;
    }

    std::shared_ptr<Buffer<idx_t *>> create_device_elements(const std::shared_ptr<FunctionSpace> &space,
                                                            const smesh::ElemType                 element_type) {
        if (space->has_semi_structured_mesh()) {
            return smesh::to_device(space->mesh().elements(0));

        } else {
            return smesh::to_device(space->mesh().elements(0));
        }
    }

    std::shared_ptr<Buffer<idx_t>> create_device_elements_AoS(const std::shared_ptr<FunctionSpace> &space,
                                                              const smesh::ElemType                 element_type) {
        if (space->has_semi_structured_mesh()) {
            auto nxe = space->mesh().n_nodes_per_element(0);
            return smesh::to_device(soa_to_aos(1, nxe, space->mesh().elements(0)));

        } else {
            auto nxe = space->mesh().n_nodes_per_element(0);
            return smesh::to_device(soa_to_aos(1, nxe, space->mesh().elements(0)));
        }
    }

    class GPUDirichletConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace>                     space;
        std::shared_ptr<DirichletConditions>               h_dirichlet;
        std::vector<struct DirichletConditions::Condition> conditions;

        GPUDirichletConditions(const std::shared_ptr<DirichletConditions> &dc) : space(dc->space()), h_dirichlet(dc) {
            for (auto &c : dc->conditions()) {
                DirichletConditions::Condition cond{.sidesets  = (!c.sidesets.empty()) ? smesh::to_device(c.sidesets)
                                                                                       : std::vector<std::shared_ptr<Sideset>>(),
                                                    .nodeset   = smesh::to_device(c.nodeset),
                                                    .values    = (c.values) ? smesh::to_device(c.values) : nullptr,
                                                    .value     = c.value,
                                                    .component = c.component};
                conditions.push_back(cond);
            }
        }

        int apply(real_t *const x) override {
            for (auto &c : conditions) {
                d_constraint_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, c.value, x);
            }

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const g) override {
            for (auto &c : conditions) {
                d_constraint_gradient_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, c.value, x, g);
            }

            return SFEM_SUCCESS;
        }

        int apply_value(const real_t value, real_t *const x) override {
            for (auto &c : conditions) {
                d_constraint_nodes_to_value_vec(c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, value, x);
            }

            return SFEM_SUCCESS;
        }

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override {
            for (auto &c : conditions) {
                d_constraint_nodes_copy_vec(c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, src, dest);
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            for (auto &c : conditions) {
                cu_crs_constraint_nodes_to_identity_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, 1, rowptr, colidx, values);
            }

            return SFEM_SUCCESS;
        }

        std::shared_ptr<Constraint> lor() const override {
            SFEM_IMPLEMENT_ME();
            return nullptr;
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<sfem::FunctionSpace> &space, bool as_zeros) const override {
            auto h_derefined = std::static_pointer_cast<DirichletConditions>(h_dirichlet->derefine(space, as_zeros));
            return std::make_shared<GPUDirichletConditions>(h_derefined);
        }

        int mask(mask_t *mask) override {
            SFEM_TRACE_SCOPE("GPUDirichletConditions::mask");

            const int block_size = space->block_size();

            int err = SFEM_SUCCESS;
            for (auto &c : conditions) {
                auto nodeset = c.nodeset->data();
                err += cu_mask_nodes(c.nodeset->size(), nodeset, block_size, c.component, mask);
            }

            return err;
        }

        ~GPUDirichletConditions() {}
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc) {
        return std::make_shared<GPUDirichletConditions>(dc);
    }

    class GPUNeumannConditions final : public Op {
    public:
        std::shared_ptr<FunctionSpace>                   space;
        std::shared_ptr<NeumannConditions>               h_neumann;
        std::vector<struct NeumannConditions::Condition> conditions;

        GPUNeumannConditions(const std::shared_ptr<NeumannConditions> &nc) : space(nc->space()), h_neumann(nc) {
            for (auto &c : nc->conditions()) {
                NeumannConditions::Condition cond{.element_type = c.element_type,
                                                  .sidesets     = (!c.sidesets.empty()) ? smesh::to_device(c.sidesets)
                                                                                        : std::vector<std::shared_ptr<Sideset>>(),
                                                  .surface      = smesh::to_device(c.surface),
                                                  .values       = (c.values) ? smesh::to_device(c.values) : nullptr,
                                                  .value        = c.value,
                                                  .component    = c.component};
                conditions.push_back(cond);
            }
        }

        const char *name() const override { return "gpu:NeumannConditions"; }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("GPUNeumannConditions::gradient");

            // Prepare points on host (mesh or semi-structured mesh)
            auto space = this->space;
            auto mesh  = space->mesh_ptr();

            auto points = mesh->points();
            if (space->has_semi_structured_mesh()) {
                points = space->mesh().points();
            }

            // Device-only: assume 'out' is a device pointer
            real_t *out_dev = out;

            int err = 0;

            // Use host NeumannConditions for host-side data (surface indices and values),
            // and device-side copies held in this->conditions for GPU kernels
            const auto &hconds = h_neumann->conditions();

            for (size_t i = 0; i < conditions.size(); ++i) {
                const auto &hc = hconds[i];
                const auto &dc = conditions[i];

                const smesh::ElemType st   = hc.element_type;
                const int             nnxs = elem_num_nodes(st);
                const int             dim  = mesh->spatial_dimension();
                const ptrdiff_t       ne   = dc.surface->extent(1);

                if (ne == 0) continue;

                if (st != smesh::QUADSHELL4) {
                    SFEM_ERROR(
                            "GPUNeumannConditions::gradient: unsupported element type %s (GPU supports only smesh::QUADSHELL4)\n",
                            type_to_string(st));
                    return SFEM_FAILURE;
                }

                auto coords_h = sfem::create_host_buffer<geom_t>(dim, ne * nnxs);
                for (int d = 0; d < dim; ++d) {
                    const geom_t *const px = points->data()[d];
                    geom_t *const       cx = coords_h->data()[d];

                    for (ptrdiff_t e = 0; e < ne; ++e) {
                        for (int v = 0; v < nnxs; ++v) {
                            const idx_t node = hc.surface->data()[v][e];
                            cx[v * ne + e]   = px[node];
                        }
                    }
                }

                auto coords_d = smesh::to_device(coords_h);

                if (hc.values && hc.values->size() && dc.values) {
                    auto scaled = smesh::create_device_buffer<real_t>(ne);

                    d_copy(ne, dc.values->data(), scaled->data());
                    d_scal(ne, -hc.value, scaled->data());

                    int gpu_ret = cu_integrate_values(st,
                                                      ne,
                                                      dc.surface->data(),
                                                      (const geom_t **)coords_d->data(),
                                                      smesh::SMESH_DEFAULT,
                                                      (void *)scaled->data(),
                                                      space->block_size(),
                                                      hc.component,
                                                      (void *)out_dev,
                                                      SFEM_DEFAULT_STREAM);
                    err |= gpu_ret;
                } else {
                    int gpu_ret = cu_integrate_value(st,
                                                     ne,
                                                     dc.surface->data(),
                                                     (const geom_t **)coords_d->data(),
                                                     -hc.value,
                                                     space->block_size(),
                                                     hc.component,
                                                     smesh::SMESH_DEFAULT,
                                                     (void *)out_dev,
                                                     SFEM_DEFAULT_STREAM);
                    err |= gpu_ret;
                }
            }

            if (err) return err;

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override { return SFEM_SUCCESS; }

        int value(const real_t *x, real_t *const out) override { return SFEM_SUCCESS; }

        int hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) override { return SFEM_SUCCESS; }

        inline bool      is_linear() const override { return true; }
        inline ptrdiff_t n_dofs_domain() const override { return space->n_dofs(); }
        inline ptrdiff_t n_dofs_image() const override { return space->n_dofs(); }

        int n_conditions() const;

        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override { return no_op(); }
    };

    std::shared_ptr<Op> to_device(const std::shared_ptr<NeumannConditions> &nc) {
        return std::make_shared<GPUNeumannConditions>(nc);
    }

    namespace {
        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); i++) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }

            SFEM_ERROR("GPULaplacian: mesh block pointer not found in mesh.blocks()");
            return 0;
        }

        class GPULaplacianOpData {
        public:
            smesh::ElemType                     element_type{smesh::INVALID};
            std::shared_ptr<Buffer<idx_t *>>    elements;
            std::shared_ptr<Buffer<jacobian_t>> fff;

            ptrdiff_t nelements() const {
                assert(elements);
                return elements->extent(1);
            }

            const jacobian_t *fff_data() const {
                assert(fff);
                return fff->data();
            }
        };

        static std::shared_ptr<Buffer<jacobian_t>> create_gpu_laplacian_fff(const std::shared_ptr<FunctionSpace> &space,
                                                                            const smesh::block_idx_t              block_id) {
            constexpr ptrdiff_t fff_size = 6;

            auto fff_src = smesh::FFF::create_SoA(space->mesh_ptr(), smesh::MEMORY_SPACE_HOST, block_id);
            if (!fff_src || !fff_src->fff_SoA()) {
                return nullptr;
            }

            const auto nelements = space->mesh_ptr()->n_elements(block_id);
            auto       flat_fff  = sfem::create_host_buffer<jacobian_t>(fff_size * nelements);

            auto *const       dst = flat_fff->data();
            const auto *const src = fff_src->fff_SoA()->data();
            for (ptrdiff_t d = 0; d < fff_size; d++) {
                memcpy(&dst[d * nelements], src[d], nelements * sizeof(jacobian_t));
            }

            return smesh::to_device(flat_fff);
        }

        static std::shared_ptr<GPULaplacianOpData> create_gpu_laplacian_op_data(const std::shared_ptr<FunctionSpace> &space,
                                                                                const OpDomain                       &domain) {
            auto ret          = std::make_shared<GPULaplacianOpData>();
            ret->element_type = domain.element_type;
            ret->elements     = smesh::to_device(domain.block->elements());

            const auto block_id = block_id_for_domain(*space->mesh_ptr(), *domain.block);
            ret->fff            = create_gpu_laplacian_fff(space, block_id);
            return ret;
        }

        class GPULinearElasticityOpData {
        public:
            std::shared_ptr<Buffer<idx_t *>>    elements;
            std::shared_ptr<Buffer<jacobian_t>> jacobian_adjugate;
            std::shared_ptr<Buffer<geom_t>>     jacobian_determinant;

            ptrdiff_t nelements() const {
                assert(elements);
                return elements->extent(1);
            }
        };

        static std::shared_ptr<Buffer<jacobian_t>> create_gpu_jacobian_adjugate(const std::shared_ptr<FunctionSpace> &space,
                                                                                const smesh::block_idx_t              block_id) {
            constexpr ptrdiff_t adjugate_size = 9;

            auto jac_src =
                    smesh::JacobianAdjugateAndDeterminant::create_SoA(space->mesh_ptr(), smesh::MEMORY_SPACE_HOST, block_id);
            if (!jac_src || !jac_src->jacobian_adjugate_SoA()) {
                return nullptr;
            }

            const auto nelements = space->mesh_ptr()->n_elements(block_id);
            auto       flat_adj  = sfem::create_host_buffer<jacobian_t>(adjugate_size * nelements);

            auto *const       dst = flat_adj->data();
            const auto *const src = jac_src->jacobian_adjugate_SoA()->data();
            for (ptrdiff_t d = 0; d < adjugate_size; d++) {
                memcpy(&dst[d * nelements], src[d], nelements * sizeof(jacobian_t));
            }

            return smesh::to_device(flat_adj);
        }

        static std::shared_ptr<Buffer<geom_t>> create_gpu_jacobian_determinant(const std::shared_ptr<FunctionSpace> &space,
                                                                               const smesh::block_idx_t              block_id) {
            auto jac_src =
                    smesh::JacobianAdjugateAndDeterminant::create_SoA(space->mesh_ptr(), smesh::MEMORY_SPACE_HOST, block_id);
            if (!jac_src || !jac_src->jacobian_determinant()) {
                return nullptr;
            }

            return smesh::to_device(jac_src->jacobian_determinant());
        }

        static void gpu_linear_elasticity_seed_material(MultiDomainOp &m, const real_t mu, const real_t lambda) {
            for (auto &kv : m.domains()) {
                kv.second.parameters->set_value("mu", mu);
                kv.second.parameters->set_value("lambda", lambda);
            }
        }

        static void gpu_linear_elasticity_copy_material(const MultiDomainOp &from, MultiDomainOp &to) {
            for (const auto &kv : from.domains()) {
                auto it = to.domains().find(kv.first);
                if (it == to.domains().end()) {
                    continue;
                }

                it->second.parameters->set_value("mu", kv.second.parameters->require_real_value("mu"));
                it->second.parameters->set_value("lambda", kv.second.parameters->require_real_value("lambda"));
            }
        }

        static std::shared_ptr<GPULinearElasticityOpData> create_gpu_linear_elasticity_op_data(
                const std::shared_ptr<FunctionSpace> &space,
                const OpDomain                       &domain) {
            auto ret      = std::make_shared<GPULinearElasticityOpData>();
            ret->elements = smesh::to_device(domain.block->elements());

            const auto block_id       = block_id_for_domain(*space->mesh_ptr(), *domain.block);
            ret->jacobian_adjugate    = create_gpu_jacobian_adjugate(space, block_id);
            ret->jacobian_determinant = create_gpu_jacobian_determinant(space, block_id);
            return ret;
        }

        class GPUKelvinVoigtNewmarkOpData {
        public:
            std::shared_ptr<Buffer<idx_t *>>    elements;
            std::shared_ptr<Buffer<jacobian_t>> jacobian_adjugate;
            std::shared_ptr<Buffer<geom_t>>     jacobian_determinant;

            ptrdiff_t nelements() const {
                assert(elements);
                return elements->extent(1);
            }
        };

        static void gpu_kv_seed_material(MultiDomainOp &m,
                                         const real_t   k,
                                         const real_t   K,
                                         const real_t   eta,
                                         const real_t   dt,
                                         const real_t   gamma,
                                         const real_t   beta,
                                         const real_t   rho) {
            for (auto &kv : m.domains()) {
                kv.second.parameters->set_value("k", k);
                kv.second.parameters->set_value("K", K);
                kv.second.parameters->set_value("eta", eta);
                kv.second.parameters->set_value("dt", dt);
                kv.second.parameters->set_value("gamma", gamma);
                kv.second.parameters->set_value("beta", beta);
                kv.second.parameters->set_value("rho", rho);
            }
        }

        static void gpu_kv_copy_material(const MultiDomainOp &from, MultiDomainOp &to) {
            for (const auto &kv : from.domains()) {
                auto it = to.domains().find(kv.first);
                if (it == to.domains().end()) {
                    continue;
                }

                it->second.parameters->set_value("k", kv.second.parameters->require_real_value("k"));
                it->second.parameters->set_value("K", kv.second.parameters->require_real_value("K"));
                it->second.parameters->set_value("eta", kv.second.parameters->require_real_value("eta"));
                it->second.parameters->set_value("dt", kv.second.parameters->require_real_value("dt"));
                it->second.parameters->set_value("gamma", kv.second.parameters->require_real_value("gamma"));
                it->second.parameters->set_value("beta", kv.second.parameters->require_real_value("beta"));
                it->second.parameters->set_value("rho", kv.second.parameters->require_real_value("rho"));
            }
        }

        static std::shared_ptr<GPUKelvinVoigtNewmarkOpData> create_gpu_kv_op_data(const std::shared_ptr<FunctionSpace> &space,
                                                                                  const OpDomain                       &domain) {
            auto ret      = std::make_shared<GPUKelvinVoigtNewmarkOpData>();
            ret->elements = smesh::to_device(domain.block->elements());

            const auto block_id       = block_id_for_domain(*space->mesh_ptr(), *domain.block);
            ret->jacobian_adjugate    = create_gpu_jacobian_adjugate(space, block_id);
            ret->jacobian_determinant = create_gpu_jacobian_determinant(space, block_id);
            return ret;
        }
    }  // namespace

    class GPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        enum smesh::PrimitiveType      real_type{smesh::SMESH_DEFAULT};
        void                          *stream{SFEM_DEFAULT_STREAM};
        smesh::ElemType                element_type{smesh::INVALID};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(1 == space->block_size());
            return std::make_unique<GPULaplacian>(space);
        }

        const char *name() const override { return is_semistructured_type(element_type) ? "ss:gpu::Laplacian" : "gpu:Laplacian"; }
        inline bool is_linear() const override { return true; }
        ptrdiff_t   n_dofs_domain() const override { return space->n_dofs(); }
        ptrdiff_t   n_dofs_image() const override { return space->n_dofs(); }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            SFEM_TRACE_SCOPE("GPULaplacian:initialize");
            domains = std::make_shared<MultiDomainOp>(space, block_names);

            for (auto &n2d : domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_laplacian_op_data(space, domain);
                if (!domain_op || !domain_op->fff) {
                    return SFEM_FAILURE;
                }

                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }

            return SFEM_SUCCESS;
        }

        GPULaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space), element_type(space->element_type()) {}

        int iterate(const std::function<int(const OpDomain &)> &func) const {
            assert(domains);
            return domains->iterate(func);
        }

        std::vector<std::string> selected_block_names() const {
            std::vector<std::string> ret;
            if (!domains) {
                return ret;
            }

            for (const auto &it : domains->domains()) {
                ret.push_back(it.first);
            }

            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            const auto block_names = selected_block_names();

            if (derefined_space->has_semi_structured_mesh() && is_semistructured_type(derefined_space->element_type())) {
                auto ret = std::make_shared<GPULaplacian>(derefined_space);
                ret->initialize(block_names);
                return ret;
            }

            if (space->has_semi_structured_mesh() && is_semistructured_type(element_type)) {
                auto ret = std::make_shared<GPULaplacian>(derefined_space);
                ret->initialize(block_names);
                assert(derefined_space->n_blocks() == 1);
                ret->override_element_types({derefined_space->element_type()});
                return ret;
            }

            auto ret     = std::make_shared<GPULaplacian>(derefined_space);
            ret->domains = domains->derefine_op(derefined_space, block_names);
            for (auto &n2d : ret->domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_laplacian_op_data(derefined_space, domain);
                if (!domain_op || !domain_op->fff) {
                    return nullptr;
                }
                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }
            return ret;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return iterate([&](const OpDomain &domain) {
                auto op_data = std::static_pointer_cast<GPULaplacianOpData>(domain.user_data);
                if (is_semistructured_type(domain.element_type)) {
                    SFEM_IMPLEMENT_ME();
                    return SFEM_FAILURE;
                }

                return cu_laplacian_crs(domain.element_type,
                                        op_data->nelements(),
                                        op_data->elements->data(),
                                        op_data->nelements(),
                                        op_data->fff_data(),
                                        rowptr,
                                        colidx,
                                        real_type,
                                        values,
                                        stream);
            });
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            return iterate([&](const OpDomain &domain) {
                auto op_data = std::static_pointer_cast<GPULaplacianOpData>(domain.user_data);
                return cu_laplacian_diag(domain.element_type,
                                         op_data->nelements(),
                                         op_data->elements->data(),
                                         op_data->nelements(),
                                         op_data->fff_data(),
                                         real_type,
                                         values,
                                         stream);
            });
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return iterate([&](const OpDomain &domain) {
                auto op_data = std::static_pointer_cast<GPULaplacianOpData>(domain.user_data);
                return cu_laplacian_apply(domain.element_type,
                                          op_data->nelements(),
                                          op_data->elements->data(),
                                          op_data->nelements(),
                                          op_data->fff_data(),
                                          real_type,
                                          x,
                                          out,
                                          stream);
            });
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return iterate([&](const OpDomain &domain) {
                auto op_data = std::static_pointer_cast<GPULaplacianOpData>(domain.user_data);
                return cu_laplacian_apply(domain.element_type,
                                          op_data->nelements(),
                                          op_data->elements->data(),
                                          op_data->nelements(),
                                          op_data->fff_data(),
                                          real_type,
                                          h,
                                          out,
                                          stream);
            });
        }

        int hessian_crs_sym(const real_t *const /*x*/,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override {
            return iterate([&](const OpDomain &domain) {
                auto op_data = std::static_pointer_cast<GPULaplacianOpData>(domain.user_data);
                if (is_semistructured_type(domain.element_type)) {
                    SFEM_IMPLEMENT_ME();
                    return SFEM_FAILURE;
                }

                return cu_laplacian_crs_sym(domain.element_type,
                                            op_data->nelements(),
                                            op_data->elements->data(),
                                            op_data->nelements(),
                                            op_data->fff_data(),
                                            rowptr,
                                            colidx,
                                            real_type,
                                            diag_values,
                                            off_diag_values,
                                            stream);
            });
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        std::shared_ptr<Op> clone() const override {
            auto ret       = std::make_shared<GPULaplacian>(space);
            ret->domains   = domains;
            ret->real_type = real_type;
            ret->stream    = stream;
            return ret;
        }

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override {
            if (domains) {
                domains->set_value_in_block(block_name, var_name, value);
            }
        }

        void override_element_types(const std::vector<smesh::ElemType> &element_types) override {
            if (domains) {
                domains->override_element_types(element_types);
            }
        }
    };

    class GPULinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        enum smesh::PrimitiveType      real_type{smesh::SMESH_DEFAULT};
        void                          *stream{SFEM_DEFAULT_STREAM};
        smesh::ElemType                element_type{smesh::INVALID};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
            return std::make_unique<GPULinearElasticity>(space);
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            const auto block_names = selected_block_names();

            if (derefined_space->has_semi_structured_mesh() && is_semistructured_type(derefined_space->element_type())) {
                auto ret = std::make_shared<GPULinearElasticity>(derefined_space);
                ret->initialize(block_names);
                gpu_linear_elasticity_copy_material(*domains, *ret->domains);
                return ret;
            }

            if (space->has_semi_structured_mesh() && is_semistructured_type(element_type)) {
                auto ret = std::make_shared<GPULinearElasticity>(derefined_space);
                ret->initialize(block_names);
                gpu_linear_elasticity_copy_material(*domains, *ret->domains);
                assert(derefined_space->n_blocks() == 1);
                ret->override_element_types({derefined_space->element_type()});
                return ret;
            }

            auto ret     = std::make_shared<GPULinearElasticity>(derefined_space);
            ret->domains = domains->derefine_op(derefined_space, block_names);
            gpu_linear_elasticity_copy_material(*domains, *ret->domains);

            for (auto &n2d : ret->domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_linear_elasticity_op_data(derefined_space, domain);
                if (!domain_op || !domain_op->jacobian_adjugate || !domain_op->jacobian_determinant) {
                    return nullptr;
                }

                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }

            return ret;
        }

        const char *name() const override {
            return is_semistructured_type(element_type) ? "ss:gpu:LinearElasticity" : "gpu:LinearElasticity";
        }
        inline bool is_linear() const override { return true; }
        ptrdiff_t   n_dofs_domain() const override { return space->n_dofs(); }
        ptrdiff_t   n_dofs_image() const override { return space->n_dofs(); }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            SFEM_TRACE_SCOPE("GPULinearElasticity:initialize");
            domains = std::make_shared<MultiDomainOp>(space, block_names);

            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;
            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
            gpu_linear_elasticity_seed_material(*domains, SFEM_SHEAR_MODULUS, SFEM_FIRST_LAME_PARAMETER);

            for (auto &n2d : domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_linear_elasticity_op_data(space, domain);
                if (!domain_op || !domain_op->jacobian_adjugate || !domain_op->jacobian_determinant) {
                    return SFEM_FAILURE;
                }

                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }

            return SFEM_SUCCESS;
        }

        GPULinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space), element_type(space->element_type()) {}

        int iterate(const std::function<int(const OpDomain &)> &func) const {
            assert(domains);
            return domains->iterate(func);
        }

        std::vector<std::string> selected_block_names() const {
            std::vector<std::string> ret;
            if (!domains) {
                return ret;
            }

            for (const auto &it : domains->domains()) {
                ret.push_back(it.first);
            }

            return ret;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_bsr");
            return iterate([&](const OpDomain &domain) {
                if (is_semistructured_type(domain.element_type)) {
                    SFEM_IMPLEMENT_ME();
                    return SFEM_FAILURE;
                }

                auto domain_op = std::static_pointer_cast<GPULinearElasticityOpData>(domain.user_data);
                auto mu        = domain.parameters->require_real_value("mu");
                auto lambda    = domain.parameters->require_real_value("lambda");

                return cu_linear_elasticity_bsr(domain.element_type,
                                                domain_op->nelements(),
                                                domain_op->elements->data(),
                                                domain_op->nelements(),
                                                domain_op->jacobian_adjugate->data(),
                                                domain_op->jacobian_determinant->data(),
                                                mu,
                                                lambda,
                                                real_type,
                                                rowptr,
                                                colidx,
                                                values,
                                                stream);
            });
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_diag");
            return iterate([&](const OpDomain &domain) {
                auto domain_op = std::static_pointer_cast<GPULinearElasticityOpData>(domain.user_data);
                auto mu        = domain.parameters->require_real_value("mu");
                auto lambda    = domain.parameters->require_real_value("lambda");

                return cu_linear_elasticity_diag(domain.element_type,
                                                 domain_op->nelements(),
                                                 domain_op->elements->data(),
                                                 domain_op->nelements(),
                                                 domain_op->jacobian_adjugate->data(),
                                                 domain_op->jacobian_determinant->data(),
                                                 mu,
                                                 lambda,
                                                 real_type,
                                                 values,
                                                 stream);
            });
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_apply");
            return iterate([&](const OpDomain &domain) {
                auto domain_op = std::static_pointer_cast<GPULinearElasticityOpData>(domain.user_data);
                auto mu        = domain.parameters->require_real_value("mu");
                auto lambda    = domain.parameters->require_real_value("lambda");

                return cu_linear_elasticity_apply(domain.element_type,
                                                  domain_op->nelements(),
                                                  domain_op->elements->data(),
                                                  domain_op->nelements(),
                                                  domain_op->jacobian_adjugate->data(),
                                                  domain_op->jacobian_determinant->data(),
                                                  mu,
                                                  lambda,
                                                  real_type,
                                                  x,
                                                  out,
                                                  stream);
            });
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_apply");
            return iterate([&](const OpDomain &domain) {
                auto domain_op = std::static_pointer_cast<GPULinearElasticityOpData>(domain.user_data);
                auto mu        = domain.parameters->require_real_value("mu");
                auto lambda    = domain.parameters->require_real_value("lambda");

                return cu_linear_elasticity_apply(domain.element_type,
                                                  domain_op->nelements(),
                                                  domain_op->elements->data(),
                                                  domain_op->nelements(),
                                                  domain_op->jacobian_adjugate->data(),
                                                  domain_op->jacobian_determinant->data(),
                                                  mu,
                                                  lambda,
                                                  real_type,
                                                  h,
                                                  out,
                                                  stream);
            });
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym");
            return iterate([&](const OpDomain &domain) {
                auto domain_op = std::static_pointer_cast<GPULinearElasticityOpData>(domain.user_data);
                auto mu        = domain.parameters->require_real_value("mu");
                auto lambda    = domain.parameters->require_real_value("lambda");

                return cu_linear_elasticity_block_diag_sym_aos(domain.element_type,
                                                               domain_op->nelements(),
                                                               domain_op->elements->data(),
                                                               domain_op->nelements(),
                                                               domain_op->jacobian_adjugate->data(),
                                                               domain_op->jacobian_determinant->data(),
                                                               mu,
                                                               lambda,
                                                               real_type,
                                                               values,
                                                               stream);
            });
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        std::shared_ptr<Op> clone() const override {
            auto ret       = std::make_shared<GPULinearElasticity>(space);
            ret->domains   = domains;
            ret->real_type = real_type;
            ret->stream    = stream;
            return ret;
        }

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override {
            if (domains) {
                domains->set_value_in_block(block_name, var_name, value);
            }
        }

        void override_element_types(const std::vector<smesh::ElemType> &element_types) override {
            if (domains) {
                domains->override_element_types(element_types);
            }
        }
    };

    class GPUKelvinVoigtNewmark final : public Op {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<MultiDomainOp>  domains;
        enum smesh::PrimitiveType       real_type{smesh::SMESH_DEFAULT};
        void                           *stream{SFEM_DEFAULT_STREAM};
        smesh::ElemType                 element_type{smesh::INVALID};
        real_t                          dt{0.1};
        real_t                          gamma{0.5};
        real_t                          beta{0.25};
        std::shared_ptr<Buffer<real_t>> vel_[3];
        std::shared_ptr<Buffer<real_t>> acc_[3];

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
            return std::make_unique<GPUKelvinVoigtNewmark>(space);
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            const auto block_names = selected_block_names();

            if (derefined_space->has_semi_structured_mesh() && is_semistructured_type(derefined_space->element_type())) {
                auto ret = std::make_shared<GPUKelvinVoigtNewmark>(derefined_space);
                ret->initialize(block_names);
                gpu_kv_copy_material(*domains, *ret->domains);
                ret->real_type = real_type;
                ret->stream    = stream;
                ret->dt        = dt;
                ret->gamma     = gamma;
                ret->beta      = beta;
                for (int c = 0; c < 3; c++) {
                    ret->vel_[c] = vel_[c];
                    ret->acc_[c] = acc_[c];
                }
                return ret;
            }

            if (space->has_semi_structured_mesh() && is_semistructured_type(element_type)) {
                auto ret = std::make_shared<GPUKelvinVoigtNewmark>(derefined_space);
                ret->initialize(block_names);
                gpu_kv_copy_material(*domains, *ret->domains);
                assert(derefined_space->n_blocks() == 1);
                ret->override_element_types({derefined_space->element_type()});
                ret->real_type = real_type;
                ret->stream    = stream;
                ret->dt        = dt;
                ret->gamma     = gamma;
                ret->beta      = beta;
                for (int c = 0; c < 3; c++) {
                    ret->vel_[c] = vel_[c];
                    ret->acc_[c] = acc_[c];
                }
                return ret;
            }

            auto ret     = std::make_shared<GPUKelvinVoigtNewmark>(derefined_space);
            ret->domains = domains->derefine_op(derefined_space, block_names);
            gpu_kv_copy_material(*domains, *ret->domains);
            ret->real_type = real_type;
            ret->stream    = stream;
            ret->dt        = dt;
            ret->gamma     = gamma;
            ret->beta      = beta;
            for (int c = 0; c < 3; c++) {
                ret->vel_[c] = vel_[c];
                ret->acc_[c] = acc_[c];
            }

            for (auto &n2d : ret->domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_kv_op_data(derefined_space, domain);
                if (!domain_op || !domain_op->jacobian_adjugate || !domain_op->jacobian_determinant) {
                    return nullptr;
                }
                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }

            return ret;
        }

        const char *name() const override {
            return is_semistructured_type(element_type) ? "ss:gpu:KelvinVoigtNewmark" : "gpu:KelvinVoigtNewmark";
        }
        inline bool is_linear() const override { return true; }
        ptrdiff_t   n_dofs_domain() const override { return space->n_dofs(); }
        ptrdiff_t   n_dofs_image() const override { return space->n_dofs(); }

        void set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &vel, int component) override {
            if (strcmp(name, "velocity") == 0) {
                vel_[component] = vel;
            } else if (strcmp(name, "acceleration") == 0) {
                acc_[component] = vel;
            } else {
                SFEM_ERROR(
                        "Invalid field name! Call set_field(\"velocity\", buffer, 0/1/2) or set_field(\"acceleration\", buffer, "
                        "0/1/2) first.\n");
            }
        }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            SFEM_TRACE_SCOPE("GPUKelvinVoigtNewmark:initialize");
            domains = std::make_shared<MultiDomainOp>(space, block_names);

            real_t SFEM_SHEAR_STIFFNESS_KV = 4;
            real_t SFEM_BULK_MODULUS       = 3;
            real_t SFEM_DAMPING_RATIO      = 0.1;
            real_t SFEM_DENSITY            = 1.0;

            SFEM_READ_ENV(SFEM_SHEAR_STIFFNESS_KV, atof);
            SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
            SFEM_READ_ENV(SFEM_DAMPING_RATIO, atof);
            SFEM_READ_ENV(SFEM_DENSITY, atof);
            // Optional Newmark parameters from env (defaults: beta=1/4, gamma=1/2)
            real_t SFEM_DT            = dt;
            real_t SFEM_NEWMARK_GAMMA = gamma;
            real_t SFEM_NEWMARK_BETA  = beta;
            SFEM_READ_ENV(SFEM_DT, atof);
            SFEM_READ_ENV(SFEM_NEWMARK_GAMMA, atof);
            SFEM_READ_ENV(SFEM_NEWMARK_BETA, atof);
            dt    = SFEM_DT;
            gamma = SFEM_NEWMARK_GAMMA;
            beta  = SFEM_NEWMARK_BETA;

            gpu_kv_seed_material(
                    *domains, SFEM_SHEAR_STIFFNESS_KV, SFEM_BULK_MODULUS, SFEM_DAMPING_RATIO, dt, gamma, beta, SFEM_DENSITY);

            for (auto &n2d : domains->domains()) {
                OpDomain &domain    = n2d.second;
                auto      domain_op = create_gpu_kv_op_data(space, domain);
                if (!domain_op || !domain_op->jacobian_adjugate || !domain_op->jacobian_determinant) {
                    return SFEM_FAILURE;
                }
                domain.user_data = std::static_pointer_cast<void>(domain_op);
            }
            return SFEM_SUCCESS;
        }

        GPUKelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space) : space(space), element_type(space->element_type()) {}

        int iterate(const std::function<int(const OpDomain &)> &func) const {
            assert(domains);
            return domains->iterate(func);
        }

        std::vector<std::string> selected_block_names() const {
            std::vector<std::string> ret;
            if (!domains) {
                return ret;
            }

            for (const auto &it : domains->domains()) {
                ret.push_back(it.first);
            }

            return ret;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            return iterate([&](const OpDomain &domain) {
                auto         domain_op = std::static_pointer_cast<GPUKelvinVoigtNewmarkOpData>(domain.user_data);
                const auto   params    = domain.parameters;
                const real_t k         = params->require_real_value("k");
                const real_t K         = params->require_real_value("K");
                const real_t eta       = params->require_real_value("eta");
                const real_t rho       = params->require_real_value("rho");
                const real_t dt_       = params->require_real_value("dt");
                const real_t gamma_    = params->require_real_value("gamma");
                const real_t beta_     = params->require_real_value("beta");
                return cu_kelvin_voigt_newmark_diag(domain.element_type,
                                                    domain_op->nelements(),
                                                    domain_op->elements->data(),
                                                    domain_op->nelements(),
                                                    domain_op->jacobian_adjugate->data(),
                                                    domain_op->jacobian_determinant->data(),
                                                    k,
                                                    K,
                                                    eta,
                                                    rho,
                                                    dt_,
                                                    gamma_,
                                                    beta_,
                                                    real_type,
                                                    values,
                                                    stream);
            });
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return iterate([&](const OpDomain &domain) {
                auto          domain_op = std::static_pointer_cast<GPUKelvinVoigtNewmarkOpData>(domain.user_data);
                const auto    params    = domain.parameters;
                const real_t  k         = params->require_real_value("k");
                const real_t  K         = params->require_real_value("K");
                const real_t  eta       = params->require_real_value("eta");
                const real_t  rho       = params->require_real_value("rho");
                const real_t  dt_       = params->require_real_value("dt");
                const real_t  gamma_    = params->require_real_value("gamma");
                const real_t  beta_     = params->require_real_value("beta");
                const real_t *v         = vel_[0]->data();
                const real_t *a         = acc_[0]->data();

                SFEM_TRACE_SCOPE("cu_kelvin_voigt_newmark_apply");
                return cu_kelvin_voigt_newmark_apply(domain.element_type,
                                                     domain_op->nelements(),
                                                     domain_op->elements->data(),
                                                     domain_op->nelements(),
                                                     domain_op->jacobian_adjugate->data(),
                                                     domain_op->jacobian_determinant->data(),
                                                     k,
                                                     K,
                                                     eta,
                                                     rho,
                                                     dt_,
                                                     gamma_,
                                                     beta_,
                                                     real_type,
                                                     x,
                                                     v,
                                                     a,
                                                     out,
                                                     stream);
            });
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            const ptrdiff_t ndofs = space->n_dofs();
            return iterate([&](const OpDomain &domain) {
                auto         domain_op = std::static_pointer_cast<GPUKelvinVoigtNewmarkOpData>(domain.user_data);
                const auto   params    = domain.parameters;
                const real_t k         = params->require_real_value("k");
                const real_t K         = params->require_real_value("K");
                const real_t eta       = params->require_real_value("eta");
                const real_t rho       = params->require_real_value("rho");
                const real_t dt_       = params->require_real_value("dt");
                const real_t gamma_    = params->require_real_value("gamma");
                const real_t beta_     = params->require_real_value("beta");

                const real_t v_scale = (dt_ != 0 && beta_ != 0) ? (gamma_ / (beta_ * dt_)) : 0.0;
                const real_t a_scale = (dt_ != 0 && beta_ != 0) ? (1.0 / (beta_ * dt_ * dt_)) : 0.0;

                auto v_lin_tmp = smesh::create_device_buffer<real_t>(ndofs);
                auto a_lin_tmp = smesh::create_device_buffer<real_t>(ndofs);
                d_copy(ndofs, h, v_lin_tmp->data());
                d_scal(ndofs, v_scale, v_lin_tmp->data());
                d_copy(ndofs, h, a_lin_tmp->data());
                d_scal(ndofs, a_scale, a_lin_tmp->data());

                SFEM_TRACE_SCOPE("cu_kelvin_voigt_newmark_apply");
                return cu_kelvin_voigt_newmark_apply(domain.element_type,
                                                     domain_op->nelements(),
                                                     domain_op->elements->data(),
                                                     domain_op->nelements(),
                                                     domain_op->jacobian_adjugate->data(),
                                                     domain_op->jacobian_determinant->data(),
                                                     k,
                                                     K,
                                                     eta,
                                                     rho,
                                                     dt_,
                                                     gamma_,
                                                     beta_,
                                                     real_type,
                                                     h,
                                                     v_lin_tmp->data(),
                                                     a_lin_tmp->data(),
                                                     out,
                                                     stream);
            });
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        std::shared_ptr<Op> clone() const override {
            auto ret       = std::make_shared<GPUKelvinVoigtNewmark>(space);
            ret->domains   = domains;
            ret->real_type = real_type;
            ret->stream    = stream;
            ret->dt        = dt;
            ret->gamma     = gamma;
            ret->beta      = beta;
            for (int c = 0; c < 3; c++) {
                ret->vel_[c] = vel_[c];
                ret->acc_[c] = acc_[c];
            }
            return ret;
        }

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override {
            if (domains) {
                domains->set_value_in_block(block_name, var_name, value);
            }
        }

        void override_element_types(const std::vector<smesh::ElemType> &element_types) override {
            if (domains) {
                domains->override_element_types(element_types);
            }
        }
    };

    class GPUEMOp : public Op {
    public:
        std::shared_ptr<FunctionSpace>    space;
        enum smesh::PrimitiveType         real_type{smesh::SMESH_DEFAULT};
        std::shared_ptr<Buffer<idx_t *>>  elements;
        std::shared_ptr<Buffer<real_t *>> element_matrix;

        GPUEMOp(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("GPUEMOp::create");
            auto ret = std::make_unique<GPUEMOp>(space);
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_ERROR("IMPLEMENT ME!\n");
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_ERROR("IMPLEMENT ME!\n");
            return nullptr;
        }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            auto mesh             = space->mesh_ptr();
            auto h_element_matrix = sfem::create_host_buffer<real_t>(mesh->n_elements() * 64);

            int err = 0;
            if (space->has_semi_structured_mesh()) {
                auto &ssm = space->mesh();
                err       = sshex8_laplacian_element_matrix(smesh::semistructured_level(ssm),
                                                      mesh->n_elements(),
                                                      mesh->n_nodes(),
                                                      mesh->elements(0)->data(),
                                                      mesh->points()->data(),
                                                      h_element_matrix->data());
            } else {
                err = sshex8_laplacian_element_matrix(1,
                                                      mesh->n_elements(),
                                                      mesh->n_nodes(),
                                                      mesh->elements(0)->data(),
                                                      mesh->points()->data(),
                                                      h_element_matrix->data());
            }

            auto soa       = sfem::aos_to_soa(64, mesh->n_elements(), 1, 64, h_element_matrix);
            element_matrix = smesh::to_device(soa);

            elements = create_device_elements(space, space->element_type());
            return err;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("GPUEMOp::apply");

            int err = 0;
            if (space->has_semi_structured_mesh()) {
                auto     &ssm   = space->mesh();
                const int level = smesh::semistructured_level(ssm);
                err             = cu_affine_sshex8_elemental_matrix_apply(level,
                                                              ssm.n_elements(),
                                                              elements->data(),
                                                              real_type,
                                                              (void **)element_matrix->data(),
                                                              h,
                                                              out,
                                                              SFEM_DEFAULT_STREAM);
            } else {
                err = cu_affine_hex8_elemental_matrix_apply(space->mesh_ptr()->n_elements(),
                                                            elements->data(),
                                                            real_type,
                                                            (void **)element_matrix->data(),
                                                            h,
                                                            out,
                                                            SFEM_DEFAULT_STREAM);
            }

            return err;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] GPUEMOp::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_ERROR("[Error] GPUEMOp::gradient NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override { return apply(nullptr, x, out); }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] GPUEMOp::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        const char *name() const override { return "ss::gpu::EMOp"; }
        inline bool is_linear() const override { return true; }
        ptrdiff_t   n_dofs_domain() const override { return space->n_dofs(); }
        ptrdiff_t   n_dofs_image() const override { return space->n_dofs(); }
    };

    class GPUEMWarpOp : public Op {
    public:
        std::shared_ptr<FunctionSpace>  space;
        enum smesh::PrimitiveType       real_type{smesh::SMESH_DEFAULT};
        std::shared_ptr<Buffer<idx_t>>  elements;
        std::shared_ptr<Buffer<real_t>> element_matrix;
        bool                            cartesian_ordering{true};

        GPUEMWarpOp(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("GPUEMWarpOp::create");
            auto ret = std::make_unique<GPUEMWarpOp>(space);
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_ERROR("IMPLEMENT ME!\n");
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_ERROR("IMPLEMENT ME!\n");
            return nullptr;
        }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            auto mesh             = space->mesh_ptr();
            auto h_element_matrix = sfem::create_host_buffer<real_t>(mesh->n_elements() * 64);

            int err = 0;
            if (space->has_semi_structured_mesh()) {
                auto &ssm = space->mesh();
                if (cartesian_ordering) {
                    err = sshex8_laplacian_element_matrix_cartesian(smesh::semistructured_level(ssm),
                                                                    mesh->n_elements(),
                                                                    mesh->n_nodes(),
                                                                    mesh->elements(0)->data(),
                                                                    mesh->points()->data(),
                                                                    h_element_matrix->data());
                } else {
                    err = sshex8_laplacian_element_matrix(smesh::semistructured_level(ssm),
                                                          mesh->n_elements(),
                                                          mesh->n_nodes(),
                                                          mesh->elements(0)->data(),
                                                          mesh->points()->data(),
                                                          h_element_matrix->data());
                }

            } else {
                SFEM_ERROR("Only works with SSMesh!\n");
            }

            elements       = create_device_elements_AoS(space, space->element_type());
            element_matrix = smesh::to_device(h_element_matrix);

            // h_element_matrix->print(std::cout);
            return err;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("GPUEMWarpOp::apply");

            int err = 0;
            if (space->has_semi_structured_mesh()) {
                auto     &ssm   = space->mesh();
                const int level = smesh::semistructured_level(ssm);
                err             = cu_affine_sshex8_elemental_matrix_apply_AoS(level,
                                                                  ssm.n_elements(),
                                                                  elements->data(),
                                                                  real_type,
                                                                  (void *)element_matrix->data(),
                                                                  h,
                                                                  out,
                                                                  SFEM_DEFAULT_STREAM);
            } else {
                SFEM_ERROR("Only works with SSMesh!\n");
            }

            return err;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] GPUEMWarpOp::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_ERROR("[Error] GPUEMWarpOp::gradient NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override { return apply(nullptr, x, out); }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] GPUEMWarpOp::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }

        const char *name() const override { return "ss::gpu::EMWarpOp"; }
        inline bool is_linear() const override { return true; }
        ptrdiff_t   n_dofs_domain() const override { return space->n_dofs(); }
        ptrdiff_t   n_dofs_image() const override { return space->n_dofs(); }
    };

    void register_device_ops() {
        Factory::register_op("gpu:LinearElasticity", &GPULinearElasticity::create);
        Factory::register_op("gpu:Laplacian", &GPULaplacian::create);
        Factory::register_op("ss:gpu:Laplacian", &GPULaplacian::create);
        Factory::register_op("ss:gpu:LinearElasticity", &GPULinearElasticity::create);
        Factory::register_op("ss:gpu:EMOp", &GPUEMOp::create);
        Factory::register_op("ss:gpu:EMWarpOp", &GPUEMWarpOp::create);
        Factory::register_op("gpu:EMOp", &GPUEMOp::create);
        Factory::register_op("gpu:KelvinVoigtNewmark", &GPUKelvinVoigtNewmark::create);
        Factory::register_op("ss:gpu:KelvinVoigtNewmark", &GPUKelvinVoigtNewmark::create);
    }

}  // namespace sfem
