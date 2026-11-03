#include "sfem_VectorLaplacian.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_glob.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"
#include "vector_laplacian.hpp"
#include "laplacian.hpp"
#include "sfem_team.h"

#include <map>
#include <vector>

namespace sfem {

    namespace {

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); i++) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("VectorLaplacian: mesh block pointer not found in mesh.blocks()");
            return 0;
        }

        /// out += A h over the elements [begin, begin + ne) of the domain, whose columns start at
        /// `elements`: by the threads of the caller's region (TEAM) or opening its own.
        template <bool TEAM>
        int vector_laplacian_dispatch_apply(const OpDomain &domain,
                                            smesh::Mesh    &mesh,
                                            const int       block_size,
                                            idx_t **const   elements,
                                            const ptrdiff_t begin,
                                            const ptrdiff_t ne,
                                            real_t **const SFEM_RESTRICT vec_in,
                                            real_t **const SFEM_RESTRICT vec_out) {
            if (domain.user_data) {
                auto                fff      = std::static_pointer_cast<smesh::FFF>(domain.user_data);
                constexpr ptrdiff_t fff_size = 6;
                return SFEM_TEAM_KERNEL(TEAM, vector_laplacian_apply_opt)(domain.element_type,
                                                                          ne,
                                                                          elements,
                                                                          fff->fff_AoS()->data() + begin * fff_size,
                                                                          block_size,
                                                                          block_size,
                                                                          vec_in,
                                                                          vec_out);
            }

            return SFEM_TEAM_KERNEL(TEAM, vector_laplacian_apply)(domain.element_type,
                                                                  ne,
                                                                  mesh.n_nodes(),
                                                                  elements,
                                                                  mesh.points()->data(),
                                                                  block_size,
                                                                  block_size,
                                                                  vec_in,
                                                                  vec_out);
        }

    }  // namespace

    class VectorLaplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {
#if SFEM_PRINT_THROUGHPUT
            const std::string op_name =
                    std::string("VectorLaplacian[") + sfem::type_to_string(sp->element_type()) + "]::apply";
            op_profiler = std::make_unique<OpTracer>(space, op_name);
#endif
        }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }

        // Each block's element columns from its first SHARED_AND_AURA element on, made outside any
        // region on a distributed mesh (prepare_interface): a scoped product builds no view.
        struct InterfaceColumns {
            ptrdiff_t            begin{0};
            std::vector<idx_t *> columns;
        };
        std::map<const smesh::Mesh::Block *, InterfaceColumns> interface;

        void prepare_interface() {
            auto mesh = space->mesh();
            if (!mesh_is_distributed(*mesh)) return;
            for (const auto &kv : domains->domains()) {
                const auto *block = kv.second.block.get();
                if (interface.count(block)) continue;
                InterfaceColumns ic;
                ic.begin         = element_range(*mesh, *block, ElementScope::SHARED_AND_AURA).begin;
                const int     nxe = block->n_nodes_per_element();
                idx_t **const own = block->elements()->data();
                ic.columns.resize(nxe);
                for (int v = 0; v < nxe; ++v) ic.columns[v] = own[v] + ic.begin;
                interface[block] = std::move(ic);
            }
        }

        /// out += A h over the domain's elements of `scope`.
        template <bool TEAM>
        int apply_domain(const OpDomain &domain, const real_t *const h, real_t *const out, const ElementScope scope) {
            auto               mesh  = space->mesh();
            const ElementRange range = element_range(*mesh, *domain.block, scope);
            if (range.empty()) return SFEM_SUCCESS;

            idx_t **elements = domain.block->elements()->data();
            if (range.begin != 0) {
                auto it = interface.find(domain.block.get());
                if (it == interface.end() || it->second.begin != range.begin) {
                    SFEM_ERROR("VectorLaplacian: no element columns from element %ld\n", (long)range.begin);
                    return SFEM_FAILURE;
                }
                elements = it->second.columns.data();
            }

            constexpr int max_components = 8;
            const int     block_size     = space->block_size();
            if (block_size > max_components) {
                SFEM_ERROR("VectorLaplacian: %d components, at most %d supported\n", block_size, max_components);
                return SFEM_FAILURE;
            }
            real_t *vec_in[max_components], *vec_out[max_components];
            for (int d = 0; d < block_size; d++) {
                vec_in[d]  = const_cast<real_t *>(&h[d]);
                vec_out[d] = &out[d];
            }
            return vector_laplacian_dispatch_apply<TEAM>(
                    domain, *mesh, block_size, elements, range.begin, range.size(), vec_in, vec_out);
        }
    };

    inline ptrdiff_t VectorLaplacian::n_dofs_domain() const { return impl_->space->n_dofs(); }

    inline ptrdiff_t VectorLaplacian::n_dofs_image() const { return impl_->space->n_dofs(); }

    int VectorLaplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("VectorLaplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        if (impl_->domains->require_supported_types("VectorLaplacian", laplacian_has_kernel) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        auto mesh = impl_->space->mesh();

        int SFEM_VECTOR_LAPLACIAN_FFF = 1;
        SFEM_READ_ENV(SFEM_VECTOR_LAPLACIAN_FFF, atoi);

        for (auto &n2d : impl_->domains->domains()) {
            OpDomain &domain = n2d.second;

            const bool want_fff =
                    SFEM_VECTOR_LAPLACIAN_FFF || is_semistructured_type(domain.element_type);

            if (want_fff) {
                const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
                auto                     fff      = smesh::FFF::create_AoS(mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!fff) {
                    SFEM_ERROR("VectorLaplacian: smesh::FFF::create_AoS failed (block '%s')\n",
                               domain.block->name().c_str());
                    return SFEM_FAILURE;
                }
                domain.user_data = std::static_pointer_cast<void>(fff);
            } else {
                domain.user_data = nullptr;
            }
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> VectorLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::create");

        assert(1 != space->block_size());
        return std::make_unique<VectorLaplacian>(space);
    }

    std::shared_ptr<Op> VectorLaplacian::lor(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::lor");

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            SMESH_ERROR("VectorLaplacian::lor NOT IMPLEMENTED for semi-structured mesh!\n");
            return nullptr;
        }

        auto ret              = std::make_shared<VectorLaplacian>(space);
        ret->impl_->domains = impl_->domains->lor(space, {});
        return ret;
    }

    std::shared_ptr<Op> VectorLaplacian::derefine(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::derefine");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<VectorLaplacian>(space);
            ret->initialize({});
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<VectorLaplacian>(space);
            ret->initialize({});
            return ret;
        }

        auto ret              = std::make_shared<VectorLaplacian>(space);
        ret->impl_->domains = impl_->domains->derefine(space, {});
        return ret;
    }

    VectorLaplacian::VectorLaplacian(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    VectorLaplacian::~VectorLaplacian() = default;

    int VectorLaplacian::hessian_crs(const real_t *const  /*x*/,
                                     const count_t *const /*rowptr*/,
                                     const idx_t *const   /*colidx*/,
                                     real_t *const        /*values*/) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_crs_sym(const real_t *const  /*x*/,
                                        const count_t *const /*rowptr*/,
                                        const idx_t *const   /*colidx*/,
                                        real_t *const        /*diag_values*/,
                                        real_t *const        /*off_diag_values*/) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs_sym");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_diag");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::gradient(const real_t *const /*x*/, real_t *const /*out*/) {
        SFEM_TRACE_SCOPE("VectorLaplacian::gradient");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::apply");
        SFEM_OP_CAPTURE();
        if (!team_supported()) return apply(x, h, out, ElementScope::ALL);
        return op_team_region(*this, impl_->space->n_dofs(), [&](const Team &team) {
            return team_apply(team, x, h, out, ElementScope::ALL);
        });
    }

    int VectorLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out, const ElementScope scope) {
        impl_->prepare_interface();
        return impl_->iterate([&](const OpDomain &domain) { return impl_->apply_domain<false>(domain, h, out, scope); });
    }

    // The semi-structured kernel has no team form (vector_laplacian_apply_opt).
    bool VectorLaplacian::team_supported() const {
        for (const auto &kv : impl_->domains->domains()) {
            if (is_semistructured_type(kv.second.element_type)) return false;
        }
        return true;
    }

    int VectorLaplacian::team_prepare() {
        impl_->prepare_interface();
        return SFEM_SUCCESS;
    }

    int VectorLaplacian::team_apply(const Team &team,
                                    const real_t *const /*x*/,
                                    const real_t *const h,
                                    real_t *const       out,
                                    const ElementScope  scope) {
        (void)team;
        return impl_->iterate([&](const OpDomain &domain) { return impl_->apply_domain<true>(domain, h, out, scope); });
    }

    int VectorLaplacian::value(const real_t * /*x*/, real_t *const /*out*/) {
        SFEM_TRACE_SCOPE("VectorLaplacian::value");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> VectorLaplacian::clone() const {
        auto ret              = std::make_shared<VectorLaplacian>(impl_->space);
        ret->impl_->domains = impl_->domains;
        return ret;
    }

    void VectorLaplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void VectorLaplacian::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem
