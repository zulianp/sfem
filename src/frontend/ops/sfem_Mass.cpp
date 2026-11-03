#include "sfem_Mass.hpp"


#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_mesh.hpp"

#include "mass.hpp"
#include "sfem_team.h"


#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"

#include <functional>

namespace sfem {

    class Mass::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;
        smesh::ElemType                  element_type { smesh::INVALID };  ///< Element type
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
            element_type = (smesh::ElemType)space->element_type();
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "Mass::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }

        /// The mesh's points and their number, taken by team_prepare for the team product.
        geom_t  **points{nullptr};
        ptrdiff_t n_nodes{0};
    };

    namespace {

        /// out += M h over the elements of `range` in the domain's block: by the threads of the
        /// caller's region (TEAM) or opening its own.
        template <bool TEAM>
        int mass_apply_domain(const OpDomain     &domain,
                              const ptrdiff_t     n_nodes,
                              geom_t **const      points,
                              const real_t *const h,
                              real_t *const       out,
                              const ElementRange  range) {
            if (range.empty()) return SFEM_SUCCESS;

            const int nxe = smesh::elem_num_nodes(domain.element_type);
            if (nxe > 8) {
                SFEM_ERROR("Mass: %d nodes per element, at most 8 supported\n", nxe);
                return SFEM_FAILURE;
            }

            idx_t **const elems = domain.block->elements()->data();
            idx_t        *view[8];
            for (int v = 0; v < nxe; ++v) {
                view[v] = elems[v] + range.begin;
            }

            return SFEM_TEAM_KERNEL(TEAM, apply_mass)(
                    domain.element_type, range.size(), n_nodes, view, points, 1, h, 1, out);
        }

    }  // namespace

    inline ptrdiff_t Mass::n_dofs_domain() const { return impl_->space->n_dofs(); }

    inline ptrdiff_t Mass::n_dofs_image() const { return impl_->space->n_dofs(); }

    int Mass::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Mass::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Mass::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Mass::create");

        assert(1 == space->block_size());

        auto ret = std::make_unique<Mass>(space);
        return ret;
    }

    Mass::Mass(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    Mass::~Mass() = default;

    int Mass::hessian_crs(const real_t *const /*x*/, const count_t *const /*rowptr*/, const idx_t *const /*colidx*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Mass::hessian_crs");

        auto mesh  = impl_->space->mesh();
        auto graph = impl_->space->dof_to_dof_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            assemble_mass(domain.element_type,
                          domain.block->n_elements(),
                          mesh->n_nodes(),
                          domain.block->elements()->data(),
                          mesh->points()->data(),
                          graph->rowptr()->data(),
                          graph->colidx()->data(),
                          values);
            return SFEM_SUCCESS;
        });

        return err;
    }

    int Mass::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Mass::gradient");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) {
            return mass_apply_domain<false>(
                    domain, mesh->n_nodes(), mesh->points()->data(), x, out, element_range(*mesh, *domain.block, ElementScope::ALL));
        });
    }

    int Mass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Mass::apply");
        SFEM_OP_CAPTURE();
        if (!team_supported()) return apply(x, h, out, ElementScope::ALL);
        return op_team_region(*this, impl_->space->n_dofs(), [&](const Team &team) {
            return team_apply(team, x, h, out, ElementScope::ALL);
        });
    }

    int Mass::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out, const ElementScope scope) {
        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) {
            return mass_apply_domain<false>(
                    domain, mesh->n_nodes(), mesh->points()->data(), h, out, element_range(*mesh, *domain.block, scope));
        });
    }

    bool Mass::team_supported() const {
        for (const auto &kv : impl_->domains->domains()) {
            if (!apply_mass_supported(kv.second.element_type)) return false;
        }
        return true;
    }

    int Mass::team_apply(const Team &team,
                         const real_t *const /*x*/,
                         const real_t *const h,
                         real_t *const       out,
                         const ElementScope  scope) {
        (void)team;
        smesh::Mesh &mesh = *impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) {
            return mass_apply_domain<true>(domain, impl_->n_nodes, impl_->points, h, out, element_range(mesh, *domain.block, scope));
        });
    }

    int Mass::team_prepare() {
        impl_->points  = impl_->space->mesh()->points()->data();
        impl_->n_nodes = impl_->space->mesh()->n_nodes();
        return SFEM_SUCCESS;
    }

    int Mass::value(const real_t * /*x*/, real_t *const /*out*/) {
        assert(0);
        return SFEM_FAILURE;
    }

    int Mass::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Mass::clone() const {
        SFEM_ERROR("Not implemented");
        return nullptr;
    }

    void Mass::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Mass::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem

