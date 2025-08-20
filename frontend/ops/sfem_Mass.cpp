#include "sfem_Mass.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "mass.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"

#include <functional>

namespace sfem {

    class Mass::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;
        enum ElemType                  element_type { INVALID };  ///< Element type
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
            element_type = (enum ElemType)space->element_type();
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "Mass::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

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

    int Mass::hessian_crs(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values) {
        SFEM_TRACE_SCOPE("Mass::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
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

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            apply_mass(domain.element_type,
                       domain.block->n_elements(),
                       mesh->n_nodes(),
                       domain.block->elements()->data(),
                       mesh->points()->data(),
                       1,
                       x,
                       1,
                       out);
            //    FIXME
            return SFEM_SUCCESS;
        });
    }

    int Mass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Mass::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            apply_mass(domain.element_type,
                       domain.block->n_elements(),
                       mesh->n_nodes(),
                       domain.block->elements()->data(),
                       mesh->points()->data(),
                       1,
                       h,
                       1,
                       out);
            //    FIXME
            return SFEM_SUCCESS;
        });
    }

    int Mass::value(const real_t *x, real_t *const out) {
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

    void Mass::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem
