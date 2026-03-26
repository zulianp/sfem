#include "sfem_LumpedMass.hpp"

#include "mass.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_mesh.hpp"

#include <cassert>
#include <cstdio>
#include <functional>

namespace sfem {

    class LumpedMass::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {}

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    inline ptrdiff_t LumpedMass::n_dofs_domain() const { return impl_->space->n_dofs(); }

    inline ptrdiff_t LumpedMass::n_dofs_image() const { return impl_->space->n_dofs(); }

    int LumpedMass::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("LumpedMass::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> LumpedMass::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LumpedMass::create");
        return std::make_unique<LumpedMass>(space);
    }

    std::shared_ptr<Op> LumpedMass::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            fprintf(stderr, "[Error] LumpedMass::lor_op NOT IMPLEMENTED for semi-structured mesh!\n");
            assert(false);
            return nullptr;
        }
        auto ret            = std::make_shared<LumpedMass>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> LumpedMass::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LumpedMass::derefine_op");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<LumpedMass>(space);
            ret->initialize({});
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<LumpedMass>(space);
            ret->initialize({});
            assert(space->n_blocks() == 1);
            ret->override_element_types({space->element_type()});
            return ret;
        }

        auto ret            = std::make_shared<LumpedMass>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    LumpedMass::LumpedMass(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    LumpedMass::~LumpedMass() = default;

    int LumpedMass::hessian_crs(const real_t *const  x,
                                const count_t *const rowptr,
                                const idx_t *const   colidx,
                                real_t *const        values) {
        assert(0);
        (void)x;
        (void)rowptr;
        (void)colidx;
        (void)values;
        return SFEM_FAILURE;
    }

    int LumpedMass::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("LumpedMass::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();

        if (impl_->space->block_size() == 1) {
            return impl_->iterate([&](const OpDomain &domain) {
                assemble_lumped_mass(static_cast<int>(domain.element_type),
                                     domain.block->n_elements(),
                                     mesh->n_nodes(),
                                     domain.block->elements()->data(),
                                     mesh->points()->data(),
                                     values);
                return SFEM_SUCCESS;
            });
        }

        const ptrdiff_t n_nodes = mesh->n_nodes();
        real_t         *temp    = (real_t *)calloc((size_t)n_nodes, sizeof(real_t));
        if (!temp) {
            return SFEM_FAILURE;
        }

        const int err = impl_->iterate([&](const OpDomain &domain) {
            assemble_lumped_mass(static_cast<int>(domain.element_type),
                                 domain.block->n_elements(),
                                 n_nodes,
                                 domain.block->elements()->data(),
                                 mesh->points()->data(),
                                 temp);
            return SFEM_SUCCESS;
        });

        if (err != SFEM_SUCCESS) {
            free(temp);
            return err;
        }

        const int bs = impl_->space->block_size();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            for (int b = 0; b < bs; b++) {
                values[i * bs + b] += temp[i];
            }
        }

        free(temp);
        return SFEM_SUCCESS;
    }

    int LumpedMass::gradient(const real_t *const x, real_t *const out) {
        assert(0);
        (void)x;
        (void)out;
        return SFEM_FAILURE;
    }

    int LumpedMass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        assert(0);
        (void)x;
        (void)h;
        (void)out;
        return SFEM_FAILURE;
    }

    int LumpedMass::value(const real_t *x, real_t *const out) {
        assert(0);
        (void)x;
        (void)out;
        return SFEM_FAILURE;
    }

    int LumpedMass::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> LumpedMass::clone() const {
        auto ret            = std::make_shared<LumpedMass>(impl_->space);
        ret->impl_->domains = impl_->domains;
        return ret;
    }

    void LumpedMass::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void LumpedMass::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void LumpedMass::set_option(const std::string & /*name*/, bool /*val*/) {}

}  // namespace sfem

