#include "sfem_LinearElasticity.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "hex8_jacobian.h"
#include "linear_elasticity.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"

#include <functional>

namespace sfem {

    /**
     * @brief Jacobian storage for performance optimization
     *
     * Precomputed Jacobian determinants and adjugates for HEX8 elements
     * to avoid repeated computation during matrix-vector products.
     */
    class Jacobians {
    public:
        std::shared_ptr<Buffer<jacobian_t>> adjugate;     ///< Adjugate matrices
        std::shared_ptr<Buffer<jacobian_t>> determinant;  ///< Determinants

        /**
         * @brief Constructor
         * @param n_elements Number of elements
         * @param size_adjugate Size of adjugate storage per element
         */
        Jacobians(const ptrdiff_t n_elements, const int size_adjugate)
            : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
              determinant(sfem::create_host_buffer<jacobian_t>(n_elements)) {}
    };

    class LinearElasticity::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;

        real_t mu{1};      ///< Shear modulus (second Lamé parameter)
        real_t lambda{1};  ///< First Lamé parameter

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "LinearElasticity::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    int LinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("LinearElasticity::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        // FIXME: Must work for all element types
        if (impl_->space->element_type() == HEX8) {
            auto mesh = impl_->space->mesh_ptr();
            int dim = mesh->spatial_dimension();

            for (auto &domain : impl_->domains->domains()) {
                auto block     = domain.second.block;
                auto jacobians = std::make_shared<Jacobians>(block->n_elements(), dim * dim);
                hex8_adjugate_and_det_fill(block->n_elements(),
                                           block->elements()->data(),
                                           mesh->points()->data(),
                                           jacobians->adjugate->data(),
                                           jacobians->determinant->data());

                domain.second.user_data = std::static_pointer_cast<void>(jacobians);
            }
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> LinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::create");

        auto ret = std::make_unique<LinearElasticity>(space);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

        ret->impl_->mu           = SFEM_SHEAR_MODULUS;
        ret->impl_->lambda       = SFEM_FIRST_LAME_PARAMETER;

        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::lor_op");

        auto ret                 = std::make_shared<LinearElasticity>(space);
        ret->impl_->domains      = impl_->domains->lor_op(space, {});
        ret->impl_->mu           = impl_->mu;
        ret->impl_->lambda       = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::derefine_op");

        auto ret                 = std::make_shared<LinearElasticity>(space);
        ret->impl_->domains      = impl_->domains->derefine_op(space, {});
        ret->impl_->mu           = impl_->mu;
        ret->impl_->lambda       = impl_->lambda;
        return ret;
    }

    LinearElasticity::LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {
        // Initialize with empty block names to include all blocks
        initialize({});
    }

    LinearElasticity::~LinearElasticity() = default;

    int LinearElasticity::hessian_crs(const real_t *const  x,
                                      const count_t *const rowptr,
                                      const idx_t *const   colidx,
                                      real_t *const        values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_crs_aos(element_type,
                                             block->n_elements(),
                                             mesh->n_nodes(),
                                             block->elements()->data(),
                                             mesh->points()->data(),
                                             mu,
                                             lambda,
                                             graph->rowptr()->data(),
                                             graph->colidx()->data(),
                                             values);
        });

        return err;
    }

    int LinearElasticity::hessian_bsr(const real_t *const  x,
                                      const count_t *const rowptr,
                                      const idx_t *const   colidx,
                                      real_t *const        values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_bsr");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_bsr(element_type,
                                         block->n_elements(),
                                         mesh->n_nodes(),
                                         block->elements()->data(),
                                         mesh->points()->data(),
                                         mu,
                                         lambda,
                                         graph->rowptr()->data(),
                                         graph->colidx()->data(),
                                         values);
        });

        return err;
    }

    int LinearElasticity::hessian_bcrs_sym(const real_t *const  x,
                                           const count_t *const rowptr,
                                           const idx_t *const   colidx,
                                           const ptrdiff_t      block_stride,
                                           real_t **const       diag_values,
                                           real_t **const       off_diag_values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_bcrs_sym");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_bcrs_sym(element_type,
                                              block->n_elements(),
                                              mesh->n_nodes(),
                                              block->elements()->data(),
                                              mesh->points()->data(),
                                              mu,
                                              lambda,
                                              graph->rowptr()->data(),
                                              graph->colidx()->data(),
                                              block_stride,
                                              diag_values,
                                              off_diag_values);
        });

        return err;
    }

    int LinearElasticity::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_block_diag_sym_aos(element_type,
                                                        block->n_elements(),
                                                        mesh->n_nodes(),
                                                        block->elements()->data(),
                                                        mesh->points()->data(),
                                                        mu,
                                                        lambda,
                                                        values);
        });

        return err;
    }

    int LinearElasticity::hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym_soa");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_block_diag_sym_soa(element_type,
                                                        block->n_elements(),
                                                        mesh->n_nodes(),
                                                        block->elements()->data(),
                                                        mesh->points()->data(),
                                                        mu,
                                                        lambda,
                                                        values);
        });

        return err;
    }

    int LinearElasticity::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_assemble_diag_aos(element_type,
                                                       block->n_elements(),
                                                       mesh->n_nodes(),
                                                       block->elements()->data(),
                                                       mesh->points()->data(),
                                                       impl_->mu,
                                                       impl_->lambda,
                                                       out);
        });

        return err;
    }

    int LinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_assemble_gradient_aos(element_type,
                                                           block->n_elements(),
                                                           mesh->n_nodes(),
                                                           block->elements()->data(),
                                                           mesh->points()->data(),
                                                           mu,
                                                           lambda,
                                                           x,
                                                           out);
        });
    }

    int LinearElasticity::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            auto block    = domain.block;
            auto lambda   = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu       = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;
            
            if (domain.user_data) {
                auto jacobians = std::static_pointer_cast<Jacobians>(domain.user_data);
                SFEM_TRACE_SCOPE("linear_elasticity_apply_adjugate_aos");
                return linear_elasticity_apply_adjugate_aos(element_type,
                                                           block->n_elements(),
                                                           mesh->n_nodes(),
                                                           block->elements()->data(),
                                                           jacobians->adjugate->data(),
                                                           jacobians->determinant->data(),
                                                           mu,
                                                           lambda,
                                                           h,
                                                           out);
            } else {
                return linear_elasticity_apply_aos(element_type,
                                                  block->n_elements(),
                                                  mesh->n_nodes(),
                                                  block->elements()->data(),
                                                  mesh->points()->data(),
                                                  mu,
                                                  lambda,
                                                  h,
                                                  out);
            }
        });
    }

    int LinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            // return linear_elasticity_value_aos(domain.element_type,
            //                                   domain.block->n_elements(),
            //                                   mesh->n_nodes(),
            //                                   domain.block->elements()->data(),
            //                                   mesh->points()->data(),
            //                                   impl_->mu,
            //                                   impl_->lambda,
            //                                   x,
            //                                   out);
            SFEM_ERROR("LinearElasticity::value not implemented");
            return SFEM_FAILURE;
        });
    }

    int LinearElasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> LinearElasticity::clone() const {
        SFEM_ERROR("Not implemented");
        return nullptr;
    }

    void LinearElasticity::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void LinearElasticity::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    // Accessors for compatibility with semi-structured wrappers
    real_t        LinearElasticity::get_mu() const { return impl_->mu; }
    void          LinearElasticity::set_mu(real_t val) { impl_->mu = val; }
    real_t        LinearElasticity::get_lambda() const { return impl_->lambda; }
    void          LinearElasticity::set_lambda(real_t val) { impl_->lambda = val; }
}  // namespace sfem
