#include "sfem_LinearElasticity.hpp"

#include "linear_elasticity.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_glob.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"
#include "smesh_spaces.hpp"

#include <assert.h>
#include <functional>

namespace sfem {

    namespace {

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); i++) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("LinearElasticity: mesh block pointer not found in mesh.blocks()");
            return 0;
        }

        bool domain_supports_adjugate_cache(const smesh::ElemType et) {
            return et == smesh::HEX8 || sfem::is_semistructured_type(et);
        }

        int linear_elasticity_dispatch_domain_vector(const OpDomain     &domain,
                                                     smesh::Mesh        &mesh,
                                                     const real_t        mu,
                                                     const real_t        lambda,
                                                     const real_t *const h,
                                                     real_t *const       out) {
            auto block        = domain.block;
            auto element_type = domain.element_type;
            if (domain.user_data) {
                auto jac = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(domain.user_data);
                SFEM_TRACE_SCOPE("linear_elasticity_apply_adjugate_aos");
                return linear_elasticity_apply_adjugate_aos(element_type,
                                                            block->n_elements(),
                                                            mesh.n_nodes(),
                                                            block->elements()->data(),
                                                            mesh.points()->data(),
                                                            jac->jacobian_adjugate_AoS()->data(),
                                                            jac->jacobian_determinant()->data(),
                                                            mu,
                                                            lambda,
                                                            h,
                                                            out);
            }
            return linear_elasticity_apply_aos(element_type,
                                               block->n_elements(),
                                               mesh.n_nodes(),
                                               block->elements()->data(),
                                               mesh.points()->data(),
                                               mu,
                                               lambda,
                                               h,
                                               out);
        }

    }  // namespace

    class LinearElasticity::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;

        real_t mu{1};      ///< Shear modulus (second Lamé parameter)
        real_t lambda{1};  ///< First Lamé parameter
        bool   use_affine_approximation{true};

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

    ptrdiff_t LinearElasticity::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t LinearElasticity::n_dofs_image() const { return impl_->space->n_dofs(); }

    int LinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("LinearElasticity::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        auto mesh = impl_->space->mesh_ptr();

        for (auto &n2d : impl_->domains->domains()) {
            OpDomain &domain = n2d.second;
            if (!domain_supports_adjugate_cache(domain.element_type)) {
                continue;
            }

            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
            auto jac = smesh::JacobianAdjugateAndDeterminant::create_AoS(mesh, smesh::MEMORY_SPACE_HOST, block_id);
            if (!jac) {
                return SFEM_FAILURE;
            }

            domain.user_data = std::static_pointer_cast<void>(jac);
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> LinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::create");

        auto ret = std::make_unique<LinearElasticity>(space);

        real_t SFEM_SHEAR_MODULUS          = 1;
        real_t SFEM_FIRST_LAME_PARAMETER   = 1;
        int    SFEM_HEX8_ASSUME_AFFINE     = ret->impl_->use_affine_approximation ? 1 : 0;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

        ret->impl_->mu                       = SFEM_SHEAR_MODULUS;
        ret->impl_->lambda                   = SFEM_FIRST_LAME_PARAMETER;
        ret->impl_->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::lor_op");

        if (impl_->space->has_semi_structured_mesh() && sfem::is_semistructured_type(impl_->space->element_type())) {
            fprintf(stderr, "[Error] LinearElasticity::lor_op NOT IMPLEMENTED for semi-structured mesh!\n");
            assert(false);
            return nullptr;
        }

        // FIXME: Must work for all element types and multi-block

        auto ret            = std::make_shared<LinearElasticity>(space);
        ret->impl_->domains                  = impl_->domains->lor_op(space, {});
        ret->impl_->mu                       = impl_->mu;
        ret->impl_->lambda                   = impl_->lambda;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::derefine_op");

        if (space->has_semi_structured_mesh() && sfem::is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->set_mu(impl_->mu);
            ret->set_lambda(impl_->lambda);
            ret->set_option("ASSUME_AFFINE", impl_->use_affine_approximation);
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && sfem::is_semistructured_type(impl_->space->element_type()) &&
            !sfem::is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->set_mu(impl_->mu);
            ret->set_lambda(impl_->lambda);
            ret->set_option("ASSUME_AFFINE", impl_->use_affine_approximation);
            assert(space->n_blocks() == 1);
            ret->override_element_types({space->element_type()});
            return ret;
        }

        // FIXME: Must work for all element types and multi-block

        auto ret            = std::make_shared<LinearElasticity>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        ret->impl_->mu      = impl_->mu;
        ret->impl_->lambda  = impl_->lambda;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
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
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
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
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
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
        auto graph = impl_->space->mesh_ptr()->node_to_node_graph_upper_triangular();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
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
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
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
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
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
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            return linear_elasticity_assemble_diag_aos(element_type,
                                                       block->n_elements(),
                                                       mesh->n_nodes(),
                                                       block->elements()->data(),
                                                       mesh->points()->data(),
                                                       mu,
                                                       lambda,
                                                       out);
        });

        return err;
    }

    int LinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            auto lambda = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu     = domain.parameters->get_real_value("mu", impl_->mu);
            return linear_elasticity_dispatch_domain_vector(domain, *mesh, mu, lambda, x, out);
        });
    }

    int LinearElasticity::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            auto lambda = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu     = domain.parameters->get_real_value("mu", impl_->mu);
            return linear_elasticity_dispatch_domain_vector(domain, *mesh, mu, lambda, h, out);
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
        auto ret = std::make_shared<LinearElasticity>(impl_->space);
        ret->impl_->domains                  = impl_->domains;
        ret->impl_->mu                       = impl_->mu;
        ret->impl_->lambda                   = impl_->lambda;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    void LinearElasticity::set_option(const std::string &name, bool val) {
        if (name == "ASSUME_AFFINE") {
            impl_->use_affine_approximation = val;
        }
    }

    void LinearElasticity::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void LinearElasticity::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    // TODO: remove these and use the block version where needed
    real_t LinearElasticity::get_mu() const { return impl_->mu; }
    void   LinearElasticity::set_mu(real_t val) { impl_->mu = val; }
    real_t LinearElasticity::get_lambda() const { return impl_->lambda; }
    void   LinearElasticity::set_lambda(real_t val) { impl_->lambda = val; }
}  // namespace sfem
