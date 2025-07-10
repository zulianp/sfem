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

namespace sfem {

    int LinearElasticity::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    std::unique_ptr<Op> LinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::create");

        auto mesh = space->mesh_ptr();
        assert(mesh->spatial_dimension() == space->block_size());

        auto ret = std::make_unique<LinearElasticity>(space);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

        ret->mu           = SFEM_SHEAR_MODULUS;
        ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
        ret->element_type = (enum ElemType)space->element_type();

        if (space->element_type() == HEX8) {
            ret->jacobians = std::make_shared<Jacobians>(mesh->n_elements(), 9);
            hex8_adjugate_and_det_fill(mesh->n_elements(),
                                       mesh->elements()->data(),
                                       mesh->points()->data(),
                                       ret->jacobians->adjugate->data(),
                                       ret->jacobians->determinant->data());
        }

        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::lor_op");

        auto ret          = std::make_shared<LinearElasticity>(space);
        ret->element_type = macro_type_variant(element_type);
        ret->mu           = mu;
        ret->lambda       = lambda;
        return ret;
    }

    std::shared_ptr<Op> LinearElasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LinearElasticity::derefine_op");

        auto ret          = std::make_shared<LinearElasticity>(space);
        ret->element_type = macro_base_elem(element_type);
        ret->mu           = mu;
        ret->lambda       = lambda;
        return ret;
    }

    LinearElasticity::LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    LinearElasticity::~LinearElasticity() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("LinearElasticity::apply called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    int LinearElasticity::hessian_crs(const real_t *const  x,
                                      const count_t *const rowptr,
                                      const idx_t *const   colidx,
                                      real_t *const        values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_crs");

        auto mesh  = space->mesh_ptr();
        auto graph = space->node_to_node_graph();

        return linear_elasticity_crs_aos(element_type,
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements()->data(),
                                         mesh->points()->data(),
                                         this->mu,
                                         this->lambda,
                                         graph->rowptr()->data(),
                                         graph->colidx()->data(),
                                         values);
    }

    int LinearElasticity::hessian_bsr(const real_t *const  x,
                                      const count_t *const rowptr,
                                      const idx_t *const   colidx,
                                      real_t *const        values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_bsr");

        auto mesh  = space->mesh_ptr();
        auto graph = space->node_to_node_graph();

        return linear_elasticity_bsr(element_type,
                                     mesh->n_elements(),
                                     mesh->n_nodes(),
                                     mesh->elements()->data(),
                                     mesh->points()->data(),
                                     this->mu,
                                     this->lambda,
                                     graph->rowptr()->data(),
                                     graph->colidx()->data(),
                                     values);
    }

    int LinearElasticity::hessian_bcrs_sym(const real_t *const  x,
                                           const count_t *const rowptr,
                                           const idx_t *const   colidx,
                                           const ptrdiff_t      block_stride,
                                           real_t **const       diag_values,
                                           real_t **const       off_diag_values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_bcrs_sym");

        auto mesh  = space->mesh_ptr();
        auto graph = space->node_to_node_graph();

        return linear_elasticity_bcrs_sym(element_type,
                                          mesh->n_elements(),
                                          mesh->n_nodes(),
                                          mesh->elements()->data(),
                                          mesh->points()->data(),
                                          this->mu,
                                          this->lambda,
                                          graph->rowptr()->data(),
                                          graph->colidx()->data(),
                                          block_stride,
                                          diag_values,
                                          off_diag_values);
    }

    int LinearElasticity::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym");

        auto mesh = space->mesh_ptr();

        return linear_elasticity_block_diag_sym_aos(element_type,
                                                    mesh->n_elements(),
                                                    mesh->n_nodes(),
                                                    mesh->elements()->data(),
                                                    mesh->points()->data(),
                                                    this->mu,
                                                    this->lambda,
                                                    values);
    }

    int LinearElasticity::hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym_soa");

        auto mesh = space->mesh_ptr();

        return linear_elasticity_block_diag_sym_soa(element_type,
                                                    mesh->n_elements(),
                                                    mesh->n_nodes(),
                                                    mesh->elements()->data(),
                                                    mesh->points()->data(),
                                                    this->mu,
                                                    this->lambda,
                                                    values);
    }

    int LinearElasticity::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::hessian_diag");

        auto mesh = space->mesh_ptr();

        return linear_elasticity_assemble_diag_aos(element_type,
                                                   mesh->n_elements(),
                                                   mesh->n_nodes(),
                                                   mesh->elements()->data(),
                                                   mesh->points()->data(),
                                                   this->mu,
                                                   this->lambda,
                                                   out);
    }

    int LinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::gradient");

        auto mesh = space->mesh_ptr();

        return linear_elasticity_assemble_gradient_aos(element_type,
                                                       mesh->n_elements(),
                                                       mesh->n_nodes(),
                                                       mesh->elements()->data(),
                                                       mesh->points()->data(),
                                                       this->mu,
                                                       this->lambda,
                                                       x,
                                                       out);
    }

    int LinearElasticity::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::apply");

        auto mesh = space->mesh_ptr();

        double tick = MPI_Wtime();

        int err;
        if (jacobians) {
            SFEM_TRACE_SCOPE("linear_elasticity_apply_adjugate_aos");
            err = linear_elasticity_apply_adjugate_aos(element_type,
                                                       mesh->n_elements(),
                                                       mesh->n_nodes(),
                                                       mesh->elements()->data(),
                                                       jacobians->adjugate->data(),
                                                       jacobians->determinant->data(),
                                                       this->mu,
                                                       this->lambda,
                                                       h,
                                                       out);
        } else {
            err = linear_elasticity_apply_aos(element_type,
                                              mesh->n_elements(),
                                              mesh->n_nodes(),
                                              mesh->elements()->data(),
                                              mesh->points()->data(),
                                              this->mu,
                                              this->lambda,
                                              h,
                                              out);
        }

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int LinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("LinearElasticity::value");

        auto mesh = space->mesh_ptr();

        // return linear_elasticity_value_aos(element_type,
        //                                   mesh->n_elements(),
        //                                   mesh->n_nodes(),
        //                                   mesh->elements()->data(),
        //                                   mesh->points()->data(),
        //                                   this->mu,
        //                                   this->lambda,
        //                                   x,
        //                                   out);
        SFEM_ERROR("LinearElasticity::value not implemented");
        return SFEM_FAILURE;
    }

    int LinearElasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> LinearElasticity::clone() const {
        auto ret = std::make_shared<LinearElasticity>(space);
        *ret     = *this;
        return ret;
    }

}  // namespace sfem