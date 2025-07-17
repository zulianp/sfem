#include "sfem_NeoHookeanOgden.hpp"

#include "neohookean_ogden.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_glob.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_Tracer.hpp"

#include <mpi.h>

namespace sfem {

    std::unique_ptr<Op> NeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());

        auto ret = std::make_unique<NeoHookeanOgden>(space);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

        ret->mu           = SFEM_SHEAR_MODULUS;
        ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
        ret->element_type = (enum ElemType)space->element_type();
        return ret;
    }

    NeoHookeanOgden::NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int NeoHookeanOgden::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_crs");

        auto mesh  = space->mesh_ptr();
        auto graph = space->node_to_node_graph();

        return neohookean_ogden_hessian_aos(element_type,
                                            mesh->n_elements(),
                                            mesh->n_nodes(),
                                            mesh->elements()->data(),
                                            mesh->points()->data(),
                                            this->mu,
                                            this->lambda,
                                            x,
                                            graph->rowptr()->data(),
                                            graph->colidx()->data(),
                                            values);
    }

    int NeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_diag");

        auto mesh = space->mesh_ptr();

        return neohookean_ogden_diag_aos(element_type,
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements()->data(),
                                         mesh->points()->data(),
                                         this->mu,
                                         this->lambda,
                                         x,
                                         out);
    }

    int NeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::gradient");

        auto mesh = space->mesh_ptr();

        return neohookean_ogden_gradient_aos(element_type,
                                             mesh->n_elements(),
                                             mesh->n_nodes(),
                                             mesh->elements()->data(),
                                             mesh->points()->data(),
                                             this->mu,
                                             this->lambda,
                                             x,
                                             out);
    }

    int NeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::apply");

        auto mesh = space->mesh_ptr();

        return neohookean_ogden_apply_aos(element_type,
                                          mesh->n_elements(),
                                          mesh->n_nodes(),
                                          mesh->elements()->data(),
                                          mesh->points()->data(),
                                          this->mu,
                                          this->lambda,
                                          x,
                                          h,
                                          out);
    }

    int NeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

        auto mesh = space->mesh_ptr();

        return neohookean_ogden_value_aos(element_type,
                                          mesh->n_elements(),
                                          mesh->n_nodes(),
                                          mesh->elements()->data(),
                                          mesh->points()->data(),
                                          this->mu,
                                          this->lambda,
                                          x,
                                          out);
    }

    int NeoHookeanOgden::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    int NeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgden::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::lor_op");

        auto ret          = std::make_shared<NeoHookeanOgden>(space);
        ret->element_type = macro_type_variant(element_type);
        ret->mu           = mu;
        ret->lambda       = lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::derefine_op");

        auto ret          = std::make_shared<NeoHookeanOgden>(space);
        ret->element_type = macro_base_elem(element_type);
        ret->mu           = mu;
        ret->lambda       = lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::clone() const {
        auto ret = std::make_shared<NeoHookeanOgden>(space);
        *ret     = *this;
        return ret;
    }

    NeoHookeanOgden::~NeoHookeanOgden() = default;

}  // namespace sfem