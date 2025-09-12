#include "sfem_NeoHookeanOgden.hpp"

#include "neohookean_ogden.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_Env.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

// FIXME
#include "tet4_partial_assembly_neohookean_inline.h"

#include <mpi.h>

namespace sfem {

    class NeoHookeanOgden::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        enum ElemType                   element_type { INVALID };
        real_t                          mu{1}, lambda{1};
        SharedBuffer<metric_tensor_t *> partial_assembly_buffer;
        bool                            use_partial_assembly{false};
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
    };

    std::unique_ptr<Op> NeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());

        auto ret = std::make_unique<NeoHookeanOgden>(space);

        ret->impl_->mu                   = sfem::Env::read("SFEM_SHEAR_MODULUS", ret->impl_->mu);
        ret->impl_->lambda               = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", ret->impl_->lambda);
        ret->impl_->use_partial_assembly = sfem::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", ret->impl_->use_partial_assembly);
        ret->impl_->element_type         = (enum ElemType)space->element_type();
        return ret;
    }

    NeoHookeanOgden::NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    int NeoHookeanOgden::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        return neohookean_ogden_hessian_aos(impl_->element_type,
                                            mesh->n_elements(),
                                            mesh->n_nodes(),
                                            mesh->elements()->data(),
                                            mesh->points()->data(),
                                            this->impl_->mu,
                                            this->impl_->lambda,
                                            x,
                                            graph->rowptr()->data(),
                                            graph->colidx()->data(),
                                            values);
    }

    int NeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        return neohookean_ogden_diag_aos(impl_->element_type,
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements()->data(),
                                         mesh->points()->data(),
                                         this->impl_->mu,
                                         this->impl_->lambda,
                                         x,
                                         out);
    }

    int NeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return neohookean_ogden_gradient_aos(impl_->element_type,
                                             mesh->n_elements(),
                                             mesh->n_nodes(),
                                             mesh->elements()->data(),
                                             mesh->points()->data(),
                                             this->impl_->mu,
                                             this->impl_->lambda,
                                             x,
                                             out);
    }

    int NeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::apply");
        auto mesh = impl_->space->mesh_ptr();
        if (impl_->partial_assembly_buffer) {
            return neohookean_ogden_partial_assembly_apply(impl_->element_type,
                                                           mesh->n_elements(),
                                                           mesh->elements()->data(),
                                                           1,
                                                           impl_->partial_assembly_buffer->data(),
                                                           3,
                                                           &h[0],
                                                           &h[1],
                                                           &h[2],
                                                           3,
                                                           &out[0],
                                                           &out[1],
                                                           &out[2]);
        }

        return neohookean_ogden_apply_aos(impl_->element_type,
                                          mesh->n_elements(),
                                          mesh->n_nodes(),
                                          mesh->elements()->data(),
                                          mesh->points()->data(),
                                          this->impl_->mu,
                                          this->impl_->lambda,
                                          x,
                                          h,
                                          out);
    }

    int NeoHookeanOgden::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    int NeoHookeanOgden::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::update");

        if (!impl_->use_partial_assembly) return SFEM_SUCCESS;

        if (impl_->element_type == TET4) {
            // FIXME: Add support for other element types
            auto mesh = impl_->space->mesh_ptr();

            if (!impl_->partial_assembly_buffer) {
                impl_->partial_assembly_buffer = sfem::create_host_buffer<metric_tensor_t>(TET4_S_IKMN_SIZE, mesh->n_elements());
            }

            return neohookean_ogden_hessian_partial_assembly(impl_->element_type,
                                                             mesh->n_elements(),
                                                             mesh->elements()->data(),
                                                             mesh->points()->data(),
                                                             impl_->mu,
                                                             impl_->lambda,
                                                             3,
                                                             &u[0],
                                                             &u[1],
                                                             &u[2],
                                                             1,
                                                             impl_->partial_assembly_buffer->data());
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

        auto mesh = impl_->space->mesh_ptr();
        return neohookean_ogden_value_aos(impl_->element_type,
                                          mesh->n_elements(),
                                          mesh->n_nodes(),
                                          mesh->elements()->data(),
                                          mesh->points()->data(),
                                          this->impl_->mu,
                                          this->impl_->lambda,
                                          x,
                                          out);
    }

    int NeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgden::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::lor_op");

        auto ret                 = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->element_type = macro_type_variant(impl_->element_type);
        ret->impl_->mu           = impl_->mu;
        ret->impl_->lambda       = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::derefine_op");

        auto ret                 = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->element_type = macro_base_elem(impl_->element_type);
        ret->impl_->mu           = impl_->mu;
        ret->impl_->lambda       = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        auto ret = std::make_shared<NeoHookeanOgden>(impl_->space);
        return ret;
    }

    NeoHookeanOgden::~NeoHookeanOgden() = default;

}  // namespace sfem