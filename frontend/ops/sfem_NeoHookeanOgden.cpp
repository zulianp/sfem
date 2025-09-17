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
#include "tet4_neohookean_ogden.h"
#include "hex8_neohookean_ogden.h"
#include "tet4_partial_assembly_neohookean_inline.h"

#include <mpi.h>

namespace sfem {

    class NeoHookeanOgden::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };
        real_t                         mu{1}, lambda{1};
        SharedBuffer<metric_tensor_t>  partial_assembly_buffer;

        SharedBuffer<scaling_t>    compression_scaling;
        SharedBuffer<compressed_t> partial_assembly_compressed;
        SharedBuffer<idx_t *>      elements;
        ptrdiff_t                  elements_stride{1};

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int compress_partial_assembly() {
            auto mesh = space->mesh_ptr();

            if (use_compression) {
                if (!compression_scaling) {
                    compression_scaling         = sfem::create_host_buffer<scaling_t>(mesh->n_elements());
                    partial_assembly_compressed = sfem::create_host_buffer<compressed_t>(mesh->n_elements() * TET4_S_IKMN_SIZE);
                }

                auto      cs         = compression_scaling->data();
                auto      pa         = partial_assembly_buffer->data();
                auto      pac        = partial_assembly_compressed->data();
                ptrdiff_t n_elements = mesh->n_elements();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai = &pa[i * TET4_S_IKMN_SIZE];
                    cs[i]    = pai[0];
                    for (int v = 1; v < TET4_S_IKMN_SIZE; v++) {
                        cs[i] = MAX(cs[i], fabs(pai[v]));
                    }
                }

                real_t max_scaling = 0;

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    if (cs[i] > real_t(FP16_MAX)) {
                        max_scaling = MAX(max_scaling, cs[i]);
                        cs[i]       = real_t(cs[i] + 1e-8) / real_t(FP16_MAX);
                    } else {
                        cs[i] = 1;
                    }
                }

#ifndef NDEBUG
                ptrdiff_t num_nans = 0;
#endif
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai  = &pa[i * TET4_S_IKMN_SIZE];
                    auto paci = &pac[i * TET4_S_IKMN_SIZE];
                    for (int v = 0; v < TET4_S_IKMN_SIZE; v++) {
                        paci[v] = pai[v] / cs[i];

#ifndef NDEBUG
                        assert(cs[i] > 0);
                        assert(std::isfinite(paci[v]));
                        num_nans += !std::isfinite(paci[v]);
#endif
                    }
                }
#ifndef NDEBUG
                printf("Max scaling: %g, num_nans: %ld\n", max_scaling, num_nans);
#endif
            }

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> NeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());

        auto ret = std::make_unique<NeoHookeanOgden>(space);

        ret->impl_->mu                   = sfem::Env::read("SFEM_SHEAR_MODULUS", ret->impl_->mu);
        ret->impl_->lambda               = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", ret->impl_->lambda);
        ret->impl_->use_partial_assembly = sfem::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", ret->impl_->use_partial_assembly);
        ret->impl_->use_compression      = sfem::Env::read("SFEM_USE_COMPRESSION", ret->impl_->use_compression);
        ret->impl_->use_AoS              = sfem::Env::read("SFEM_NEOHOOKEAN_OGDEN_USE_AOS", ret->impl_->use_AoS);
        if (ret->impl_->use_partial_assembly) {
            printf("SFEM_SHEAR_MODULUS=%g\n", (double)ret->impl_->mu);
            printf("SFEM_FIRST_LAME_PARAMETER=%g\n", (double)ret->impl_->lambda);
            printf("sizeof(metric_tensor_t) = %lu\n", sizeof(metric_tensor_t));
            printf("sizeof(compressed_t) = %lu\n", sizeof(compressed_t));
        }

        ret->impl_->elements        = space->mesh_ptr()->elements();
        ret->impl_->elements_stride = 1;
        ret->impl_->element_type    = (enum ElemType)space->element_type();

        if (ret->impl_->use_AoS) {
            auto nxe                    = space->mesh_ptr()->n_nodes_per_element();
            ret->impl_->elements        = convert_host_buffer_to_fake_SoA(nxe, soa_to_aos(1, nxe, space->mesh_ptr()->elements()));
            ret->impl_->elements_stride = nxe;
        }

        if (ret->impl_->element_type == HEX8) {
            // This is the only implementation available for HEX8
            ret->impl_->use_partial_assembly = true;
            // ret->impl_->use_compression      = true;
        }

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
            if (impl_->use_compression) {
                return neohookean_ogden_compressed_partial_assembly_apply(impl_->element_type,
                                                                          mesh->n_elements(),
                                                                          impl_->elements_stride,
                                                                          impl_->elements->data(),
                                                                          impl_->partial_assembly_compressed->data(),
                                                                          impl_->compression_scaling->data(),
                                                                          3,
                                                                          &h[0],
                                                                          &h[1],
                                                                          &h[2],
                                                                          3,
                                                                          &out[0],
                                                                          &out[1],
                                                                          &out[2]);

            } else {
                return neohookean_ogden_partial_assembly_apply(impl_->element_type,
                                                               mesh->n_elements(),
                                                               impl_->elements_stride,
                                                               impl_->elements->data(),
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

        if (impl_->element_type == TET4 || impl_->element_type == HEX8) {
            // FIXME: Add support for other element types
            auto mesh = impl_->space->mesh_ptr();

            if (!impl_->partial_assembly_buffer) {
                impl_->partial_assembly_buffer = sfem::create_host_buffer<metric_tensor_t>(mesh->n_elements() * TET4_S_IKMN_SIZE);
            }

            int ok = neohookean_ogden_hessian_partial_assembly(impl_->element_type,
                                                               mesh->n_elements(),
                                                               impl_->elements_stride,
                                                               impl_->elements->data(),
                                                               mesh->points()->data(),
                                                               impl_->mu,
                                                               impl_->lambda,
                                                               3,
                                                               &u[0],
                                                               &u[1],
                                                               &u[2],
                                                               impl_->partial_assembly_buffer->data());

            return impl_->compress_partial_assembly();
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

        auto mesh = impl_->space->mesh_ptr();

        if (impl_->element_type == HEX8) {
            return hex8_neohookean_ogden_objective(mesh->n_elements(),
                                                   impl_->elements_stride,
                                                   mesh->n_nodes(),
                                                   impl_->elements->data(),
                                                   mesh->points()->data(),
                                                   this->impl_->mu,
                                                   this->impl_->lambda,
                                                   3,
                                                   &x[0],
                                                   &x[1],
                                                   &x[2],
                                                   false,
                                                   out);
        } else {
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