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
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

// FIXME
#include "hex8_neohookean_ogden.h"
#include "tet4_neohookean_ogden.h"
#include "tet4_partial_assembly_neohookean_inline.h"

#include <mpi.h>
#include <math.h>

namespace sfem {

    struct AssemblyData {
        SharedBuffer<metric_tensor_t> partial_assembly_buffer;
        SharedBuffer<scaling_t>       compression_scaling;
        SharedBuffer<compressed_t>    partial_assembly_compressed;
        SharedBuffer<idx_t *>         elements;
        ptrdiff_t                     elements_stride{1};

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};

        int compress_partial_assembly(OpDomain &domain) {
            auto mesh = domain.block;

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

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai  = &pa[i * TET4_S_IKMN_SIZE];
                    auto paci = &pac[i * TET4_S_IKMN_SIZE];
                    for (int v = 0; v < TET4_S_IKMN_SIZE; v++) {
                        paci[v] = (compressed_t)(pai[v] / cs[i]);

                        assert(cs[i] > 0);
                        // Avoid _Float16 overload ambiguity on some libstdc++ versions
                        assert(std::isfinite(static_cast<double>(paci[v])));
                    }
                }
            }

            return SFEM_SUCCESS;
        }
    };

    class NeoHookeanOgden::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        real_t                         mu{1}, lambda{1};

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    std::unique_ptr<Op> NeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
        auto ret           = std::make_unique<NeoHookeanOgden>(space);
        ret->impl_->mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", ret->impl_->mu);
        ret->impl_->lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", ret->impl_->lambda);
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
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_hessian_aos(domain.element_type,
                                                domain.block->n_elements(),
                                                mesh->n_nodes(),
                                                domain.block->elements()->data(),
                                                mesh->points()->data(),
                                                domain.parameters->get_real_value("mu", impl_->mu),
                                                domain.parameters->get_real_value("lambda", impl_->lambda),
                                                x,
                                                graph->rowptr()->data(),
                                                graph->colidx()->data(),
                                                values);
        });
    }

    int NeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_diag_aos(domain.element_type,
                                             mesh->n_elements(),
                                             mesh->n_nodes(),
                                             domain.block->elements()->data(),
                                             mesh->points()->data(),
                                             this->impl_->mu,
                                             this->impl_->lambda,
                                             x,
                                             out);
        });
    }

    int NeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_gradient_aos(domain.element_type,
                                                 domain.block->n_elements(),
                                                 mesh->n_nodes(),
                                                 domain.block->elements()->data(),
                                                 mesh->points()->data(),
                                                 domain.parameters->get_real_value("mu", impl_->mu),
                                                 domain.parameters->get_real_value("lambda", impl_->lambda),
                                                 x,
                                                 out);
        });
    }

    int NeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::apply");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<struct AssemblyData>(domain.user_data);
            if (ua->partial_assembly_buffer) {
                if (ua->use_compression) {
                    return neohookean_ogden_compressed_partial_assembly_apply(domain.element_type,
                                                                              domain.block->n_elements(),
                                                                              ua->elements_stride,
                                                                              ua->elements->data(),
                                                                              ua->partial_assembly_compressed->data(),
                                                                              ua->compression_scaling->data(),
                                                                              3,
                                                                              &h[0],
                                                                              &h[1],
                                                                              &h[2],
                                                                              3,
                                                                              &out[0],
                                                                              &out[1],
                                                                              &out[2]);

                } else {
                    return neohookean_ogden_partial_assembly_apply(domain.element_type,
                                                                   domain.block->n_elements(),
                                                                   ua->elements_stride,
                                                                   ua->elements->data(),
                                                                   ua->partial_assembly_buffer->data(),
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

            return neohookean_ogden_apply_aos(domain.element_type,
                                              domain.block->n_elements(),
                                              mesh->n_nodes(),
                                              domain.block->elements()->data(),
                                              mesh->points()->data(),
                                              domain.parameters->get_real_value("mu", impl_->mu),
                                              domain.parameters->get_real_value("lambda", impl_->lambda),
                                              x,
                                              h,
                                              out);
        });
    }

    int NeoHookeanOgden::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        bool use_partial_assembly = sfem::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", false);
        bool use_compression      = sfem::Env::read("SFEM_USE_COMPRESSION", false);
        bool use_AoS              = sfem::Env::read("SFEM_NEOHOOKEAN_OGDEN_USE_AOS", false);

        for (auto &domain : impl_->domains->domains()) {
            auto ua                  = std::make_shared<struct AssemblyData>();
            ua->use_partial_assembly = use_partial_assembly || domain.second.element_type == HEX8;
            ua->use_compression      = use_compression;
            ua->use_AoS              = use_AoS;
            ua->elements             = domain.second.block->elements();
            ua->elements_stride      = 1;
            domain.second.user_data  = ua;

            if (use_AoS) {
                auto nxe            = domain.second.block->n_nodes_per_element();
                ua->elements        = convert_host_buffer_to_fake_SoA(nxe, soa_to_aos(1, nxe, domain.second.block->elements()));
                ua->elements_stride = nxe;
            }
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::update");

        auto mesh = impl_->space->mesh_ptr();

        for (auto &domain : impl_->domains->domains()) {
            auto assembly_data = std::static_pointer_cast<struct AssemblyData>(domain.second.user_data);
            if (!assembly_data->use_partial_assembly) continue;

            auto lambda       = domain.second.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.second.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.second.element_type;

            if (element_type == TET4 || element_type == HEX8) {
                // FIXME: Add support for other element types
                if (!assembly_data->partial_assembly_buffer) {
                    assembly_data->partial_assembly_buffer =
                            sfem::create_host_buffer<metric_tensor_t>(domain.second.block->n_elements() * TET4_S_IKMN_SIZE);
                }

                int ok = neohookean_ogden_hessian_partial_assembly(
                        domain.second.element_type,
                        domain.second.block->n_elements(),
                        assembly_data->elements_stride,
                        assembly_data->elements->data(),
                        mesh->points()->data(),
                        domain.second.parameters->get_real_value("mu", impl_->mu),
                        domain.second.parameters->get_real_value("lambda", impl_->lambda),
                        3,
                        &u[0],
                        &u[1],
                        &u[2],
                        assembly_data->partial_assembly_buffer->data());
                assembly_data->compress_partial_assembly(domain.second);
            }
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<AssemblyData>(domain.user_data);

            if (domain.element_type == HEX8) {
                return hex8_neohookean_ogden_objective(domain.block->n_elements(),
                                                       ua->elements_stride,
                                                       mesh->n_nodes(),
                                                       ua->elements->data(),
                                                       mesh->points()->data(),
                                                       domain.parameters->get_real_value("mu", impl_->mu),
                                                       domain.parameters->get_real_value("lambda", impl_->lambda),
                                                       3,
                                                       &x[0],
                                                       &x[1],
                                                       &x[2],
                                                       false,
                                                       out);
            } else {
                return neohookean_ogden_value_aos(domain.element_type,
                                                  domain.block->n_elements(),
                                                  mesh->n_nodes(),
                                                  domain.block->elements()->data(),
                                                  mesh->points()->data(),
                                                  domain.parameters->get_real_value("mu", impl_->mu),
                                                  domain.parameters->get_real_value("lambda", impl_->lambda),
                                                  x,
                                                  out);
            }
        });
    }

    int NeoHookeanOgden::value_steps(const real_t       *x,
                                     const real_t       *h,
                                     const int           nsteps,
                                     const real_t *const steps,
                                     real_t *const       out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value_steps");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<AssemblyData>(domain.user_data);
            if (domain.element_type == HEX8) {
                return hex8_neohookean_ogden_objective_steps(mesh->n_elements(),
                                                             ua->elements_stride,
                                                             mesh->n_nodes(),
                                                             ua->elements->data(),
                                                             mesh->points()->data(),
                                                             domain.parameters->get_real_value("mu", impl_->mu),
                                                             domain.parameters->get_real_value("lambda", impl_->lambda),
                                                             3,
                                                             &x[0],
                                                             &x[1],
                                                             &x[2],
                                                             3,
                                                             &h[0],
                                                             &h[1],
                                                             &h[2],
                                                             nsteps,
                                                             steps,
                                                             out);
            } else {
                // Must be implemented
                return SFEM_FAILURE;
            }
        });
    }

    int NeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgden::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::lor_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        ret->impl_->mu      = impl_->mu;
        ret->impl_->lambda  = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::derefine_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        ret->impl_->mu      = impl_->mu;
        ret->impl_->lambda  = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        auto ret = std::make_shared<NeoHookeanOgden>(impl_->space);
        return ret;
    }

    NeoHookeanOgden::~NeoHookeanOgden() = default;

    void NeoHookeanOgden::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void NeoHookeanOgden::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void NeoHookeanOgden::set_mu(const real_t mu) { impl_->mu = mu; }
    void NeoHookeanOgden::set_lambda(const real_t lambda) { impl_->lambda = lambda; }

}  // namespace sfem