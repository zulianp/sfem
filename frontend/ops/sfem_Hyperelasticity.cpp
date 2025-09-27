#include "sfem_Hyperelasticity.hpp"

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

#include <math.h>
#include <mpi.h>
#include <functional>

namespace sfem {

    struct HyperelasticityAssemblyData {
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
                        assert(std::isfinite(paci[v]));
                    }
                }
            }

            return SFEM_SUCCESS;
        }
    };

  

    struct HyperelasticityKernels {
        // Function objects for element-wise kernels (optionally bound to material params)
        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const real_t *const,
                          const count_t *const,
                          const idx_t *const,
                          real_t *const)>
                hessian_aos;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const real_t *const,
                          real_t *const)>
                hessian_diag_aos;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const real_t *const,
                          real_t *const)>
                gradient_aos;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const real_t *const,
                          const real_t *const,
                          real_t *const)>
                apply_aos;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          const metric_tensor_t *const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const ptrdiff_t,
                          real_t *const,
                          real_t *const,
                          real_t *const)>
                partial_assembly_apply;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          const compressed_t *const,
                          const scaling_t *const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const ptrdiff_t,
                          real_t *const,
                          real_t *const,
                          real_t *const)>
                compressed_partial_assembly_apply;

        std::function<int(const enum ElemType,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          metric_tensor_t *const)>
                hessian_partial_assembly;

        // Objective and path-evaluated objective (HEX8 specialized forms)
        std::function<int(const ptrdiff_t,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const int,
                          real_t *const)>
                objective;

        std::function<int(const ptrdiff_t,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const int,
                          const real_t *const,
                          real_t *const)>
                objective_steps;
        ~HyperelasticityKernels() = default;
        
    };

    class Hyperelasticity::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }

        std::unordered_map<enum ElemType, std::shared_ptr<HyperelasticityKernels>> kernels;
        
        std::shared_ptr<HyperelasticityKernels> find_kernels(const OpDomain &domain) {
            return kernels[domain.element_type];
        }

        int hyperelasticity_load_plugins(const std::string &folder) {
            // TODO: load plugins (shared libraries) from the folder and make them available as HyperelasticityKernels
            return SFEM_FAILURE;
        }
    };

    std::unique_ptr<Op> Hyperelasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
        auto ret           = std::make_unique<Hyperelasticity>(space);
        auto plugin_folder = sfem::Env::read_string("SFEM_HYPERELASTICITY_PLUGIN_FOLDER", "./");
        if(!plugin_folder.empty()) {
            ret->impl_->hyperelasticity_load_plugins(plugin_folder);
        }
        return ret;
    }

    Hyperelasticity::Hyperelasticity(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    int Hyperelasticity::hessian_crs(const real_t *const  x,
                                    const count_t *const rowptr,
                                    const idx_t *const   colidx,
                                    real_t *const        values) {
        SFEM_TRACE_SCOPE("Hyperelasticity::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->hessian_aos(domain.element_type,
                                        domain.block->n_elements(),
                                        mesh->n_nodes(),
                                        domain.block->elements()->data(),
                                        mesh->points()->data(),
                                        x,
                                        graph->rowptr()->data(),
                                        graph->colidx()->data(),
                                        values);
        });
    }

    int Hyperelasticity::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->hessian_diag_aos(domain.element_type,
                                     mesh->n_elements(),
                                     mesh->n_nodes(),
                                     domain.block->elements()->data(),
                                     mesh->points()->data(),
                                     x,
                                     out);
        });
    }

    int Hyperelasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->gradient_aos(domain.element_type,
                                         domain.block->n_elements(),
                                         mesh->n_nodes(),
                                         domain.block->elements()->data(),
                                         mesh->points()->data(),
                                         x,
                                         out);
        });
    }

    int Hyperelasticity::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::apply");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<struct HyperelasticityAssemblyData>(domain.user_data);
            if (ua->partial_assembly_buffer) {
                if (ua->use_compression) {
                    auto kernels = impl_->find_kernels(domain);
                    return kernels->compressed_partial_assembly_apply(domain.element_type,
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
                    auto kernels = impl_->find_kernels(domain);
                    return kernels->partial_assembly_apply(domain.element_type,
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

            auto kernels = impl_->find_kernels(domain);
            return kernels->apply_aos(domain.element_type,
                                      domain.block->n_elements(),
                                      mesh->n_nodes(),
                                      domain.block->elements()->data(),
                                      mesh->points()->data(),
                                      x,
                                      h,
                                      out);
        });
    }

    int Hyperelasticity::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        bool use_partial_assembly = sfem::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", false);
        bool use_compression      = sfem::Env::read("SFEM_USE_COMPRESSION", false);
        bool use_AoS              = sfem::Env::read("SFEM_HYPERELAStICITY_USE_AOS", false);

        for (auto &domain : impl_->domains->domains()) {
            auto ua                  = std::make_shared<struct HyperelasticityAssemblyData>();
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

    int Hyperelasticity::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("Hyperelasticity::update");

        auto mesh = impl_->space->mesh_ptr();

        for (auto &domain : impl_->domains->domains()) {
            auto assembly_data = std::static_pointer_cast<struct HyperelasticityAssemblyData>(domain.second.user_data);
            if (!assembly_data->use_partial_assembly) continue;

            auto element_type = domain.second.element_type;

            if (element_type == TET4 || element_type == HEX8) {
                // FIXME: Add support for other element types
                if (!assembly_data->partial_assembly_buffer) {
                    assembly_data->partial_assembly_buffer =
                            sfem::create_host_buffer<metric_tensor_t>(domain.second.block->n_elements() * TET4_S_IKMN_SIZE);
                }
                auto kernels = impl_->find_kernels(domain.second);
                int ok = kernels->hessian_partial_assembly(domain.second.element_type,
                                                           domain.second.block->n_elements(),
                                                           assembly_data->elements_stride,
                                                           assembly_data->elements->data(),
                                                           mesh->points()->data(),
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

    int Hyperelasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<HyperelasticityAssemblyData>(domain.user_data);

            auto kernels = impl_->find_kernels(domain);
            return kernels->objective(domain.block->n_elements(),
                                      ua->elements_stride,
                                      mesh->n_nodes(),
                                      ua->elements->data(),
                                      mesh->points()->data(),
                                      3,
                                      &x[0],
                                      &x[1],
                                      &x[2],
                                      false,
                                      out);
        });
    }

    int Hyperelasticity::value_steps(const real_t       *x,
                                    const real_t       *h,
                                    const int           nsteps,
                                    const real_t *const steps,
                                    real_t *const       out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::value_steps");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<HyperelasticityAssemblyData>(domain.user_data);
            if (domain.element_type == HEX8) {
                auto kernels = impl_->find_kernels(domain);
                return kernels->objective_steps(mesh->n_elements(),
                                                ua->elements_stride,
                                                mesh->n_nodes(),
                                                ua->elements->data(),
                                                mesh->points()->data(),
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

    int Hyperelasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Hyperelasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::lor_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<Hyperelasticity>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Hyperelasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::derefine_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<Hyperelasticity>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Hyperelasticity::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        auto ret = std::make_shared<Hyperelasticity>(impl_->space);
        return ret;
    }

    Hyperelasticity::~Hyperelasticity() = default;

    void Hyperelasticity::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Hyperelasticity::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem
