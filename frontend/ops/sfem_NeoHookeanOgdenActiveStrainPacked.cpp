#include "sfem_NeoHookeanOgdenActiveStrainPacked.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_Env.hpp"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "hex8_neohookean_ogden_active_strain.h"
#include "hex8_partial_assembly_neohookean_ogden_active_strain_inline.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_ElasticityAssemblyData.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Packed.hpp"
#include "sfem_Parameters.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    class NeoHookeanOgdenActiveStrainPacked::Impl {
    public:
        std::shared_ptr<FunctionSpace>                              space;
        std::shared_ptr<MultiDomainOp>                              domains;
        std::shared_ptr<Packed<sfem::FunctionSpace::PackedIdxType>> packed;
        real_t                                                      mu{1}, lambda{1};
        std::vector<std::shared_ptr<struct ElasticityAssemblyData>> assembly_data;

        // Active strain data per block (AoS pointer and stride)
        std::vector<const real_t *> Fa_aos_per_block;
        std::vector<ptrdiff_t>      Fa_stride_per_block;

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "NeoHookeanOgdenActiveStrainPacked::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    std::unique_ptr<Op> NeoHookeanOgdenActiveStrainPacked::create(const std::shared_ptr<FunctionSpace> &space) {
        auto ret = std::make_unique<NeoHookeanOgdenActiveStrainPacked>(space);
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgdenActiveStrainPacked::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<NeoHookeanOgdenActiveStrainPacked>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgdenActiveStrainPacked::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<NeoHookeanOgdenActiveStrainPacked>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    NeoHookeanOgdenActiveStrainPacked::NeoHookeanOgdenActiveStrainPacked(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    NeoHookeanOgdenActiveStrainPacked::~NeoHookeanOgdenActiveStrainPacked() = default;

    int NeoHookeanOgdenActiveStrainPacked::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        if (!impl_->space->has_packed_mesh()) {
            fprintf(stderr, "[Warning] NeoHookeanOgdenActiveStrainPacked: Initializing packed mesh, outer states may be inconsistent!\n");
            impl_->space->initialize_packed_mesh();
            fprintf(stderr, "[Warning] NeoHookeanOgdenActiveStrainPacked: Packed mesh initialized\n");
        }
        impl_->packed = impl_->space->packed_mesh();

        impl_->assembly_data.resize(impl_->packed->n_blocks());
        impl_->Fa_aos_per_block.resize(impl_->packed->n_blocks(), nullptr);
        impl_->Fa_stride_per_block.resize(impl_->packed->n_blocks(), 9);

        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name   = impl_->packed->block_name(b);
            auto domain = impl_->domains->domains().find(name);
            if (domain == impl_->domains->domains().end()) {
                SFEM_ERROR("Domain %s not found", name.c_str());
            }
            domain->second.user_data = std::static_pointer_cast<void>(std::make_shared<int>(b));
        }

        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name                                     = impl_->packed->block_name(b);
            auto domain                                   = impl_->domains->domains().find(name);
            impl_->assembly_data[b]                       = std::make_shared<struct ElasticityAssemblyData>();
            impl_->assembly_data[b]->use_partial_assembly = true;
            impl_->assembly_data[b]->use_compression      = false;
            impl_->assembly_data[b]->use_AoS              = true;
            impl_->assembly_data[b]->elements             = domain->second.block->elements();
            impl_->assembly_data[b]->elements_stride      = 1;
            impl_->assembly_data[b]->elements             = convert_host_buffer_to_fake_SoA(
                    domain->second.block->n_nodes_per_element(),
                    soa_to_aos(1, domain->second.block->n_nodes_per_element(), domain->second.block->elements()));
            impl_->assembly_data[b]->elements_stride = domain->second.block->n_nodes_per_element();
            impl_->assembly_data[b]->partial_assembly_buffer =
                    sfem::create_host_buffer<metric_tensor_t>(domain->second.block->n_elements() * HEX8_S_IKMN_SIZE);
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgdenActiveStrainPacked::hessian_crs(const real_t *const,
                                                       const count_t *const,
                                                       const idx_t *const,
                                                       real_t *const) {
        SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::hessian_crs not implemented");
        return SFEM_FAILURE;
    }

    int NeoHookeanOgdenActiveStrainPacked::hessian_crs_sym(const real_t *const,
                                                           const count_t *const,
                                                           const idx_t *const,
                                                           real_t *const,
                                                           real_t *const) {
        SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::hessian_crs_sym not implemented");
        return SFEM_FAILURE;
    }

    int NeoHookeanOgdenActiveStrainPacked::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::hessian_diag");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::hessian_diag only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            const real_t *Fa   = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_elasticity_diag(domain.block->n_elements(),
                                                                       assembly_data->elements_stride,
                                                                       mesh->n_nodes(),
                                                                       assembly_data->elements->data(),
                                                                       mesh->points()->data(),
                                                                       domain.parameters->get_real_value("mu", impl_->mu),
                                                                       domain.parameters->get_real_value("lambda", impl_->lambda),
                                                                       Fa_stride,
                                                                       Fa_soa,
                                                                       3,
                                                                       &x[0],
                                                                       &x[1],
                                                                       &x[2],
                                                                       3,
                                                                       &values[0],
                                                                       &values[1],
                                                                       &values[2]);
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::update");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::update only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            const real_t *Fa   = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_hessian_partial_assembly(
                    domain.block->n_elements(),
                    assembly_data->elements_stride,
                    assembly_data->elements->data(),
                    mesh->points()->data(),
                    domain.parameters->get_real_value("mu", impl_->mu),
                    domain.parameters->get_real_value("lambda", impl_->lambda),
                    Fa_stride,
                    Fa_soa,
                    3,
                    &u[0],
                    &u[1],
                    &u[2],
                    assembly_data->partial_assembly_buffer->data());
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::gradient");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::gradient only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            const real_t *Fa   = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_gradient(domain.block->n_elements(),
                                                                assembly_data->elements_stride,
                                                                mesh->n_nodes(),
                                                                assembly_data->elements->data(),
                                                                mesh->points()->data(),
                                                                domain.parameters->get_real_value("mu", impl_->mu),
                                                                domain.parameters->get_real_value("lambda", impl_->lambda),
                                                                Fa_stride,
                                                                Fa_soa,
                                                                3,
                                                                &x[0],
                                                                &x[1],
                                                                &x[2],
                                                                3,
                                                                &out[0],
                                                                &out[1],
                                                                &out[2]);
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::apply");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::apply only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            return hex8_neohookean_ogden_active_strain_partial_assembly_apply(
                    domain.block->n_elements(),
                    assembly_data->elements_stride,
                    assembly_data->elements->data(),
                    assembly_data->partial_assembly_buffer->data(),
                    3,
                    &h[0],
                    &h[1],
                    &h[2],
                    3,
                    &out[0],
                    &out[1],
                    &out[2]);
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::value");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::value only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            const real_t *Fa   = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_objective(domain.block->n_elements(),
                                                                 assembly_data->elements_stride,
                                                                 mesh->n_nodes(),
                                                                 assembly_data->elements->data(),
                                                                 mesh->points()->data(),
                                                                 domain.parameters->get_real_value("mu", impl_->mu),
                                                                 domain.parameters->get_real_value("lambda", impl_->lambda),
                                                                 Fa_stride,
                                                                 Fa_soa,
                                                                 3,
                                                                 &x[0],
                                                                 &x[1],
                                                                 &x[2],
                                                                 false,
                                                                 out);
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::value_steps(const real_t       *x,
                                                       const real_t       *h,
                                                       const int           nsteps,
                                                       const real_t *const steps,
                                                       real_t *const       out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::value_steps only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b             = *std::static_pointer_cast<int>(domain.user_data);
            auto assembly_data = impl_->assembly_data[b];
            const real_t *Fa   = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_objective_steps(domain.block->n_elements(),
                                                                       assembly_data->elements_stride,
                                                                       mesh->n_nodes(),
                                                                       assembly_data->elements->data(),
                                                                       mesh->points()->data(),
                                                                       domain.parameters->get_real_value("mu", impl_->mu),
                                                                       domain.parameters->get_real_value("lambda", impl_->lambda),
                                                                       Fa_stride,
                                                                       Fa_soa,
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
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgdenActiveStrainPacked::clone() const {
        SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::clone not implemented\n");
        return nullptr;
    }

    void NeoHookeanOgdenActiveStrainPacked::set_value_in_block(const std::string &block_name,
                                                               const std::string &var_name,
                                                               const real_t       value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void NeoHookeanOgdenActiveStrainPacked::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void NeoHookeanOgdenActiveStrainPacked::set_mu(const real_t mu) { impl_->mu = mu; }
    void NeoHookeanOgdenActiveStrainPacked::set_lambda(const real_t lambda) { impl_->lambda = lambda; }

    void NeoHookeanOgdenActiveStrainPacked::set_active_strain_global(const real_t *Fa_aos, const ptrdiff_t Fa_stride) {
        for (size_t b = 0; b < impl_->Fa_aos_per_block.size(); ++b) {
            impl_->Fa_aos_per_block[b]    = Fa_aos;
            impl_->Fa_stride_per_block[b] = Fa_stride;
        }
    }

    void NeoHookeanOgdenActiveStrainPacked::set_active_strain_in_block(const std::string &block_name,
                                                                       const real_t      *Fa_aos,
                                                                       const ptrdiff_t    Fa_stride) {
        // Map block name to index b
        for (int b = 0; b < impl_->packed->n_blocks(); ++b) {
            if (impl_->packed->block_name(b) == block_name) {
                impl_->Fa_aos_per_block[b]    = Fa_aos;
                impl_->Fa_stride_per_block[b] = Fa_stride;
                return;
            }
        }
        SFEM_ERROR("Block %s not found in set_active_strain_in_block\n", block_name.c_str());
    }

    int NeoHookeanOgdenActiveStrainPacked::hessian_bsr(const real_t *const  x,
                                                       const count_t *const rowptr,
                                                       const idx_t *const   colidx,
                                                       real_t *const        values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::hessian_bsr");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::hessian_bsr only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b           = *std::static_pointer_cast<int>(domain.user_data);
            const real_t *Fa = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_bsr(domain.block->n_elements(),
                                                           1,
                                                           domain.block->elements()->data(),
                                                           mesh->points()->data(),
                                                           domain.parameters->get_real_value("mu", impl_->mu),
                                                           domain.parameters->get_real_value("lambda", impl_->lambda),
                                                           Fa_stride,
                                                           Fa_soa,
                                                           3,
                                                           &x[0],
                                                           &x[1],
                                                           &x[2],
                                                           rowptr,
                                                           colidx,
                                                           values);
        });
    }

    int NeoHookeanOgdenActiveStrainPacked::hessian_bcrs_sym(const real_t *const  x,
                                                            const count_t *const rowptr,
                                                            const idx_t *const   colidx,
                                                            const ptrdiff_t      block_stride,
                                                            real_t **const       diag_values,
                                                            real_t **const       off_diag_values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgdenActiveStrainPacked::hessian_bcrs_sym");
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            if (domain.element_type != HEX8) {
                SFEM_ERROR("NeoHookeanOgdenActiveStrainPacked::hessian_bcrs_sym only implemented for HEX8\n");
                return SFEM_FAILURE;
            }
            auto b           = *std::static_pointer_cast<int>(domain.user_data);
            const real_t *Fa = impl_->Fa_aos_per_block[b];
            if (!Fa) {
                SFEM_ERROR("Active strain Fa not set for block %s\n", impl_->packed->block_name(b).c_str());
                return SFEM_FAILURE;
            }
            const ptrdiff_t Fa_stride = impl_->Fa_stride_per_block[b];

            const real_t *Fa_soa[9];
            for (int k = 0; k < 9; ++k) Fa_soa[k] = Fa + k;

            return hex8_neohookean_ogden_active_strain_bcrs_sym(domain.block->n_elements(),
                                                                1,
                                                                domain.block->elements()->data(),
                                                                mesh->points()->data(),
                                                                domain.parameters->get_real_value("mu", impl_->mu),
                                                                domain.parameters->get_real_value("lambda", impl_->lambda),
                                                                Fa_stride,
                                                                Fa_soa,
                                                                3,
                                                                &x[0],
                                                                &x[1],
                                                                &x[2],
                                                                rowptr,
                                                                colidx,
                                                                block_stride,
                                                                diag_values,
                                                                off_diag_values);
        });
    }
}  // namespace sfem


