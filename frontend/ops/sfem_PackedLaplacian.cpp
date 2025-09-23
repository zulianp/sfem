#include "sfem_PackedLaplacian.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "laplacian.h"
#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Packed.hpp"
#include "sfem_Parameters.hpp"

template <typename pack_idx_t>
int packed_laplacian_apply(enum ElemType                         element_type,
                           const ptrdiff_t                       n_packs,
                           const ptrdiff_t                       n_elements_per_pack,
                           const ptrdiff_t                       n_elements,
                           const ptrdiff_t                       max_nodes_per_pack,
                           pack_idx_t **const SFEM_RESTRICT      elements,
                           const jacobian_t *const SFEM_RESTRICT fff,
                           const ptrdiff_t *const SFEM_RESTRICT  owned_nodes_ptr,
                           const count_t *const SFEM_RESTRICT    ghost_ptr,
                           const idx_t *const SFEM_RESTRICT      ghost_idx,
                           const real_t *const SFEM_RESTRICT     u,
                           real_t *const SFEM_RESTRICT           values) {
    if (element_type != TET4) {
        SFEM_ERROR("tet4_laplacian_packed_apply only supports TET4 elements");
    }

    static const int nxe = 4;
#pragma omp parallel
    {
        real_t *in  = (real_t *)malloc(max_nodes_per_pack * sizeof(real_t));
        real_t *out = (real_t *)malloc(max_nodes_per_pack * sizeof(real_t));

#pragma omp for
        for (ptrdiff_t p = 0; p < n_packs; p++) {
            const ptrdiff_t e_start = p * n_elements_per_pack;
            const ptrdiff_t e_end   = MIN(n_elements, (p + 1) * n_elements_per_pack);
            const ptrdiff_t n_owned = owned_nodes_ptr[p + 1] - owned_nodes_ptr[p];
            const ptrdiff_t n_ghost = ghost_ptr[p + 1] - ghost_ptr[p];
            
            const auto ghosts = &ghost_idx[ghost_ptr[p]];

            memset(out, 0, max_nodes_per_pack * sizeof(real_t));
            memcpy(in, &u[owned_nodes_ptr[p]], n_owned * sizeof(real_t));
            
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                in[n_owned + k] = u[ghosts[k]];
            }

            for (ptrdiff_t e = e_start; e < e_end; e++) {
                idx_t ev[nxe];
                for (int v = 0; v < nxe; ++v) {
                    ev[v] = elements[v][e];
                }

                scalar_t element_out[nxe] = {0};
                scalar_t fff_i[6];

                for (int d = 0; d < 6; ++d) {
                    fff_i[d] = fff[e * 6 + d];
                }

                tet4_laplacian_apply_fff(fff_i,
                                         in[ev[0]],
                                         in[ev[1]],
                                         in[ev[2]],
                                         in[ev[3]],
                                         &element_out[0],
                                         &element_out[1],
                                         &element_out[2],
                                         &element_out[3]);

                for (int v = 0; v < nxe; ++v) {
                    out[ev[v]] += element_out[v];
                }
            }

            for (ptrdiff_t k = 0; k < n_owned; ++k) {
#pragma omp atomic update
                values[owned_nodes_ptr[p] + k] += out[k];
            }

            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                values[ghosts[k]] += out[n_owned + k];
            }
        }

        free(in);
        free(out);
    }

    return SFEM_SUCCESS;
}

namespace sfem {

    class PackedLaplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace>         space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp>         domains;
        std::shared_ptr<Packed<PackedIdxType>> packed;
        std::vector<SharedBuffer<jacobian_t>>  fff;

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "PackedLaplacian::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    int PackedLaplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("PackedLaplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        impl_->packed  = Packed<PackedIdxType>::create(impl_->space->mesh_ptr(), block_names);

        impl_->fff.resize(impl_->packed->n_blocks());
        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name   = impl_->packed->block_name(b);
            auto domain = impl_->domains->domains().find(name);
            if (domain == impl_->domains->domains().end()) {
                SFEM_ERROR("Domain %s not found", name.c_str());
            }

            domain->second.user_data = std::static_pointer_cast<void>(std::make_shared<int>(b));
            impl_->fff[b]            = create_host_buffer<jacobian_t>(domain->second.block->n_elements() * 6);
            if (SFEM_SUCCESS != tet4_fff_fill(domain->second.block->n_elements(),
                                              domain->second.block->elements()->data(),
                                              impl_->space->mesh_ptr()->points()->data(),
                                              impl_->fff[b]->data())) {
                SFEM_ERROR("Unable to create fff");
            }
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> PackedLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("PackedLaplacian::create");

        assert(1 == space->block_size());

        auto ret = std::make_unique<PackedLaplacian>(space);
        return ret;
    }

    std::shared_ptr<Op> PackedLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<PackedLaplacian>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> PackedLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<PackedLaplacian>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    PackedLaplacian::PackedLaplacian(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    PackedLaplacian::~PackedLaplacian() = default;

    int PackedLaplacian::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        // SFEM_TRACE_SCOPE("PackedLaplacian::hessian_crs");

        // auto mesh  = impl_->space->mesh_ptr();
        // auto graph = impl_->space->dof_to_dof_graph();
        // int  err   = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_crs(domain.element_type,
        //                          domain.block->n_elements(),
        //                          mesh->n_nodes(),
        //                          domain.block->elements()->data(),
        //                          mesh->points()->data(),
        //                          graph->rowptr()->data(),
        //                          graph->colidx()->data(),
        //                          values);
        // });

        // return err;
        SFEM_ERROR("PackedLaplacian::hessian_crs not implemented");
        return SFEM_FAILURE;
    }

    int PackedLaplacian::hessian_crs_sym(const real_t *const  x,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         real_t *const        diag_values,
                                         real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("PackedLaplacian::hessian_crs_sym");

        // auto mesh = impl_->space->mesh_ptr();
        // int  err  = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_crs_sym(domain.element_type,
        //                              domain.block->n_elements(),
        //                              mesh->n_nodes(),
        //                              domain.block->elements()->data(),
        //                              mesh->points()->data(),
        //                              rowptr,
        //                              colidx,
        //                              diag_values,
        //                              off_diag_values);
        // });

        // return err;
        SFEM_ERROR("PackedLaplacian::hessian_crs_sym not implemented");
        return SFEM_FAILURE;
    }

    int PackedLaplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("PackedLaplacian::hessian_diag");

        // auto mesh = impl_->space->mesh_ptr();
        // int  err  = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_diag(domain.element_type,
        //                           domain.block->n_elements(),
        //                           mesh->n_nodes(),
        //                           domain.block->elements()->data(),
        //                           mesh->points()->data(),
        //                           values);
        // });

        // return err;
        SFEM_ERROR("PackedLaplacian::hessian_diag not implemented");
        return SFEM_FAILURE;
    }

    int PackedLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("PackedLaplacian::gradient");

        // auto mesh = impl_->space->mesh_ptr();
        // return impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_assemble_gradient(domain.element_type,
        //                                        domain.block->n_elements(),
        //                                        mesh->n_nodes(),
        //                                        domain.block->elements()->data(),
        //                                        mesh->points()->data(),
        //                                        x,
        //                                        out);
        // });
        SFEM_ERROR("PackedLaplacian::gradient not implemented");
        return SFEM_FAILURE;
    }

    int PackedLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("PackedLaplacian::apply");
        SFEM_OP_CAPTURE();

        return impl_->iterate([&](const OpDomain &domain) {
            auto b                  = *std::static_pointer_cast<int>(domain.user_data);
            auto elements           = impl_->packed->elements(b);
            auto owned_nodes_ptr    = impl_->packed->owned_nodes_ptr(b);
            auto ghost_ptr          = impl_->packed->ghost_ptr(b);
            auto ghost_idx          = impl_->packed->ghost_idx(b);
            auto fff                = impl_->fff[b]->data();
            auto max_nodes_per_pack = impl_->packed->max_nodes_per_pack();
            return packed_laplacian_apply<PackedIdxType>(domain.element_type,
                                                         impl_->packed->n_packs(b),
                                                         impl_->packed->n_elements_per_pack(b),
                                                         elements->extent(1),
                                                         max_nodes_per_pack,
                                                         elements->data(),
                                                         fff,
                                                         owned_nodes_ptr->data(),
                                                         ghost_ptr->data(),
                                                         ghost_idx->data(),
                                                         h,
                                                         out);
        });
    }

    int PackedLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("PackedLaplacian::value");

        // auto mesh = impl_->space->mesh_ptr();
        // return impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_assemble_value(domain.element_type,
        //                                     domain.block->n_elements(),
        //                                     mesh->n_nodes(),
        //                                     domain.block->elements()->data(),
        //                                     mesh->points()->data(),
        //                                     x,
        //                                     out);
        // });
        SFEM_ERROR("PackedLaplacian::value not implemented");
        return SFEM_FAILURE;
    }

    int PackedLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> PackedLaplacian::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        return nullptr;
    }

    void PackedLaplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void PackedLaplacian::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }
}  // namespace sfem
