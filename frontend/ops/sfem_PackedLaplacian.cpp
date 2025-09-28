#include "sfem_PackedLaplacian.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_Env.hpp"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "hex8_fff.h"
#include "hex8_laplacian_inline_cpu.h"
#include "laplacian.h"
#include "tet10_laplacian_inline_cpu.h"
#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Packed.hpp"
#include "sfem_Parameters.hpp"

using PackedIdxType = sfem::FunctionSpace::PackedIdxType;

// #if defined(__GNUC__) || defined(__clang__)
// #define SFEM_PREFETCH_R(addr, locality) __builtin_prefetch((addr), 0, (locality))
// #define SFEM_PREFETCH_W(addr, locality) __builtin_prefetch((addr), 1, (locality))
// #else
// #define SFEM_PREFETCH_R(addr, locality)
// #define SFEM_PREFETCH_W(addr, locality)
// #endif

template <typename pack_idx_t, int NXE, typename MicroKernel>
struct PackedLaplacian {
    static int apply(const ptrdiff_t                       n_packs,
                     const ptrdiff_t                       n_elements_per_pack,
                     const ptrdiff_t                       n_elements,
                     const ptrdiff_t                       max_nodes_per_pack,
                     pack_idx_t **const SFEM_RESTRICT      elements,
                     const jacobian_t *const SFEM_RESTRICT fff,
                     const ptrdiff_t *const SFEM_RESTRICT  owned_nodes_ptr,
                     const ptrdiff_t *const SFEM_RESTRICT  n_shared_nodes,
                     const ptrdiff_t *const SFEM_RESTRICT  ghost_ptr,
                     const idx_t *const SFEM_RESTRICT      ghost_idx,
                     const real_t *const SFEM_RESTRICT     u,
                     real_t *const SFEM_RESTRICT           values) {
#pragma omp parallel
        {
            real_t *in  = (real_t *)malloc(max_nodes_per_pack * sizeof(real_t));
            real_t *out = (real_t *)calloc(max_nodes_per_pack, sizeof(real_t));

#pragma omp for schedule(static)
            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t e_start      = p * n_elements_per_pack;
                const ptrdiff_t e_end        = MIN(n_elements, (p + 1) * n_elements_per_pack);
                const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned_nodes_ptr[p];
                const ptrdiff_t n_shared     = n_shared_nodes[p];
                const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - ghost_ptr[p];
                const ptrdiff_t n_not_shared = n_contiguous - n_shared;
                const auto      ghosts       = &ghost_idx[ghost_ptr[p]];
                scalar_t *const g_out        = &out[n_contiguous];

                memcpy(in, &u[owned_nodes_ptr[p]], n_contiguous * sizeof(real_t));

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    in[n_contiguous + k] = u[ghosts[k]];
                }

                for (ptrdiff_t e = e_start; e < e_end; e++) {
                    pack_idx_t ev[NXE];
                    for (int v = 0; v < NXE; ++v) {
                        ev[v] = elements[v][e];
                    }

                    scalar_t element_u[NXE];
                    for (int v = 0; v < NXE; ++v) {
                        element_u[v] = in[ev[v]];
                    }

                    accumulator_t element_out[NXE] = {0};
                    scalar_t      fff_i[6];

                    for (int d = 0; d < 6; ++d) {
                        fff_i[d] = fff[e * 6 + d];
                    }

                    MicroKernel::apply(fff_i, element_u, element_out);

                    for (int v = 0; v < NXE; ++v) {
                        out[ev[v]] += element_out[v];
                    }
                }

                real_t *const SFEM_RESTRICT acc = &values[owned_nodes_ptr[p]];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    // No need for atomic there are no collisions
                    acc[k] += out[k];
                    out[k] = 0;  // Clean-up while hot
                }

                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    acc[k] += out[k];
                    out[k] = 0;  // Clean-up while hot
                }

                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    values[ghosts[k]] += g_out[k];
                    g_out[k] = 0;  // Clean-up while hot
                }
            }

            free(in);
            free(out);
        }

        return SFEM_SUCCESS;
    }
};

struct Tet4MicroKernel {
    static SFEM_INLINE void apply(const scalar_t *const fff, const scalar_t *const element_u, accumulator_t *const element_out) {
        tet4_laplacian_apply_fff(fff,
                                 element_u[0],
                                 element_u[1],
                                 element_u[2],
                                 element_u[3],
                                 &element_out[0],
                                 &element_out[1],
                                 &element_out[2],
                                 &element_out[3]);
    }
};

struct Hex8MicroKernel {
    static SFEM_INLINE void apply(const scalar_t *const fff, const scalar_t *const element_u, accumulator_t *const element_out) {
        hex8_laplacian_apply_fff_integral(fff, element_u, element_out);
    }
};

struct Tet10MicroKernel {
    static SFEM_INLINE void apply(const scalar_t *const fff, const scalar_t *const element_u, accumulator_t *const element_out) {
        for (int i = 0; i < 10; i++) {
            element_out[i] = 0;
        }

        tet10_laplacian_apply_add_fff(fff, element_u, element_out);
    }
};

template <typename pack_idx_t>
static int packed_laplacian_apply(enum ElemType                         element_type,
                                  const ptrdiff_t                       n_packs,
                                  const ptrdiff_t                       n_elements_per_pack,
                                  const ptrdiff_t                       n_elements,
                                  const ptrdiff_t                       max_nodes_per_pack,
                                  pack_idx_t **const SFEM_RESTRICT      elements,
                                  const jacobian_t *const SFEM_RESTRICT fff,
                                  const ptrdiff_t *const SFEM_RESTRICT  owned_nodes_ptr,
                                  const ptrdiff_t *const SFEM_RESTRICT  n_shared_nodes,
                                  const ptrdiff_t *const SFEM_RESTRICT  ghost_ptr,
                                  const idx_t *const SFEM_RESTRICT      ghost_idx,
                                  const real_t *const SFEM_RESTRICT     u,
                                  real_t *const SFEM_RESTRICT           values) {
    switch (element_type) {
        case TET4:
            return PackedLaplacian<PackedIdxType, 4, Tet4MicroKernel>::apply(n_packs,
                                                                             n_elements_per_pack,
                                                                             n_elements,
                                                                             max_nodes_per_pack,
                                                                             elements,
                                                                             fff,
                                                                             owned_nodes_ptr,
                                                                             n_shared_nodes,
                                                                             ghost_ptr,
                                                                             ghost_idx,
                                                                             u,
                                                                             values);
        case HEX8:
            return PackedLaplacian<PackedIdxType, 8, Hex8MicroKernel>::apply(n_packs,
                                                                             n_elements_per_pack,
                                                                             n_elements,
                                                                             max_nodes_per_pack,
                                                                             elements,
                                                                             fff,
                                                                             owned_nodes_ptr,
                                                                             n_shared_nodes,
                                                                             ghost_ptr,
                                                                             ghost_idx,
                                                                             u,
                                                                             values);
        case TET10:
            return PackedLaplacian<PackedIdxType, 10, Tet10MicroKernel>::apply(n_packs,
                                                                               n_elements_per_pack,
                                                                               n_elements,
                                                                               max_nodes_per_pack,
                                                                               elements,
                                                                               fff,
                                                                               owned_nodes_ptr,
                                                                               n_shared_nodes,
                                                                               ghost_ptr,
                                                                               ghost_idx,
                                                                               u,
                                                                               values);
        default: {
            SFEM_ERROR("packed_laplacian_apply not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
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

        if (!impl_->space->has_packed_mesh()) {
            fprintf(stderr, "[Warning] PackedLaplacian: Initializing packed mesh, outer states may be inconsistent!\n");
            impl_->space->initialize_packed_mesh();
            fprintf(stderr, "[Warning] PackedLaplacian: Packed mesh initialized\n");
        }
        impl_->packed = impl_->space->packed_mesh();

        impl_->fff.resize(impl_->packed->n_blocks());
        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            auto name   = impl_->packed->block_name(b);
            auto domain = impl_->domains->domains().find(name);
            if (domain == impl_->domains->domains().end()) {
                SFEM_ERROR("Domain %s not found", name.c_str());
            }

            domain->second.user_data = std::static_pointer_cast<void>(std::make_shared<int>(b));
            impl_->fff[b]            = create_host_buffer<jacobian_t>(domain->second.block->n_elements() * 6);

            if (domain->second.element_type == HEX8 || domain->second.element_type == SSHEX8) {
                hex8_fff_fill(domain->second.block->n_elements(),
                              domain->second.block->elements()->data(),
                              impl_->space->mesh_ptr()->points()->data(),
                              impl_->fff[b]->data());
            } else {
                tet4_fff_fill(domain->second.block->n_elements(),
                              domain->second.block->elements()->data(),
                              impl_->space->mesh_ptr()->points()->data(),
                              impl_->fff[b]->data());
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
            auto n_shared           = impl_->packed->n_shared(b);
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
                                                         n_shared->data(),
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
