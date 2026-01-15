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

#ifdef _OPENMP
#include <omp.h>
#endif

using PackedIdxType = sfem::FunctionSpace::PackedIdxType;

// #if defined(__GNUC__) || defined(__clang__)
// #define SFEM_PREFETCH_R(addr, locality) __builtin_prefetch((addr), 0, (locality))
// #define SFEM_PREFETCH_W(addr, locality) __builtin_prefetch((addr), 1, (locality))
// #else
// #define SFEM_PREFETCH_R(addr, locality)
// #define SFEM_PREFETCH_W(addr, locality)
// #endif

struct PackedLaplacianScratch {
    struct ThreadData {
        ptrdiff_t max_nodes_per_pack;
        real_t   *in;
        real_t   *out;

        ThreadData(const ptrdiff_t max_nodes_per_pack) : max_nodes_per_pack(max_nodes_per_pack) {
            in  = (real_t *)malloc(max_nodes_per_pack * sizeof(real_t));
            out = (real_t *)calloc(max_nodes_per_pack, sizeof(real_t));
        }

        ~ThreadData() {
            free(in);
            free(out);
        }

        void reset() { memset(out, 0, max_nodes_per_pack * sizeof(real_t)); }
    };

    std::vector<std::unique_ptr<ThreadData>> thread_data;

    real_t *in(int thread_id) { return thread_data[thread_id]->in; }
    real_t *out(int thread_id) { return thread_data[thread_id]->out; }

    void reset() {
#ifdef _OPENMP
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            thread_data[thread_id]->reset();
        }
#else
        thread_data[0]->reset();
#endif
    }

    PackedLaplacianScratch(const ptrdiff_t max_nodes_per_pack) {
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        thread_data.resize(n_threads);
#pragma omp parallel
        {
            int thread_id          = omp_get_thread_num();
            thread_data[thread_id] = std::make_unique<ThreadData>(max_nodes_per_pack);
        }
#else
        thread_data.emplace_back(std::make_unique<ThreadData>(max_nodes_per_pack));
#endif
    }

    ~PackedLaplacianScratch() {}
};

template <typename pack_idx_t, int NXE, typename MicroKernel>
struct PackedLaplacianApply {
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
                     real_t *const SFEM_RESTRICT           values,
                     PackedLaplacianScratch               &scratch) {
#pragma omp parallel
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else
            int thread_id = 0;
#endif

            real_t *in  = scratch.in(thread_id);
            real_t *out = scratch.out(thread_id);
            memset(out, 0, max_nodes_per_pack * sizeof(real_t));

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

                MicroKernel::apply(e_start, e_end, elements, fff, in, out);

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
        }

        return SFEM_SUCCESS;
    }
};

template <typename pack_idx_t>
struct Tet4MicroKernel {
    static SFEM_INLINE void apply(const ptrdiff_t                       e_start,
                                  const ptrdiff_t                       e_end,
                                  pack_idx_t **const SFEM_RESTRICT      elements,
                                  const jacobian_t *const SFEM_RESTRICT fff,
                                  const scalar_t *const SFEM_RESTRICT   in,
                                  accumulator_t *const SFEM_RESTRICT    out) {
        static constexpr int VECTOR_SIZE = 64;

        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];

        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];

        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

        for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
            const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);

            // NOTE(zulianp): This transposition could be avoided by having the SoA layout from the start
            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = (evbegin + e) * 6;
                fff0[e]              = fff[eidx];
                fff1[e]              = fff[eidx + 1];
                fff2[e]              = fff[eidx + 2];
                fff3[e]              = fff[eidx + 3];
                fff4[e]              = fff[eidx + 4];
                fff5[e]              = fff[eidx + 5];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                u0[e]                = in[elements[0][eidx]];
                u1[e]                = in[elements[1][eidx]];
                u2[e]                = in[elements[2][eidx]];
                u3[e]                = in[elements[3][eidx]];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                tet4_laplacian_apply_fff_soa(fff0[e],
                                             fff1[e],
                                             fff2[e],
                                             fff3[e],
                                             fff4[e],
                                             fff5[e],
                                             u0[e],
                                             u1[e],
                                             u2[e],
                                             u3[e],
                                             &out0[e],
                                             &out1[e],
                                             &out2[e],
                                             &out3[e]);
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                out[elements[0][eidx]] += out0[e];
                out[elements[1][eidx]] += out1[e];
                out[elements[2][eidx]] += out2[e];
                out[elements[3][eidx]] += out3[e];
            }
        }
    }
};

template <typename pack_idx_t>
struct Hex8MicroKernel {
    static constexpr int    VECTOR_SIZE = 32;
    static SFEM_INLINE void apply(const ptrdiff_t                       e_start,
                                  const ptrdiff_t                       e_end,
                                  pack_idx_t **const SFEM_RESTRICT      elements,
                                  const jacobian_t *const SFEM_RESTRICT fff,
                                  const scalar_t *const SFEM_RESTRICT   in,
                                  accumulator_t *const SFEM_RESTRICT    out) {
        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];
        scalar_t out4[VECTOR_SIZE];
        scalar_t out5[VECTOR_SIZE];
        scalar_t out6[VECTOR_SIZE];
        scalar_t out7[VECTOR_SIZE];

        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];
        scalar_t u4[VECTOR_SIZE];
        scalar_t u5[VECTOR_SIZE];
        scalar_t u6[VECTOR_SIZE];
        scalar_t u7[VECTOR_SIZE];

        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

        for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
            const ptrdiff_t nelems = MIN(VECTOR_SIZE, e_end - evbegin);
            // NOTE(zulianp): This transposition could be avoided by having the SoA layout from the start
            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = (evbegin + e) * 6;
                fff0[e]              = fff[eidx];
                fff1[e]              = fff[eidx + 1];
                fff2[e]              = fff[eidx + 2];
                fff3[e]              = fff[eidx + 3];
                fff4[e]              = fff[eidx + 4];
                fff5[e]              = fff[eidx + 5];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                u0[e]                = in[elements[0][eidx]];
                u1[e]                = in[elements[1][eidx]];
                u2[e]                = in[elements[2][eidx]];
                u3[e]                = in[elements[3][eidx]];
                u4[e]                = in[elements[4][eidx]];
                u5[e]                = in[elements[5][eidx]];
                u6[e]                = in[elements[6][eidx]];
                u7[e]                = in[elements[7][eidx]];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                hex8_laplacian_apply_fff_integral_soa(fff0[e],
                                                      fff1[e],
                                                      fff2[e],
                                                      fff3[e],
                                                      fff4[e],
                                                      fff5[e],
                                                      u0[e],
                                                      u1[e],
                                                      u2[e],
                                                      u3[e],
                                                      u4[e],
                                                      u5[e],
                                                      u6[e],
                                                      u7[e],
                                                      &out0[e],
                                                      &out1[e],
                                                      &out2[e],
                                                      &out3[e],
                                                      &out4[e],
                                                      &out5[e],
                                                      &out6[e],
                                                      &out7[e]);
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                out[elements[0][eidx]] += out0[e];
                out[elements[1][eidx]] += out1[e];
                out[elements[2][eidx]] += out2[e];
                out[elements[3][eidx]] += out3[e];
                out[elements[4][eidx]] += out4[e];
                out[elements[5][eidx]] += out5[e];
                out[elements[6][eidx]] += out6[e];
                out[elements[7][eidx]] += out7[e];
            }
        }
    }
};

template <typename pack_idx_t>
struct Tet10MicroKernel {
    static SFEM_INLINE void apply(const ptrdiff_t                       e_start,
                                  const ptrdiff_t                       e_end,
                                  pack_idx_t **const SFEM_RESTRICT      elements,
                                  const jacobian_t *const SFEM_RESTRICT fff,
                                  const scalar_t *const SFEM_RESTRICT   in,
                                  accumulator_t *const SFEM_RESTRICT    out) {
        static constexpr int VECTOR_SIZE = 32;

        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];
        scalar_t out4[VECTOR_SIZE];
        scalar_t out5[VECTOR_SIZE];
        scalar_t out6[VECTOR_SIZE];
        scalar_t out7[VECTOR_SIZE];
        scalar_t out8[VECTOR_SIZE];
        scalar_t out9[VECTOR_SIZE];

        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];
        scalar_t u4[VECTOR_SIZE];
        scalar_t u5[VECTOR_SIZE];
        scalar_t u6[VECTOR_SIZE];
        scalar_t u7[VECTOR_SIZE];
        scalar_t u8[VECTOR_SIZE];
        scalar_t u9[VECTOR_SIZE];

        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

        for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
            const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);

            // NOTE(zulianp): This transposition could be avoided by having the SoA layout from the start
            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = (evbegin + e) * 6;
                fff0[e]              = fff[eidx];
                fff1[e]              = fff[eidx + 1];
                fff2[e]              = fff[eidx + 2];
                fff3[e]              = fff[eidx + 3];
                fff4[e]              = fff[eidx + 4];
                fff5[e]              = fff[eidx + 5];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                u0[e]                = in[elements[0][eidx]];
                u1[e]                = in[elements[1][eidx]];
                u2[e]                = in[elements[2][eidx]];
                u3[e]                = in[elements[3][eidx]];
                u4[e]                = in[elements[4][eidx]];
                u5[e]                = in[elements[5][eidx]];
                u6[e]                = in[elements[6][eidx]];
                u7[e]                = in[elements[7][eidx]];
                u8[e]                = in[elements[8][eidx]];
                u9[e]                = in[elements[9][eidx]];
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                tet10_laplacian_apply_fff_soa(fff0[e],
                                              fff1[e],
                                              fff2[e],
                                              fff3[e],
                                              fff4[e],
                                              fff5[e],
                                              u0[e],
                                              u1[e],
                                              u2[e],
                                              u3[e],
                                              u4[e],
                                              u5[e],
                                              u6[e],
                                              u7[e],
                                              u8[e],
                                              u9[e],
                                              &out0[e],
                                              &out1[e],
                                              &out2[e],
                                              &out3[e],
                                              &out4[e],
                                              &out5[e],
                                              &out6[e],
                                              &out7[e],
                                              &out8[e],
                                              &out9[e]);
            }

            for (ptrdiff_t e = 0; e < nelems; e++) {
                const ptrdiff_t eidx = evbegin + e;
                out[elements[0][eidx]] += out0[e];
                out[elements[1][eidx]] += out1[e];
                out[elements[2][eidx]] += out2[e];
                out[elements[3][eidx]] += out3[e];
                out[elements[4][eidx]] += out4[e];
                out[elements[5][eidx]] += out5[e];
                out[elements[6][eidx]] += out6[e];
                out[elements[7][eidx]] += out7[e];
                out[elements[8][eidx]] += out8[e];
                out[elements[9][eidx]] += out9[e];
            }
        }
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
                                  real_t *const SFEM_RESTRICT           values,
                                  PackedLaplacianScratch               &scratch) {
    switch (element_type) {
        case TET4:
            return PackedLaplacianApply<PackedIdxType, 4, Tet4MicroKernel<PackedIdxType>>::apply(n_packs,
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
                                                                                                 values,
                                                                                                 scratch);
        case HEX8:
            return PackedLaplacianApply<PackedIdxType, 8, Hex8MicroKernel<PackedIdxType>>::apply(n_packs,
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
                                                                                                 values,
                                                                                                 scratch);
        case TET10:
            return PackedLaplacianApply<PackedIdxType, 10, Tet10MicroKernel<PackedIdxType>>::apply(n_packs,
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
                                                                                                   values,
                                                                                                   scratch);
        default: {
            SFEM_ERROR("packed_laplacian_apply not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
}

namespace sfem {

    class PackedLaplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace>                       space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp>                       domains;
        std::shared_ptr<Packed<PackedIdxType>>               packed;
        std::vector<SharedBuffer<jacobian_t>>                fff;
        std::vector<std::shared_ptr<PackedLaplacianScratch>> scratch;

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

        impl_->scratch.resize(impl_->packed->n_blocks());
        for (int b = 0; b < impl_->packed->n_blocks(); b++) {
            impl_->scratch[b] = std::make_shared<PackedLaplacianScratch>(impl_->packed->max_nodes_per_pack());
        }

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
            auto scratch            = impl_->scratch[b];
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
                                                         out,
                                                         *scratch);
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
