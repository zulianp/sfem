#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "sfem_ElementMatrixStore.hpp"
#include "sfem_InexactInterface.hpp"
#include "sfem_InexactOptions.hpp"
#include "sfem_InexactSetup.hpp"
#include "sfem_InexactTraversal.hpp"
#include "sfem_PointStore.hpp"
#include "sfem_QuantBSRStore.hpp"
#include "sfem_aliases.hpp"
#include "sfem_blas_factory.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_QuantBSRStore.hpp"
#endif

namespace sfem {

    namespace inexact {

        /// The host lane width: a block of elements is one store block (16 for int8 to vectorise;
        /// 8 cripples it, RESULTS_QP.md).
        static constexpr int HOST_VS = 16;

        template <class Store, typename tau>
        class StoreImpl final : public InexactStore {
        public:
            StoreImpl(const ptrdiff_t nelements, const InexactOptions &options, std::string name)
                : store_(nelements), options_(options), name_(std::move(name)) {}

            void update(const SetupInput &input) override {
                if (input.nelements != store_.nelements()) {
                    SFEM_ERROR("inexact: the store holds %ld elements, the set-up has %ld\n",
                               (long)store_.nelements(),
                               (long)input.nelements);
                }
                setup(store_, input);
            }

            void apply(const ptrdiff_t           nelements,
                       const idx_t *const *const elements,
                       const PackedView *const   packed,
                       const ptrdiff_t           stride,
                       const real_t *const       h,
                       real_t *const             out) const override {
                if (!packed && options_.traversal == InexactOptions::Traversal::PACKED) {
                    SFEM_ERROR("inexact: traversal packed asked for, but the space has no packed mesh\n");
                }
                const ptrdiff_t rows = packed ? packed->n_nodes * stride : 0;
#pragma omp parallel
                team_apply(Team::current(rows), nelements, 0, nelements, elements, packed, stride, h, out);
            }

            void team_apply(const Team               &team,
                            const ptrdiff_t           nelements,
                            const ptrdiff_t           element_begin,
                            const ptrdiff_t           element_end,
                            const idx_t *const *const elements,
                            const PackedView *const   packed,
                            const ptrdiff_t           stride,
                            const real_t *const       h,
                            real_t *const             out) const override {
                const bool whole = element_begin == 0 && element_end == nelements;
                if (whole && packed && options_.traversal != InexactOptions::Traversal::ATOMIC) {
                    team_apply_packed<Store, tau>(team, store_, *packed, nelements, stride, h, out);
                } else {
                    team_apply_atomic<Store, tau>(team, store_, element_begin, element_end, elements, stride, h, out);
                }
            }

            void diagonal(const ptrdiff_t           nelements,
                          const idx_t *const *const elements,
                          const ptrdiff_t           stride,
                          real_t *const             diag) const override {
                using El                = typename Store::Element;
                static constexpr int VS = Store::VS;
                static constexpr int N  = El::N;
                static constexpr int D  = El::DIM;
#pragma omp parallel for schedule(static)
                for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
                    const int ne = (int)std::min<ptrdiff_t>(VS, nelements - evb);
                    real_t    d[D][N][VS];
                    store_.diagonal_block(evb, ne, d);
                    for (int s = 0; s < N; ++s) {
                        const idx_t *const RSTR nodes = elements[ElementTables<El>::shape_node(s)];
                        for (int lane = 0; lane < ne; ++lane)
                            for (int c = 0; c < D; ++c) {
#pragma omp atomic update
                                diag[(ptrdiff_t)nodes[evb + lane] * stride + c] += d[c][s][lane];
                            }
                    }
                }
            }

            std::size_t bytes() const override { return store_.bytes(); }
            std::string describe() const override { return name_; }

        private:
            Store          store_;
            InexactOptions options_;
            std::string    name_;
        };

        template <class Store, typename tau>
        std::unique_ptr<InexactStore> make(const ptrdiff_t nelements, const InexactOptions &options, const char *const what) {
            return std::make_unique<StoreImpl<Store, tau>>(
                    nelements, options, std::string(what) + " " + Store::Unit::element::name);
        }

        template <class M, bool FOLD>
        std::unique_ptr<InexactStore> make_points(const ptrdiff_t nelements, const InexactOptions &options) {
            using E = InexactOptions::Encoding;
            switch (options.encoding) {
                case E::F64:
                    return make<PointStore<M, double, Scheme::Plain, HOST_VS, FOLD>, double>(nelements, options, "points f64");
                case E::F32:
                    return make<PointStore<M, float, Scheme::Plain, HOST_VS, FOLD>, float>(nelements, options, "points f32");
                case E::F16:
                    return make<PointStore<M, half_t, Scheme::Plain, HOST_VS, FOLD>, float>(nelements, options, "points f16");
                case E::I8_MEAN_DEVIATION:
                    return make<PointStore<M, int8_t, Scheme::MeanDeviation, HOST_VS, FOLD>, float>(
                            nelements, options, "points i8md");
                default:
                    break;
            }
            SFEM_ERROR("inexact: the point store has no encoding %s\n", options.describe().c_str());
            return nullptr;
        }

        template <class M>
        std::unique_ptr<InexactStore> make_element_matrices(const ptrdiff_t nelements, const InexactOptions &options) {
            using E = InexactOptions::Encoding;
            switch (options.encoding) {
                case E::F64:
                    return make<ElementMatrixStore<M, double, false, HOST_VS>, double>(nelements, options, "element matrices f64");
                case E::F32:
                    return make<ElementMatrixStore<M, float, false, HOST_VS>, float>(nelements, options, "element matrices f32");
                case E::F16:
                    return make<ElementMatrixStore<M, half_t, false, HOST_VS>, float>(nelements, options, "element matrices f16");
                case E::I8_PAIR_SCALE:
                    return make<ElementMatrixStore<M, int8_t, true, HOST_VS>, float>(
                            nelements, options, "element matrices i8pair");
                default:
                    break;
            }
            SFEM_ERROR("inexact: the element-matrix store has no encoding %s\n", options.describe().c_str());
            return nullptr;
        }

        /// The host store the options ask for, for a generated unit `M` on `nelements` elements.
        template <class M>
        std::unique_ptr<InexactStore> create_store(const ptrdiff_t nelements, const InexactOptions &options) {
            options.validate();
            if (options.store == InexactOptions::Store::BSR) {
                SFEM_ERROR("inexact: the bsr store is the operator's, not a domain's (OpState::update_bsr)\n");
                return nullptr;
            }
            if (options.store == InexactOptions::Store::ELEMENT_MATRIX) {
                return make_element_matrices<M>(nelements, options);
            }
            if constexpr (M::element::foldable) {
                // The fold is exact and measured faster for every encoding on Grace, so `auto` folds.
                if (options.fold != InexactOptions::Switch::OFF) return make_points<M, true>(nelements, options);
            } else {
                if (options.fold == InexactOptions::Switch::ON) {
                    SFEM_ERROR("inexact: fold asked for on %s, which is not a tensor-product element\n",
                               M::element::name);
                }
            }
            return make_points<M, false>(nelements, options);
        }

        /// What a generated operator holds for its stored tangent: the options, one store per
        /// domain, the state the stores were built at, and the packed traversal's ghost buffers.
        struct OpState {
            /// Where the operator runs: a device operator's stores are the device's, on its stream.
            ExecutionSpace                                             space{EXECUTION_SPACE_HOST};
            void                                                      *stream{nullptr};
            InexactOptions                                             options;
            std::map<const void *, std::unique_ptr<InexactStore>>      stores;
            std::map<int, SharedBuffer<real_t>>                        ghost;
            std::unique_ptr<BsrStore>                                  bsr;
            const real_t                                              *state{nullptr};
            /// apply_bsr's extra diagonal blocks and whether it overwrites (Op::set_inexact_extra);
            /// null: the plain y += A x.
            const inexact::DiagonalExtra                              *extra{nullptr};
            bool                                                       overwrite{false};
            /// The time scheme's inertia beside the `bsr` store, when it is diagonal and fused: its
            /// diagonal, which the store adds in its pass (BsrStore::set_diagonal).
            SharedBuffer<real_t>                                       inertia_diagonal;
            bool                                                       inertia_fused{false};

            std::size_t bytes() const {
                std::size_t ret = bsr ? bsr->bytes() : 0;
                if (inertia_fused && inertia_diagonal) ret += inertia_diagonal->size() * sizeof(real_t);
                for (const auto &entry : stores) ret += entry.second ? entry.second->bytes() : 0;
                return ret;
            }

            bool assembled() const { return options.store == InexactOptions::Store::BSR; }

            /// What update_bsr is told of the time scheme's inertia: whether it is its diagonal (a lumped
            /// mass, Op::hessian_is_diagonal) and how to add that diagonal to a vector; empty: none.
            struct Inertia {
                bool                              diagonal{false};
                std::function<int(real_t *const)> add_diagonal;
            };

            /// The `bsr` store: the operator's tangent on the space's node graph without its inertia
            /// (`assemble` leaves it out, InertiaLeftOut): the material's, whose rows sum to zero, so
            /// its difference form is exact. A diagonal inertia goes into the store's pass (inertia
            /// auto or fused; inertia_fused); any other the caller applies beside the store.
            int update_bsr(const std::shared_ptr<CRSGraph> &graph, const bool symmetric, const AssembleBSR &assemble) {
                return update_bsr(graph, symmetric, assemble, Inertia{});
            }

            int update_bsr(const std::shared_ptr<CRSGraph> &graph, const bool symmetric, const AssembleBSR &assemble,
                           const Inertia &inertia) {
                SFEM_TRACE_SCOPE("inexact::update_bsr");
                if (!bsr) {
                    SFEM_TRACE_SCOPE("inexact::make_bsr");
                    bsr = make_bsr(graph, symmetric);
                }
                if (bsr->update(assemble) != SFEM_SUCCESS) return SFEM_FAILURE;
                inertia_fused = inertia.add_diagonal && inertia.diagonal && options.inertia != InexactOptions::Inertia::SEPARATE;
                if (!inertia_fused) {
                    bsr->set_diagonal(nullptr);
                    return SFEM_SUCCESS;
                }
                const ptrdiff_t n = graph->n_nodes() * 3;
                if (!inertia_diagonal || (ptrdiff_t)inertia_diagonal->size() != n) inertia_diagonal = create_buffer<real_t>(n, space);
                sfem::blas<real_t>(space)->zeros(n, inertia_diagonal->data());
                if (inertia.add_diagonal(inertia_diagonal->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
                bsr->set_diagonal(inertia_diagonal->data());
                return SFEM_SUCCESS;
            }

            /// From this many block rows `auto` takes SELL-C-sigma on the device. SELL runs a thread per
            /// block row and the rows layout 4-8, so below it SELL does not fill an H100: on the cubes and
            /// the valve meshes (locality order) the rows layout is up to 5x faster under 4e5 dof, SELL
            /// 3-36% faster on the cubes from 3e6 dof, and the two within 4% on the valves at 4e6-7e6 dof
            /// (bench_inexact, 2026-11-18).
            static constexpr ptrdiff_t sell_min_block_rows = ptrdiff_t(1) << 20;

            /// The layout: SELL-C-sigma on the host; on the device the node graph's rows (rows), SELL-C-sigma
            /// (sell), or by the number of block rows (auto, sell_min_block_rows).
            std::unique_ptr<BsrStore> make_bsr(const std::shared_ptr<CRSGraph> &graph, const bool symmetric) const {
                if (space == EXECUTION_SPACE_DEVICE) {
#ifdef SFEM_ENABLE_CUDA
                    const bool sell = options.layout == InexactOptions::Layout::SELL ||
                                      (options.layout == InexactOptions::Layout::AUTO && graph->n_nodes() >= sell_min_block_rows);
                    return cuda::create_bsr_store(graph, options.encoding, symmetric, stream, sell);
#else
                    SFEM_ERROR("inexact: a device bsr store in a build without CUDA\n");
#endif
                }
                if (options.layout == InexactOptions::Layout::ROWS) {
                    SFEM_ERROR("inexact: the host's bsr store is SELL-C-sigma only (layout auto or sell)\n");
                }
                return create_bsr_store(graph, options.encoding, symmetric);
            }

            int diagonal_bsr(real_t *const diag) const {
                if (!bsr) {
                    SFEM_ERROR("inexact: the bsr store's diagonal is asked for before inexact_update\n");
                    return SFEM_FAILURE;
                }
                bsr->add_diagonal(diag);
                return SFEM_SUCCESS;
            }

            int apply_bsr(const real_t *const h, real_t *const out) const {
                if (!bsr) {
                    SFEM_ERROR("inexact: the bsr store is applied before inexact_update\n");
                    return SFEM_FAILURE;
                }
                if (extra) {
                    bsr->apply(h, out, *extra, overwrite);
                } else {
                    bsr->apply(h, out);
                }
                return SFEM_SUCCESS;
            }

            /// The extra diagonal blocks of apply_bsr, taken only by the assembled store.
            bool set_extra(const inexact::DiagonalExtra *const e, const bool overwrite_output) {
                if (!assembled()) return false;
                extra     = e;
                overwrite = overwrite_output;
                return true;
            }

            /// apply_bsr by the threads of a region (BsrStore::team_apply); inexact_update first.
            void team_apply_bsr(const Team &team, const real_t *const h, real_t *const out) const { bsr->team_apply(team, h, out); }

            /// One packed block as the traversal reads it, with this state's ghost buffer.
            template <class PackedMesh>
            PackedView packed_view(const PackedMesh &mesh, const int block, const int components) {
                PackedView view;
                view.n_packs             = mesh.n_packs(block);
                view.n_elements_per_pack = mesh.n_elements_per_pack(block);
                view.max_nodes_per_pack  = mesh.max_nodes_per_pack();
                view.elements            = mesh.elements(block)->data();
                view.owned_nodes_ptr     = mesh.owned_nodes_ptr(block)->data();
                view.n_ghost_entries     = mesh.n_ghost_entries(block);
                view.ghost_ptr           = mesh.ghost_ptr(block)->data();
                view.ghost_idx           = mesh.ghost_idx(block)->data();
                view.n_ghost_reduce_rows = mesh.n_ghost_reduce_rows(block);
                view.ghost_reduce_ptr    = mesh.ghost_reduce_ptr(block)->data();
                view.ghost_reduce_idx    = mesh.ghost_reduce_idx(block)->data();
                view.ghost_reduce_dest   = mesh.ghost_reduce_dest(block)->data();
                view.n_nodes             = mesh.mesh()->n_nodes();
                const std::size_t slots  = (std::size_t)components * std::max<ptrdiff_t>(1, view.n_ghost_entries);
                auto             &buffer = ghost[block];
                if (!buffer || buffer->size() < slots) buffer = create_host_buffer<real_t>(slots);
                view.ghost_buf = buffer->data();
                return view;
            }
        };

    }  // namespace inexact
}  // namespace sfem
