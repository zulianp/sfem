#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "sfem_ElementMatrixStore.hpp"
#include "sfem_InexactOptions.hpp"
#include "sfem_InexactSetup.hpp"
#include "sfem_InexactTraversal.hpp"
#include "sfem_PointStore.hpp"
#include "sfem_QuantBSRStore.hpp"

namespace sfem {

    /// One domain's stored tangent, whatever its encoding: what a generated operator holds
    /// between `inexact_update` and the applies of a Newton step.
    class InexactStore {
    public:
        virtual ~InexactStore() = default;

        /// Evaluates the tangent at every point from the state and encodes it.
        virtual void update(const inexact::SetupInput &input) = 0;

        /// `out += A h`, interleaved vectors of `stride` components.  Packed when `packed` is
        /// given and the options allow it, atomic otherwise.
        virtual void apply(const ptrdiff_t                  nelements,
                           const idx_t *const *const        elements,
                           const inexact::PackedView *const packed,
                           const ptrdiff_t                  stride,
                           const real_t *const              h,
                           real_t *const                    out) const = 0;

        /// diag += the stored operator's diagonal, interleaved with `stride` components.
        virtual void diagonal(const ptrdiff_t           nelements,
                              const idx_t *const *const elements,
                              const ptrdiff_t           stride,
                              real_t *const             diag) const = 0;

        virtual std::size_t bytes() const    = 0;
        virtual std::string describe() const = 0;
    };

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
                if (packed && options_.traversal != InexactOptions::Traversal::ATOMIC) {
                    apply_packed<Store, tau>(store_, *packed, nelements, stride, h, out);
                    return;
                }
                if (options_.traversal == InexactOptions::Traversal::PACKED) {
                    SFEM_ERROR("inexact: traversal packed asked for, but the space has no packed mesh\n");
                }
                apply_atomic<Store, tau>(store_, nelements, elements, stride, h, out);
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
            InexactOptions                                             options;
            std::map<const void *, std::unique_ptr<InexactStore>>      stores;
            std::map<int, SharedBuffer<real_t>>                        ghost;
            std::unique_ptr<BsrStore>                                  bsr;
            const real_t                                              *state{nullptr};

            std::size_t bytes() const {
                std::size_t ret = bsr ? bsr->bytes() : 0;
                for (const auto &entry : stores) ret += entry.second ? entry.second->bytes() : 0;
                return ret;
            }

            bool assembled() const { return options.store == InexactOptions::Store::BSR; }

            /// The `bsr` store: the operator's whole tangent on the space's node graph, its inertia
            /// term included, so the caller adds nothing to its apply.
            int update_bsr(const std::shared_ptr<CRSGraph> &graph, const bool symmetric, const AssembleBSR &assemble) {
                if (!bsr) bsr = create_bsr_store(graph, options.encoding == InexactOptions::Encoding::I8_BLOCK, symmetric);
                return bsr->update(assemble);
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
                bsr->apply(h, out);
                return SFEM_SUCCESS;
            }

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
                const std::size_t slots  = (std::size_t)components * std::max<ptrdiff_t>(1, view.n_ghost_entries);
                auto             &buffer = ghost[block];
                if (!buffer || buffer->size() < slots) buffer = create_host_buffer<real_t>(slots);
                view.ghost_buf = buffer->data();
                return view;
            }
        };

    }  // namespace inexact
}  // namespace sfem
