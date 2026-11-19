#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_InexactInterface.hpp"
#include "sfem_InexactOptions.hpp"
#include "sfem_SELL.hpp"
#include "sfem_aliases.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_types.hpp"

namespace sfem {
    namespace inexact {

        template <class Q>
        class BsrStoreImpl final : public BsrStore {
        public:
            /// `windowed`: the operator's assembly visits only the rows of a window's graph (owner computes,
            /// owner_computes.hpp), so the store assembles by windows; otherwise in one piece, as a window
            /// would cost a pass over every element.
            BsrStoreImpl(const std::shared_ptr<CRSGraph> &graph, std::string name, const bool windowed)
                : graph_(graph),
                  name_(std::move(name)),
                  windowed_(windowed),
                  n_(graph->n_nodes()),
                  quant_(graph->n_nodes(), graph->rowptr()->data(), graph->colidx()->data()),
                  window_rp_(create_host_buffer<count_t>((size_t)graph->n_nodes() + 1)) {}

            /// Window by window (Q::update_windows, qbsr_window_count), as the device's store: the window's
            /// graph has the node graph's rows of the window and no entries in the others, which the
            /// assembly kernels skip; its single-precision values exist while the window is encoded.
            int update(const AssembleBSR &assemble) override {
                const count_t *const rp      = graph_->rowptr()->data();
                const idx_t *const   ci      = graph_->colidx()->data();
                count_t *const       wrp     = window_rp_->data();
                const auto           windows = quant_.update_windows(
                        rp,
                        windowed_ ? qbsr_window_count(9.0 * sizeof(float) * (double)rp[n_], (double)std::max<size_t>(1, quant_.record_bytes()), true)
                                  : 1);
                count_t largest = 0;
                for (size_t w = 0; w + 1 < windows.size(); ++w) largest = std::max(largest, rp[windows[w + 1]] - rp[windows[w]]);
                auto values = create_host_buffer<float>((size_t)largest * 9);
                float *const v = values->data();
                rows_.clear();
                blocks_.clear();
                quant_.reserve_records();
                for (size_t w = 0; w + 1 < windows.size(); ++w) {
                    const ptrdiff_t r0 = windows[w], r1 = windows[w + 1];
                    const count_t   k0 = rp[r0], nk = rp[r1] - k0;
                    {
                        SFEM_TRACE_SCOPE("inexact::BsrStore::assemble");
#pragma omp parallel for
                        for (ptrdiff_t i = 0; i <= n_; ++i) wrp[i] = i <= r0 ? 0 : i <= r1 ? rp[i] - k0 : nk;
#pragma omp parallel for
                        for (ptrdiff_t k = 0; k < (ptrdiff_t)nk * 9; ++k) v[k] = 0;
                        if (assemble(wrp, ci + k0, smesh::SMESH_FLOAT32, v) != SFEM_SUCCESS) return SFEM_FAILURE;
                    }
                    {
                        SFEM_TRACE_SCOPE("inexact::BsrStore::correction");
                        correction(rp, v, k0, r0, r1);
                    }
                    SFEM_TRACE_SCOPE("inexact::BsrStore::encode");
                    quant_.encode_rows(rp, ci, v, k0, r0, r1);
                }
                windows_ = (int)windows.size() - 1;
                return SFEM_SUCCESS;
            }

            void apply(const real_t *const x, real_t *const y) const override {
#pragma omp parallel
                team_apply(Team::current(n_ * 3), x, y);
            }

            /// The plain product, then the extra blocks node by node: the host's composed tangent
            /// runs the team product, so this is the reference, not a fast path.
            void apply(const real_t *const x, real_t *const y, const DiagonalExtra &extra, const bool overwrite) const override {
                const ptrdiff_t n3 = n_ * 3;
                if (overwrite) {
#pragma omp parallel for
                    for (ptrdiff_t k = 0; k < n3; ++k) y[k] = 0;
                }
                apply(x, y);
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_; ++i) {
                    if (extra.contact_of && extra.contact_of[i] >= 0) {
                        packed_sym_apply(&extra.blocks[(size_t)extra.contact_of[i] * 6], &x[i * 3], &y[i * 3]);
                    }
                    if (extra.shift != 0)
                        for (int d = 0; d < 3; ++d) y[i * 3 + d] += extra.shift * (extra.reference[i * 3 + d] * x[i * 3 + d]);
                }
            }

            void set_diagonal(const real_t *const d) override { diagonal_ = d; }

            /// The quantized product, each row's correction and diagonal added by the thread that
            /// completed it.
            void team_apply(const Team &team, const real_t *const x, real_t *const y) const override {
                const idx_t *const  rows = rows_.data();
                const ptrdiff_t     n    = (ptrdiff_t)rows_.size();
                const real_t *const d    = diagonal_;
                quant_.template team_apply<float, true>(team, x, y, [=](const ptrdiff_t b, const ptrdiff_t e) {
                    if (d)
                        for (ptrdiff_t k = b * 3; k < e * 3; ++k) y[k] += d[k] * x[k];
                    // The corrected rows are in row order.
                    const ptrdiff_t first = std::lower_bound(rows, rows + n, (idx_t)b) - rows;
                    const ptrdiff_t last  = std::lower_bound(rows, rows + n, (idx_t)e) - rows;
                    for (ptrdiff_t r = first; r < last; ++r) {
                        const ptrdiff_t     i  = rows[r];
                        const float *const  R  = &blocks_[(size_t)r * 9];
                        const real_t *const xi = &x[i * 3];
                        for (int a = 0; a < 3; ++a) y[i * 3 + a] += R[a * 3 + 0] * xi[0] + R[a * 3 + 1] * xi[1] + R[a * 3 + 2] * xi[2];
                    }
                });
            }

            /// What the store holds: its structure and records, the corrections and the window graph's row
            /// pointers.
            std::size_t bytes() const override {
                return quant_.bytes() + rows_.size() * sizeof(idx_t) + blocks_.size() * sizeof(float) + window_rp_->size() * sizeof(count_t);
            }

            void add_diagonal(real_t *const diag) const override {
                quant_.add_diagonal(diag);
                for (size_t r = 0; r < rows_.size(); ++r)
                    for (int d = 0; d < 3; ++d) diag[(size_t)rows_[r] * 3 + d] += blocks_[r * 9 + d * 3 + d];
            }

            std::string describe() const override {
                return name_ + ", " + std::to_string(windows_) + " windows, " + std::to_string(rows_.size()) + " corrected rows";
            }

        private:
            std::shared_ptr<CRSGraph> graph_;
            std::string                      name_;
            bool                             windowed_{false};
            ptrdiff_t                        n_{0};
            Q                                quant_;
            SharedBuffer<count_t>            window_rp_;
            int                              windows_{0};
            std::vector<idx_t>               rows_;
            std::vector<float>               blocks_;
            const real_t                    *diagonal_{nullptr};

            /// The rows of [r0, r1) whose blocks do not sum to zero beyond the assembly's rounding, appended in
            /// row order (v holds the window's blocks from k0): each thread collects its own rows' sums, then
            /// they are joined in thread order.
            void correction(const count_t *const rp, const float *const v, const count_t k0, const ptrdiff_t r0, const ptrdiff_t r1) {
                std::vector<std::vector<idx_t>> rows;
                std::vector<std::vector<float>> blocks;
#pragma omp parallel
                {
#ifdef _OPENMP
                    const int nt = omp_get_num_threads(), t = omp_get_thread_num();
#else
                    const int nt = 1, t = 0;
#endif
#pragma omp single
                    {
                        rows.resize((size_t)nt);
                        blocks.resize(rows.size());
                    }
#pragma omp for schedule(static)
                    for (ptrdiff_t i = r0; i < r1; ++i) {
                        double R[9] = {0}, top = 0;
                        for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                            for (int x = 0; x < 9; ++x) {
                                R[x] += (double)v[(size_t)(k - k0) * 9 + x];
                                top = std::max(top, std::fabs((double)v[(size_t)(k - k0) * 9 + x]));
                            }
                        double sum = 0;
                        for (int x = 0; x < 9; ++x) sum = std::max(sum, std::fabs(R[x]));
                        // Single-precision sums of a row's few dozen blocks round to about 1e-6 of
                        // its largest entry; anything above that is a term that does not annihilate
                        // translations.
                        if (!row_sum_is_a_term(sum, top)) continue;
                        rows[(size_t)t].push_back((idx_t)i);
                        for (int x = 0; x < 9; ++x) blocks[(size_t)t].push_back((float)R[x]);
                    }
                }
                for (size_t t = 0; t < rows.size(); ++t) {
                    rows_.insert(rows_.end(), rows[t].begin(), rows[t].end());
                    blocks_.insert(blocks_.end(), blocks[t].begin(), blocks[t].end());
                }
            }
        };

        /// The store for a symmetric or an unsymmetric operator on `graph`, SELL-C-sigma (C = 16): int8
        /// codes with a scale per block, f16 codes with one scale, or f32 values; the upper form with
        /// 16-bit column offsets for a symmetric one (about 10% smaller at the same speed on Grace),
        /// the full difference form otherwise.
        template <typename code_t>
        inline std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph, const bool symmetric, const char *const enc,
                                                          const bool windowed) {
            constexpr int VS = 16;
            if (symmetric) {
                return std::make_unique<BsrStoreImpl<SellMatrix<code_t, SellForm::Upper, VS, uint16_t>>>(
                        graph, std::string("bsr ") + enc + " upper", windowed);
            }
            return std::make_unique<BsrStoreImpl<SellMatrix<code_t, SellForm::Difference, VS, idx_t, false>>>(
                    graph, std::string("bsr ") + enc + " difference", windowed);
        }

        inline std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                          const InexactOptions::Encoding   encoding,
                                                          const bool                       symmetric,
                                                          const bool                       windowed) {
            switch (encoding) {
                case InexactOptions::Encoding::I8_BLOCK:
                    return create_bsr_store<int8_t>(graph, symmetric, "i8", windowed);
                case InexactOptions::Encoding::F16:
                    return create_bsr_store<half_t>(graph, symmetric, "f16", windowed);
                case InexactOptions::Encoding::F32:
                    return create_bsr_store<float>(graph, symmetric, "f32", windowed);
                default:
                    SFEM_ERROR("inexact: the bsr store takes f32, f16 or i8\n");
                    return nullptr;
            }
        }

    }  // namespace inexact
}  // namespace sfem
