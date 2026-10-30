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
#include "sfem_QuantBSR.hpp"
#include "sfem_aliases.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_types.hpp"

namespace sfem {
    namespace inexact {

        template <class Q>
        class BsrStoreImpl final : public BsrStore {
        public:
            BsrStoreImpl(const std::shared_ptr<CRSGraph> &graph, std::string name)
                : graph_(graph),
                  name_(std::move(name)),
                  n_(graph->n_nodes()),
                  quant_(graph->n_nodes(), graph->rowptr()->data(), graph->colidx()->data()) {
                values_ = create_host_buffer<float>((size_t)graph->nnz() * 9);
            }

            int update(const AssembleBSR &assemble) override {
                const count_t *const rp = graph_->rowptr()->data();
                const idx_t *const   ci = graph_->colidx()->data();
                float *const         v  = values_->data();
                const ptrdiff_t      nv = (ptrdiff_t)values_->size();
#pragma omp parallel for
                for (ptrdiff_t k = 0; k < nv; ++k) v[k] = 0;
                if (assemble(rp, ci, smesh::SMESH_FLOAT32, v) != SFEM_SUCCESS) return SFEM_FAILURE;
                correction(rp, v);
                if constexpr (Q::upper) {
                    quant_.encode_in_place(rp, ci, v);
                } else {
                    quant_.encode(rp, ci, v);
                }
                return SFEM_SUCCESS;
            }

            void apply(const real_t *const x, real_t *const y) const override {
                quant_.template apply<float, true>(x, y);
                const ptrdiff_t n = (ptrdiff_t)rows_.size();
#pragma omp parallel for
                for (ptrdiff_t r = 0; r < n; ++r) {
                    const ptrdiff_t     i = rows_[(size_t)r];
                    const float *const  R = &blocks_[(size_t)r * 9];
                    const real_t *const xi = &x[i * 3];
                    for (int a = 0; a < 3; ++a) y[i * 3 + a] += R[a * 3 + 0] * xi[0] + R[a * 3 + 1] * xi[1] + R[a * 3 + 2] * xi[2];
                }
            }

            /// What the store holds: its structure and records, the corrections, and the single-precision
            /// assembly the records are encoded from (the upper form's records live inside it).
            std::size_t bytes() const override {
                std::size_t ret = quant_.bytes() + rows_.size() * sizeof(idx_t) + blocks_.size() * sizeof(float) +
                                  values_->size() * sizeof(float);
                if constexpr (Q::upper) ret -= quant_.record_bytes();
                return ret;
            }

            void add_diagonal(real_t *const diag) const override {
                quant_.add_diagonal(diag);
                for (size_t r = 0; r < rows_.size(); ++r)
                    for (int d = 0; d < 3; ++d) diag[(size_t)rows_[r] * 3 + d] += blocks_[r * 9 + d * 3 + d];
            }

            std::string describe() const override { return name_ + ", " + std::to_string(rows_.size()) + " corrected rows"; }

        private:
            std::shared_ptr<CRSGraph> graph_;
            std::string                      name_;
            ptrdiff_t                        n_{0};
            Q                                quant_;
            SharedBuffer<float>              values_;
            std::vector<idx_t>               rows_;
            std::vector<float>               blocks_;

            /// The rows whose blocks do not sum to zero beyond the assembly's rounding, in row order:
            /// each thread collects its own rows' sums, then they are joined in thread order.
            void correction(const count_t *const rp, const float *const v) {
                rows_.clear();
                blocks_.clear();
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
                    for (ptrdiff_t i = 0; i < n_; ++i) {
                        double R[9] = {0}, top = 0;
                        for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                            for (int x = 0; x < 9; ++x) {
                                R[x] += (double)v[(size_t)k * 9 + x];
                                top = std::max(top, std::fabs((double)v[(size_t)k * 9 + x]));
                            }
                        double sum = 0;
                        for (int x = 0; x < 9; ++x) sum = std::max(sum, std::fabs(R[x]));
                        // Single-precision sums of a row's few dozen blocks round to about 1e-6 of
                        // its largest entry; anything above that is a term that does not annihilate
                        // translations.
                        if (!(sum > 64 * FLT_EPSILON * top)) continue;
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

        /// The store for a symmetric or an unsymmetric operator on `graph`: int8 codes with a scale
        /// per block or f16 codes with one scale; the upper form with 16-bit column offsets for a
        /// symmetric one (about 10% smaller at the same speed on Grace), the full difference form
        /// otherwise.
        inline std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                          const bool                              int8,
                                                          const bool                              symmetric) {
            constexpr int VS = 16;
            if (symmetric) {
                if (int8) {
                    return std::make_unique<BsrStoreImpl<QuantBsr<int8_t, BsrForm::Upper, VS, uint16_t>>>(graph, "bsr i8 upper");
                }
                return std::make_unique<BsrStoreImpl<QuantBsr<half_t, BsrForm::Upper, VS, uint16_t>>>(graph, "bsr f16 upper");
            }
            if (int8) {
                return std::make_unique<BsrStoreImpl<QuantBsr<int8_t, BsrForm::Difference, VS, idx_t, false>>>(
                        graph, "bsr i8 difference");
            }
            return std::make_unique<BsrStoreImpl<QuantBsr<half_t, BsrForm::Difference, VS, idx_t, false>>>(
                    graph, "bsr f16 difference");
        }

    }  // namespace inexact
}  // namespace sfem
