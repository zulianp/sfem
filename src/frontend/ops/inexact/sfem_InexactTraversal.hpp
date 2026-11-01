#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "packed_thread_scratch.hpp"
#include "sfem_InexactElement.hpp"
#include "sfem_InexactInterface.hpp"
#include "sfem_base.hpp"

namespace sfem {
    namespace inexact {

        /// Unordered traversal, atomic scatter: `out += A h` over the elements of one block.
        template <class Store, typename tau>
        void apply_atomic(const Store              &st,
                          const ptrdiff_t           nelements,
                          const idx_t *const *const elements,
                          const ptrdiff_t           stride,
                          const real_t *const RSTR  h,
                          real_t *const RSTR        out) {
            using El                = typename Store::Element;
            static constexpr int VS = Store::VS;
            static constexpr int N  = El::N;
            static constexpr int D  = El::DIM;

#pragma omp parallel for schedule(static)
            for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
                const int ne = (int)std::min<ptrdiff_t>(VS, nelements - evb);
                idx_t     bev[N][VS];
                real_t    bh[D][N][VS];
                for (int s = 0; s < N; ++s) {
                    const idx_t *const RSTR nodes = elements[ElementTables<El>::shape_node(s)];
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
                        const idx_t v = nodes[evb + lane];
                        bev[s][lane]  = v;
                        for (int c = 0; c < D; ++c) bh[c][s][lane] = h[(ptrdiff_t)v * stride + c];
                    }
                }
                real_t bout[D][N][VS];
                st.template block<tau>(evb, ne, bh, bout);
                for (int s = 0; s < N; ++s) {
                    for (int lane = 0; lane < ne; ++lane) {
                        const ptrdiff_t v = (ptrdiff_t)bev[s][lane] * stride;
                        for (int c = 0; c < D; ++c) {
#pragma omp atomic update
                            out[v + c] += bout[c][s][lane];
                        }
                    }
                }
            }
        }

        /// The packed layout's two-pass traversal: each pack gathers its owned and ghost nodes
        /// into the thread's scratch, accumulates there without atomics, writes its owned nodes,
        /// and parks its ghost contributions; a second pass reduces the ghosts.  Deterministic.
        /// The scratch is node-major -- a node's components in one cache line for the gather and
        /// the scatter -- and kept per thread across calls (measured against component-major
        /// scratch allocated per call: 5-15% faster on Grace).
        template <class Store, typename tau>
        void apply_packed(const Store             &st,
                          const PackedView        &pk,
                          const ptrdiff_t          nelements,
                          const ptrdiff_t          stride,
                          const real_t *const RSTR h,
                          real_t *const RSTR       out) {
            using El                = typename Store::Element;
            static constexpr int VS = Store::VS;
            static constexpr int N  = El::N;
            static constexpr int D  = El::DIM;
            const ptrdiff_t      M  = pk.max_nodes_per_pack;
            const ptrdiff_t      G  = pk.n_ghost_entries;

#pragma omp parallel
            {
                real_t *const RSTR ph = codegen::thread_scratch<real_t>(0, (size_t)D * M);
                real_t *const RSTR po = codegen::thread_scratch<real_t>(1, (size_t)D * M);
#pragma omp for schedule(static)
                for (ptrdiff_t pack = 0; pack < pk.n_packs; ++pack) {
                    const ptrdiff_t e_start = pack * pk.n_elements_per_pack;
                    const ptrdiff_t e_end   = std::min(nelements, (pack + 1) * pk.n_elements_per_pack);
                    const ptrdiff_t own0    = pk.owned_nodes_ptr[pack];
                    const ptrdiff_t n_own   = pk.owned_nodes_ptr[pack + 1] - own0;
                    const ptrdiff_t g0      = pk.ghost_ptr[pack];
                    const ptrdiff_t n_ghost = pk.ghost_ptr[pack + 1] - g0;
                    for (ptrdiff_t k = 0; k < D * (n_own + n_ghost); ++k) po[k] = 0;
                    for (ptrdiff_t k = 0; k < n_own; ++k)
                        for (int c = 0; c < D; ++c) ph[D * k + c] = h[(own0 + k) * stride + c];
                    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                        const ptrdiff_t g = (ptrdiff_t)pk.ghost_idx[g0 + k] * stride;
                        for (int c = 0; c < D; ++c) ph[D * (n_own + k) + c] = h[g + c];
                    }
                    // Packs need not start on a store block: blocks are cut at multiples of VS so
                    // that none straddles two store blocks.
                    for (ptrdiff_t evb = e_start; evb < e_end;) {
                        const ptrdiff_t room = (ptrdiff_t)VS - evb % VS;
                        const int       ne   = (int)std::min(e_end - evb, room);
                        uint16_t        bev[N][VS];
                        real_t          bh[D][N][VS];
                        for (int s = 0; s < N; ++s) {
                            const uint16_t *const RSTR nodes = pk.elements[ElementTables<El>::shape_node(s)];
#pragma omp simd
                            for (int lane = 0; lane < ne; ++lane) {
                                const uint16_t v = nodes[evb + lane];
                                bev[s][lane]     = v;
                                // The node's offset once, in ptrdiff_t: the components are then
                                // constant offsets from it.  `D * v + c` in int has gcc build a
                                // sign-extended index vector per component and move every lane
                                // out of it (TET4 f16 points on Grace: the traversal 9% slower).
                                const ptrdiff_t at = D * (ptrdiff_t)v;
                                for (int c = 0; c < D; ++c) bh[c][s][lane] = ph[at + c];
                            }
                        }
                        real_t bout[D][N][VS];
                        st.template block<tau>(evb, ne, bh, bout);
                        for (int s = 0; s < N; ++s)
                            for (int lane = 0; lane < ne; ++lane) {
                                const ptrdiff_t at = D * (ptrdiff_t)bev[s][lane];
                                for (int c = 0; c < D; ++c) po[at + c] += bout[c][s][lane];
                            }
                        evb += ne;
                    }
                    for (int c = 0; c < D; ++c) {
                        for (ptrdiff_t k = 0; k < n_own; ++k) out[(own0 + k) * stride + c] += po[D * k + c];
                        for (ptrdiff_t k = 0; k < n_ghost; ++k) pk.ghost_buf[c * G + g0 + k] = po[D * (n_own + k) + c];
                    }
                }
            }

#pragma omp parallel for schedule(static)
            for (ptrdiff_t row = 0; row < pk.n_ghost_reduce_rows; ++row) {
                const ptrdiff_t dest = pk.ghost_reduce_dest[row];
                for (int c = 0; c < D; ++c) {
                    real_t sum = 0;
                    for (ptrdiff_t j = pk.ghost_reduce_ptr[row]; j < pk.ghost_reduce_ptr[row + 1]; ++j)
                        sum += pk.ghost_buf[c * G + pk.ghost_reduce_idx[j]];
                    out[dest * stride + c] += sum;
                }
            }
        }

    }  // namespace inexact
}  // namespace sfem
