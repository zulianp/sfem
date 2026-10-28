// The application kernel of the compressed quadrature-point partial assembly
// (docs/compressed_partial_assembly, Algorithms 2 and 3).
//
// One kernel for every element and encoding: the element supplies its fixed stages (stage_in,
// stage_out) and the pairing of stored tensors with gradient vectors; the store supplies decoding.
// Points (HEX8): tensor j = q contracts gradient q into output q.  Moments (TET10, TET4 with
// r = 1): tensor j = (a, b) contracts gradient a into output b and, when a != b, b into a.
//
// Template parameters: the store, the arithmetic type `tau` of the contraction, and nothing
// else -- the lane width is the store's block, because the blocked layout is laid out by it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "qp_contract.hpp"
#include "qp_element.hpp"
#include "qp_store.hpp"
#include "ea_store.hpp"

namespace qp {

// How the decode and the contraction are vectorised over the lanes of a block:
//   Registers  one lane loop around the whole decode and contraction, the 45 decoded values
//              held per lane (what clang vectorises for the plain codecs)
//   Lanes      lane loop innermost around every statement, the decoded block in L1 (what every
//              compiler vectorises, whatever the scheme)
enum class Form { Registers, Lanes };

// The element block: gathered increment in, element outputs out (scaled), for elements
// evb .. evb + ne of the store.  Shared by the atomic and the packed application.
template <class St, typename tau, Form FORM>
static SFEM_INLINE void element_block(const St &st,
                                      const ptrdiff_t evb,
                                      const int ne,
                                      const double bh[3][St::Element::N][St::VS],
                                      double out[3][St::Element::N][St::VS]) {
    using El = typename St::Element;
    static constexpr int VS = St::VS;
    static constexpr int N = El::N;
    static constexpr int R = El::R;
    {
        {
        tau P[R][9][VS];
        El::template stage_in<double, tau, VS>(ne, bh, P);

        tau Y[R][9][VS];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < 9; ++c)
#pragma omp simd
                for (int lane = 0; lane < VS; ++lane) Y[r][c][lane] = tau(0);

        // What decoding needs besides the codes, read once per block into local arrays: the
        // per-component scales or the mean, and the element scale.  Decoding is eq. "decode" of
        // the paper; the plain scaled codecs leave their scale to the outputs.
        static constexpr Scheme SCH = St::scheme;
        static constexpr bool blocked = St::layout == Layout::Blocked;
        tau A[(SCH == Scheme::PerComponent || SCH == Scheme::MeanDeviation) ? 45 : 1][VS];
        tau sc[VS];
        if constexpr (SCH == Scheme::PerComponent || SCH == Scheme::MeanDeviation) {
            const float *const RSTR src = SCH == Scheme::PerComponent ? st.scales.data() : st.aux.data();
            for (int c = 0; c < 45; ++c)
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) A[c][lane] = tau(src[st.comp_index(evb, lane, c)]);
        }
        if constexpr (SCH == Scheme::MeanDeviation) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) sc[lane] = tau(st.scales[(size_t)evb + lane]);
        }
        using code_t = typename St::code_type;
        const ptrdiff_t cs = blocked ? (ptrdiff_t)VS : st.stride;
        for (int j = 0; j < El::NV; ++j) {
            const int a = El::PA[j], b = El::PB[j];
            const tau xj = tau(El::X[j]);
            const code_t *const RSTR cj =
                    st.codes.data() + (blocked ? ((size_t)(evb / VS) * El::NV + j) * 45 * VS + (size_t)(evb % VS)
                                               : (size_t)j * 45 * st.stride + evb);
            if constexpr (FORM == Form::Registers) {
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    tau Sl[45];
                    for (int c = 0; c < 45; ++c) {
                        const tau v = tau(cj[c * cs + lane]);
                        if constexpr (SCH == Scheme::Plain) {
                            Sl[c] = v;
                        } else if constexpr (SCH == Scheme::PerComponent) {
                            Sl[c] = A[c][lane] * v;
                        } else {
                            Sl[c] = xj * A[c][lane] + sc[lane] * v;
                        }
                    }
                    tau g[9], y[9];
                    for (int c = 0; c < 9; ++c) {
                        g[c] = P[a][c][lane];
                        y[c] = Y[b][c][lane];
                    }
                    contract(Sl, g, y);
                    for (int c = 0; c < 9; ++c) Y[b][c][lane] = y[c];
                    if (a != b) {
                        for (int c = 0; c < 9; ++c) {
                            g[c] = P[b][c][lane];
                            y[c] = Y[a][c][lane];
                        }
                        contract(Sl, g, y);
                        for (int c = 0; c < 9; ++c) Y[a][c][lane] = y[c];
                    }
                }
                continue;
            }
            // Lane loop innermost, one statement per loop: the shape every compiler vectorises.
            tau S[45][VS];
            for (int c = 0; c < 45; ++c) {
                const code_t *const RSTR cjc = cj + c * cs;
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    const tau v = tau(cjc[lane]);
                    if constexpr (SCH == Scheme::Plain) {
                        S[c][lane] = v;
                    } else if constexpr (SCH == Scheme::PerComponent) {
                        S[c][lane] = A[c][lane] * v;
                    } else {
                        S[c][lane] = xj * A[c][lane] + sc[lane] * v;
                    }
                }
            }
            contract_lanes<tau, VS>(ne, S, P[a], Y[b]);
            if (a != b) contract_lanes<tau, VS>(ne, S, P[b], Y[a]);
        }

        El::template stage_out<tau, double, VS>(ne, Y, out);

        if constexpr (St::scale_on_outputs) {
            for (int i = 0; i < 3; ++i)
                for (int p = 0; p < N; ++p)
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) out[i][p][lane] *= st.output_scale(evb + lane);
        }
        (void)N;
        }
    }
}

// The element block of either store: quadrature-point tensors or element matrices.
template <class St, typename tau, Form FORM>
static SFEM_INLINE void any_block(const St &st,
                                  const ptrdiff_t evb,
                                  const int ne,
                                  const double bh[3][St::Element::N][St::VS],
                                  double out[3][St::Element::N][St::VS]) {
    if constexpr (St::is_element_matrix)
        st.template block<tau>(evb, ne, bh, out);
    else
        element_block<St, tau, FORM>(st, evb, ne, bh, out);
}

// Unordered traversal, atomic scatter.
template <class St, typename tau, Form FORM = Form::Lanes>
int apply(const St &st,
          const ptrdiff_t nelements,
          idx_t **const RSTR elements,
          const ptrdiff_t h_stride,
          const double *const RSTR hx,
          const double *const RSTR hy,
          const double *const RSTR hz,
          const ptrdiff_t out_stride,
          double *const RSTR ox,
          double *const RSTR oy,
          double *const RSTR oz) {
    using El = typename St::Element;
    static constexpr int VS = St::VS;
    static constexpr int N = El::N;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int ne = (int)((nelements - evb) < (ptrdiff_t)VS ? (nelements - evb) : (ptrdiff_t)VS);
        idx_t bev[N][VS];
        double bh[3][N][VS];
        for (int p = 0; p < N; ++p) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                const idx_t v = elements[p][evb + lane];
                bev[p][lane] = v;
                bh[0][p][lane] = hx[v * h_stride];
                bh[1][p][lane] = hy[v * h_stride];
                bh[2][p][lane] = hz[v * h_stride];
            }
        }
        double out[3][N][VS];
        any_block<St, tau, FORM>(st, evb, ne, bh, out);
        for (int p = 0; p < N; ++p) {
            for (int lane = 0; lane < ne; ++lane) {
                const ptrdiff_t v = (ptrdiff_t)bev[p][lane] * out_stride;
#pragma omp atomic update
                ox[v] += out[0][p][lane];
#pragma omp atomic update
                oy[v] += out[1][p][lane];
#pragma omp atomic update
                oz[v] += out[2][p][lane];
            }
        }
    }
    return SFEM_SUCCESS;
}

// The packed layout's two-pass application: each pack gathers its owned and ghost nodes into
// thread-local scratch, accumulates there without atomics, writes its owned nodes, and parks its
// ghost contributions; a second pass reduces the ghosts.  Deterministic.  `Packed` is any type
// with the fields of the spike's PackedLayout; packs may start anywhere.  The scratch is kept per
// thread across calls and is node-major, a node's three components in one cache line for the
// gather and the scatter (measured against component-major scratch allocated per call: 5-15%
// faster traversal on Grace).
template <class St, typename tau, Form FORM = Form::Lanes, class Packed>
int apply_packed(const St &st,
                 const Packed &pk,
                 uint16_t **const RSTR elements,
                 const ptrdiff_t nelements,
                 const ptrdiff_t h_stride,
                 const double *const RSTR hx,
                 const double *const RSTR hy,
                 const double *const RSTR hz,
                 const ptrdiff_t out_stride,
                 double *const RSTR ox,
                 double *const RSTR oy,
                 double *const RSTR oz,
                 double *const RSTR ghost_buf) {
    using El = typename St::Element;
    static constexpr int VS = St::VS;
    static constexpr int N = El::N;
    const double *const hc[3] = {hx, hy, hz};
    double *const oc[3] = {ox, oy, oz};
    const ptrdiff_t M = pk.max_nodes_per_pack;
    const ptrdiff_t G = pk.n_ghost_entries;

    // Index of component d of pack-local node v in the scratch.
    const auto at = [](const ptrdiff_t v, const int d) -> ptrdiff_t { return 3 * v + d; };
#pragma omp parallel
    {
        static thread_local std::vector<double> keep_h, keep_o;
        if (keep_h.size() < (size_t)3 * M) {
            keep_h.resize((size_t)3 * M);
            keep_o.resize((size_t)3 * M);
        }
        double *const RSTR ph = keep_h.data();
        double *const RSTR po = keep_o.data();
#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < pk.n_packs; ++pack) {
            const ptrdiff_t e_start = pack * pk.n_elements_per_pack;
            const ptrdiff_t e_end = std::min(nelements, (pack + 1) * pk.n_elements_per_pack);
            const ptrdiff_t own0 = pk.owned_nodes_ptr[pack];
            const ptrdiff_t n_own = pk.owned_nodes_ptr[pack + 1] - own0;
            const ptrdiff_t g0 = pk.ghost_ptr[pack];
            const ptrdiff_t n_ghost = pk.ghost_ptr[pack + 1] - g0;
            for (ptrdiff_t k = 0; k < 3 * (n_own + n_ghost); ++k) po[k] = 0;
            for (ptrdiff_t k = 0; k < n_own; ++k)
                for (int d = 0; d < 3; ++d) ph[3 * k + d] = hc[d][(own0 + k) * h_stride];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t g = (ptrdiff_t)pk.ghost_idx[g0 + k] * h_stride;
                for (int d = 0; d < 3; ++d) ph[3 * (n_own + k) + d] = hc[d][g];
            }
            // Packs need not start on a store block: blocks are cut at multiples of VS so that
            // none straddles two store blocks.
            for (ptrdiff_t evb = e_start; evb < e_end;) {
                const ptrdiff_t room = (ptrdiff_t)VS - evb % VS;
                const int ne = (int)((e_end - evb) < room ? (e_end - evb) : room);
                uint16_t bev[N][VS];
                double bh[3][N][VS];
                for (int p = 0; p < N; ++p) {
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
                        const uint16_t v = elements[p][evb + lane];
                        bev[p][lane] = v;
                        bh[0][p][lane] = ph[at(v, 0)];
                        bh[1][p][lane] = ph[at(v, 1)];
                        bh[2][p][lane] = ph[at(v, 2)];
                    }
                }
                double out[3][N][VS];
                any_block<St, tau, FORM>(st, evb, ne, bh, out);
                for (int p = 0; p < N; ++p)
                    for (int lane = 0; lane < ne; ++lane) {
                        const uint16_t v = bev[p][lane];
                        po[at(v, 0)] += out[0][p][lane];
                        po[at(v, 1)] += out[1][p][lane];
                        po[at(v, 2)] += out[2][p][lane];
                    }
                evb += ne;
            }
            for (int d = 0; d < 3; ++d) {
                for (ptrdiff_t k = 0; k < n_own; ++k) oc[d][(own0 + k) * out_stride] += po[at(k, d)];
                for (ptrdiff_t k = 0; k < n_ghost; ++k) ghost_buf[d * G + g0 + k] = po[at(n_own + k, d)];
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < pk.n_ghost_reduce_rows; ++row) {
        const ptrdiff_t dest = pk.ghost_reduce_dest[row];
        for (int d = 0; d < 3; ++d) {
            double sum = 0;
            for (ptrdiff_t j = pk.ghost_reduce_ptr[row]; j < pk.ghost_reduce_ptr[row + 1]; ++j)
                sum += ghost_buf[d * G + pk.ghost_reduce_idx[j]];
            oc[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

}  // namespace qp
