#pragma once

// Element-wise Galerkin coarsening for the semi-structured CVFEM operator.
//
// The coarse operator is R A P. Because A = sum_e A_e and the prolongation's support does
// not leave the macro-element that carries it, the sum commutes with the triple product:
//
//     R A P = sum_e P_e^T A_e P_e
//
// so every coarse operator in the hierarchy can be built one macro-element at a time, with
// no global sparse matrix, no SpGEMM and -- the point -- no probing. Element locality is not
// assumed here; the driver gates it (SFEM_GMG_CHECK=1, "element-locality"), and it holds
// because Rhie-Chow freezes the nodal pressure gradient from the state rather than
// differentiating it. A formulation that differentiated it would couple across macro-element
// faces and silently invalidate everything below.
//
// Three structural facts make this cheap.
//
// 1. The micro-cell *matrix* is already available. Passing identity slots to
//    cvfem_hex8_ns_upwind_jacobian_add_slots writes a dense 8x8-block (32x32) cell matrix
//    into a local buffer -- the trick assemble_block_diag already uses. So no level needs
//    the operator's action probed with unit vectors: the entries come out directly.
//
// 2. A micro-cell's coarse support is exactly eight nodes. The cell spans fine indices
//    [xi, xi+1], whose coarse floors differ by at most one, so per axis it reaches coarse
//    indices {ax, ax+1} and no more -- for *any* ratio q, not just 2:1. The triple product
//    is therefore a fixed 8x8 -> 8x8 contraction rather than anything that grows with L.
//
// 3. The interpolation weights depend only on the offset class (xi%q, yi%q, zi%q). There are
//    q^3 classes, shared by every element and every macro-element, so the weights are a
//    small table built once -- never an array indexed per entry. This is the same principle
//    cvfem_ss_transfer.hpp applies to the prolongation itself.
//
// Levels do not chain. Piecewise-linear interpolation on nested uniform lattices composes to
// the direct interpolation (level 4 <- 2 <- 1 gives fine node 1 the weights 3/4, 1/4, which
// is what the direct level-4 <- level-1 map gives), so a level at ratio q is built straight
// from the fine micro-cells with q in the weight table. No error accumulates through repeated
// Galerkin products and a level can be rebuilt without touching its neighbours.
//
// The local coarse matrix is stored as a 27-point lattice stencil, a fixed slot per offset,
// because a coarse node couples only to its 3x3x3 lattice neighbourhood: it belongs to at
// most eight coarse cells and each reaches one node away. Fixed slots mean the triple
// product scatters with no search and no indirection, and the 4x4 blocks stay contiguous.
//
// Cost, per micro-cell, in 4x4-block multiply-accumulates: the two stages are 8*27 and 27*8,
// so 432 against the 1024 a dense 8x8 -> 8x8 contraction would need. The 27 is the average
// row count of the prolongation restricted to a cell (one corner interpolates from 1 coarse
// node, three from 2, three from 4, one from 8: 1+6+12+8 = 27).

#include "cvfem_sshex8_ns.hpp"

#include "sfem_Function.hpp"
#include "smesh_sshex8.hpp"

#include <algorithm>
#include <vector>

namespace cvfem_ss {

    // Local hex8 corner ordering, matching sscvfem_corner_offsets: the fine micro-cell's
    // corners and the coarse patch's corners are indexed the same way, so the contraction
    // below maps corner index to corner index.
    static constexpr int GAL_CORNER[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                             {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

    static SFEM_INLINE int gal_corner_id(const int i, const int j, const int k) {
        // Inverse of GAL_CORNER.
        static constexpr int id[2][2][2] = {{{0, 4}, {3, 7}}, {{1, 5}, {2, 6}}};
        return id[i][j][k];
    }

    // Stencil slot for the lattice offset between two patch corners. Both lie in the same
    // 2x2x2 patch, so every offset is in {-1,0,1} and the slot is always valid.
    static SFEM_INLINE int gal_slot(const int a, const int b) {
        const int di = GAL_CORNER[b][0] - GAL_CORNER[a][0];
        const int dj = GAL_CORNER[b][1] - GAL_CORNER[a][1];
        const int dk = GAL_CORNER[b][2] - GAL_CORNER[a][2];
        return (dk + 1) * 9 + (dj + 1) * 3 + (di + 1);
    }

    // The prolongation restricted to one micro-cell: fine corner -> coarse patch corner.
    // One of these per offset class.
    struct CellQ {
        int      nnz[8];
        int      col[8][8];
        scalar_t w[8][8];

        // The same map transposed: for each coarse patch corner, the fine corners feeding
        // it. Stage 1 of the triple product contracts over the fine index for a fixed coarse
        // one, so with the transpose it writes each output block once instead of zeroing all
        // 64 of them and accumulating -- 1024 scalars of zeroing saved per micro-cell, for
        // the same arithmetic.
        int      tnnz[8];
        int      trow[8][8];
        scalar_t tw[8][8];
    };

    // Build the q^3 offset classes. Per axis, fine index xi+ai with xi = ax*q + r sits at
    // coarse offset (r+ai)/q with remainder (r+ai)%q, so the weights are (1 - rr/q, rr/q) on
    // consecutive coarse nodes -- and a single weight of 1 when the fine node is on-lattice.
    inline void build_cellq_table(const int q, std::vector<CellQ> &tab) {
        tab.assign((size_t)(q * q * q), CellQ{});

        // Per-axis: for remainder r and corner offset ai, the columns and weights.
        auto axis = [q](const int r, const int ai, int c[2], scalar_t w[2]) -> int {
            const int s = r + ai;
            const int c0 = s / q, rr = s % q;
            if (rr == 0) {
                c[0] = c0;
                w[0] = scalar_t(1);
                return 1;
            }
            c[0] = c0;
            c[1] = c0 + 1;
            w[0] = scalar_t(1) - (scalar_t)rr / (scalar_t)q;
            w[1] = (scalar_t)rr / (scalar_t)q;
            return 2;
        };

        for (int rz = 0; rz < q; ++rz)
            for (int ry = 0; ry < q; ++ry)
                for (int rx = 0; rx < q; ++rx) {
                    CellQ &Q = tab[(size_t)((rz * q + ry) * q + rx)];
                    for (int a = 0; a < 8; ++a) {
                        const int ai = GAL_CORNER[a][0], aj = GAL_CORNER[a][1], ak = GAL_CORNER[a][2];
                        int       cx[2], cy[2], cz[2];
                        scalar_t  wx[2], wy[2], wz[2];
                        const int nx = axis(rx, ai, cx, wx);
                        const int ny = axis(ry, aj, cy, wy);
                        const int nz = axis(rz, ak, cz, wz);

                        int n = 0;
                        for (int k = 0; k < nz; ++k)
                            for (int j = 0; j < ny; ++j)
                                for (int i = 0; i < nx; ++i) {
                                    Q.col[a][n] = gal_corner_id(cx[i], cy[j], cz[k]);
                                    Q.w[a][n]   = wx[i] * wy[j] * wz[k];
                                    ++n;
                                }
                        Q.nnz[a] = n;
                    }

                    for (int b = 0; b < 8; ++b) Q.tnnz[b] = 0;
                    for (int a = 0; a < 8; ++a)
                        for (int k = 0; k < Q.nnz[a]; ++k) {
                            const int b          = Q.col[a][k];
                            Q.trow[b][Q.tnnz[b]] = a;
                            Q.tw[b][Q.tnnz[b]]   = Q.w[a][k];
                            ++Q.tnnz[b];
                        }

                    // Every coarse patch corner is reached by at least one fine corner of the
                    // cell, for any q: per axis the two fine positions are r and r+1, and one
                    // of them always carries weight onto each of the two coarse nodes. Stage 1
                    // relies on that to seed its accumulator with trow[b][0] instead of
                    // zeroing, so the property is checked once here, at setup, rather than
                    // branched on per micro-cell.
                    for (int b = 0; b < 8; ++b)
                        if (!Q.tnnz[b])
                            SFEM_ERROR("build_cellq_table: coarse corner %d unreached at q=%d\n", b, q);
                }
    }

    // One coarse level, held as element matrices in the 27-point lattice stencil.
    struct GalerkinLevel {
        int       q{1};
        int       Lc{0};
        int       nc{0};       // (Lc+1)^3, coarse nodes per macro-element
        ptrdiff_t nmacro{0};
        ptrdiff_t n_coarse{0};  // global coarse nodes

        std::vector<CellQ>    qtab;

        // Local coarse matrices for the element chunk [e0, e1), as (e - e0) * nc * 27 blocks.
        //
        // Chunked rather than held for the whole mesh because the full array is
        // nmacro * nc * 27 * 16 scalars -- 272 MB at 108 macro-elements with a level-8 coarse
        // lattice, for a transient. Determinism survives: a block's contributions are summed
        // in chunk order, and within a chunk in inverted-index order, both fixed.
        std::vector<scalar_t> C;
        ptrdiff_t             e0{0}, e1{0};

        // Optional fine-side constraint mask, one byte per fine dof (node * 4 + component),
        // empty when unconstrained. Zeroing the constrained *columns* of each micro-cell
        // matrix gives P^T (A Z) P directly, which is the composite the probe path recovered
        // and the rap path reproduces as mask_block_columns followed by patch_identity_rows.
        // Row masking is not done here: the caller patches the coarse identity rows, which is
        // what the validated formula does and what keeps the two paths comparable.
        std::vector<uint8_t> fine_constrained;

        // Coarse global id of local coarse node a in element e. Filled by the caller from
        // the coarse space's own connectivity, exactly as cvfem_ss_transfer does.
        std::vector<smesh::idx_t> gid;  // nmacro * nc

        SFEM_INLINE const scalar_t *block(const ptrdiff_t e, const int a, const int s) const {
            return C.data() + ((((size_t)(e - e0) * (size_t)nc + (size_t)a) * 27) + (size_t)s) * 16;
        }
    };

    inline void galerkin_init(const SSMeshData &d, const int q, GalerkinLevel &g) {
        if (q < 1 || d.level % q) SFEM_ERROR("galerkin_init: level %d is not divisible by q=%d\n", d.level, q);
        g.q      = q;
        g.Lc     = d.level / q;
        g.nc     = (g.Lc + 1) * (g.Lc + 1) * (g.Lc + 1);
        g.nmacro = d.nmacro;
        build_cellq_table(q, g.qtab);
    }

    // Elements per assembly chunk, chosen so the staging buffer stays near 32 MiB.
    inline ptrdiff_t galerkin_chunk(const GalerkinLevel &g) {
        const size_t per = (size_t)g.nc * 27 * 16 * sizeof(scalar_t);
        const size_t cap = 32u << 20;
        return std::max<ptrdiff_t>(1, std::min<ptrdiff_t>(g.nmacro, (ptrdiff_t)(cap / std::max<size_t>(1, per))));
    }

    // The kernel. Mirrors sscvfem_apply_macro_local_hoisted's gather and geometry exactly --
    // the same hoisted micro-cell-0 adjugate, the same Rhie-Chow struct -- so that what is
    // assembled here is the operator that path applies, and the q=1 gate can say so.
    inline SFEM_NOINLINE void galerkin_assemble(const SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                GalerkinLevel &g, const ptrdiff_t e0, const ptrdiff_t e1) {
        SFEM_TRACE_SCOPE("cvfem_ss::galerkin_assemble");
        const int L   = d.level;
        const int q   = g.q;
        const int Lc  = g.Lc;
        const int nc  = g.nc;
        int       off[8];
        sscvfem_corner_offsets(L, off);

        // Identity slots turn the BSR assembly kernel into a dense 32x32 cell matrix.
        smesh::count_t sl[64];
        for (int k = 0; k < 64; ++k) sl[k] = (smesh::count_t)k;

        // Constant: which stencil slot each ordered patch-corner pair writes to.
        int slot_ab[8][8];
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b) slot_ab[a][b] = gal_slot(a, b);

        g.e0 = e0;
        g.e1 = e1;
        g.C.assign((size_t)(e1 - e0) * (size_t)nc * 27 * 16, scalar_t(0));

#pragma omp parallel
        {
            const int nxe = d.nxe;
            std::vector<scalar_t> lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
            std::vector<scalar_t> lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
            std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
            std::vector<smesh::idx_t> lg((size_t)nxe);

#pragma omp for schedule(static)
            for (ptrdiff_t e = e0; e < e1; ++e) {
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t gn = d.elems[a][e];
                    lg[(size_t)a]         = gn;
                    lx[(size_t)a]         = (scalar_t)d.points[0][gn];
                    ly[(size_t)a]         = (scalar_t)d.points[1][gn];
                    lz[(size_t)a]         = (scalar_t)d.points[2][gn];
                    lux[(size_t)a]        = d.ux[(size_t)gn];
                    luy[(size_t)a]        = d.uy[(size_t)gn];
                    luz[(size_t)a]        = d.uz[(size_t)gn];
                    lp[(size_t)a]         = d.p[(size_t)gn];
                    lpgx[(size_t)a]       = d.pgx[(size_t)gn];
                    lpgy[(size_t)a]       = d.pgy[(size_t)gn];
                    lpgz[(size_t)a]       = d.pgz[(size_t)gn];
                }

                SSMacroGeom mg;
                {
                    scalar_t ex[8], ey[8], ez[8];
                    for (int a = 0; a < 8; ++a) {
                        const int l = off[a];
                        ex[a]       = lx[(size_t)l];
                        ey[a]       = ly[(size_t)l];
                        ez[a]       = lz[(size_t)l];
                    }
                    sscvfem_macro_geom(ex, ey, ez, rho, mu, d.rhie_chow_scale, mg);
                }

                scalar_t *const Ce = g.C.data() + (size_t)(e - e0) * (size_t)nc * 27 * 16;

                for (int zi = 0; zi < L; ++zi) {
                    for (int yi = 0; yi < L; ++yi) {
                        for (int xi = 0; xi < L; ++xi) {
                            const int base = sscvfem_lidx(L, xi, yi, zi);

                            scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                            for (int a = 0; a < 8; ++a) {
                                const int l = base + off[a];
                                x[a]        = lx[(size_t)l];
                                y[a]        = ly[(size_t)l];
                                z[a]        = lz[(size_t)l];
                                ux[a]       = lux[(size_t)l];
                                uy[a]       = luy[(size_t)l];
                                uz[a]       = luz[(size_t)l];
                                p[a]        = lp[(size_t)l];
                                pgx[a]      = lpgx[(size_t)l];
                                pgy[a]      = lpgy[(size_t)l];
                                pgz[a]      = lpgz[(size_t)l];
                            }

                            scalar_t loc[64 * 16];
                            for (int k = 0; k < 64 * 16; ++k) loc[k] = scalar_t(0);

                            const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
                            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, mg.adj, mg.det, ux, uy, uz, sl,
                                                                           loc, rc, p);
                            boundary_scs_add_jacobian<false>(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                             ux, uy, uz, sl, loc);

                            if (!g.fine_constrained.empty()) {
                                for (int a = 0; a < 8; ++a) {
                                    const size_t gn = (size_t)lg[(size_t)(base + off[a])];
                                    for (int c = 0; c < 4; ++c) {
                                        if (!g.fine_constrained[gn * 4 + (size_t)c]) continue;
                                        for (int b = 0; b < 8; ++b)
                                            for (int r = 0; r < 4; ++r)
                                                loc[(size_t)(b * 8 + a) * 16 + (size_t)(r * 4 + c)] = scalar_t(0);
                                    }
                                }
                            }

                            const CellQ &Q = g.qtab[(size_t)((zi % q) * q + (yi % q)) * (size_t)q + (size_t)(xi % q)];

                            // Stage 1: W[i][b] = sum_j loc[i][j] * Q[j][b], contracted over
                            // the fine index j through the transpose so each block is stored
                            // once rather than zeroed and accumulated into.
                            scalar_t W[8][8][16];
                            for (int i = 0; i < 8; ++i) {
                                const scalar_t *const SFEM_RESTRICT rowi = loc + (size_t)(i * 8) * 16;
                                for (int b = 0; b < 8; ++b) {
                                    scalar_t *const SFEM_RESTRICT dst = W[i][b];
                                    const int                     n0  = Q.trow[b][0];
                                    const scalar_t                w0  = Q.tw[b][0];
#pragma omp simd
                                    for (int c = 0; c < 16; ++c) dst[c] = w0 * rowi[(size_t)n0 * 16 + (size_t)c];
                                    for (int k = 1; k < Q.tnnz[b]; ++k) {
                                        const scalar_t *const SFEM_RESTRICT src =
                                                rowi + (size_t)Q.trow[b][k] * 16;
                                        const scalar_t wk = Q.tw[b][k];
#pragma omp simd
                                        for (int c = 0; c < 16; ++c) dst[c] += wk * src[c];
                                    }
                                }
                            }

                            // Stage 2: C[a][b] += sum_i Q[i][a] * W[i][b], scattered by the
                            // patch corner's lattice position and the fixed offset slot.
                            const int cax = xi / q, cay = yi / q, caz = zi / q;
                            int       pl[8];
                            for (int a = 0; a < 8; ++a)
                                pl[a] = sscvfem_lidx(Lc, cax + GAL_CORNER[a][0], cay + GAL_CORNER[a][1],
                                                     caz + GAL_CORNER[a][2]);

                            for (int i = 0; i < 8; ++i) {
                                for (int k = 0; k < Q.nnz[i]; ++k) {
                                    const int      a  = Q.col[i][k];
                                    const scalar_t wi = Q.w[i][k];
                                    scalar_t *const SFEM_RESTRICT row = Ce + ((size_t)pl[a] * 27) * 16;
                                    for (int b = 0; b < 8; ++b) {
                                        scalar_t *const SFEM_RESTRICT dst = row + (size_t)slot_ab[a][b] * 16;
                                        const scalar_t *const SFEM_RESTRICT src = W[i][b];
#pragma omp simd
                                        for (int c = 0; c < 16; ++c) dst[c] += wi * src[c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Assembly to BSR, needed only where a level is factorised.
    //
    // The pattern is *derived*, not guessed and not widened: an entry exists exactly where
    // two coarse nodes share a macro-element and sit within one lattice step of each other.
    // That is the true Galerkin pattern, so nothing can fall outside it and be folded into
    // the wrong slot -- the failure mode that made the probe path produce a wrong matrix when
    // its guess was too narrow.

    // Local coarse lattice neighbour for a stencil slot, or -1 when it leaves the lattice.
    static SFEM_INLINE int gal_neighbour(const int Lc, const int a, const int s) {
        const int Lp1 = Lc + 1;
        const int ci = a % Lp1, cj = (a / Lp1) % Lp1, ck = a / (Lp1 * Lp1);
        const int di = (s % 3) - 1, dj = ((s / 3) % 3) - 1, dk = (s / 9) - 1;
        const int ni = ci + di, nj = cj + dj, nk = ck + dk;
        if (ni < 0 || ni > Lc || nj < 0 || nj > Lc || nk < 0 || nk > Lc) return -1;
        return (nk * Lp1 + nj) * Lp1 + ni;
    }

    inline void galerkin_build_pattern(const GalerkinLevel &g, std::vector<sfem::count_t> &rowptr,
                                       std::vector<sfem::idx_t> &colidx) {
        const ptrdiff_t n  = g.n_coarse;
        const int       nc = g.nc, Lc = g.Lc;

        std::vector<sfem::count_t> cnt((size_t)n + 1, 0);
        for (ptrdiff_t e = 0; e < g.nmacro; ++e)
            for (int a = 0; a < nc; ++a)
                for (int s = 0; s < 27; ++s)
                    if (gal_neighbour(Lc, a, s) >= 0) cnt[(size_t)g.gid[(size_t)e * nc + a] + 1]++;

        std::vector<sfem::count_t> tptr((size_t)n + 1, 0);
        for (ptrdiff_t i = 0; i < n; ++i) tptr[(size_t)i + 1] = tptr[(size_t)i] + cnt[(size_t)i + 1];

        std::vector<sfem::idx_t>   tcol((size_t)tptr[(size_t)n]);
        std::vector<sfem::count_t> at(tptr.begin(), tptr.end());
        for (ptrdiff_t e = 0; e < g.nmacro; ++e)
            for (int a = 0; a < nc; ++a) {
                const sfem::idx_t r = g.gid[(size_t)e * nc + a];
                for (int s = 0; s < 27; ++s) {
                    const int b = gal_neighbour(Lc, a, s);
                    if (b < 0) continue;
                    tcol[(size_t)at[(size_t)r]++] = g.gid[(size_t)e * nc + b];
                }
            }

        rowptr.assign((size_t)n + 1, 0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            auto b = tcol.begin() + (ptrdiff_t)tptr[(size_t)i];
            auto e = tcol.begin() + (ptrdiff_t)tptr[(size_t)i + 1];
            std::sort(b, e);
            rowptr[(size_t)i + 1] = (sfem::count_t)(std::unique(b, e) - b);
        }
        for (ptrdiff_t i = 0; i < n; ++i) rowptr[(size_t)i + 1] += rowptr[(size_t)i];

        colidx.assign((size_t)rowptr[(size_t)n], 0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            const sfem::count_t len = rowptr[(size_t)i + 1] - rowptr[(size_t)i];
            std::copy(tcol.begin() + (ptrdiff_t)tptr[(size_t)i], tcol.begin() + (ptrdiff_t)tptr[(size_t)i] + len,
                      colidx.begin() + (ptrdiff_t)rowptr[(size_t)i]);
        }
    }

    // (e, a, s) -> position in the BSR value array, or -1 when the slot leaves the lattice.
    // Columns are sorted by construction above, so this is a binary search rather than the
    // linear scan an mm-produced pattern would force.
    inline void galerkin_build_scatter(const GalerkinLevel &g, const std::vector<sfem::count_t> &rowptr,
                                       const std::vector<sfem::idx_t> &colidx, std::vector<ptrdiff_t> &pos) {
        const int nc = g.nc, Lc = g.Lc;
        pos.assign((size_t)g.nmacro * (size_t)nc * 27, -1);

#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < g.nmacro; ++e)
            for (int a = 0; a < nc; ++a) {
                const sfem::idx_t r = g.gid[(size_t)e * nc + a];
                const auto        b = colidx.begin() + (ptrdiff_t)rowptr[(size_t)r];
                const auto        f = colidx.begin() + (ptrdiff_t)rowptr[(size_t)r + 1];
                for (int s = 0; s < 27; ++s) {
                    const int nb = gal_neighbour(Lc, a, s);
                    if (nb < 0) continue;
                    const sfem::idx_t c  = g.gid[(size_t)e * nc + nb];
                    const auto        it = std::lower_bound(b, f, c);
                    if (it == f || *it != c) SFEM_ERROR("galerkin_build_scatter: column %d missing from row %d\n",
                                                        (int)c, (int)r);
                    pos[((size_t)e * nc + a) * 27 + (size_t)s] = (ptrdiff_t)(it - colidx.begin());
                }
            }
    }

    // Invert the scatter so accumulation runs over destinations rather than sources: each
    // block sums its own contributions in a fixed order, with no atomics and therefore the
    // same bits on any thread count. This is the two-pass packed idea applied to assembly.
    inline void galerkin_build_inverse(const std::vector<ptrdiff_t> &pos, const ptrdiff_t nblocks,
                                       const size_t kbegin, const size_t kend, std::vector<ptrdiff_t> &iptr,
                                       std::vector<ptrdiff_t> &iidx) {
        iptr.assign((size_t)nblocks + 1, 0);
        for (size_t k = kbegin; k < kend; ++k)
            if (pos[k] >= 0) iptr[(size_t)pos[k] + 1]++;
        for (ptrdiff_t i = 0; i < nblocks; ++i) iptr[(size_t)i + 1] += iptr[(size_t)i];

        iidx.assign((size_t)iptr[(size_t)nblocks], 0);
        std::vector<ptrdiff_t> at(iptr.begin(), iptr.end());
        for (size_t k = kbegin; k < kend; ++k)
            if (pos[k] >= 0) iidx[(size_t)at[(size_t)pos[k]]++] = (ptrdiff_t)(k - kbegin);
    }

    inline void galerkin_accumulate(const GalerkinLevel &g, const std::vector<ptrdiff_t> &iptr,
                                    const std::vector<ptrdiff_t> &iidx, scalar_t *const SFEM_RESTRICT values) {
        SFEM_TRACE_SCOPE("cvfem_ss::galerkin_accumulate");
        const ptrdiff_t nblocks = (ptrdiff_t)iptr.size() - 1;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t b = 0; b < nblocks; ++b) {
            scalar_t acc[16];
            for (int c = 0; c < 16; ++c) acc[c] = scalar_t(0);
            for (ptrdiff_t k = iptr[(size_t)b]; k < iptr[(size_t)b + 1]; ++k) {
                const scalar_t *const SFEM_RESTRICT src = g.C.data() + (size_t)iidx[(size_t)k] * 16;
#pragma omp simd
                for (int c = 0; c < 16; ++c) acc[c] += src[c];
            }
            for (int c = 0; c < 16; ++c) values[(size_t)b * 16 + c] += acc[c];
        }
    }

    // Coarse global ids per macro-element, taken from the coarse space's own connectivity.
    // Mirrors cvfem_ss_transfer::build_from_spaces, including its one subtlety: on the last
    // hop the coarse space is unstructured HEX8, and the coarse ids must be read from the
    // *fine* semi-structured mesh's macro-corner slots rather than from the coarse mesh's
    // element array. Reading them the other way gave a transfer with relative error 1.14.
    inline void galerkin_gid_from_spaces(const std::shared_ptr<sfem::FunctionSpace> &from_space,  // coarse
                                         const std::shared_ptr<sfem::FunctionSpace> &to_space,    // fine, level 0
                                         GalerkinLevel                              &g) {
        auto &to_m   = to_space->mesh();
        auto &from_m = from_space->mesh();

        if (!to_space->has_semi_structured_mesh())
            SFEM_ERROR("galerkin_gid_from_spaces: the fine space must be semi-structured\n");
        if (to_m.n_blocks() != 1 || from_m.n_blocks() != 1)
            SFEM_ERROR("galerkin_gid_from_spaces: multi-block is not implemented\n");

        auto      to_b     = to_m.block(0);
        auto      from_b   = from_m.block(0);
        const int to_level = smesh::semistructured_level(to_m);
        const int Lc       = from_space->has_semi_structured_mesh() ? smesh::semistructured_level(from_m) : 1;

        if (Lc != g.Lc)
            SFEM_ERROR("galerkin_gid_from_spaces: coarse level %d does not match q=%d on level %d\n", Lc, g.q,
                       to_level);

        g.n_coarse = from_space->n_dofs() / from_space->block_size();

        std::vector<const smesh::idx_t *> rows((size_t)g.nc, nullptr);
        if (from_space->has_semi_structured_mesh()) {
            for (int a = 0; a < g.nc; ++a) rows[(size_t)a] = from_b->elements()->data()[a];
        } else {
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i)
                        rows[(size_t)smesh::sshex8_lidx(1, i, j, k)] =
                                to_b->elements()->data()[smesh::sshex8_lidx(to_level, i * to_level, j * to_level,
                                                                            k * to_level)];
        }

        g.gid.assign((size_t)g.nmacro * (size_t)g.nc, 0);
        for (ptrdiff_t e = 0; e < g.nmacro; ++e)
            for (int a = 0; a < g.nc; ++a) g.gid[(size_t)e * g.nc + a] = rows[(size_t)a][e];
    }

    // ------------------------------------------------------------------
    // Applying a level as element matrices, so only the coarsest needs a BSR.
    //
    // A level held this way is never assembled: the apply gathers the coarse nodes of a
    // macro-element, runs the 27-point stencil over its local matrix, and reduces. Storage
    // is (Lc+1)^3 * 27 blocks per element against n_coarse * 27 for the assembled form, so
    // the excess is duplication at shared macro-element faces alone -- 1.42x at a level-8
    // coarse lattice, 1.95x at level 4 -- and it improves as the lattice deepens. The flops
    // scale the same way, which is the honest cost: the apply trades about 1.4 to 2 times
    // the block multiplies for contiguous blocks and no column indirection, so which wins is
    // a measurement rather than a deduction.
    //
    // The coarsest level is still assembled, because it is factorised.

    struct GalerkinReduce {
        std::vector<ptrdiff_t> ptr;  // n_coarse + 1
        std::vector<ptrdiff_t> idx;  // (e * nc + a) sources feeding each coarse node
    };

    inline void galerkin_build_node_reduce(const GalerkinLevel &g, GalerkinReduce &r) {
        r.ptr.assign((size_t)g.n_coarse + 1, 0);
        for (size_t k = 0; k < g.gid.size(); ++k) r.ptr[(size_t)g.gid[k] + 1]++;
        for (ptrdiff_t i = 0; i < g.n_coarse; ++i) r.ptr[(size_t)i + 1] += r.ptr[(size_t)i];

        r.idx.assign((size_t)r.ptr[(size_t)g.n_coarse], 0);
        std::vector<ptrdiff_t> at(r.ptr.begin(), r.ptr.end());
        for (size_t k = 0; k < g.gid.size(); ++k) r.idx[(size_t)at[(size_t)g.gid[k]]++] = (ptrdiff_t)k;
    }

    // Two passes, like the packed scatter: elements write their own staging slots, then each
    // coarse node sums the slots that feed it in a fixed order. No atomics, same bits on any
    // thread count.
    inline SFEM_NOINLINE void galerkin_apply(const GalerkinLevel &g, const GalerkinReduce &r,
                                             std::vector<scalar_t> &stage,
                                             const scalar_t *const SFEM_RESTRICT x,
                                             scalar_t *const SFEM_RESTRICT       y) {
        SFEM_TRACE_SCOPE("cvfem_ss::galerkin_apply");
        const int nc = g.nc, Lc = g.Lc;
        stage.resize((size_t)g.nmacro * (size_t)nc * N_FIELDS);

        // Stencil slot -> local lattice offset, once: the neighbour of node a at slot s is
        // a + step[s] whenever it stays inside the lattice, which the bounds below decide.
        const int Lp1 = Lc + 1;
        int       step[27];
        for (int s = 0; s < 27; ++s)
            step[s] = ((s / 9) - 1) * Lp1 * Lp1 + (((s / 3) % 3) - 1) * Lp1 + ((s % 3) - 1);

#pragma omp parallel
        {
            std::vector<scalar_t> xl((size_t)nc * N_FIELDS);
#pragma omp for schedule(static)
            for (ptrdiff_t e = 0; e < g.nmacro; ++e) {
                for (int a = 0; a < nc; ++a) {
                    const size_t gn = (size_t)g.gid[(size_t)e * nc + a];
                    for (int c = 0; c < N_FIELDS; ++c)
                        xl[(size_t)a * N_FIELDS + (size_t)c] = x[gn * N_FIELDS + (size_t)c];
                }

                const scalar_t *const SFEM_RESTRICT Ce = g.C.data() + (size_t)(e - g.e0) * (size_t)nc * 27 * 16;
                scalar_t *const SFEM_RESTRICT       st = stage.data() + (size_t)e * (size_t)nc * N_FIELDS;

                for (int a = 0; a < nc; ++a) {
                    const int ci = a % Lp1, cj = (a / Lp1) % Lp1, ck = a / (Lp1 * Lp1);
                    scalar_t  acc[N_FIELDS] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)};

                    for (int s = 0; s < 27; ++s) {
                        const int di = (s % 3) - 1, dj = ((s / 3) % 3) - 1, dk = (s / 9) - 1;
                        if (ci + di < 0 || ci + di > Lc || cj + dj < 0 || cj + dj > Lc || ck + dk < 0 ||
                            ck + dk > Lc)
                            continue;
                        const scalar_t *const SFEM_RESTRICT blk = Ce + ((size_t)a * 27 + (size_t)s) * 16;
                        const scalar_t *const SFEM_RESTRICT v   = &xl[(size_t)(a + step[s]) * N_FIELDS];
                        for (int rr = 0; rr < N_FIELDS; ++rr)
                            for (int cc = 0; cc < N_FIELDS; ++cc) acc[rr] += blk[rr * 4 + cc] * v[cc];
                    }
                    for (int c = 0; c < N_FIELDS; ++c) st[(size_t)a * N_FIELDS + (size_t)c] = acc[c];
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (ptrdiff_t n = 0; n < g.n_coarse; ++n) {
            scalar_t acc[N_FIELDS] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)};
            for (ptrdiff_t k = r.ptr[(size_t)n]; k < r.ptr[(size_t)n + 1]; ++k) {
                const scalar_t *const SFEM_RESTRICT src = stage.data() + (size_t)r.idx[(size_t)k] * N_FIELDS;
                for (int c = 0; c < N_FIELDS; ++c) acc[c] += src[c];
            }
            for (int c = 0; c < N_FIELDS; ++c) y[(size_t)n * N_FIELDS + (size_t)c] = acc[c];
        }
    }

    // The smoother's block diagonal: the centre slot summed over the elements at each node.
    inline void galerkin_block_diag(const GalerkinLevel &g, const GalerkinReduce &r,
                                    scalar_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t n = 0; n < g.n_coarse; ++n) {
            scalar_t acc[16];
            for (int c = 0; c < 16; ++c) acc[c] = scalar_t(0);
            for (ptrdiff_t k = r.ptr[(size_t)n]; k < r.ptr[(size_t)n + 1]; ++k) {
                const scalar_t *const SFEM_RESTRICT blk =
                        g.C.data() + ((size_t)r.idx[(size_t)k] * 27 + 13) * 16;
#pragma omp simd
                for (int c = 0; c < 16; ++c) acc[c] += blk[c];
            }
            for (int c = 0; c < 16; ++c) diag[(size_t)n * 16 + c] = acc[c];
        }
    }

    // ------------------------------------------------------------------
    // Coarsening a level's element matrices by one hop, entirely inside the macro-element.
    //
    // This is what makes the whole hierarchy element-wise rather than only level 1. The
    // obstacle was never the algebra, it was that the transfers zero constrained dofs at each
    // hop, so a level built straight from level 0 skips the intermediate Z's. SFEM already
    // carries what is needed to not skip them: create_gmg_data derefines the Function at every
    // level (`f_prev->derefine(fs_next, true)`), so each level has its own constraints, and the
    // transfers apply them as R = Z_coarse Rhat and P = Z_fine Phat. The composite at a hop is
    // therefore
    //
    //     Z_i Rhat A_{i-1} Z_{i-1} Phat
    //
    // -- mask the source level's columns with the source level's own mask, contract with the
    // plain interpolation, and patch identity rows at the target. That is the same recipe the
    // rap branch uses (mask_block_columns, rap, patch_identity_rows), and the same one the
    // fine-level assembly above already applies to its micro-cell matrices. Because the mask
    // acts on the *matrix* and not on the transfer, the interpolation stays scalar: no
    // per-component prolongation is needed even though the constraints are per component.
    //
    // Both source and target lattices live in the same macro-element, so the hop is local and
    // needs no global matrix.
    //
    // The coarse stencil stays 27-point. A source node and its 27-neighbour land on coarse
    // nodes at most one lattice step apart: reaching two would need the two source nodes'
    // coarse floors to differ while the upper one is off-lattice, and a differing floor forces
    // it to be on-lattice. That holds for any ratio, not only 2:1.

    struct HopSupp {
        int      n{0};
        int      col[8];
        int      cc[8][3];
        scalar_t w[8];
    };

    // Where a source-lattice node interpolates from on the target lattice.
    inline void hop_support(const int Lout, const int q, const int x, const int y, const int z, HopSupp &h) {
        int      cx[2], cy[2], cz[2];
        scalar_t wx[2], wy[2], wz[2];
        auto     axis = [q](const int v, int c[2], scalar_t w[2]) -> int {
            const int a = v / q, r = v % q;
            c[0] = a;
            if (!r) {
                w[0] = scalar_t(1);
                return 1;
            }
            c[1] = a + 1;
            w[0] = scalar_t(1) - (scalar_t)r / (scalar_t)q;
            w[1] = (scalar_t)r / (scalar_t)q;
            return 2;
        };
        const int nx = axis(x, cx, wx), ny = axis(y, cy, wy), nz = axis(z, cz, wz);

        h.n = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    h.cc[h.n][0]  = cx[i];
                    h.cc[h.n][1]  = cy[j];
                    h.cc[h.n][2]  = cz[k];
                    h.col[h.n]    = sscvfem_lidx(Lout, cx[i], cy[j], cz[k]);
                    h.w[h.n]      = wx[i] * wy[j] * wz[k];
                    ++h.n;
                }
    }

    // out must already carry Lc, nc, nmacro, gid and n_coarse for the target level.
    // mask_in is one byte per source-level dof (node * 4 + component).
    inline SFEM_NOINLINE void galerkin_hop(const GalerkinLevel &in, const uint8_t *const mask_in,
                                           GalerkinLevel &out) {
        SFEM_TRACE_SCOPE("cvfem_ss::galerkin_hop");
        const int Lin = in.Lc, Lout = out.Lc;
        const int q = Lin / Lout;
        if (q < 1 || Lin % Lout) SFEM_ERROR("galerkin_hop: level %d does not divide %d\n", Lout, Lin);

        // Depends on lattice position alone, so it is shared by every macro-element.
        std::vector<HopSupp> hs((size_t)in.nc);
        for (int z = 0; z <= Lin; ++z)
            for (int y = 0; y <= Lin; ++y)
                for (int x = 0; x <= Lin; ++x)
                    hop_support(Lout, q, x, y, z, hs[(size_t)sscvfem_lidx(Lin, x, y, z)]);

        out.e0 = 0;
        out.e1 = out.nmacro;
        out.C.assign((size_t)out.nmacro * (size_t)out.nc * 27 * 16, scalar_t(0));

#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < out.nmacro; ++e) {
            const scalar_t *const SFEM_RESTRICT Ci = in.C.data() + (size_t)(e - in.e0) * (size_t)in.nc * 27 * 16;
            scalar_t *const SFEM_RESTRICT       Co = out.C.data() + (size_t)e * (size_t)out.nc * 27 * 16;

            for (int i = 0; i < in.nc; ++i) {
                const HopSupp &hi = hs[(size_t)i];
                for (int s = 0; s < 27; ++s) {
                    const int nb = gal_neighbour(Lin, i, s);
                    if (nb < 0) continue;

                    scalar_t m[16];
                    const scalar_t *const SFEM_RESTRICT blk = Ci + ((size_t)i * 27 + (size_t)s) * 16;
                    for (int t = 0; t < 16; ++t) m[t] = blk[t];

                    // Z on the source level's columns: the block's column node is the
                    // neighbour, so a constrained (node, component) kills that component's
                    // column of this block.
                    if (mask_in) {
                        const size_t gn = (size_t)in.gid[(size_t)e * in.nc + nb];
                        for (int c = 0; c < 4; ++c)
                            if (mask_in[gn * 4 + (size_t)c])
                                for (int r = 0; r < 4; ++r) m[r * 4 + c] = scalar_t(0);
                    }

                    const HopSupp &hj = hs[(size_t)nb];
                    for (int ka = 0; ka < hi.n; ++ka) {
                        const scalar_t wa = hi.w[ka];
                        scalar_t *const SFEM_RESTRICT row = Co + ((size_t)hi.col[ka] * 27) * 16;
                        for (int kb = 0; kb < hj.n; ++kb) {
                            const int slot = (hj.cc[kb][2] - hi.cc[ka][2] + 1) * 9 +
                                             (hj.cc[kb][1] - hi.cc[ka][1] + 1) * 3 +
                                             (hj.cc[kb][0] - hi.cc[ka][0] + 1);
                            const scalar_t                w   = wa * hj.w[kb];
                            scalar_t *const SFEM_RESTRICT dst = row + (size_t)slot * 16;
#pragma omp simd
                            for (int t = 0; t < 16; ++t) dst[t] += w * m[t];
                        }
                    }
                }
            }
        }
    }

}  // namespace cvfem_ss
