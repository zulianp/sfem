#pragma once

// Diagonal Vanka smoother for the semi-structured colocated CVFEM Navier-Stokes operator.
//
// Why: the point-block Jacobi smoother measures rho = 0.981 at Re = 1 -- 1.9% error removal
// per sweep -- and diverges for omega >= 0.5; SIMPLE measures identically. The cause is
// structural rather than a tuning failure. Inverting the nodal 4x4
//
//     D_i = [ F_ii  G_ii ; D_ii  A_pp,ii ]
//
// damps the momentum equation while discarding the continuity constraint, because continuity
// at a node is a statement about fluxes across the faces *surrounding* it: the coupling that
// determines the pressure lives in the off-diagonal divergence entries D_ij, j != i, which a
// point-block smoother throws away.
//
// Vanka replaces the nodal block with a small *coupled* local system over a patch containing a
// pressure together with the velocities its continuity equation references:
//
//     x <- x + omega * sum_k R_k^T (R_k A R_k^T)^-1 R_k (b - A x)
//
// The patch here is one micro-element: 8 corners x 4 fields = 32 dofs. That choice is specific
// to this code -- the patch operator R_k A R_k^T is *exactly* the dense micro-cell matrix that
// identity slots into cvfem_hex8_ns_upwind_jacobian_add_slots already produce, the same trick
// assemble_block_diag and the element-wise Galerkin assembly use. No new element kernel.
//
// Diagonal Vanka approximates the velocity block F_k by its diagonal, which lets the
// velocities be eliminated and leaves an 8x8 pressure Schur complement per cell:
//
//     S_k = A_pp - D diag(F)^-1 G ,     S_k dp = r_p - D diag(F)^-1 r_u
//     du  = diag(F)^-1 ( r_u - G dp )
//
// Full Vanka would invert the 32x32 exactly; at 242,500 dofs that is 432 MiB of cached
// factors against 208 MiB here, and it is not worth the memory until the diagonal variant has
// been measured.
//
// Caveat recorded where it belongs: the published Vanka rates (rho = 0.08 for Stokes) are for
// *staggered* discretisations and describe the saddle-point coupling only. This is colocated,
// and convection dominance at high Re is a separate difficulty that no Stokes rate speaks to.
// The acceptance test is SFEM_GMG_CHECK=3 run to its asymptote at the physical Reynolds
// number, compared against block-Jacobi's 0.981.

// sfem::count_t / sfem::idx_t come from cvfem_ss_galerkin_api.hpp, which the only
// translation unit including this header includes first. sfem_base.h is not on the
// include path of every install (it is not on alps).
#include "cvfem_sshex8_ns.hpp"

#include <cmath>
#include <vector>

namespace cvfem_ss {

    // Per micro-cell state, in the layout the sweep streams.
    //
    // The products are stored pre-divided by diag(F) so the sweep does no divisions:
    //   Dhat[a][ic] = D[a][ic] / dF[ic]      (8 x 24)
    //   Ghat[ic][b] = G[ic][b] / dF[ic]      (24 x 8)
    //   invdF[ic]   = 1 / dF[ic]             (24)
    //   S_lu, piv   = LU of the 8x8 Schur complement
    // 472 doubles + 8 ints per cell, about 3.8 KiB.
    struct VankaCell {
        scalar_t Dhat[8][24];
        scalar_t Ghat[24][8];
        scalar_t invdF[24];
        scalar_t S[8][8];
        int      piv[8];
    };

    struct VankaData {
        // The assembled fine operator, borrowed. Needed by the multiplicative sweep, which
        // forms each patch's residual from the current iterate and therefore needs full rows,
        // not just the patch block.
        const sfem::count_t *rowptr{nullptr};
        const sfem::idx_t   *colidx{nullptr};
        const real_t        *values{nullptr};

        int                    L{0};
        ptrdiff_t              nmacro{0};
        ptrdiff_t              ncell{0};   // L^3 per macro-element
        std::vector<VankaCell> cells;      // nmacro * ncell
        std::vector<scalar_t>  weight;     // 1 / (micro-cell patches touching a node), additive form
        std::vector<scalar_t>  ewt;        // 1 / (macro-elements touching a node), multiplicative form
        std::vector<uint8_t>   constrained;
    };

    // Dense LU with partial pivoting, n = 8. Small enough that the pivoting cost is noise and
    // large enough that skipping it is a real risk: the Schur complement of a convection
    // dominated block is not diagonally dominant.
    static SFEM_INLINE bool vanka_lu8(scalar_t A[8][8], int piv[8]) {
        for (int k = 0; k < 8; ++k) {
            int      p = k;
            scalar_t m = std::fabs(A[k][k]);
            for (int i = k + 1; i < 8; ++i) {
                const scalar_t v = std::fabs(A[i][k]);
                if (v > m) { m = v; p = i; }
            }
            if (m < scalar_t(1e-300)) return false;
            piv[k] = p;
            if (p != k)
                for (int j = 0; j < 8; ++j) std::swap(A[k][j], A[p][j]);
            const scalar_t inv = scalar_t(1) / A[k][k];
            for (int i = k + 1; i < 8; ++i) {
                const scalar_t f = A[i][k] * inv;
                A[i][k]          = f;
                for (int j = k + 1; j < 8; ++j) A[i][j] -= f * A[k][j];
            }
        }
        return true;
    }

    static SFEM_INLINE void vanka_solve8(const scalar_t A[8][8], const int piv[8], scalar_t b[8]) {
        for (int k = 0; k < 8; ++k) {
            const int p = piv[k];
            if (p != k) std::swap(b[k], b[p]);
            for (int i = k + 1; i < 8; ++i) b[i] -= A[i][k] * b[k];
        }
        for (int k = 7; k >= 0; --k) {
            for (int j = k + 1; j < 8; ++j) b[k] -= A[k][j] * b[j];
            b[k] /= A[k][k];
        }
    }

    // Build the per-cell factorisations for the current linearisation. Valid for one Newton
    // step, the same lifetime as the element-wise Galerkin coarse operators.
    // Build the per-cell factorisations from the ASSEMBLED fine operator.
    //
    // This is the part that has to be right, and was got wrong twice. The Vanka patch operator
    // is R_k A R_k^T -- the assembled operator restricted to the patch's dofs -- not one
    // micro-cell's contribution to it. A node interior to a macro-element is touched by eight
    // cells, so a single cell carries roughly an eighth of the true entries; using it made the
    // smoother diverge at a rate of 18. Taking only the diagonal from the assembled matrix and
    // leaving the off-diagonals single-cell is worse than either extreme, because the patch
    // system is then internally inconsistent: it still diverged, at 7.8.
    //
    // So the patch is read from the assembled matrix. That matrix is produced by the
    // element-wise Galerkin assembly at ratio q = 1, where the prolongation is the identity and
    // P^T A P is A itself -- the path whose identity gate matches the matrix-free apply at
    // 1.54e-16, so the entries are known to be the operator's own.
    //
    // `rowptr`, `colidx`, `values` are that matrix in BSR form with sorted columns; `elems`
    // maps a macro-element's lattice slot to a global node.
    inline SFEM_NOINLINE void vanka_setup(const SSMeshData &d, const uint8_t *const constrained,
                                          const sfem::count_t *const rowptr, const sfem::idx_t *const colidx,
                                          const real_t *const values, VankaData &v) {
        SFEM_TRACE_SCOPE("cvfem_ss::vanka_setup");
        const int L = d.level;
        int       off[8];
        sscvfem_corner_offsets(L, off);

        v.rowptr = rowptr;
        v.colidx = colidx;
        v.values = values;
        v.L      = L;
        v.nmacro = d.nmacro;
        v.ncell  = (ptrdiff_t)L * L * L;
        v.cells.assign((size_t)v.nmacro * (size_t)v.ncell, VankaCell{});
        v.constrained.assign(constrained, constrained + (size_t)d.nnodes * N_FIELDS);

        // Patch multiplicity: additive Vanka sums overlapping corrections, so each node's
        // contributions are averaged over the patches touching it. Without this a node interior
        // to a macro-element, lying in eight micro-cells, gets eight times the correction of a
        // corner node.
        v.weight.assign((size_t)d.nnodes, scalar_t(0));
        for (ptrdiff_t e = 0; e < d.nmacro; ++e)
            for (int zi = 0; zi < L; ++zi)
                for (int yi = 0; yi < L; ++yi)
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        for (int a = 0; a < 8; ++a)
                            v.weight[(size_t)d.elems[base + off[a]][e]] += scalar_t(1);
                    }
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            v.weight[(size_t)i] = v.weight[(size_t)i] > 0 ? scalar_t(1) / v.weight[(size_t)i] : scalar_t(0);

        v.ewt.assign((size_t)d.nnodes, scalar_t(0));
        for (ptrdiff_t e = 0; e < d.nmacro; ++e)
            for (int a = 0; a < d.nxe; ++a) v.ewt[(size_t)d.elems[a][e]] += scalar_t(1);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            v.ewt[(size_t)i] = v.ewt[(size_t)i] > 0 ? scalar_t(1) / v.ewt[(size_t)i] : scalar_t(0);

#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            for (int zi = 0; zi < L; ++zi)
                for (int yi = 0; yi < L; ++yi)
                    for (int xi = 0; xi < L; ++xi) {
                        const int       base = sscvfem_lidx(L, xi, yi, zi);
                        const ptrdiff_t ci   = (ptrdiff_t)e * v.ncell + ((ptrdiff_t)zi * L + yi) * L + xi;
                        VankaCell      &c    = v.cells[(size_t)ci];

                        smesh::idx_t n[8];
                        for (int a = 0; a < 8; ++a) n[a] = d.elems[base + off[a]][e];

                        // Gather the 8x8 blocks of the assembled operator over the patch nodes.
                        scalar_t A[8][8][16];
                        for (int a = 0; a < 8; ++a) {
                            const sfem::count_t rb = rowptr[(size_t)n[a]], re = rowptr[(size_t)n[a] + 1];
                            for (int b = 0; b < 8; ++b) {
                                scalar_t *dst = A[a][b];
                                for (int k = 0; k < 16; ++k) dst[k] = scalar_t(0);
                                // columns are sorted, so this is a binary search
                                sfem::count_t lo = rb, hi = re;
                                while (lo < hi) {
                                    const sfem::count_t mid = lo + (hi - lo) / 2;
                                    if (colidx[(size_t)mid] < n[b]) lo = mid + 1;
                                    else                            hi = mid;
                                }
                                if (lo < re && colidx[(size_t)lo] == n[b])
                                    for (int k = 0; k < 16; ++k) dst[k] = (scalar_t)values[(size_t)lo * 16 + k];
                            }
                        }

                        for (int a = 0; a < 8; ++a)
                            for (int t = 0; t < 3; ++t) {
                                const scalar_t dd = A[a][a][t * 4 + t];
                                c.invdF[a * 3 + t] =
                                        std::fabs(dd) > scalar_t(1e-300) ? scalar_t(1) / dd : scalar_t(0);
                            }

                        for (int a = 0; a < 8; ++a)
                            for (int i = 0; i < 8; ++i)
                                for (int t = 0; t < 3; ++t)
                                    c.Dhat[a][i * 3 + t] = A[a][i][12 + t] * c.invdF[i * 3 + t];
                        for (int i = 0; i < 8; ++i)
                            for (int t = 0; t < 3; ++t)
                                for (int b = 0; b < 8; ++b)
                                    c.Ghat[i * 3 + t][b] = A[i][b][t * 4 + 3] * c.invdF[i * 3 + t];

                        for (int a = 0; a < 8; ++a)
                            for (int b = 0; b < 8; ++b) {
                                scalar_t sv = A[a][b][15];
                                for (int i = 0; i < 8; ++i)
                                    for (int t = 0; t < 3; ++t)
                                        sv -= A[a][i][12 + t] * c.invdF[i * 3 + t] * A[i][b][t * 4 + 3];
                                c.S[a][b] = sv;
                            }

                        if (!vanka_lu8(c.S, c.piv)) {
                            for (int a = 0; a < 8; ++a) {
                                for (int b = 0; b < 8; ++b) c.S[a][b] = (a == b) ? scalar_t(1) : scalar_t(0);
                                c.piv[a] = a;
                            }
                        }
                    }
        }
    }

    // One additive Vanka application: z = omega * sum_k R_k^T Ahat_k^-1 R_k r, averaged over
    // the patches touching each node. Two-pass scatter, so no atomics and the same bits on any
    // thread count.
    // NOTE ON THE INTERFACE: this ACCUMULATES into `y`, because that is what every other
    // preconditioner here does -- BlockJacobi::apply ends in `yy[r] += s`. Overwriting instead
    // was a real bug: the SFEM_GMG_CHECK=3 gate zeroes its own vector before calling, so an
    // overwriting apply measures correctly there and then destroys the caller's accumulator
    // inside the cycle, where the smoother diverged to 1e+24 while measuring 0.63 standalone.
    // `work` is scratch of ndof, owned by the caller so the sweep does not allocate.
    inline SFEM_NOINLINE void vanka_apply(const SSMeshData &d, const VankaData &v, const scalar_t omega,
                                          const scalar_t *const SFEM_RESTRICT r,
                                          scalar_t *const SFEM_RESTRICT       y,
                                          std::vector<scalar_t>              &work) {
        work.resize((size_t)d.nnodes * N_FIELDS);
        scalar_t *const z = work.data();
        SFEM_TRACE_SCOPE("cvfem_ss::vanka_apply");
        const int L   = v.L;
        const int nxe = d.nxe;
        int       off[8];
        sscvfem_corner_offsets(L, off);

        const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

        for (ptrdiff_t k = 0; k < d.nnodes * N_FIELDS; ++k) z[(size_t)k] = scalar_t(0);

#pragma omp parallel
        {
            std::vector<smesh::idx_t> lg((size_t)nxe);
            std::vector<scalar_t>     lr((size_t)nxe * N_FIELDS);
            std::vector<scalar_t>     lz((size_t)nxe * N_FIELDS);

#pragma omp for schedule(static)
            for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = d.elems[a][e];
                    lg[(size_t)a]        = g;
                    for (int t = 0; t < N_FIELDS; ++t)
                        lr[(size_t)a * N_FIELDS + (size_t)t] = r[(size_t)g * N_FIELDS + (size_t)t];
                }
                std::fill(lz.begin(), lz.end(), scalar_t(0));

                for (int zi = 0; zi < L; ++zi)
                    for (int yi = 0; yi < L; ++yi)
                        for (int xi = 0; xi < L; ++xi) {
                            const int       base = sscvfem_lidx(L, xi, yi, zi);
                            const ptrdiff_t ci   = (ptrdiff_t)e * v.ncell + ((ptrdiff_t)zi * L + yi) * L + xi;
                            const VankaCell &c   = v.cells[(size_t)ci];

                            scalar_t ru[24], rp[8];
                            for (int a = 0; a < 8; ++a) {
                                const size_t l = (size_t)(base + off[a]) * N_FIELDS;
                                ru[a * 3 + 0]  = lr[l + 0];
                                ru[a * 3 + 1]  = lr[l + 1];
                                ru[a * 3 + 2]  = lr[l + 2];
                                rp[a]          = lr[l + 3];
                            }

                            // S dp = r_p - D diag(F)^-1 r_u
                            scalar_t dp[8];
                            for (int a = 0; a < 8; ++a) {
                                scalar_t s = rp[a];
                                for (int j = 0; j < 24; ++j) s -= c.Dhat[a][j] * ru[j];
                                dp[a] = s;
                            }
                            vanka_solve8(c.S, c.piv, dp);

                            // du = diag(F)^-1 r_u - Ghat dp
                            for (int a = 0; a < 8; ++a) {
                                const size_t l = (size_t)(base + off[a]) * N_FIELDS;
                                for (int t = 0; t < 3; ++t) {
                                    const int j = a * 3 + t;
                                    scalar_t  s = c.invdF[j] * ru[j];
                                    for (int b = 0; b < 8; ++b) s -= c.Ghat[j][b] * dp[b];
                                    lz[l + (size_t)t] += s;
                                }
                                lz[l + 3] += dp[a];
                            }
                        }

                if (sc)
                    sscvfem_scatter_element(*sc, nxe, e, lg.data(), lz.data(), z);
                else
                    for (int a = 0; a < nxe; ++a)
                        for (int t = 0; t < N_FIELDS; ++t)
                            atomic_add(z + (ptrdiff_t)lg[(size_t)a] * N_FIELDS + t, 0,
                                       lz[(size_t)a * N_FIELDS + (size_t)t]);
            }
        }

        if (sc) sscvfem_reduce_shared(*sc, z);

        // Average over the patches touching each node, damp, leave constrained dofs alone, and
        // ACCUMULATE into the caller's vector.
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t w = omega * v.weight[(size_t)i];
            for (int t = 0; t < N_FIELDS; ++t) {
                const size_t k = (size_t)i * N_FIELDS + (size_t)t;
                if (!v.constrained[k]) y[k] += w * z[k];
            }
        }
    }

    // Multiplicative Vanka: eight colours inside a macro-element, additive between them.
    //
    // The colouring argument -- two micro-cells share a node iff their indices differ by at most
    // one per axis, so the 2x2x2 parity classes are independent -- holds only WITHIN a
    // macro-element. Same-coloured cells in adjacent macro-elements share face nodes, so a
    // sweep parallel over macro-elements races on them. That is not hypothetical: it made the
    // solve take 36 linear iterations on one thread and 156 on eight, which is the determinism
    // gate failing rather than a performance quirk.
    //
    // So each macro-element sweeps its own L^3 cells multiplicatively, on a private local
    // vector, reading only its own nodes; corrections are then combined additively across
    // elements through the two-pass scatter. Multiplicative where the coupling is dense and
    // cheap, additive across the few shared faces -- the standard domain-decomposition
    // compromise, and it restores bitwise reproducibility.
    inline SFEM_NOINLINE void vanka_apply_mult(const SSMeshData &d, const VankaData &v, const scalar_t omega,
                                               const scalar_t *const SFEM_RESTRICT r,
                                               scalar_t *const SFEM_RESTRICT       y,
                                               std::vector<scalar_t>              &work) {
        SFEM_TRACE_SCOPE("cvfem_ss::vanka_apply_mult");
        const int       L    = v.L;
        const int       nxe  = d.nxe;
        const ptrdiff_t ndof = d.nnodes * N_FIELDS;
        int             off[8];
        sscvfem_corner_offsets(L, off);

        work.resize((size_t)ndof);
        scalar_t *const z = work.data();
        for (ptrdiff_t k = 0; k < ndof; ++k) z[(size_t)k] = scalar_t(0);

        const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

#pragma omp parallel
        {
            // Global node -> local slot, so a row of A can be restricted to this element.
            // Entries are set and cleared per element, so the fill cost is 2 * nxe, not nnodes.
            std::vector<int>          loc_of((size_t)d.nnodes, -1);
            std::vector<smesh::idx_t> lg((size_t)nxe);
            std::vector<scalar_t>     rl((size_t)nxe * N_FIELDS), zl((size_t)nxe * N_FIELDS);

#pragma omp for schedule(static)
            for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = d.elems[a][e];
                    lg[(size_t)a]        = g;
                    loc_of[(size_t)g]    = a;
                    for (int t = 0; t < N_FIELDS; ++t)
                        rl[(size_t)a * N_FIELDS + (size_t)t] = r[(size_t)g * N_FIELDS + (size_t)t];
                }
                std::fill(zl.begin(), zl.end(), scalar_t(0));

                for (int colour = 0; colour < 8; ++colour) {
                    const int cx = colour & 1, cy = (colour >> 1) & 1, cz = (colour >> 2) & 1;
                    for (int zi = cz; zi < L; zi += 2)
                        for (int yi = cy; yi < L; yi += 2)
                            for (int xi = cx; xi < L; xi += 2) {
                                const int       base = sscvfem_lidx(L, xi, yi, zi);
                                const ptrdiff_t ci = (ptrdiff_t)e * v.ncell + ((ptrdiff_t)zi * L + yi) * L + xi;
                                const VankaCell &c = v.cells[(size_t)ci];

                                scalar_t ru[24], rp[8];
                                for (int a = 0; a < 8; ++a) {
                                    const int    la = base + off[a];
                                    scalar_t     acc[N_FIELDS] = {0, 0, 0, 0};
                                    const size_t gn            = (size_t)lg[(size_t)la];
                                    for (sfem::count_t k = v.rowptr[gn]; k < v.rowptr[gn + 1]; ++k) {
                                        const int lj = loc_of[(size_t)v.colidx[(size_t)k]];
                                        if (lj < 0) continue;  // outside this element: additive
                                        const real_t *const blk = v.values + (size_t)k * 16;
                                        const scalar_t *const zz = &zl[(size_t)lj * N_FIELDS];
                                        for (int rr = 0; rr < N_FIELDS; ++rr)
                                            for (int cc = 0; cc < N_FIELDS; ++cc)
                                                acc[rr] += (scalar_t)blk[rr * 4 + cc] * zz[cc];
                                    }
                                    const size_t lo = (size_t)la * N_FIELDS;
                                    ru[a * 3 + 0]   = rl[lo + 0] - acc[0];
                                    ru[a * 3 + 1]   = rl[lo + 1] - acc[1];
                                    ru[a * 3 + 2]   = rl[lo + 2] - acc[2];
                                    rp[a]           = rl[lo + 3] - acc[3];
                                }

                                scalar_t dp[8];
                                for (int a = 0; a < 8; ++a) {
                                    scalar_t sacc = rp[a];
                                    for (int jj = 0; jj < 24; ++jj) sacc -= c.Dhat[a][jj] * ru[jj];
                                    dp[a] = sacc;
                                }
                                vanka_solve8(c.S, c.piv, dp);

                                for (int a = 0; a < 8; ++a) {
                                    const size_t lo = (size_t)(base + off[a]) * N_FIELDS;
                                    for (int t = 0; t < 3; ++t) {
                                        const int jj = a * 3 + t;
                                        scalar_t  sv = c.invdF[jj] * ru[jj];
                                        for (int b = 0; b < 8; ++b) sv -= c.Ghat[jj][b] * dp[b];
                                        zl[lo + (size_t)t] += omega * sv;
                                    }
                                    zl[lo + 3] += omega * dp[a];
                                }
                            }
                }

                if (sc)
                    sscvfem_scatter_element(*sc, nxe, e, lg.data(), zl.data(), z);
                else
                    for (int a = 0; a < nxe; ++a)
                        for (int t = 0; t < N_FIELDS; ++t)
                            atomic_add(z + (ptrdiff_t)lg[(size_t)a] * N_FIELDS + t, 0,
                                       zl[(size_t)a * N_FIELDS + (size_t)t]);

                for (int a = 0; a < nxe; ++a) loc_of[(size_t)lg[(size_t)a]] = -1;
            }
        }

        if (sc) sscvfem_reduce_shared(*sc, z);

        // Average over the macro-elements sharing a node (additive between subdomains), skip
        // constrained dofs, and ACCUMULATE, matching BlockJacobi::apply.
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t w = v.ewt[(size_t)i];
            for (int t = 0; t < N_FIELDS; ++t) {
                const size_t k = (size_t)i * N_FIELDS + (size_t)t;
                if (!v.constrained[k]) y[k] += w * z[k];
            }
        }
    }

}  // namespace cvfem_ss
