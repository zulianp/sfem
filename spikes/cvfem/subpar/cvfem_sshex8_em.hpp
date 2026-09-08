#pragma once

// Element-matrix variants of the semi-structured CVFEM apply, quarantined.
//
// Both assemble the linear part of the operator as a small dense matrix for the macro
// and apply it to all L^3 micro-elements with one gemm, the shape SFEM uses for
// semi-structured linear elasticity. Both are correct -- they agree with the reference
// to better than 5e-16 -- and both are slower than evaluating the terms directly. See
// subpar/README.md for the numbers and why.
//
// Included only under -DCVFEM_ENABLE_SUBPAR, so the comparison stays reproducible on
// other hardware without carrying the variants in the default build.

#include "cvfem_sshex8_ns.hpp"

// Momentum element matrix, assembled on the fly, applied with a gemm.
//
// This follows the semi-structured linear-elasticity path: assemble the element matrix
// for the macro, then apply it to every micro-element in one gemm
// (sfem_SemiStructuredEMLinearElasticity, operators/stencil/sshex8_stencil_element_matrix_apply*).
//
// The matrix is the viscous momentum block and nothing else, which is exactly what the
// existing cvfem_hex8_ns_upwind_jacobian_add_slots_linear assembles -- it takes mu, adj
// and det and no velocity at all. So it is 24x24, three velocity components on eight
// nodes, rather than the full 32x32: the pressure and continuity couplings are not in it,
// and carrying them as structural zeros would be 44% of the gemm doing nothing.
//
// Assembling it by calling the kernel, rather than probing the action with unit vectors,
// means there is one definition of the viscous term and this uses it.

static constexpr int SSCVFEM_EM_N = 24;  // 8 nodes x 3 velocity components

inline void sscvfem_build_momentum_em(const SSMacroGeom &g, const scalar_t mu,
                                      scalar_t *const SFEM_RESTRICT Mv) {
    // Assemble into a local 8x8 block array addressed by identity slots, then compress
    // the 3x3 velocity sub-blocks into a dense 24x24.
    scalar_t blocks[64 * 16];
    for (int i = 0; i < 64 * 16; ++i) blocks[i] = scalar_t(0);
    smesh::count_t slots[64];
    for (int i = 0; i < 64; ++i) slots[i] = (smesh::count_t)i;

    cvfem_hex8_ns_upwind_jacobian_add_slots_linear<false>(mu, g.adj, g.det, slots, blocks);

    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            const scalar_t *const blk = blocks + (size_t)(a * 8 + b) * 16;
            for (int rr = 0; rr < 3; ++rr)
                for (int cc = 0; cc < 3; ++cc)
                    Mv[(size_t)(a * 3 + rr) * SSCVFEM_EM_N + (b * 3 + cc)] = blk[rr * 4 + cc];
        }
    }
}

// The pressure terms are linear too, so they can go in the matrix as well.
//
// Under the affine-macro assumption everything except the convective flux has constant
// coefficients: the pressure gradient qmid*A in the momentum rows, the continuity
// divergence rho/2 (v.A), and the Rhie-Chow c (q_i - q_j). Adding them makes the matrix
// the full 32x32 and leaves the sub-control-surface loop carrying convection alone.
//
// It is not obvious this is a win, which is why both exist. The matrix grows from 576 to
// 1024 entries -- 78% more gemm work -- to remove roughly 180 FLOPs per element from the
// loop, and the added blocks are sparse: each pressure row has only its handful of
// sub-control-surface neighbours, and a dense gemm cannot exploit that.
inline void sscvfem_build_full_em(const SSMacroGeom &g, const scalar_t rho, const scalar_t mu,
                                  scalar_t *const SFEM_RESTRICT Mf) {
    const int N = CVFEM_HEX8_N_DOF;  // 32
    for (int i = 0; i < N * N; ++i) Mf[i] = scalar_t(0);

    // Viscous momentum, from the same assembly kernel the 24x24 variant uses.
    scalar_t blocks[64 * 16];
    for (int i = 0; i < 64 * 16; ++i) blocks[i] = scalar_t(0);
    smesh::count_t slots[64];
    for (int i = 0; i < 64; ++i) slots[i] = (smesh::count_t)i;
    cvfem_hex8_ns_upwind_jacobian_add_slots_linear<false>(mu, g.adj, g.det, slots, blocks);
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) {
            const scalar_t *const blk = blocks + (size_t)(a * 8 + b) * 16;
            for (int rr = 0; rr < 3; ++rr)
                for (int cc = 0; cc < 3; ++cc)
                    Mf[(size_t)(a * 4 + rr) * N + (b * 4 + cc)] += blk[rr * 4 + cc];
        }

    const scalar_t half = scalar_t(0.5);
    for (int sc = 0; sc < CVFEM_HEX8_N_SCS; ++sc) {
        const int      i = CVFEM_HEX8_SCS[sc].i;
        const int      j = CVFEM_HEX8_SCS[sc].j;
        const int      d = sc >> 2;
        const scalar_t c = g.coeff[sc];
        for (int comp = 0; comp < 3; ++comp) {
            const scalar_t ac = g.A[d][comp];
            // Momentum: + qmid * A, qmid = (q_i + q_j)/2.
            Mf[(size_t)(i * 4 + comp) * N + (i * 4 + 3)] += half * ac;
            Mf[(size_t)(i * 4 + comp) * N + (j * 4 + 3)] += half * ac;
            Mf[(size_t)(j * 4 + comp) * N + (i * 4 + 3)] -= half * ac;
            Mf[(size_t)(j * 4 + comp) * N + (j * 4 + 3)] -= half * ac;
            // Continuity: + rho/2 (v_i + v_j) . A.
            Mf[(size_t)(i * 4 + 3) * N + (i * 4 + comp)] += rho * half * ac;
            Mf[(size_t)(i * 4 + 3) * N + (j * 4 + comp)] += rho * half * ac;
            Mf[(size_t)(j * 4 + 3) * N + (i * 4 + comp)] -= rho * half * ac;
            Mf[(size_t)(j * 4 + 3) * N + (j * 4 + comp)] -= rho * half * ac;
        }
        // Continuity: Rhie-Chow, + c (q_i - q_j).
        Mf[(size_t)(i * 4 + 3) * N + (i * 4 + 3)] += c;
        Mf[(size_t)(i * 4 + 3) * N + (j * 4 + 3)] -= c;
        Mf[(size_t)(j * 4 + 3) * N + (i * 4 + 3)] -= c;
        Mf[(size_t)(j * 4 + 3) * N + (j * 4 + 3)] += c;
    }
}

// Convection alone: what the full 32x32 matrix cannot carry, because its upwind weights
// depend on the state. No pressure gradient here -- that is in the matrix.
static SFEM_INLINE void sscvfem_convection_only(const scalar_t rho, const SSMacroGeom &g,
                                                const scalar_t *const SFEM_RESTRICT ux,
                                                const scalar_t *const SFEM_RESTRICT uy,
                                                const scalar_t *const SFEM_RESTRICT uz,
                                                const scalar_t *const SFEM_RESTRICT vx,
                                                const scalar_t *const SFEM_RESTRICT vy,
                                                const scalar_t *const SFEM_RESTRICT vz,
                                                const scalar_t *const SFEM_RESTRICT q,
                                                const scalar_t *const SFEM_RESTRICT p,
                                                const scalar_t *const SFEM_RESTRICT pgx,
                                                const scalar_t *const SFEM_RESTRICT pgy,
                                                const scalar_t *const SFEM_RESTRICT pgz,
                                                scalar_t *const SFEM_RESTRICT       r) {
    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      dd = s >> 2;
        const scalar_t ax = g.A[dd][0], ay = g.A[dd][1], az = g.A[dd][2];
        const scalar_t c  = g.coeff[s];

        const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                               half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                               half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
        const scalar_t mdot = rho * (half * (ux[i] + ux[j]) * ax + half * (uy[i] + uy[j]) * ay +
                                     half * (uz[i] + uz[j]) * az) - c * corr;
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);

        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               c * (q[i] - q[j]);
        const scalar_t apos = d_pos * dmdot;
        const scalar_t aneg = d_neg * dmdot;
        const scalar_t fx   = apos * ux[i] + mpos * vx[i] + aneg * ux[j] + mneg * vx[j];
        const scalar_t fy   = apos * uy[i] + mpos * vy[i] + aneg * uy[j] + mneg * vy[j];
        const scalar_t fz   = apos * uz[i] + mpos * vz[i] + aneg * uz[j] + mneg * vz[j];
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
    }
}

// Everything the momentum matrix does not carry: the whole sub-control-surface loop --
// convection, the pressure gradient, the continuity row and Rhie-Chow. Identical to the
// second half of sscvfem_action_hoisted; only the viscous block is missing, because that
// is what the gemm supplies.
static SFEM_INLINE void sscvfem_convective_remainder(const scalar_t rho, const SSMacroGeom &g,
                                          const scalar_t *const SFEM_RESTRICT ux,
                                          const scalar_t *const SFEM_RESTRICT uy,
                                          const scalar_t *const SFEM_RESTRICT uz,
                                          const scalar_t *const SFEM_RESTRICT vx,
                                          const scalar_t *const SFEM_RESTRICT vy,
                                          const scalar_t *const SFEM_RESTRICT vz,
                                          const scalar_t *const SFEM_RESTRICT q,
                                          const scalar_t *const SFEM_RESTRICT p,
                                          const scalar_t *const SFEM_RESTRICT pgx,
                                          const scalar_t *const SFEM_RESTRICT pgy,
                                          const scalar_t *const SFEM_RESTRICT pgz,
                                          scalar_t *const SFEM_RESTRICT       r) {
    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      dd = s >> 2;
        const scalar_t ax = g.A[dd][0], ay = g.A[dd][1], az = g.A[dd][2];
        const scalar_t c  = g.coeff[s];

        const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                               half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                               half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
        const scalar_t mdot = rho * (half * (ux[i] + ux[j]) * ax + half * (uy[i] + uy[j]) * ay +
                                     half * (uz[i] + uz[j]) * az) -
                              c * corr;
        const scalar_t sgn   = mdot > scalar_t(0) ? one : (mdot < scalar_t(0) ? -one : scalar_t(0));
        const scalar_t mpos  = half * (mdot + sgn * mdot);
        const scalar_t mneg  = half * (mdot - sgn * mdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);

        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               c * (q[i] - q[j]);
        const scalar_t dpos = d_pos * dmdot;
        const scalar_t dneg = d_neg * dmdot;
        const scalar_t qmid = half * (q[i] + q[j]);
        const scalar_t fx   = dpos * ux[i] + mpos * vx[i] + dneg * ux[j] + mneg * vx[j] + qmid * ax;
        const scalar_t fy   = dpos * uy[i] + mpos * vy[i] + dneg * uy[j] + mneg * vy[j] + qmid * ay;
        const scalar_t fz   = dpos * uz[i] + mpos * vz[i] + dneg * uz[j] + mneg * vz[j] + qmid * az;
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += dmdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= dmdot;
    }
}

// Assemble the element matrix on the fly, then apply it to every micro-element of the
// macro in ONE gemm rather than L^3 separate matvecs.
//
// The per-element matvec was the wrong shape and measured it: 16.15 ns/dof against 11.33
// for the direct evaluation, because a 32x32 matvec has no reuse -- the matrix is re-read
// for every element and the arithmetic intensity is that of a memory copy. Batched, the
// same matrix multiplies a 32 x L^3 block: 512 right-hand sides at L=8, so each entry of
// M is loaded once and used 512 times, which is what a gemm is for. This is the shape
// SFEM already uses for semi-structured linear elasticity.
//
// packed_elements_matmul_nonsym is the shared wrapper -- BLAS dgemm where SFEM_ENABLE_BLAS
// is on, a blocked loop otherwise. X and Y are element-major, contiguous per element.
inline SFEM_NOINLINE void sscvfem_apply_macro_local_em(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                       const scalar_t *const SFEM_RESTRICT dir,
                                                       scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_macro_local_em");
    const int L      = d.level;
    const int nxe    = d.nxe;
    const int nmicro = L * L * L;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lvx((size_t)nxe), lvy((size_t)nxe), lvz((size_t)nxe), lq((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        std::vector<scalar_t>     lout((size_t)nxe * N_FIELDS);
        std::vector<scalar_t>     M((size_t)SSCVFEM_EM_N * SSCVFEM_EM_N);
        // Velocity direction and viscous result for every micro-element, element-major.
        std::vector<scalar_t> X((size_t)nmicro * SSCVFEM_EM_N), Y((size_t)nmicro * SSCVFEM_EM_N);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)d.points[0][g];
                ly[(size_t)a]        = (scalar_t)d.points[1][g];
                lz[(size_t)a]        = (scalar_t)d.points[2][g];
                lux[(size_t)a]       = d.ux[(size_t)g];
                luy[(size_t)a]       = d.uy[(size_t)g];
                luz[(size_t)a]       = d.uz[(size_t)g];
                lp[(size_t)a]        = d.p[(size_t)g];
                lvx[(size_t)a]       = dir[(size_t)g * 4 + 0];
                lvy[(size_t)a]       = dir[(size_t)g * 4 + 1];
                lvz[(size_t)a]       = dir[(size_t)g * 4 + 2];
                lq[(size_t)a]        = dir[(size_t)g * 4 + 3];
                lpgx[(size_t)a]      = d.pgx[(size_t)g];
                lpgy[(size_t)a]      = d.pgy[(size_t)g];
                lpgz[(size_t)a]      = d.pgz[(size_t)g];
            }
            std::fill(lout.begin(), lout.end(), scalar_t(0));

            SSMacroGeom mg;
            scalar_t    ex[8], ey[8], ez[8];
            for (int a = 0; a < 8; ++a) {
                const int l = off[a];
                ex[a]       = lx[(size_t)l];
                ey[a]       = ly[(size_t)l];
                ez[a]       = lz[(size_t)l];
            }
            sscvfem_macro_geom(ex, ey, ez, rho, mu, d.rhie_chow_scale, mg);
            sscvfem_build_momentum_em(mg, mu, M.data());

            // Gather the direction for every micro-element into one block.
            int me = 0;
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi, ++me) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t *const SFEM_RESTRICT xe = X.data() + (size_t)me * SSCVFEM_EM_N;
                        for (int a = 0; a < 8; ++a) {
                            const int l   = base + off[a];
                            xe[a * 3 + 0] = lvx[(size_t)l];
                            xe[a * 3 + 1] = lvy[(size_t)l];
                            xe[a * 3 + 2] = lvz[(size_t)l];
                        }
                    }
                }
            }

            // One gemm for the whole macro-element: the same 24x24 viscous matrix applied
            // to all L^3 micro-elements at once, so each of its entries is loaded once and
            // used L^3 times. The _nonsym helper, not _sym: a Navier-Stokes element matrix
            // is not symmetric, and _sym's two branches disagree for such a matrix.
            packed_elements_matmul_nonsym(SSCVFEM_EM_N, nmicro, SSCVFEM_EM_N, M.data(), X.data(), Y.data());

            // The convective flux the matrix cannot carry, then scatter.
            me = 0;
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi, ++me) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                        }

                        // Viscous momentum from the gemm; the pressure row starts at zero
                        // and the sub-control-surface loop fills in the rest.
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        const scalar_t *const SFEM_RESTRICT ye = Y.data() + (size_t)me * SSCVFEM_EM_N;
                        for (int a = 0; a < 8; ++a) {
                            r[a * 4 + 0] = ye[a * 3 + 0];
                            r[a * 4 + 1] = ye[a * 3 + 1];
                            r[a * 4 + 2] = ye[a * 3 + 2];
                            r[a * 4 + 3] = scalar_t(0);
                        }

                        sscvfem_convective_remainder(rho, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz, r);
                        boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                         ux, uy, uz, vx, vy, vz, q, r);

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = lg[(size_t)a];
                for (int c = 0; c < N_FIELDS; ++c)
                    atomic_add(jv + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
            }
        }
    }
}

inline SFEM_NOINLINE void sscvfem_apply_macro_local_emfull(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                       const scalar_t *const SFEM_RESTRICT dir,
                                                       scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_macro_local_emfull");
    const int L      = d.level;
    const int nxe    = d.nxe;
    const int nmicro = L * L * L;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lvx((size_t)nxe), lvy((size_t)nxe), lvz((size_t)nxe), lq((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        std::vector<scalar_t>     lout((size_t)nxe * N_FIELDS);
        std::vector<scalar_t>     M((size_t)CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF);
        // Velocity direction and viscous result for every micro-element, element-major.
        std::vector<scalar_t> X((size_t)nmicro * CVFEM_HEX8_N_DOF), Y((size_t)nmicro * CVFEM_HEX8_N_DOF);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)d.points[0][g];
                ly[(size_t)a]        = (scalar_t)d.points[1][g];
                lz[(size_t)a]        = (scalar_t)d.points[2][g];
                lux[(size_t)a]       = d.ux[(size_t)g];
                luy[(size_t)a]       = d.uy[(size_t)g];
                luz[(size_t)a]       = d.uz[(size_t)g];
                lp[(size_t)a]        = d.p[(size_t)g];
                lvx[(size_t)a]       = dir[(size_t)g * 4 + 0];
                lvy[(size_t)a]       = dir[(size_t)g * 4 + 1];
                lvz[(size_t)a]       = dir[(size_t)g * 4 + 2];
                lq[(size_t)a]        = dir[(size_t)g * 4 + 3];
                lpgx[(size_t)a]      = d.pgx[(size_t)g];
                lpgy[(size_t)a]      = d.pgy[(size_t)g];
                lpgz[(size_t)a]      = d.pgz[(size_t)g];
            }
            std::fill(lout.begin(), lout.end(), scalar_t(0));

            SSMacroGeom mg;
            scalar_t    ex[8], ey[8], ez[8];
            for (int a = 0; a < 8; ++a) {
                const int l = off[a];
                ex[a]       = lx[(size_t)l];
                ey[a]       = ly[(size_t)l];
                ez[a]       = lz[(size_t)l];
            }
            sscvfem_macro_geom(ex, ey, ez, rho, mu, d.rhie_chow_scale, mg);
            sscvfem_build_full_em(mg, rho, mu, M.data());

            // Gather the direction for every micro-element into one block.
            int me = 0;
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi, ++me) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t *const SFEM_RESTRICT xe = X.data() + (size_t)me * CVFEM_HEX8_N_DOF;
                        for (int a = 0; a < 8; ++a) {
                            const int l   = base + off[a];
                            xe[a * 4 + 0] = lvx[(size_t)l];
                            xe[a * 4 + 1] = lvy[(size_t)l];
                            xe[a * 4 + 2] = lvz[(size_t)l];
                            xe[a * 4 + 3] = lq[(size_t)l];
                        }
                    }
                }
            }

            // One gemm for the whole macro-element: the same 24x24 viscous matrix applied
            // to all L^3 micro-elements at once, so each of its entries is loaded once and
            // used L^3 times. The _nonsym helper, not _sym: a Navier-Stokes element matrix
            // is not symmetric, and _sym's two branches disagree for such a matrix.
            packed_elements_matmul_nonsym(CVFEM_HEX8_N_DOF, nmicro, CVFEM_HEX8_N_DOF, M.data(), X.data(), Y.data());

            // The convective flux the matrix cannot carry, then scatter.
            me = 0;
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi, ++me) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                        }

                        // Everything but convection comes from the gemm, pressure rows included.
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        const scalar_t *const SFEM_RESTRICT ye = Y.data() + (size_t)me * CVFEM_HEX8_N_DOF;
                        for (int i2 = 0; i2 < CVFEM_HEX8_N_DOF; ++i2) r[i2] = ye[i2];

                        sscvfem_convection_only(rho, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz, r);
                        boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                         ux, uy, uz, vx, vy, vz, q, r);

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = lg[(size_t)a];
                for (int c = 0; c < N_FIELDS; ++c)
                    atomic_add(jv + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
            }
        }
    }
}
