#pragma once

// HEX8 CVFEM Navier-Stokes on a semi-structured (sshex8) mesh.
//
// This exists to answer one question: how much of the flat kernel's cost is the indexed
// gather? T2 measured the flat matrix-free action at 2.36 ns/dof on Grace while its
// compulsory traffic ran at 4% of memory peak, so it is limited by neither the data it
// must move nor arithmetic -- and every element re-reads its eight nodes through
// d.elems[a][e], so each node is fetched about eight times per sweep.
//
// A semi-structured mesh removes that by construction. Nodes within a macro-element are
// numbered lexicographically, lidx(L,x,y,z) = z(L+1)^2 + y(L+1) + x, so the eight corners
// of every micro-element sit at the SAME eight constant offsets from their base:
//
//     {0, 1, Lp1+1, Lp1, Lp1^2, Lp1^2+1, Lp1^2+Lp1+1, Lp1^2+Lp1}
//
// So a macro-element's (L+1)^3 nodes can be gathered once and its L^3 micro-elements read
// from contiguous local buffers with no indirection at all. Indexed loads per element
// fall from 8 to (L+1)^3/L^3 -- 1.95 at L=4, 1.42 at L=8, 1.20 at L=16 -- and the atomic
// scatter falls by the same factor, since a macro-element writes its nodes once instead
// of once per element-node incidence.
//
// Two variants are provided and they must agree to round-off. `naive` keeps the flat
// gather, reading every node through the global id, and exists only as the control:
// it is the same physics on the same mesh, differing from `macro_local` in the gather
// alone, so the difference between them is the transformation and nothing else.
//
// The element kernels are reused verbatim from cvfem_hex8_ns_core.hpp. Nothing about the
// physics is reimplemented here, which is what makes the comparison meaningful.

#include "cvfem_hex8_ns_core.hpp"

#include "packed_elements.hpp"   // packed_elements_matmul_nonsym: BLAS gemm, loop fallback
#include "smesh_mesh.hpp"

#include <cmath>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------

struct SSMeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    int                          level{0};
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nmacro{0};
    int                          nxe{0};  // (L+1)^3, nodes per macro-element
    smesh::idx_t               **elems{nullptr};
    smesh::geom_t              **points{nullptr};
    scalar_t                     Lx{1}, Ly{1}, Lz{1};
    scalar_t                     rhie_chow_scale{1};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> pgx, pgy, pgz;
};

static SFEM_INLINE int sscvfem_lidx(const int L, const int x, const int y, const int z) {
    const int Lp1 = L + 1;
    return z * (Lp1 * Lp1) + y * Lp1 + x;
}

// The eight corner offsets, constant for every micro-element in the macro-element.
static SFEM_INLINE void sscvfem_corner_offsets(const int L, int off[8]) {
    const int Lp1 = L + 1;
    off[0]        = 0;
    off[1]        = 1;
    off[2]        = Lp1 + 1;
    off[3]        = Lp1;
    off[4]        = Lp1 * Lp1;
    off[5]        = Lp1 * Lp1 + 1;
    off[6]        = Lp1 * Lp1 + Lp1 + 1;
    off[7]        = Lp1 * Lp1 + Lp1;
}

inline void sscvfem_init(SSMeshData &d, const std::shared_ptr<smesh::Mesh> &mesh, const int level) {
    d.mesh   = mesh;
    d.level  = level;
    d.nnodes = mesh->n_nodes();
    d.nmacro = mesh->n_elements(0);
    d.nxe    = (level + 1) * (level + 1) * (level + 1);
    d.elems  = mesh->elements(0)->data();
    d.points = mesh->points()->data();

    scalar_t hi[3] = {0, 0, 0};
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        hi[0] = std::max(hi[0], (scalar_t)d.points[0][i]);
        hi[1] = std::max(hi[1], (scalar_t)d.points[1][i]);
        hi[2] = std::max(hi[2], (scalar_t)d.points[2][i]);
    }
    d.Lx = hi[0];
    d.Ly = hi[1];
    d.Lz = hi[2];

    d.ux.assign((size_t)d.nnodes, 0);
    d.uy.assign((size_t)d.nnodes, 0);
    d.uz.assign((size_t)d.nnodes, 0);
    d.p.assign((size_t)d.nnodes, 0);
}

inline void sscvfem_unpack(SSMeshData &d, const scalar_t *const SFEM_RESTRICT x) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[(size_t)i] = x[(size_t)i * 4 + 0];
        d.uy[(size_t)i] = x[(size_t)i * 4 + 1];
        d.uz[(size_t)i] = x[(size_t)i * 4 + 2];
        d.p[(size_t)i]  = x[(size_t)i * 4 + 3];
    }
}

// Geometry of one micro-element from its eight corners. The macro-elements here come from
// a box mesh, so each micro-element is affine and the adjugate is constant; evaluating at
// the centre is therefore exact rather than an approximation.
static SFEM_INLINE void sscvfem_micro_geom(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                                           scalar_t adj[9], scalar_t *det) {
    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, det);
}

// ---------------------------------------------------------------------------
// Nodal pressure gradient, the pre-pass Rhie-Chow interpolation needs. Mirrors
// assemble_nodal_p_grad: a volume-weighted average of the element gradients.

inline void sscvfem_nodal_p_grad(SSMeshData &d) {
    d.pgx.assign((size_t)d.nnodes, 0);
    d.pgy.assign((size_t)d.nnodes, 0);
    d.pgz.assign((size_t)d.nnodes, 0);
    std::vector<scalar_t> w((size_t)d.nnodes, 0);

    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        std::vector<scalar_t> lx((size_t)d.nxe), ly((size_t)d.nxe), lz((size_t)d.nxe), lp((size_t)d.nxe);
        std::vector<smesh::idx_t> lg((size_t)d.nxe);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            for (int a = 0; a < d.nxe; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)d.points[0][g];
                ly[(size_t)a]        = (scalar_t)d.points[1][g];
                lz[(size_t)a]        = (scalar_t)d.points[2][g];
                lp[(size_t)a]        = d.p[(size_t)g];
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t  ex[8], ey[8], ez[8], ep[8];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            ex[a]       = lx[(size_t)l];
                            ey[a]       = ly[(size_t)l];
                            ez[a]       = lz[(size_t)l];
                            ep[a]       = lp[(size_t)l];
                        }
                        scalar_t adj[9], det;
                        sscvfem_micro_geom(ex, ey, ez, adj, &det);
                        const scalar_t vol = std::fabs(det);
                        if (vol < scalar_t(1e-30)) continue;
                        scalar_t gx, gy, gz;
                        cvfem_hex8_grad_scalar(adj, det, ep, gx, gy, gz);
                        for (int a = 0; a < 8; ++a) {
                            const smesh::idx_t id = lg[(size_t)(base + off[a])];
                            atomic_add(d.pgx.data(), id, vol * gx);
                            atomic_add(d.pgy.data(), id, vol * gy);
                            atomic_add(d.pgz.data(), id, vol * gz);
                            atomic_add(w.data(), id, vol);
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        if (w[(size_t)i] <= scalar_t(0)) continue;
        const scalar_t inv = scalar_t(1) / w[(size_t)i];
        d.pgx[(size_t)i] *= inv;
        d.pgy[(size_t)i] *= inv;
        d.pgz[(size_t)i] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Control: the flat gather, on the semi-structured mesh. Every micro-element reads its
// eight nodes through the global id, exactly as the flat kernel does.

inline SFEM_NOINLINE void sscvfem_apply_naive(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                              const scalar_t *const SFEM_RESTRICT dir,
                                              scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_naive");
    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
        for (int zi = 0; zi < L; ++zi) {
            for (int yi = 0; yi < L; ++yi) {
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);

                    smesh::idx_t g[8];
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                    scalar_t     vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                    scalar_t     r[CVFEM_HEX8_N_DOF];
                    for (int a = 0; a < 8; ++a) {
                        g[a]   = d.elems[base + off[a]][e];
                        x[a]   = (scalar_t)d.points[0][g[a]];
                        y[a]   = (scalar_t)d.points[1][g[a]];
                        z[a]   = (scalar_t)d.points[2][g[a]];
                        ux[a]  = d.ux[(size_t)g[a]];
                        uy[a]  = d.uy[(size_t)g[a]];
                        uz[a]  = d.uz[(size_t)g[a]];
                        p[a]   = d.p[(size_t)g[a]];
                        vx[a]  = dir[(size_t)g[a] * 4 + 0];
                        vy[a]  = dir[(size_t)g[a] * 4 + 1];
                        vz[a]  = dir[(size_t)g[a] * 4 + 2];
                        q[a]   = dir[(size_t)g[a] * 4 + 3];
                        pgx[a] = d.pgx[(size_t)g[a]];
                        pgy[a] = d.pgy[(size_t)g[a]];
                        pgz[a] = d.pgz[(size_t)g[a]];
                    }

                    const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(x, y, z, adj, &det);
                    cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r, rc, p);
                    boundary_scs_add_jacobian_action(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                                     vx, vy, vz, q, r);

                    for (int a = 0; a < 8; ++a)
                        for (int c = 0; c < N_FIELDS; ++c)
                            atomic_add(jv + (ptrdiff_t)g[a] * N_FIELDS + c, 0, r[a * 4 + c]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The transformation: gather the macro-element's nodes once, run its L^3 micro-elements
// against constant offsets into contiguous local buffers, scatter once at the end.

inline SFEM_NOINLINE void sscvfem_apply_macro_local(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                    const scalar_t *const SFEM_RESTRICT dir,
                                                    scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_macro_local");
    const int L   = d.level;
    const int nxe = d.nxe;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        // One allocation per thread for the whole sweep, not per macro-element.
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lvx((size_t)nxe), lvy((size_t)nxe), lvz((size_t)nxe), lq((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        std::vector<scalar_t>     lout((size_t)nxe * N_FIELDS);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            // Gather once. This is the only indirection in the sweep.
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

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];  // no indirection
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

                        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
                        scalar_t           adj[9], det;
                        sscvfem_micro_geom(x, y, z, adj, &det);
                        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r, rc, p);
                        boundary_scs_add_jacobian_action(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                                         vx, vy, vz, q, r);

                        // Accumulate locally: no atomic, no contention, contiguous.
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            // Scatter once per macro node instead of once per element-node incidence.
            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = lg[(size_t)a];
                for (int c = 0; c < N_FIELDS; ++c)
                    atomic_add(jv + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// macro_local, plus the geometry hoisted out of the micro-element loop.
//
// The flat kernel loads a precomputed adjugate and determinant per element; the two
// variants above recompute the Jacobian from eight corners for every micro-element, so
// they were doing strictly more work than the kernel they are meant to beat. Inside an
// affine macro-element every micro-element is a translate of the same box, so adj and det
// are invariant over the whole L^3 sweep and belong outside it.
//
// This is only valid when the macro-element is affine, which is true of the box meshes
// benchmarked here and false in general -- a trilinear macro-element has a Jacobian that
// varies across its lattice. The assert guards it: the geometry of the last micro-element
// is compared against the hoisted value, so a curved macro-element fails loudly rather
// than silently returning a wrong operator.
inline SFEM_NOINLINE void sscvfem_apply_macro_local_affine(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                           const scalar_t *const SFEM_RESTRICT dir,
                                                           scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_macro_local_affine");
    const int L   = d.level;
    const int nxe = d.nxe;
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

            // Once per macro-element, from its first micro-element.
            scalar_t madj[9], mdet;
            {
                scalar_t ex[8], ey[8], ez[8];
                for (int a = 0; a < 8; ++a) {
                    const int l = off[a];
                    ex[a]       = lx[(size_t)l];
                    ey[a]       = ly[(size_t)l];
                    ez[a]       = lz[(size_t)l];
                }
                sscvfem_micro_geom(ex, ey, ez, madj, &mdet);
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
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

                        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
                        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, madj, mdet, ux, uy, uz, vx, vy, vz, q, r, rc, p);
                        boundary_scs_add_jacobian_action(rho, mu, 0, madj, mdet, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                                         vx, vy, vz, q, r);

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

// ---------------------------------------------------------------------------
// One Jacobian per macro-element, and the linear terms lifted out of the loop.
//
// Under the affine-macro assumption the geometry is invariant over all L^3
// micro-elements, and three things the action kernel recomputes per element become
// loop invariants:
//
//   * the direction areas A[3][3], from cvfem_hex8_dir_areas(adj, A)
//   * the twelve node-separation vectors d_ij across the sub-control-surfaces
//   * the twelve Rhie-Chow coefficients rho * Df * A^2 / (A.d), Df = scale h^2/(2 mu)
//
// The last is the valuable one. It is pure geometry, so it is constant here, and each
// evaluation costs a square root and a division -- twelve of them per micro-element,
// L^3 times per macro, all producing the same twelve numbers.
//
// What cannot be hoisted is the viscous term. It is linear in the direction v and its
// geometry is invariant, but it still contracts against v, which changes per element:
// cvfem_hex8_grad_sumfact has to run either way. Only its geometric factors are lifted.
// The convective term stays inside entirely -- its upwind switch depends on the state.
//
// This is the one place in this file that restates kernel arithmetic rather than calling
// the shared kernel, which is why the benchmark checks it against the naive variant like
// the others. If it stops agreeing, this is the copy that drifted.

struct SSMacroGeom {
    scalar_t adj[9];
    scalar_t det;
    scalar_t A[3][3];
    scalar_t coeff[CVFEM_HEX8_N_SCS];
    scalar_t dvec[CVFEM_HEX8_N_SCS][3];
};

inline void sscvfem_macro_geom(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                               const scalar_t rho, const scalar_t mu, const scalar_t rc_scale,
                               SSMacroGeom &g) {
    sscvfem_micro_geom(x, y, z, g.adj, &g.det);
    cvfem_hex8_dir_areas(g.adj, g.A);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
        g.dvec[s][0] = x[j] - x[i];
        g.dvec[s][1] = y[j] - y[i];
        g.dvec[s][2] = z[j] - z[i];
        g.coeff[s]   = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc_scale,
                                                       g.dvec[s][0], g.dvec[s][1], g.dvec[s][2],
                                                       g.A[d][0], g.A[d][1], g.A[d][2]);
    }
}

// Mirrors cvfem_hex8_ns_upwind_jacobian_action with the invariants passed in.
static SFEM_INLINE void sscvfem_action_hoisted(const scalar_t rho, const scalar_t mu, const SSMacroGeom &g,
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
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t dgrad[9];
    cvfem_hex8_grad_sumfact(g.adj, g.det, vx, vy, vz, dgrad);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6], dgrad[7],
                            dgrad[8], g.A[d][0], g.A[d][1], g.A[d][2], tx, ty, tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      d  = s >> 2;
        const scalar_t ax = g.A[d][0], ay = g.A[d][1], az = g.A[d][2];
        const scalar_t c  = g.coeff[s];

        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);

        // -coeff * ((p_j - p_i) - avg(grad p) . d), with coeff and d both loop invariants.
        const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                               half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                               half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
        const scalar_t mdot_rc = -c * corr;

        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
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

inline SFEM_NOINLINE void sscvfem_apply_macro_local_hoisted(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT dir,
                                                            scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_macro_local_hoisted");
    const int L   = d.level;
    const int nxe = d.nxe;
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

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
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

                        sscvfem_action_hoisted(rho, mu, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz, r);
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

// ---------------------------------------------------------------------------
// The linear part as a small dense matrix, applied as a matvec.
//
// For a fixed state the Jacobian action is linear in the direction, and under the
// affine-macro assumption everything in it except the convective flux has coefficients
// that are pure geometry. So that part is one constant 32x32 matrix for the whole macro,
// and applying it is a dense matvec -- the same shape as the element-matrix path SFEM
// already uses for semi-structured linear elasticity (sfem_SemiStructuredEMLinearElasticity
// and operators/stencil/sshex8_stencil_element_matrix_apply*).
//
// The matrix is not written out by hand. With u = 0, p = 0 and grad p = 0 the existing
// action kernel reduces exactly to the geometry-linear operator: mdot vanishes, so the
// upwind weights mpos and mneg vanish with it, and what survives is the viscous term, the
// pressure gradient qmid*A in the momentum rows, and the whole continuity row. So the
// matrix is obtained by probing the unmodified kernel with the 32 unit vectors -- it is
// consistent with the kernel by construction, which the hand-written variant above is not.
// 32 probes are amortised over L^3 micro-elements: 6% at L=8.
//
// What is left outside is the convective momentum flux alone, whose upwind weights depend
// on the state. The continuity row needs no remainder at all.
//
// Whether this is faster is not obvious and is not argued here: the matvec is about twice
// the FLOPs of evaluating those terms directly, but it is branch-free, contiguous, and the
// matrix stays in L1 across the whole macro-element. The benchmark decides.

// ---------------------------------------------------------------------------
// The default.
//
// Measured best on both machines tried, at 4343300 dofs and L=8: 1.097 ns/dof on one
// Grace socket, 2.29x the flat kernel, and 10.876 on an M1. Of that, gathering the
// macro-element's nodes once is worth about 1.44x and lifting the affine-macro
// invariants out of the micro-element loop a further 1.28x.
//
// The alternatives are kept rather than deleted. sscvfem_apply_naive is the correctness
// control and is what the benchmark checks every other variant against.
// sscvfem_apply_macro_local and _affine are the intermediate steps, which is how the
// 1.44x and 1.28x above are attributed. The two element-matrix variants lost and moved
// to subpar/cvfem_sshex8_em.hpp.
inline void sscvfem_apply(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                          const scalar_t *const SFEM_RESTRICT dir,
                          scalar_t *const SFEM_RESTRICT       jv) {
    sscvfem_apply_macro_local_hoisted(d, rho, mu, dir, jv);
}
