#pragma once

// Two families of HEX8 CVFEM headers live in this directory and they are not
// interchangeable. This one backs the steady solver and the sfem::Op; the
// cvfem_hex8_layout_*.hpp family backs the throughput benchmark. They define sixteen
// of the same names -- MeshData, BSR4, GeomKind, assemble_jacobian_atomic_sumfact and
// the residual entry points among them -- and the assembly ones differ in physics, not
// just in layout: the benchmark's carry no boundary sub-control-surface or Rhie-Chow
// terms, because the benchmark has no boundaries to close. Including both would
// otherwise produce a page of redefinition errors that says nothing about why.
#if defined(CVFEM_HEX8_LAYOUT_COMMON_HPP)
#error "cvfem_hex8_ns_core.hpp (solver) and cvfem_hex8_layout_*.hpp (benchmark) define the same names with different physics -- include one family per translation unit. Drivers that only need the operator should include cvfem_hex8_ns_op.hpp instead, which exposes neither."
#endif
#define CVFEM_HEX8_NS_CORE_HPP
// Core of the HEX8 CVFEM Navier-Stokes spike: mesh state, kernels, assembly and
// the verification helpers. Split out of cvfem_hex8_ns_steady.cpp so that a second
// translation unit -- the sfem::Op wrapper -- can drive the same code. Moving code
// only: no logic changed, and file-scope `static` became `inline` so the header can
// be included from more than one TU.

#include "sfem_BSR.hpp"
#include "sfem_Operator.hpp"
#include "sfem_base.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_context.hpp"
#include "sfem_openmp_blas.hpp"
#include "smesh_buffer.hpp"
#include "smesh_context.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_output.hpp"
#include "smesh_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline __attribute__((always_inline))
#endif

#ifndef SFEM_NOINLINE
#define SFEM_NOINLINE __attribute__((noinline))
#endif

using scalar_t = double;

static constexpr int N_FIELDS = 4;

#include "cvfem_portability.hpp"

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_pack_coloring.hpp"

enum class GeomKind { Affine, Isoparam };
enum class FlowCase { Poiseuille, Couette };
enum class InitKind { Zero, Exact };

struct PackedData;

struct MeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nelements{0};
    smesh::idx_t               **elems{nullptr};
    smesh::geom_t              **points{nullptr};
    scalar_t                     Lx{1};
    scalar_t                     Ly{1};
    scalar_t                     Lz{1};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> rx, ry, rz, rc;
    std::vector<scalar_t> pgx, pgy, pgz;
    std::vector<scalar_t> jacobian_adjugate[9];
    std::vector<scalar_t> jacobian_determinant;
    PackedData           *packed{nullptr};
    const PackColoring   *coloring{nullptr};
    scalar_t              rhie_chow_scale{1};
};

#include "cvfem_hex8_ns_packed.hpp"
#include "cvfem_hex8_ns_upwind_sympy_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

struct BSR4 {
    std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
    const smesh::count_t                         *rowptr{nullptr};
    const smesh::idx_t                           *colidx{nullptr};
    smesh::SharedBuffer<scalar_t>                 values;
    std::vector<smesh::count_t>                   element_slots;
    std::vector<smesh::count_t>                   diag_slots;
    ptrdiff_t                                     nnz{0};

    // When set, assembly writes here instead of into `values`. The sfem::Op wrapper
    // is handed a values buffer by its caller and owns neither the buffer nor the
    // graph, so it points this at the caller's array and reuses the slot caches.
    scalar_t *external_values{nullptr};

    scalar_t *data() const { return external_values ? external_values : values->data(); }
};

inline void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <output_folder>\n"
                 "\n"
                 "HEX8 CVFEM Navier-Stokes channel verification (textbook Couette / Poiseuille).\n"
                 "Domain: [0,Lx] x [0,Ly] x [0,Lz]  (default 4 x 1 x 1).\n"
                 "  walls y=0,Ly     no-slip (Couette: top lid u=(U,0,0))\n"
                 "  span  z=0,Lz     symmetry uz=0\n"
                 "  x=0 and x=Lx     fully-developed profile (inlet/outlet)\n"
                 "  pressure         one Dirichlet pin (CVs closed by boundary SCS)\n"
                 "Writes <output_folder>/mesh and <output_folder>/out.\n"
                 "ParaView: create_xdmf.sh <output_folder>\n"
                 "\n"
                 "Environment:\n"
                 "  SFEM_CASE            poiseuille | couette | coutte\n"
                 "  SFEM_N               cells in y (wall-normal, default 8)\n"
                 "  SFEM_NX SFEM_NY SFEM_NZ   override cells per direction\n"
                 "  SFEM_LX SFEM_LY SFEM_LZ   channel size (default 4, 1, 1)\n"
                 "  SFEM_RHO SFEM_MU SFEM_U   density, viscosity, velocity scale\n"
                 "  SFEM_GEOM            affine | isoparam (default affine)\n"
                 "  SFEM_INIT            zero | exact (default zero)\n"
                 "  SFEM_NL_MAX_IT       Newton iterations (default 40)\n"
                 "  SFEM_NL_RTOL SFEM_NL_ATOL\n"
                 "  SFEM_LSOLVE_RTOL SFEM_LSOLVE_ATOL SFEM_LSOLVE_MAX_IT\n"
                 "  SFEM_VERIFY_TOL      fail if velocity Linf exceeds this (default 1e-2)\n"
                 "  SFEM_VERBOSE         BiCGStab monitor (default 0)\n"
                 "  SFEM_NO_PREC         disable Jacobi (default 0)\n"
                 "  SFEM_MATRIX_FREE     1: Krylov uses J(u)v (default 0 = assembled BSR)\n"
                 "  SFEM_CHECK_JV        1: print |J_mf v - J_asm v| after first assembly\n"
                 "  SFEM_RHIE_CHOW       colocated mass-flux interpolation (default 1)\n"
                 "  SFEM_RHIE_CHOW_SCALE D_f = scale * h^2 / (2 mu) (default 1)\n"
                 "  SFEM_PACK_SIZE       affine packed SIMD (default 2048; 0 = atomic)\n"
                 "  SFEM_PC_PSCALE       Schur scaling of the pressure block:\n"
                 "                       inv_pp = PSCALE / V_p (default 0 = use 1/A_pp).\n"
                 "                       Tuned, not physical: PSCALE * rc_scale ~ 0.1 over\n"
                 "                       rc_scale 0.5..2. Helps at high Re only.\n"
                 "  SFEM_PC_PDAMP        damping on the 1 / A_pp pressure block (default 1)\n"
                 "  SFEM_NL_CONTINUATION 0: skip the Re=1 continuation stage (default 1)\n"
                 "  SFEM_PC_SIMPLE       1: pressure block from the SIMPLE Schur diagonal\n"
                 "                       diag(C - B diag(A_uu)^-1 B^T); overrides PSCALE\n",
                 argv0);
}

inline GeomKind parse_geom(const std::string &name) {
    if (name == "isoparam") return GeomKind::Isoparam;
    return GeomKind::Affine;
}

inline bool parse_case(const std::string &name, FlowCase &out) {
    if (name == "poiseuille") {
        out = FlowCase::Poiseuille;
        return true;
    }
    if (name == "couette" || name == "coutte") {
        out = FlowCase::Couette;
        return true;
    }
    return false;
}


inline void exact_state(const FlowCase flow,
                        const scalar_t mu,
                        const scalar_t U,
                        const scalar_t Lx,
                        const scalar_t Ly,
                        const scalar_t x,
                        const scalar_t y,
                        const scalar_t z,
                        scalar_t      &ux,
                        scalar_t      &uy,
                        scalar_t      &uz,
                        scalar_t      &p) {
    (void)z;
    uy = scalar_t(0);
    uz = scalar_t(0);
    if (flow == FlowCase::Couette) {
        ux = U * (y / Ly);
        p  = scalar_t(0);
        return;
    }
    const scalar_t G = scalar_t(8) * mu * U / (Ly * Ly);
    ux               = scalar_t(4) * U * y * (Ly - y) / (Ly * Ly);
    p                = G * (scalar_t(0.5) * Lx - x);
}

inline void reset_residual(MeshData &d) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.rx[i] = scalar_t(0);
        d.ry[i] = scalar_t(0);
        d.rz[i] = scalar_t(0);
        d.rc[i] = scalar_t(0);
    }
}

inline BSR4 make_bsr4(const std::shared_ptr<smesh::Mesh> &mesh) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::make_bsr4");
    BSR4 b;
    b.graph  = mesh->node_to_node_graph();
    b.rowptr = b.graph->rowptr()->data();
    b.colidx = b.graph->colidx()->data();
    b.nnz    = b.graph->nnz();
    b.values = smesh::create_host_buffer<scalar_t>((size_t)b.nnz * 16);
    return b;
}

inline void zero_bsr4(BSR4 &b) { cvfem_zero_scalars(b.data(), b.nnz * 16); }

SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const smesh::idx_t id, const scalar_t value) {
    CVFEM_ATOMIC_ADD(f[id], value);
}

SFEM_INLINE smesh::count_t find_bsr_slot(const smesh::count_t *const SFEM_RESTRICT rowptr,
                                                const smesh::idx_t *const SFEM_RESTRICT   colidx,
                                                const smesh::idx_t                        row,
                                                const smesh::idx_t                        col) {
    const smesh::count_t begin = rowptr[row];
    const smesh::count_t end   = rowptr[row + 1];
    for (smesh::count_t k = begin; k < end; ++k) {
        if (colidx[k] == col) return k;
    }
    return begin;
}

inline void precompute_element_bsr_slots(const MeshData &d, BSR4 &b) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::precompute_element_bsr_slots");
    b.element_slots.resize((size_t)d.nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES);
    b.diag_slots.resize((size_t)d.nnodes);

    smesh::idx_t **const SFEM_RESTRICT  elems = d.elems;
    smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t row = elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const smesh::idx_t col                = elems[bnode][e];
                slots[(size_t)e * 64 + a * 8 + bnode] = find_bsr_slot(b.rowptr, b.colidx, row, col);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < d.nnodes; ++row) {
        b.diag_slots[(size_t)row] = find_bsr_slot(b.rowptr, b.colidx, (smesh::idx_t)row, (smesh::idx_t)row);
    }
}

SFEM_INLINE void gather_element_fields(const MeshData               &d,
                                              const ptrdiff_t               e,
                                              scalar_t *const SFEM_RESTRICT ux,
                                              scalar_t *const SFEM_RESTRICT uy,
                                              scalar_t *const SFEM_RESTRICT uz,
                                              scalar_t *const SFEM_RESTRICT p) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t g = d.elems[a][e];
        ux[a]                = d.ux[g];
        uy[a]                = d.uy[g];
        uz[a]                = d.uz[g];
        p[a]                 = d.p[g];
    }
}

SFEM_INLINE void gather_element_coords(const MeshData               &d,
                                              const ptrdiff_t               e,
                                              scalar_t *const SFEM_RESTRICT x,
                                              scalar_t *const SFEM_RESTRICT y,
                                              scalar_t *const SFEM_RESTRICT z) {
    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t g = d.elems[a][e];
        x[a]                 = scalar_t(px[g]);
        y[a]                 = scalar_t(py[g]);
        z[a]                 = scalar_t(pz[g]);
    }
}


/* Nodal ∇p (volume-weighted element gradients). Element-local ∇p makes
   (p_j-p_i)-∇p_el·Δx vanish for any field that is linear on a HEX8, including the
   axis-aligned odd-even mode p=(-1)^i. Averaging neighboring elements restores
   the standard Rhie–Chow term 0.5(∇p_i+∇p_j) and still annihilates globally linear p. */
inline void assemble_nodal_p_grad(MeshData &d, const GeomKind geom_kind) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_nodal_p_grad");
    d.pgx.assign((size_t)d.nnodes, scalar_t(0));
    d.pgy.assign((size_t)d.nnodes, scalar_t(0));
    d.pgz.assign((size_t)d.nnodes, scalar_t(0));
    std::vector<scalar_t> w((size_t)d.nnodes, scalar_t(0));

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t p[8], gx, gy, gz, vol;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) p[a] = d.p[d.elems[a][e]];
        if (geom_kind == GeomKind::Isoparam) {
            scalar_t x[8], y[8], z[8], dN[CVFEM_HEX8_N_NODES][3];
            gather_element_coords(d, e, x, y, z);
            cvfem_hex8_dn_ref(scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), dN);
            scalar_t adj[9], det;
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
            vol = std::fabs(det);
            if (vol < scalar_t(1e-30)) continue;
            scalar_t dr = 0, ds = 0, dt = 0;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                dr += p[a] * dN[a][0];
                ds += p[a] * dN[a][1];
                dt += p[a] * dN[a][2];
            }
            cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, gx, gy, gz);
        } else {
            scalar_t adj[9], det;
            cvfem_hex8_load_adj(d, e, adj, &det);
            vol = std::fabs(det);
            if (vol < scalar_t(1e-30)) continue;
            cvfem_hex8_grad_scalar(adj, det, p, gx, gy, gz);
        }
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t id = d.elems[a][e];
            atomic_add(d.pgx.data(), id, vol * gx);
            atomic_add(d.pgy.data(), id, vol * gy);
            atomic_add(d.pgz.data(), id, vol * gz);
            atomic_add(w.data(), id, vol);
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

SFEM_INLINE void gather_element_pgrad(const MeshData               &d,
                                             const ptrdiff_t               e,
                                             scalar_t *const SFEM_RESTRICT gx,
                                             scalar_t *const SFEM_RESTRICT gy,
                                             scalar_t *const SFEM_RESTRICT gz) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t id = d.elems[a][e];
        gx[a]                 = d.pgx[id];
        gy[a]                 = d.pgy[id];
        gz[a]                 = d.pgz[id];
    }
}


SFEM_INLINE void gather_element_dir(const MeshData &d, const ptrdiff_t e, const scalar_t *const SFEM_RESTRICT dir,
                                           scalar_t *const SFEM_RESTRICT vx, scalar_t *const SFEM_RESTRICT vy,
                                           scalar_t *const SFEM_RESTRICT vz, scalar_t *const SFEM_RESTRICT q);

inline SFEM_NOINLINE void apply_boundary_scs_residual(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                      const int isoparam) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_boundary_scs_residual");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) cvfem_hex8_load_adj(d, e, adj, &det);
        boundary_scs_add_residual(rho, mu, isoparam, isoparam ? nullptr : adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                  p, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            if (r[a * 4 + 0] == scalar_t(0) && r[a * 4 + 1] == scalar_t(0) && r[a * 4 + 2] == scalar_t(0) &&
                r[a * 4 + 3] == scalar_t(0))
                continue;
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

inline SFEM_NOINLINE void apply_boundary_scs_jacobian_action(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                             const int isoparam, const scalar_t *const SFEM_RESTRICT dir,
                                                             scalar_t *const SFEM_RESTRICT jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        gather_element_dir(d, e, dir, vx, vy, vz, q);
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) cvfem_hex8_load_adj(d, e, adj, &det);
        boundary_scs_add_jacobian_action(rho, mu, isoparam, isoparam ? nullptr : adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux,
                                         uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            if (r[a * 4 + 0] == scalar_t(0) && r[a * 4 + 1] == scalar_t(0) && r[a * 4 + 2] == scalar_t(0) &&
                r[a * 4 + 3] == scalar_t(0))
                continue;
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

inline SFEM_NOINLINE void apply_residual_atomic_sumfact(MeshData &d, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual_sumfact");
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        scalar_t adj[9], det;
        cvfem_hex8_load_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r, rc);
        boundary_scs_add_residual(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

inline SFEM_NOINLINE void apply_residual_atomic_isoparam(MeshData &d, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual_isoparam");
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, ux, uy, uz, p, r, rc);
        boundary_scs_add_residual(rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

// Colored assembly. Packs sharing a color touch no common node, so the element
// kernels can accumulate straight into the global BSR with plain (non-atomic)
// updates. Compared with the atomic sweep this drops ~1024 atomic
// read-modify-writes per element and keeps each pack's rows cache-resident.
inline SFEM_NOINLINE void assemble_jacobian_colored_sumfact(MeshData           &d,
                                                            const PackedData   &p,
                                                            const PackColoring &c,
                                                            BSR4               &b,
                                                            const scalar_t      rho,
                                                            const scalar_t      mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_jacobian_colored_sumfact");
    scalar_t *const SFEM_RESTRICT             values = b.data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel
    {
        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t pack    = c.pack_order[(size_t)i];
                const ptrdiff_t e_start = pack * p.n_elements_per_pack;
                const ptrdiff_t e_end   = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], pp[8];
                    gather_element_coords(d, e, x, y, z);
                    gather_element_fields(d, e, ux, uy, uz, pp);
                    scalar_t pgx[8], pgy[8], pgz[8];
                    gather_element_pgrad(d, e, pgx, pgy, pgz);
                    const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
                    scalar_t           adj[9], det;
                    cvfem_hex8_load_adj(d, e, adj, &det);
                    const smesh::count_t *const SFEM_RESTRICT es = slots + (size_t)e * 64;
                    // One rc-aware kernel instead of the SymPy kernel plus a separate
                    // Rhie-Chow pass. The SymPy kernel picks the upwind direction from
                    // rho (u.A) alone, while the residual and the matrix-free action use
                    // rho (u.A) + mdot_rc, so the two operators disagreed wherever the
                    // Rhie-Chow flux could flip the sign. This kernel takes rc and p and
                    // routes mdot_rc into the same switch, so assembled and matrix-free
                    // are the same operator by construction.
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, ux, uy, uz, es, values, rc, pp);
                    boundary_scs_add_jacobian<false>(
                            rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, es, values);
                }
            }
        }
    }
}

inline SFEM_NOINLINE void assemble_jacobian_atomic_sumfact(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_jacobian_sumfact");
    scalar_t *const SFEM_RESTRICT             values = b.data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        scalar_t adj[9], det;
        cvfem_hex8_load_adj(d, e, adj, &det);
        // See the note in assemble_jacobian_colored_sumfact: rc and p go through the same
        // upwind switch the residual uses, so this matches the matrix-free action.
        cvfem_hex8_ns_upwind_jacobian_add_slots<true>(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values, rc, p);
        boundary_scs_add_jacobian<true>(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

inline SFEM_NOINLINE void assemble_jacobian_atomic_isoparam(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_jacobian_isoparam");
    scalar_t *const SFEM_RESTRICT             values = b.data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(rho, mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values, rc,
                                                              p);
        boundary_scs_add_jacobian<true>(rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, slots + (size_t)e * 64,
                                        values);
    }
}

// Node-indexed 4x4 diagonal blocks, 16 doubles per node, without forming the matrix.
// This is what a block-Jacobi smoother wants, and it is the reason the multigrid work
// does not have to keep assembling a BSR on the fine level.
//
// The bench has a diagonal assembly (assemble_diag_atomic in cvfem_hex8_layout_atomic.hpp)
// that masks the off-diagonal slots with -1 and lets the element kernel drop them, since
// cvfem_hex8_bsr_acc returns on a negative slot. That trick cannot be reused here, for
// two independent reasons:
//
//   - The solver's affine path runs the SymPy kernel, which writes
//     values[slots[k] * 16 + f] directly instead of going through the guarded accessor.
//     A negative slot there is an out-of-bounds write, not a dropped one.
//   - The bench's version omits both the Rhie-Chow and the boundary sub-control-surface
//     terms, because the bench has no boundary handling at all and verifies against its
//     own boundary-free assembly. Rhie-Chow is the entire pressure-pressure diagonal, so
//     a diagonal missing it is exactly the degenerate saddle point block-Jacobi cannot
//     invert, and the boundary term reaches 43% of the nodes on this channel at N=8 --
//     a larger share on the coarse grids where a smoother matters most.
//
// So each element assembles its full 8x8 block set into a local buffer addressed by
// identity slots, which is correct for every kernel however it writes, and only the eight
// diagonal blocks are scattered. The call sequence mirrors assemble_jacobian_atomic_*
// exactly; if those gain a term, this must too, and the gate will say so.
inline SFEM_NOINLINE void assemble_block_diag(MeshData             &d,
                                              const scalar_t        rho,
                                              const scalar_t        mu,
                                              const GeomKind        geom,
                                              std::vector<scalar_t> &diag) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_block_diag");
    diag.assign((size_t)d.nnodes * 16, scalar_t(0));
    assemble_nodal_p_grad(d, geom);
    scalar_t *const SFEM_RESTRICT out = diag.data();

    smesh::count_t sl[64];
    for (int k = 0; k < 64; ++k) sl[k] = (smesh::count_t)k;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t loc[64 * 16];
        for (int k = 0; k < 64 * 16; ++k) loc[k] = scalar_t(0);

        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};

        if (geom == GeomKind::Isoparam) {
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(rho, mu, x, y, z, ux, uy, uz, sl, loc, rc, p);
            boundary_scs_add_jacobian<false>(
                    rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, sl, loc);
        } else {
            scalar_t adj[9], det;
            cvfem_hex8_load_adj(d, e, adj, &det);
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj, det, ux, uy, uz, sl, loc, rc, p);
            boundary_scs_add_jacobian<false>(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, sl, loc);
        }

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t          g   = d.elems[a][e];
            const scalar_t *const       blk = loc + (size_t)(a * 8 + a) * 16;
            for (int k = 0; k < 16; ++k) CVFEM_ATOMIC_ADD(out[(size_t)g * 16 + k], blk[k]);
        }
    }
}

inline void apply_residual(MeshData &d, const scalar_t rho, const scalar_t mu, const GeomKind geom) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual");
    assemble_nodal_p_grad(d, geom);
    if (geom == GeomKind::Isoparam) {
        apply_residual_atomic_isoparam(d, rho, mu);
        return;
    }
    if (d.packed) {
        reset_residual(d);
        cvfem_hex8_apply_residual_packed(d, *d.packed, rho, mu);
        apply_boundary_scs_residual(d, rho, mu, 0);
        return;
    }
    apply_residual_atomic_sumfact(d, rho, mu);
}

// zero_first=false accumulates into whatever is already in b. sfem::Function::hessian_bsr
// runs every operator over one shared values buffer without clearing it between them, so
// an Op that cleared would silently drop the operators assembled before it. The element
// scatter accumulates either way, so this costs nothing but the skipped memset.
inline void assemble_jacobian(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu, const GeomKind geom,
                              const bool zero_first = true) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_jacobian");
    if (zero_first) zero_bsr4(b);
    assemble_nodal_p_grad(d, geom);
    if (geom == GeomKind::Isoparam)
        assemble_jacobian_atomic_isoparam(d, b, rho, mu);
    else if (d.packed && d.coloring)
        assemble_jacobian_colored_sumfact(d, *d.packed, *d.coloring, b, rho, mu);
    else
        assemble_jacobian_atomic_sumfact(d, b, rho, mu);
}

SFEM_INLINE void gather_element_dir(const MeshData &d, const ptrdiff_t e, const scalar_t *const SFEM_RESTRICT dir,
                                           scalar_t *const SFEM_RESTRICT vx, scalar_t *const SFEM_RESTRICT vy,
                                           scalar_t *const SFEM_RESTRICT vz, scalar_t *const SFEM_RESTRICT q) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)d.elems[a][e] * N_FIELDS;
        vx[a]                                  = dv[0];
        vy[a]                                  = dv[1];
        vz[a]                                  = dv[2];
        q[a]                                   = dv[3];
    }
}

inline SFEM_NOINLINE void apply_jacobian_action_atomic_sumfact(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                               const scalar_t *const SFEM_RESTRICT dir,
                                                               scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_sumfact");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        gather_element_dir(d, e, dir, vx, vy, vz, q);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        scalar_t adj[9], det;
        cvfem_hex8_load_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r, rc, p);
        boundary_scs_add_jacobian_action(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

inline SFEM_NOINLINE void apply_jacobian_action_atomic_isoparam(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                                const scalar_t *const SFEM_RESTRICT dir,
                                                                scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_isoparam");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        gather_element_dir(d, e, dir, vx, vy, vz, q);
        scalar_t pgx[8], pgy[8], pgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const Hex8RhieChow rc{x, y, z, pgx, pgy, pgz, d.rhie_chow_scale};
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, x, y, z, ux, uy, uz, vx, vy, vz, q, r, rc, p);
        boundary_scs_add_jacobian_action(rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

// Element contributions only: no zeroing of jv and no constraint handling, so the
// caller decides both. The driver wants neither left to it and uses the wrapper below;
// sfem::Op wants exactly this, because Op::apply accumulates and the Function owns the
// constraints.
inline void apply_jacobian_action_accumulate(MeshData &d, const scalar_t rho, const scalar_t mu, const GeomKind geom,
                                             const scalar_t *const SFEM_RESTRICT dir,
                                             scalar_t *const SFEM_RESTRICT       jv) {
    if (geom == GeomKind::Isoparam) {
        apply_jacobian_action_atomic_isoparam(d, rho, mu, dir, jv);
    } else if (d.packed) {
        cvfem_hex8_apply_jacobian_action_packed(d, *d.packed, rho, mu, dir, jv);
        apply_boundary_scs_jacobian_action(d, rho, mu, 0, dir, jv);
    } else {
        apply_jacobian_action_atomic_sumfact(d, rho, mu, dir, jv);
    }
}

inline void apply_jacobian_action(MeshData &d, const scalar_t rho, const scalar_t mu, const GeomKind geom,
                                  const std::vector<uint8_t> &constrained, const scalar_t *const SFEM_RESTRICT dir,
                                  scalar_t *const SFEM_RESTRICT jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action");
    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    cvfem_zero_scalars(jv, ndof);
    apply_jacobian_action_accumulate(d, rho, mu, geom, dir, jv);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        if (constrained[(size_t)i]) jv[i] = dir[i];
    }
}

inline void pack_fields(const MeshData &d, scalar_t *const SFEM_RESTRICT x) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        x[(size_t)i * 4 + 0] = d.ux[i];
        x[(size_t)i * 4 + 1] = d.uy[i];
        x[(size_t)i * 4 + 2] = d.uz[i];
        x[(size_t)i * 4 + 3] = d.p[i];
    }
}

inline void unpack_fields(MeshData &d, const scalar_t *const SFEM_RESTRICT x) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x[(size_t)i * 4 + 0];
        d.uy[i] = x[(size_t)i * 4 + 1];
        d.uz[i] = x[(size_t)i * 4 + 2];
        d.p[i]  = x[(size_t)i * 4 + 3];
    }
}

inline void pack_residual(const MeshData &d, scalar_t *const SFEM_RESTRICT r) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        r[(size_t)i * 4 + 0] = d.rx[i];
        r[(size_t)i * 4 + 1] = d.ry[i];
        r[(size_t)i * 4 + 2] = d.rz[i];
        r[(size_t)i * 4 + 3] = d.rc[i];
    }
}

// Accumulating counterpart of pack_residual, for sfem::Op::gradient.
inline void add_residual(const MeshData &d, scalar_t *const SFEM_RESTRICT r) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        r[(size_t)i * 4 + 0] += d.rx[i];
        r[(size_t)i * 4 + 1] += d.ry[i];
        r[(size_t)i * 4 + 2] += d.rz[i];
        r[(size_t)i * 4 + 3] += d.rc[i];
    }
}

inline void apply_dirichlet_residual(const std::vector<uint8_t> &constrained, scalar_t *const r, const ptrdiff_t ndof) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        if (constrained[(size_t)i]) r[i] = scalar_t(0);
    }
}

inline void apply_dirichlet_fields(const std::vector<uint8_t>  &constrained,
                                   const std::vector<scalar_t> &bc,
                                   scalar_t *const              x,
                                   const ptrdiff_t              ndof) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        if (constrained[(size_t)i]) x[i] = bc[(size_t)i];
    }
}

inline void apply_dirichlet_bsr(BSR4 &b, const std::vector<uint8_t> &constrained, const ptrdiff_t nnodes) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_dirichlet_bsr");
    scalar_t *const SFEM_RESTRICT values = b.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        const int c0 = constrained[(size_t)row * 4 + 0];
        const int c1 = constrained[(size_t)row * 4 + 1];
        const int c2 = constrained[(size_t)row * 4 + 2];
        const int c3 = constrained[(size_t)row * 4 + 3];
        if (!(c0 | c1 | c2 | c3)) continue;

        for (smesh::count_t k = b.rowptr[row]; k < b.rowptr[row + 1]; ++k) {
            scalar_t *const blk  = values + (ptrdiff_t)k * 16;
            const int       diag = (b.colidx[k] == (smesh::idx_t)row);
            if (c0) {
                blk[0] = blk[1] = blk[2] = blk[3] = scalar_t(0);
                if (diag) blk[0] = scalar_t(1);
            }
            if (c1) {
                blk[4] = blk[5] = blk[6] = blk[7] = scalar_t(0);
                if (diag) blk[5] = scalar_t(1);
            }
            if (c2) {
                blk[8] = blk[9] = blk[10] = blk[11] = scalar_t(0);
                if (diag) blk[10] = scalar_t(1);
            }
            if (c3) {
                blk[12] = blk[13] = blk[14] = blk[15] = scalar_t(0);
                if (diag) blk[15] = scalar_t(1);
            }
        }
    }
}

inline bool invert3_vel(const scalar_t *const SFEM_RESTRICT a, scalar_t *const SFEM_RESTRICT inv) {
    const scalar_t a00 = a[0], a01 = a[1], a02 = a[2];
    const scalar_t a10 = a[4], a11 = a[5], a12 = a[6];
    const scalar_t a20 = a[8], a21 = a[9], a22 = a[10];
    const scalar_t x0  = a11 * a22;
    const scalar_t x1  = a12 * a21;
    const scalar_t x2  = a01 * a12;
    const scalar_t x3  = a01 * a22;
    const scalar_t x4  = a02 * a11;
    const scalar_t det = a00 * (x0 - x1) + a02 * a10 * a21 - a10 * x3 + a20 * x2 - a20 * x4;
    if (std::fabs(det) < scalar_t(1e-30) || !std::isfinite(det)) return false;
    const scalar_t s = scalar_t(1) / det;
    inv[0]           = s * (x0 - x1);
    inv[1]           = s * (a02 * a21 - x3);
    inv[2]           = s * (x2 - x4);
    inv[4]           = s * (-a10 * a22 + a12 * a20);
    inv[5]           = s * (a00 * a22 - a02 * a20);
    inv[6]           = s * (-a00 * a12 + a02 * a10);
    inv[8]           = s * (a10 * a21 - a11 * a20);
    inv[9]           = s * (-a00 * a21 + a01 * a20);
    inv[10]          = s * (a00 * a11 - a01 * a10);
    return std::isfinite(inv[0]) && std::isfinite(inv[5]) && std::isfinite(inv[10]);
}

// Lumped pressure mass matrix: the control volume attached to each node.
//
// A HEX8 element's eight sub-control volumes partition it evenly, so each node collects
// |det| / 8 from every element it touches. This is M_p for a piecewise-constant pressure
// test space, which is what the Schur approximation below needs.
inline void build_node_volume(const MeshData &d, std::vector<scalar_t> &node_vol) {
    node_vol.assign((size_t)d.nnodes, scalar_t(0));
    // jacobian_determinant is only precomputed for affine geometry, so evaluate it here
    // when it is absent rather than reading an empty array.
    const bool have_det = (ptrdiff_t)d.jacobian_determinant.size() >= d.nelements;
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t det;
        if (have_det) {
            det = d.jacobian_determinant[e];
        } else {
            scalar_t x[8], y[8], z[8], adj[9];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                x[a] = d.points[0][g]; y[a] = d.points[1][g]; z[a] = d.points[2][g];
            }
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
        }
        const scalar_t v = std::fabs(det) / scalar_t(8);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) node_vol[d.elems[a][e]] += v;
    }
}

inline int cvfem_report_schur = 1;

// SIMPLE-style pressure Schur diagonal: diag(C - B diag(A_uu)^-1 B^T).
//
// This is the approximation colocated finite-volume codes actually use, and unlike
// mu * M_p^-1 it makes no assumption about which term dominates -- it reads both the
// Rhie-Chow pressure operator C and the velocity coupling straight out of the assembled
// Jacobian. C_ii is block(i,i)[3][3]; B is the continuity row of block(i,j) over the
// velocity columns; B^T is the momentum rows of block(j,i) over the pressure column.
//
// Dirichlet rows are zeroed by apply_dirichlet_bsr before this runs, so a constrained
// velocity dof contributes B^T = 0 and drops out on its own.
//
// This costs O(nnz * row_length) because block(j,i) has to be looked up for each (i,j).
// That is once per Jacobian, against a linear solve of several hundred iterations, so it
// is not on the hot path -- but it is not free either, which is why it is opt-in.
inline void build_schur_diag(const BSR4 &b, const ptrdiff_t nnodes, std::vector<scalar_t> &schur) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::build_schur_diag");
    schur.assign((size_t)nnodes, scalar_t(0));
    const scalar_t *const SFEM_RESTRICT values = b.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        const smesh::count_t dslot = find_bsr_slot(b.rowptr, b.colidx, (smesh::idx_t)row, (smesh::idx_t)row);
        scalar_t             s     = values[(ptrdiff_t)dslot * 16 + 15];

        for (smesh::count_t k = b.rowptr[row]; k < b.rowptr[row + 1]; ++k) {
            const smesh::idx_t   col   = b.colidx[k];
            const scalar_t *const bij  = values + (ptrdiff_t)k * 16;
            const smesh::count_t djj   = find_bsr_slot(b.rowptr, b.colidx, col, col);
            const smesh::count_t kji   = find_bsr_slot(b.rowptr, b.colidx, col, (smesh::idx_t)row);
            const scalar_t *const ajj  = values + (ptrdiff_t)djj * 16;
            const scalar_t *const bji  = values + (ptrdiff_t)kji * 16;
            for (int c = 0; c < 3; ++c) {
                const scalar_t auu = ajj[c * 4 + c];
                if (std::fabs(auu) < scalar_t(1e-30)) continue;
                s -= bij[3 * 4 + c] * (scalar_t(1) / auu) * bji[c * 4 + 3];
            }
        }
        schur[(size_t)row] = s;
    }

    // One-shot report of how the two terms of S compare. The question this answers is
    // whether S is dominated by the Rhie-Chow operator C or by the velocity coupling
    // B A^-1 B^T, which is what decides whether any S^-1 approximation can differ from
    // the 1 / A_pp that block-Jacobi already applies.
    if (cvfem_report_schur) {
        cvfem_report_schur = 0;
        double c_sum = 0, bab_sum = 0, s_sum = 0;
        ptrdiff_t cnt = 0;
        for (ptrdiff_t row = 0; row < nnodes; ++row) {
            const smesh::count_t ds = find_bsr_slot(b.rowptr, b.colidx, (smesh::idx_t)row, (smesh::idx_t)row);
            const double         c  = (double)values[(ptrdiff_t)ds * 16 + 15];
            if (std::fabs(c) < 1e-30) continue;
            c_sum += std::fabs(c);
            bab_sum += std::fabs(c - (double)schur[(size_t)row]);
            s_sum += std::fabs((double)schur[(size_t)row]);
            ++cnt;
        }
        if (cnt) {
            std::printf("  schur: mean|C|=%.6e  mean|B A^-1 B^T|=%.6e  mean|S|=%.6e  ratio=%.3e  (n=%td)\n",
                        c_sum / cnt, bab_sum / cnt, s_sum / cnt, bab_sum / c_sum, cnt);
        }
    }
}

inline void build_block_jacobi(const BSR4                  &b,
                               const std::vector<uint8_t>  &constrained,
                               const ptrdiff_t              nnodes,
                               const std::vector<scalar_t> &node_vol,
                               const std::vector<scalar_t> &schur,
                               const scalar_t               pscale,
                               const scalar_t               pdamp,
                               std::vector<scalar_t>       &inv_diag) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::build_block_jacobi");
    inv_diag.assign((size_t)nnodes * 16, scalar_t(0));
    const scalar_t *const SFEM_RESTRICT values = b.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        const scalar_t *const blk = values + (ptrdiff_t)b.diag_slots[(size_t)row] * 16;
        scalar_t *const       inv = inv_diag.data() + (size_t)row * 16;
        const int             c0  = constrained[(size_t)row * 4 + 0];
        const int             c1  = constrained[(size_t)row * 4 + 1];
        const int             c2  = constrained[(size_t)row * 4 + 2];
        const int             c3  = constrained[(size_t)row * 4 + 3];

        if (!(c0 | c1 | c2) && invert3_vel(blk, inv)) {
            /* velocity 3x3 inverse */
        } else {
            for (int f = 0; f < 3; ++f) {
                if (constrained[(size_t)row * 4 + f]) {
                    inv[f * 4 + f] = scalar_t(1);
                } else {
                    const scalar_t d = blk[f * 4 + f];
                    inv[f * 4 + f]   = (std::fabs(d) > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
                }
            }
        }

        if (c3) {
            inv[15] = scalar_t(1);
        } else {
            const scalar_t d = blk[15];
            const scalar_t v = node_vol[(size_t)row];
            if (!schur.empty()) {
                // The literature approximation, with no fitted constant.
                const scalar_t sd = schur[(size_t)row];
                inv[15] = (std::fabs(sd) > scalar_t(1e-30)) ? scalar_t(1) / sd : scalar_t(1);
            } else if (pscale != scalar_t(0) && v > scalar_t(1e-30)) {
                // Pressure block scaled by the control volume instead of by A_pp.
                //
                // The textbook reading of this is the viscous Schur approximation
                // S^-1 ~ -mu M_p^-1, which for the lumped mass matrix is -mu / V_p. That
                // is not what is going on here, and the measurements say so twice over.
                //
                // First, S is already what block-Jacobi inverts. Measured from the
                // assembled Jacobian (SFEM_PC_SIMPLE), diag(B A^-1 B^T) is about 0.19 of
                // diag(C), so S = C - B A^-1 B^T sits within a fifth of the C = A_pp that
                // block-Jacobi uses -- and building the real SIMPLE Schur diagonal
                // changes the iteration count by 0.1%. Approximating S^-1 better is not
                // where the gain comes from.
                //
                // Second, the gain is a high-Reynolds effect, not a viscous one. Split by
                // continuation stage at N=8, this scaling saves 60% of the linear
                // iterations in the Re=100 stage and 3% in the Re=10 one. In the Stokes
                // limit 1 / A_pp is already right, which is exactly where the textbook
                // approximation is supposed to hold.
                //
                // So this is not an S^-1 approximation. What it does is weaken the
                // pressure block relative to the velocity block by a factor that grows
                // with Re, which a block-diagonal preconditioner needs and a Schur
                // approximation does not supply.
                //
                // PSCALE is therefore a tuned coefficient, not a physical constant. It is
                // dimensional and it tracks the stabilisation. A_pp is proportional to
                // rc_scale, and sweeping SFEM_RHIE_CHOW_SCALE moves the optimum inversely,
                // so the product is what is conserved:
                //
                //   rc_scale   0.25   0.5    1      2
                //   PSCALE     0.3    0.2    0.1    0.05
                //   product    0.075  0.10   0.10   0.10
                //
                // The three points from 0.5 to 2 are exact. Only rc=0.25 is off, and it
                // was swept on a grid of {0.1, 0.3, 1.0} that never tested the predicted
                // 0.4, so read it as unresolved rather than as a departure from the law.
                //
                // Positive is the correct sign for the continuity row as assembled here;
                // negative diverges. Default 0 keeps plain 1 / A_pp.
                inv[15] = pscale / v;
            } else {
                // A_pp is only structurally zero without Rhie-Chow, a configuration whose
                // linear solves do not converge anyway; the guard costs one compare.
                inv[15] = (std::fabs(d) > scalar_t(1e-30)) ? pdamp / d : scalar_t(1);
            }
        }
    }
}

inline void apply_block_jacobi(const std::vector<scalar_t> &inv_diag,
                               const ptrdiff_t              nnodes,
                               const scalar_t *const        x,
                               scalar_t *const              y) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_block_jacobi");
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        const scalar_t *const inv = inv_diag.data() + (size_t)row * 16;
        const scalar_t *const xx  = x + (size_t)row * 4;
        scalar_t *const       yy  = y + (size_t)row * 4;
        yy[0]                     = inv[0] * xx[0] + inv[1] * xx[1] + inv[2] * xx[2] + inv[3] * xx[3];
        yy[1]                     = inv[4] * xx[0] + inv[5] * xx[1] + inv[6] * xx[2] + inv[7] * xx[3];
        yy[2]                     = inv[8] * xx[0] + inv[9] * xx[1] + inv[10] * xx[2] + inv[11] * xx[3];
        yy[3]                     = inv[12] * xx[0] + inv[13] * xx[1] + inv[14] * xx[2] + inv[15] * xx[3];
    }
}

inline bool all_finite(const scalar_t *const v, const ptrdiff_t n) {
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) return false;
    }
    return true;
}

inline scalar_t max_abs(const scalar_t *const v, const ptrdiff_t n) {
    scalar_t m = 0;
    for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, std::fabs(v[i]));
    return m;
}

inline void compare_hessian_apply(MeshData &d, sfem::Operator<scalar_t> &A_bsr, const scalar_t rho, const scalar_t mu,
                                  const GeomKind geom, const std::vector<uint8_t> &constrained,
                                  const scalar_t *const SFEM_RESTRICT v, const ptrdiff_t ndof) {
    std::vector<scalar_t> y_mf((size_t)ndof), y_asm((size_t)ndof);
    apply_jacobian_action(d, rho, mu, geom, constrained, v, y_mf.data());
    A_bsr.apply(v, y_asm.data());
    scalar_t linf = 0, l2 = 0, nrm = 0, linf_u = 0, linf_p = 0;
    ptrdiff_t imax = 0;
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        const scalar_t e = y_mf[i] - y_asm[i];
        const scalar_t ae = std::fabs(e);
        l2 += e * e;
        nrm += y_asm[i] * y_asm[i];
        if (ae > linf) {
            linf = ae;
            imax = i;
        }
        if ((i & 3) == 3)
            linf_p = std::max(linf_p, ae);
        else
            linf_u = std::max(linf_u, ae);
    }
    std::printf("  Jv check: |Jmf-Jasm|_inf=%.6e  rel_l2=%.6e  |du|=%.6e  |dp|=%.6e  imax=%ld (node %ld fld %ld)\n",
                linf,
                (nrm > 0) ? std::sqrt(l2 / nrm) : std::sqrt(l2),
                linf_u,
                linf_p,
                (long)imax,
                (long)(imax / 4),
                (long)(imax & 3));
}

inline bool newton_step_converged(const scalar_t rn, const scalar_t r0, const scalar_t atol, const scalar_t rtol) {
    if (!std::isfinite(rn) || !std::isfinite(r0)) return false;
    return rn < atol || (r0 > 0 && rn / r0 < rtol);
}

inline void mark_constraints(const MeshData        &d,
                             const FlowCase         flow,
                             const scalar_t         mu,
                             const scalar_t         U,
                             std::vector<uint8_t>  &constrained,
                             std::vector<scalar_t> &bc,
                             ptrdiff_t             &pin_p) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::mark_constraints");
    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    constrained.assign((size_t)ndof, 0);
    bc.assign((size_t)ndof, scalar_t(0));

    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];

    scalar_t  best = 1e300;
    ptrdiff_t pin  = 0;
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t x = scalar_t(px[i]);
        const scalar_t y = scalar_t(py[i]);
        const scalar_t z = scalar_t(pz[i]);
        const scalar_t r = x + y + z;
        if (r < best) {
            best = r;
            pin  = i;
        }

        scalar_t ux, uy, uz, p;
        exact_state(flow, mu, U, d.Lx, d.Ly, x, y, z, ux, uy, uz, p);
        bc[(size_t)i * 4 + 0] = ux;
        bc[(size_t)i * 4 + 1] = uy;
        bc[(size_t)i * 4 + 2] = uz;
        bc[(size_t)i * 4 + 3] = p;

        const bool wall_y = on_plane(y, scalar_t(0), d.Ly) || on_plane(y, d.Ly, d.Ly);
        const bool inlet  = on_plane(x, scalar_t(0), d.Lx);
        const bool outlet = on_plane(x, d.Lx, d.Lx);
        const bool span   = on_plane(z, scalar_t(0), d.Lz) || on_plane(z, d.Lz, d.Lz);

        if (wall_y) {
            constrained[(size_t)i * 4 + 0] = 1;
            constrained[(size_t)i * 4 + 1] = 1;
            constrained[(size_t)i * 4 + 2] = 1;
        } else if ((inlet || outlet)) {
            constrained[(size_t)i * 4 + 0] = 1;
            constrained[(size_t)i * 4 + 1] = 1;
            constrained[(size_t)i * 4 + 2] = 1;
        }

        if (span) constrained[(size_t)i * 4 + 2] = 1;
    }

    pin_p = pin;
    constrained[(size_t)pin * 4 + 3] = 1;
}

inline void init_fields(MeshData                    &d,
                        const InitKind               init,
                        const std::vector<uint8_t>  &constrained,
                        const std::vector<scalar_t> &bc) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::init_fields");
    d.ux.resize((size_t)d.nnodes);
    d.uy.resize((size_t)d.nnodes);
    d.uz.resize((size_t)d.nnodes);
    d.p.resize((size_t)d.nnodes);
    d.rx.assign((size_t)d.nnodes, 0);
    d.ry.assign((size_t)d.nnodes, 0);
    d.rz.assign((size_t)d.nnodes, 0);
    d.rc.assign((size_t)d.nnodes, 0);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.p[i] = bc[(size_t)i * 4 + 3];
        if (init == InitKind::Exact) {
            d.ux[i] = bc[(size_t)i * 4 + 0];
            d.uy[i] = bc[(size_t)i * 4 + 1];
            d.uz[i] = bc[(size_t)i * 4 + 2];
        } else {
            d.ux[i] = constrained[(size_t)i * 4 + 0] ? bc[(size_t)i * 4 + 0] : scalar_t(0);
            d.uy[i] = constrained[(size_t)i * 4 + 1] ? bc[(size_t)i * 4 + 1] : scalar_t(0);
            d.uz[i] = constrained[(size_t)i * 4 + 2] ? bc[(size_t)i * 4 + 2] : scalar_t(0);
        }
    }
}

struct ErrorNorms {
    scalar_t  u_linf{0};
    scalar_t  u_l2{0};
    scalar_t  p_linf{0};
    scalar_t  p_l2{0};
    scalar_t  u_linf_free{0};
    scalar_t  p_linf_free{0};
    scalar_t  p_min{0};
    scalar_t  p_max{0};
    ptrdiff_t n_free_u{0};
    ptrdiff_t n_free_p{0};
};

inline ErrorNorms compute_errors(const MeshData               &d,
                                 const FlowCase                flow,
                                 const scalar_t                mu,
                                 const scalar_t                U,
                                 const std::vector<uint8_t>   &constrained) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::compute_errors");
    ErrorNorms        err;
    scalar_t          u2 = 0;
    scalar_t          p2 = 0;
    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];

    if (d.nnodes > 0) {
        err.p_min = d.p[0];
        err.p_max = d.p[0];
    }

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        scalar_t ux, uy, uz, p;
        exact_state(flow, mu, U, d.Lx, d.Ly, scalar_t(px[i]), scalar_t(py[i]), scalar_t(pz[i]), ux, uy, uz, p);
        const scalar_t eux = d.ux[i] - ux;
        const scalar_t euy = d.uy[i] - uy;
        const scalar_t euz = d.uz[i] - uz;
        const scalar_t ep  = d.p[i] - p;
        const scalar_t eu  = std::sqrt(eux * eux + euy * euy + euz * euz);
        err.u_linf         = std::max(err.u_linf, eu);
        err.p_linf         = std::max(err.p_linf, std::fabs(ep));
        err.p_min          = std::min(err.p_min, d.p[i]);
        err.p_max          = std::max(err.p_max, d.p[i]);
        u2 += eux * eux + euy * euy + euz * euz;
        p2 += ep * ep;

        const int u_free = !constrained[(size_t)i * 4 + 0] || !constrained[(size_t)i * 4 + 1] ||
                           !constrained[(size_t)i * 4 + 2];
        if (u_free) {
            err.u_linf_free = std::max(err.u_linf_free, eu);
            err.n_free_u += 1;
        }
        if (!constrained[(size_t)i * 4 + 3]) {
            err.p_linf_free = std::max(err.p_linf_free, std::fabs(ep));
            err.n_free_p += 1;
        }
    }
    err.u_l2 = std::sqrt(u2 / scalar_t(d.nnodes));
    err.p_l2 = std::sqrt(p2 / scalar_t(d.nnodes));
    return err;
}

