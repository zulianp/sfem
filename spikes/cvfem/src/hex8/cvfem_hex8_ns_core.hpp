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
    std::vector<scalar_t> qgx, qgy, qgz;  // same reconstruction applied to the Jacobian direction

    // The Rhie-Chow coefficient, hoisted out of the element loop -- twelve values per
    // element, one per sub-control surface, rebuilt by cvfem_hex8_build_rc_coeff only when
    // rho, mu, the scale or the mesh change. See Hex8RhieChowPack::coeff for why it is not
    // computed where it is used.
    std::vector<scalar_t> rc_coeff[CVFEM_HEX8_N_SCS];
    scalar_t              rc_coeff_rho{0}, rc_coeff_mu{0}, rc_coeff_scale{0};
    // Optional body force, one value per node, and the control volume it is weighted by.
    // Left empty for every case that has no source term, in which case nothing is added and
    // the residual is bit-identical to what it was before this existed. Used by the
    // manufactured-solution case, where f = -(1/Re) lap(u) + (u.grad)u + grad(p).
    std::vector<scalar_t> fx, fy, fz;
    std::vector<scalar_t> node_vol;
    // The reconstruction's denominator: 1 / sum_{e in i} |det J_e|, one scalar per node.
    // Pure geometry, so it is constant for the whole solve -- and it was being rebuilt from
    // scratch on every matvec, with its own heap allocation and one atomic per node per
    // element, inside the pass that is 69% of that matvec. Keyed on the mesh and the
    // geometry rule, because the affine and isoparametric sweeps evaluate det differently.
    std::vector<scalar_t> grad_w_inv;
    // The partially assembled element tangent: five scalars per sub-control surface, sixty
    // per element, SoA by (surface, component). The complete dependence of the Jacobian
    // action on the Newton iterate -- see Hex8TangentPack. Built once per Newton step and
    // read on every matvec, so it is invalidated by the STATE changing, which no key made
    // of rho, mu and the mesh can see: pa_valid is cleared by whoever moves the state.
    std::vector<scalar_t> pa_tangent;
    scalar_t              pa_rho{0}, pa_mu{0}, pa_scale{0}, pa_ueps{0};
    ptrdiff_t             pa_nelements{0};
    bool                  pa_valid{false};
    int                   grad_w_isoparam{-1};
    ptrdiff_t             grad_w_nelements{0};
    // Transient term. dt <= 0 means steady, in which case nothing below is touched and the
    // residual is bit-identical to what it was before this existed -- which is what every
    // existing case and every recorded number depends on.
    //
    // The history is the velocity at the previous one or two time levels, three components
    // interleaved per node the way the state vector is. Pressure has no time derivative in
    // incompressible flow and no history: the continuity equation is a constraint, not an
    // evolution equation, and giving it a mass term would be a different set of equations.
    scalar_t              dt{0};
    int                   bdf_order{1};
    std::vector<scalar_t> u_prev;   // u^n,     3 * nnodes
    std::vector<scalar_t> u_prev2;  // u^{n-1}, 3 * nnodes, BDF2 only
    // Per-element boundary-face bitmask, one bit per CVFEM local face. Empty means "decide
    // from the bounding box", which is what every box case does and keeps those results
    // bit-identical. A non-box domain must set it: a coordinate test cannot see a re-entrant
    // face, and an unclosed control volume does not fail, it just stops conserving mass.
    std::vector<uint8_t> face_mask;
    // Faces carrying a prescribed pressure, and the values the two new boundary conditions
    // need. Empty / zero means neither is in use -- every case in the repository today --
    // and the boundary term then behaves exactly as it did.
    //
    // Plain scalars rather than a Hex8BoundaryDataT because that type is defined in
    // cvfem_hex8_boundary_scs.hpp, which this header includes further down; hex8_bd()
    // assembles the struct once the type is in scope.
    std::vector<uint8_t> pressure_mask;
    // Which natural faces carry bc_t*. Empty means none do, so every natural face is the
    // traction-free do-nothing outflow -- see the tmask note in cvfem_hex8_boundary_scs.hpp.
    std::vector<uint8_t> traction_mask;
    scalar_t             bc_tx{0}, bc_ty{0}, bc_tz{0};
    scalar_t             bc_p{0};
    // Faces carrying the do-nothing outflow. Empty/zero everywhere means no
    // natural face, which is every case except the backward-facing step, and the
    // outflow branch is then never taken.
    std::vector<uint8_t> natural_mask;

    // The effective boundary face mask -- see cvfem_hex8_build_face_mask_eff. Built once so
    // the boundary sweeps can skip the elements that have no boundary face at all, which on
    // any refined mesh is nearly all of them.
    std::vector<uint8_t> face_mask_eff;
    bool                 face_mask_eff_valid{false};
    scalar_t             face_mask_eff_lx{0}, face_mask_eff_ly{0}, face_mask_eff_lz{0};
    // The subset of elements face_mask_eff marks, compacted so the boundary sweeps are
    // load balanced rather than merely short.
    std::vector<ptrdiff_t> bnd_elems;
    // The boundary shell's node gather map -- see cvfem_hex8_build_bnd_gather. It lets the
    // closure be summed per node in a fixed order instead of scattered atomically, which
    // is what makes the operator bit-reproducible across thread counts.
    std::vector<ptrdiff_t>   bnd_gather_ptr;
    std::vector<int32_t>     bnd_gather_slot;
    std::vector<smesh::idx_t> bnd_gather_dest;
    std::vector<scalar_t>    bnd_r;
    ptrdiff_t                bnd_gather_n_bnd{-1};
    bool                     bnd_gather_valid{false};
    std::vector<scalar_t> jacobian_adjugate[9];
    std::vector<scalar_t> jacobian_determinant;
    PackedData           *packed{nullptr};
    const PackColoring   *coloring{nullptr};
    scalar_t              rhie_chow_scale{1};
    // Harten band for the upwind switch, as an absolute mass-flux magnitude; 0 is the
    // hard switch. See cvfem_upwind_abs.
    scalar_t upwind_eps{0};
};

#include "cvfem_hex8_ns_packed.hpp"
#include "cvfem_hex8_ns_upwind_sympy_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

// The prescribed boundary data for one element. A default-constructed result -- what every
// case with neither condition in use produces -- makes the boundary term behave exactly as
// it did before the conditions existed.
static SFEM_INLINE Hex8BoundaryDataT<scalar_t> hex8_bd(const MeshData &d, const ptrdiff_t e) {
    Hex8BoundaryDataT<scalar_t> bd;
    bd.tx    = d.bc_tx;
    bd.ty    = d.bc_ty;
    bd.tz    = d.bc_tz;
    bd.p_bar = d.bc_p;
    bd.tmask = d.traction_mask.empty() ? 0 : (int)d.traction_mask[(size_t)e];
    bd.pmask = d.pressure_mask.empty() ? 0 : (int)d.pressure_mask[(size_t)e];
    return bd;
}

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
// The reconstruction, over an arbitrary strided nodal scalar. It is linear in that scalar
// and its weights depend only on geometry, so applying it to a perturbation q yields exactly
// the derivative of applying it to p -- which is the term the Jacobian was missing.
// The reconstruction itself lives in cvfem_hex8_pack_helpers.hpp, which this header already
// includes and which exists precisely so the benchmark and the solver cannot drift on a
// quantity both of them feed into the same element kernels. This was a second copy of it --
// the thing that header's own comment warns against -- and is now a two-line forward that
// keeps the trace scope, because the pass is 40-52% of every matvec in the solver's own
// trace and has to stay separately attributable.
inline void assemble_nodal_grad_strided(MeshData &d, const GeomKind geom_kind,
                                        const scalar_t *const SFEM_RESTRICT src, const int stride,
                                        std::vector<scalar_t> &ogx, std::vector<scalar_t> &ogy,
                                        std::vector<scalar_t> &ogz) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::nodal_grad_strided");
    const int iso = geom_kind == GeomKind::Isoparam ? 1 : 0;
    // Over packs where there is a pack to sweep, which is the solver's normal case. Same
    // operator, no atomics, and deterministic. SFEM_QGRAD_ATOMIC=1 forces the flat sweep,
    // as a measurement escape hatch rather than a supported mode.
    static const int force_atomic = smesh::Env::read<int>("SFEM_QGRAD_ATOMIC", 0);
    if (d.packed && !force_atomic)
        cvfem_hex8_assemble_nodal_grad_packed(d, *d.packed, iso, src, stride, ogx, ogy, ogz);
    else
        cvfem_hex8_assemble_nodal_grad(d, iso, src, stride, ogx, ogy, ogz);
}

inline void assemble_nodal_p_grad(MeshData &d, const GeomKind geom_kind) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_nodal_p_grad");
    assemble_nodal_grad_strided(d, geom_kind, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
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


SFEM_INLINE void gather_element_qgrad(const MeshData               &d,
                                      const ptrdiff_t               e,
                                      scalar_t *const SFEM_RESTRICT gx,
                                      scalar_t *const SFEM_RESTRICT gy,
                                      scalar_t *const SFEM_RESTRICT gz) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t id = d.elems[a][e];
        gx[a]                 = d.qgx[id];
        gy[a]                 = d.qgy[id];
        gz[a]                 = d.qgz[id];
    }
}

SFEM_INLINE void gather_element_dir(const MeshData &d, const ptrdiff_t e, const scalar_t *const SFEM_RESTRICT dir,
                                           scalar_t *const SFEM_RESTRICT vx, scalar_t *const SFEM_RESTRICT vy,
                                           scalar_t *const SFEM_RESTRICT vz, scalar_t *const SFEM_RESTRICT q);

inline SFEM_NOINLINE void apply_boundary_scs_residual(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                      const int isoparam) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_boundary_scs_residual");
    // The boundary layer is what this closes, so walk only it. This swept the whole mesh --
    // gathering coordinates, fields and, where applicable, the direction, 88 doubles an
    // element -- to do work on the faces touching the domain edge alone, which at N=96 is
    // about 9% of the elements. cvfem_hex8_build_face_mask_eff settles which faces those
    // are once and lists the elements that have any; an element with none contributes
    // exactly nothing, so restricting the sweep to that list is exact.
    cvfem_hex8_build_face_mask_eff(d);
    // The gather is the default: scattering the closure atomically made the operator's
    // result depend on thread timing. SFEM_BND_ATOMIC=1 restores the old scatter, as a
    // measurement escape hatch rather than a supported mode.
    static const int bnd_atomic = smesh::Env::read<int>("SFEM_BND_ATOMIC", 0);
    if (!bnd_atomic) cvfem_hex8_build_bnd_gather(d);
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) cvfem_hex8_load_adj(d, e, adj, &det);
        // The masks, which this post-pass used to omit -- it fell back to the bounding-box
        // test, so on a non-box domain it closed the wrong faces and it never applied the
        // do-nothing outflow at all. Both are exactly what the masks exist to prevent, and
        // this is the path the packed residual takes.
        boundary_scs_add_residual(rho, mu, isoparam, isoparam ? nullptr : adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz,
                                  p, r, fmask,
                                  d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
        cvfem_hex8_bnd_commit(d, i, e, r, bnd_atomic, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());
    }
    if (!bnd_atomic) cvfem_hex8_bnd_gather_soa(d, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());
}

inline SFEM_NOINLINE void apply_boundary_scs_jacobian_action(MeshData &d, const scalar_t rho, const scalar_t mu,
                                                             const int isoparam, const scalar_t *const SFEM_RESTRICT dir,
                                                             scalar_t *const SFEM_RESTRICT jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action");
    // The boundary layer is what this closes, so walk only it. This swept the whole mesh --
    // gathering coordinates, fields and, where applicable, the direction, 88 doubles an
    // element -- to do work on the faces touching the domain edge alone, which at N=96 is
    // about 9% of the elements. cvfem_hex8_build_face_mask_eff settles which faces those
    // are once and lists the elements that have any; an element with none contributes
    // exactly nothing, so restricting the sweep to that list is exact.
    cvfem_hex8_build_face_mask_eff(d);
    // The gather is the default: scattering the closure atomically made the operator's
    // result depend on thread timing. SFEM_BND_ATOMIC=1 restores the old scatter, as a
    // measurement escape hatch rather than a supported mode.
    static const int bnd_atomic = smesh::Env::read<int>("SFEM_BND_ATOMIC", 0);
    if (!bnd_atomic) cvfem_hex8_build_bnd_gather(d);
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        gather_element_dir(d, e, dir, vx, vy, vz, q);
        std::memset(r, 0, sizeof(r));
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) cvfem_hex8_load_adj(d, e, adj, &det);
        boundary_scs_add_jacobian_action(rho, mu, isoparam, isoparam ? nullptr : adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux,
                                         uy, uz, vx, vy, vz, q, r,
                                         fmask,
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
        cvfem_hex8_bnd_commit_interleaved(d, i, e, r, bnd_atomic, jv);
    }
    if (!bnd_atomic) cvfem_hex8_bnd_gather_interleaved(d, jv);
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
        boundary_scs_add_residual(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, p, r,
                                  d.face_mask.empty() ? -1 : (int)d.face_mask[(size_t)e],
                                  d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));

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
        boundary_scs_add_residual(rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, p, r,
                                  d.face_mask.empty() ? -1 : (int)d.face_mask[(size_t)e],
                                  d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));

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
                            rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, es, values,
                            d.face_mask.empty() ? -1 : (int)d.face_mask[(size_t)e],
                            d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
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
        boundary_scs_add_jacobian<true>(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values,
                                         cvfem_hex8_face_mask_of(d, e),
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
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
                                        values,
                                         cvfem_hex8_face_mask_of(d, e),
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
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
// Both are defined below, after the element sweeps they share helpers with. The existing
// forward declaration of build_node_volume sits further down than this function does.
inline void     build_node_volume(const MeshData &d, std::vector<scalar_t> &node_vol);
inline scalar_t transient_diag_weight(const MeshData &d, const scalar_t rho);

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
                    rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, sl, loc,
                    d.face_mask.empty() ? -1 : (int)d.face_mask[(size_t)e],
                    d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
        } else {
            scalar_t adj[9], det;
            cvfem_hex8_load_adj(d, e, adj, &det);
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj, det, ux, uy, uz, sl, loc, rc, p);
            boundary_scs_add_jacobian<false>(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, sl, loc,
                                         cvfem_hex8_face_mask_of(d, e),
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
        }

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t          g   = d.elems[a][e];
            const scalar_t *const       blk = loc + (size_t)(a * 8 + a) * 16;
            for (int k = 0; k < 16; ++k) CVFEM_ATOMIC_ADD(out[(size_t)g * 16 + k], blk[k]);
        }
    }

    // The transient term's diagonal. It is rho V a0 / dt on each velocity component and
    // nothing on pressure, so it strengthens exactly the block the smoother inverts and
    // leaves the saddle-point structure alone. Adding it here rather than in the element
    // kernels keeps it consistent with apply_transient, which is a post-pass for the same
    // reason.
    {
        const scalar_t a = transient_diag_weight(d, rho);
        if (a != scalar_t(0)) {
            if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const scalar_t w = a * d.node_vol[(size_t)i];
                for (int c = 0; c < 3; ++c) diag[(size_t)i * 16 + (size_t)c * 4 + (size_t)c] += w;
            }
        }
    }
}

inline void build_node_volume(const MeshData &d, std::vector<scalar_t> &node_vol);  // defined below

// Subtract the body force from the momentum residual.
//
// The residual is a *volume integral* over each node's control volume -- see the flux form in
// cvfem_hex8_ns_upwind_residual_sumfact, where the face contribution is added at the owner and
// subtracted at the neighbour, so r[i*4+0..2] accumulates the integral of
// div(rho u u) + grad p - div tau over CV_i. A source term therefore enters as -f(x_i) * V_i,
// with V_i the control volume from build_node_volume.
//
// This is a per-node post-pass rather than a term inside the element kernels, and that is
// deliberate: the forcing does not interact with any sub-control-surface flux, so putting it
// here covers the sumfact, isoparametric and packed sweeps at once instead of touching five
// host kernels and five CUDA kernels for the same arithmetic.
inline void apply_body_force(MeshData &d) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_body_force");
    if (d.fx.empty()) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t v = d.node_vol[(size_t)i];
        d.rx[(size_t)i] -= d.fx[(size_t)i] * v;
        d.ry[(size_t)i] -= d.fy[(size_t)i] * v;
        d.rz[(size_t)i] -= d.fz[(size_t)i] * v;
    }
}

// BDF1/BDF2 coefficients for r += rho V (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / dt.
//
// BDF2 needs two levels of history, so the first step of a run has none and must fall back
// to BDF1. That is not an approximation to apologise for -- it is the standard start-up,
// and it costs one step of first-order error in a sequence that is otherwise second-order.
// The caller signals it by leaving u_prev2 empty.
struct BdfCoeffs {
    scalar_t a0, a1, a2;
    int      order;
};

inline BdfCoeffs bdf_coeffs(const MeshData &d) {
    const bool have_two = d.bdf_order >= 2 && (ptrdiff_t)d.u_prev2.size() == 3 * d.nnodes;
    if (have_two) return {scalar_t(1.5), scalar_t(-2), scalar_t(0.5), 2};
    return {scalar_t(1), scalar_t(-1), scalar_t(0), 1};
}

// The transient term, as a per-node post-pass.
//
// In a control-volume scheme the mass matrix IS the control volume: the momentum equation
// integrated over CV_i has d/dt of (rho u_i V_i), so the term is diagonal and V_i is
// already computed for the body force and the MMS norms. There is no consistent mass
// matrix to assemble and none should be introduced -- an FEM mass matrix here would be a
// different discretisation, not a better one.
//
// This is a post-pass for the reason apply_body_force gives above: the term touches no
// sub-control-surface flux, so one pass covers the sumfact, isoparametric and packed
// sweeps at once rather than being threaded into five host kernels and five CUDA kernels.
inline void apply_transient(MeshData &d, const scalar_t rho) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_transient");
    if (d.dt <= scalar_t(0)) return;
    if ((ptrdiff_t)d.u_prev.size() != 3 * d.nnodes) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
    const BdfCoeffs c   = bdf_coeffs(d);
    const scalar_t  inv = scalar_t(1) / d.dt;
    const bool      two = c.order == 2;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w  = rho * d.node_vol[(size_t)i] * inv;
        const size_t   k  = (size_t)i * 3;
        const scalar_t p2x = two ? d.u_prev2[k + 0] : scalar_t(0);
        const scalar_t p2y = two ? d.u_prev2[k + 1] : scalar_t(0);
        const scalar_t p2z = two ? d.u_prev2[k + 2] : scalar_t(0);
        d.rx[(size_t)i] += w * (c.a0 * d.ux[(size_t)i] + c.a1 * d.u_prev[k + 0] + c.a2 * p2x);
        d.ry[(size_t)i] += w * (c.a0 * d.uy[(size_t)i] + c.a1 * d.u_prev[k + 1] + c.a2 * p2y);
        d.rz[(size_t)i] += w * (c.a0 * d.uz[(size_t)i] + c.a1 * d.u_prev[k + 2] + c.a2 * p2z);
    }
}

// The same term's contribution to the Jacobian: d/du of the above is rho V a0 / dt on each
// of the three velocity diagonal entries. Pressure is untouched, so the saddle-point
// structure -- and the reason the block-diagonal preconditioner needs Rhie-Chow to invert
// the pressure entry at all -- is unchanged.
inline scalar_t transient_diag_weight(const MeshData &d, const scalar_t rho) {
    if (d.dt <= scalar_t(0)) return scalar_t(0);
    // The history is NOT required. The derivative of the BDF term is rho a0 / dt whatever
    // u^n and u^{n-1} hold -- they are data the residual differences, not part of the
    // Jacobian. Requiring them here gave every coarse level in a hierarchy a STEADY
    // Jacobian: clone_onto builds those and they never receive a history, being applied
    // only to a correction. For a small timestep rho V a0 / dt is the dominant diagonal, so
    // that is not a small coarse-grid inconsistency.
    //
    // With a history present the coefficient still comes from it, so a BDF2 run's FIRST
    // step -- which has u^n but no u^{n-1} and correctly falls back to BDF1 -- keeps a
    // Jacobian consistent with the residual it is the derivative of. Without one, the
    // requested order is the best available and the factor of 1.5 is immaterial to a
    // preconditioner anyway.
    if ((ptrdiff_t)d.u_prev.size() == 3 * d.nnodes) return bdf_coeffs(d).a0 * rho / d.dt;
    return (d.bdf_order >= 2 ? scalar_t(1.5) : scalar_t(1)) * rho / d.dt;
}

// Applied to the matrix-free Jacobian action, where the direction plays the role of u.
inline void apply_transient_action(MeshData &d, const scalar_t rho,
                                   const scalar_t *const SFEM_RESTRICT dir,
                                   scalar_t *const SFEM_RESTRICT jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_transient_action");
    const scalar_t a = transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = a * d.node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) jv[(ptrdiff_t)i * N_FIELDS + c] += w * dir[(ptrdiff_t)i * N_FIELDS + c];
    }
}

inline void apply_residual(MeshData &d, const scalar_t rho, const scalar_t mu, const GeomKind geom) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual");
    assemble_nodal_p_grad(d, geom);
    if (geom == GeomKind::Isoparam) {
        apply_residual_atomic_isoparam(d, rho, mu);
        apply_body_force(d);
        apply_transient(d, rho);
        return;
    }
    if (d.packed) {
        reset_residual(d);
        cvfem_hex8_apply_residual_packed(d, *d.packed, rho, mu);
        apply_boundary_scs_residual(d, rho, mu, 0);
        apply_body_force(d);
        apply_transient(d, rho);
        return;
    }
    apply_residual_atomic_sumfact(d, rho, mu);
    apply_body_force(d);
    apply_transient(d, rho);
}

// zero_first=false accumulates into whatever is already in b. sfem::Function::hessian_bsr
// runs every operator over one shared values buffer without clearing it between them, so
// an Op that cleared would silently drop the operators assembled before it. The element
// scatter accumulates either way, so this costs nothing but the skipped memset.
// The transient term's diagonal, on the assembled matrix. Identical in substance to the
// block one assemble_block_diag adds -- rho V a0 / dt on each velocity diagonal entry and
// nothing on pressure -- and it reads the same transient_diag_weight, so the two cannot
// disagree about the coefficient.
//
// It was missing, and that is not cosmetic. apply_jacobian_action_accumulate applies the
// term and assemble_block_diag applies it, so in an unsteady run the assembled matrix was
// the STEADY Jacobian while the operator it preconditions and the smoother built beside it
// were the unsteady one. For a small timestep rho V a0 / dt is the dominant diagonal, so
// the preconditioner was missing the largest entry it has.
inline void assemble_transient_diag(MeshData &d, const scalar_t rho, BSR4 &b) {
    const scalar_t a = transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_transient_diag");
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) build_node_volume(d, d.node_vol);
    scalar_t *const SFEM_RESTRICT values = b.data();
    // diag_slots is what precompute_element_bsr_slots leaves behind and is the direct
    // answer; the row scan is the fallback for a matrix assembled without it.
    const bool have_diag = (ptrdiff_t)b.diag_slots.size() == d.nnodes;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < d.nnodes; ++r) {
        const scalar_t w = a * d.node_vol[(size_t)r];
        if (have_diag) {
            const smesh::count_t j = b.diag_slots[(size_t)r];
            if (j < 0) continue;
            for (int c = 0; c < 3; ++c) values[(ptrdiff_t)j * 16 + c * 4 + c] += w;
            continue;
        }
        for (smesh::count_t j = b.rowptr[r]; j < b.rowptr[r + 1]; ++j) {
            if (b.colidx[j] != (smesh::idx_t)r) continue;
            for (int c = 0; c < 3; ++c) values[(ptrdiff_t)j * 16 + c * 4 + c] += w;
        }
    }
}

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
    assemble_transient_diag(d, rho, b);
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
        scalar_t pgx[8], pgy[8], pgz[8], qgx[8], qgy[8], qgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const bool has_qg = !d.qgx.empty();
        if (has_qg) gather_element_qgrad(d, e, qgx, qgy, qgz);
        const Hex8RhieChow rc{x,   y,   z,   pgx, pgy, pgz, d.rhie_chow_scale,
                              has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                              has_qg ? qgz : nullptr};
        scalar_t adj[9], det;
        cvfem_hex8_load_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r, rc, p);
        boundary_scs_add_jacobian_action(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, vx, vy, vz, q, r,
                                         cvfem_hex8_face_mask_of(d, e),
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
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
        scalar_t pgx[8], pgy[8], pgz[8], qgx[8], qgy[8], qgz[8];
        gather_element_pgrad(d, e, pgx, pgy, pgz);
        const bool has_qg = !d.qgx.empty();
        if (has_qg) gather_element_qgrad(d, e, qgx, qgy, qgz);
        const Hex8RhieChow rc{x,   y,   z,   pgx, pgy, pgz, d.rhie_chow_scale,
                              has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                              has_qg ? qgz : nullptr};
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, x, y, z, ux, uy, uz, vx, vy, vz, q, r, rc, p);
        boundary_scs_add_jacobian_action(rho, mu, 1, (const scalar_t *)nullptr, scalar_t(0), d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, vx, vy, vz, q, r,
                                         cvfem_hex8_face_mask_of(d, e),
                                         d.natural_mask.empty() ? 0 : (int)d.natural_mask[(size_t)e], hex8_bd(d, e));
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
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_accumulate");
    // The Rhie-Chow correction differentiates through the nodal pressure-gradient
    // reconstruction, so the direction's own reconstructed gradient is needed. One extra
    // pass per Jacobian apply, the same shape as the one update() already does for p.
    // SFEM_RC_EXACT_JAC=0 restores the frozen-pg Jacobian. Parity with the semi-structured
    // path, and it is what lets cvfem_ns_op_gate compare the assembled operator against the
    // matrix-free action like for like: the assembled Jacobian keeps the frozen form on
    // purpose, so with the exact term on they are *meant* to differ.
    static const int rc_exact = smesh::Env::read<int>("SFEM_RC_EXACT_JAC", 1);
    if (rc_exact && d.rhie_chow_scale != scalar_t(0) && !d.pgx.empty()) {
        SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::assemble_nodal_q_grad");
        assemble_nodal_grad_strided(d, geom, dir + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);
    } else {
        d.qgx.clear(); d.qgy.clear(); d.qgz.clear();
    }
    if (geom == GeomKind::Isoparam) {
        apply_jacobian_action_atomic_isoparam(d, rho, mu, dir, jv);
    } else if (d.packed) {
        cvfem_hex8_apply_jacobian_action_packed(d, *d.packed, rho, mu, dir, jv);
        apply_boundary_scs_jacobian_action(d, rho, mu, 0, dir, jv);
    } else {
        apply_jacobian_action_atomic_sumfact(d, rho, mu, dir, jv);
    }
    apply_transient_action(d, rho, dir, jv);
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

