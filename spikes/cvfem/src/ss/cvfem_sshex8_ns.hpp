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
    // Harten band for the upwind switch, as an absolute mass-flux magnitude. Zero is the
    // hard switch, which is what every case that converges quadratically already uses.
    scalar_t                     upwind_eps{0};
    // Whether sscvfem_apply_blocks carries the exact Rhie-Chow term, making it the true
    // restriction of the operator. See sscvfem_wants_q_grad for why a preconditioner clears
    // this and what it measured.
    bool                         blocks_exact_rc{true};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> pgx, pgy, pgz;
    // The nodal VELOCITY gradient, [i*9 + r*3 + c] = du_r/dx_c, for the deferred-correction
    // convection scheme. Empty unless that is on. Same layout and same reconstruction as the
    // flat path's, so the two produce the same correction on the same mesh.
    std::vector<scalar_t> ugrad;
    int                   conv_ho{0};
    int                   conv_limiter{0};
    // The reconstruction's denominator: 1 / sum of |det| over the micro-elements touching a
    // node. Pure geometry, so it is built once and kept, and the sweep that uses it neither
    // allocates nor accumulates it. Keyed on the mesh it was built for; see
    // sscvfem_build_grad_weight for why the key is what it is.
    // The macro-element mesh, packed. Optional: null keeps the SSScatter path.
    //
    // Packing groups macro-elements so that a node shared between two of them IN THE SAME
    // PACK stops being shared at all -- only the pack boundary needs staging. Measured on a
    // 64-macro mesh at eight macro-elements per pack, the staged fraction falls from the
    // macro skin's 96.3 / 78.4 / 52.9 percent at levels 2, 4 and 8 to 48.1 / 24.6 / 12.4.
    // Since the gradient's cost tracks that fraction almost exactly across levels, this is
    // the lever rather than the arithmetic.
    PackedData           *packed{nullptr};
    std::vector<scalar_t> grad_w_inv;
    ptrdiff_t             grad_w_nmacro{-1};
    int                   grad_w_level{-1};
    std::vector<scalar_t> qgx, qgy, qgz;
    // Body force per node and the control volume it is weighted by; empty unless a case sets
    // one, in which case the residual is unchanged. See apply_body_force in the flat core.
    std::vector<scalar_t> fx, fy, fz;
    std::vector<scalar_t> node_vol;
    // Transient term; dt <= 0 means steady and nothing is evaluated. Mirrors the flat
    // core's fields -- see the note there on why pressure carries no history.
    scalar_t              dt{0};
    // The PREVIOUS step size, for variable-step BDF2. Zero means none recorded.
    scalar_t              dt_prev{0};
    int                   bdf_order{1};
    std::vector<scalar_t> u_prev;   // u^n,     3 * nnodes
    std::vector<scalar_t> u_prev2;  // u^{n-1}, 3 * nnodes, BDF2 only
    // Boundary-face bitmask per MACRO element. The micro mask follows from this and the
    // lattice indices, and is level-independent, so one macro-level mask serves every level
    // of the multigrid hierarchy.
    std::vector<uint8_t> macro_face_mask;
    // Faces carrying the do-nothing outflow. Empty means none, which is every case except
    // the backward-facing step, so the outflow branch is never taken elsewhere.
    std::vector<uint8_t> macro_natural_mask;
    // The value-carrying conditions, at the same macro level and for the same reason: a
    // sideset stores (macro element, local face), which a level change does not touch, so
    // one mask is correct at every level of a hierarchy.
    std::vector<uint8_t> macro_pressure_mask;
    std::vector<uint8_t> macro_traction_mask;
    scalar_t             bc_tx{0}, bc_ty{0}, bc_tz{0};
    scalar_t             bc_p{0};

    // Deterministic scatter tables, built once. Null means the atomic path.
    std::shared_ptr<struct SSScatter> scatter;
};
// Defined below, next to the macro-element geometry it configures; the element sweeps that
// need it sit above.
inline scalar_t sscvfem_transient_diag_weight(const SSMeshData &d, const scalar_t rho);
inline Hex8RcConfig sscvfem_rc_config(const SSMeshData &d);

struct SSScatter;

// Deterministic scatter, the semi-structured counterpart of the packed HEX8 layout.
//
// The flat HEX8 path won on CPU with a packed mesh: each pack writes its exclusively owned
// nodes straight to the global array, stages the shared ones, and a second pass gathers each
// shared node's contributions in a fixed order. No atomics, and therefore a fixed summation
// order and a bit-reproducible result. The semi-structured path never got that -- it returns
// early in initialize() before the packing block -- and scatters every macro-element node
// with atomic_add instead. That is why a matrix-free apply here is not reproducible: with 27
// macro-elements on 8 threads the same application gives 0.080346978588455187 and
// 0.080346978588455631, while one thread gives 0.080346695382905065 every time.
//
// The structure is simpler than the flat case because the split is geometric. Every lattice
// node strictly inside a macro-element is touched by that macro-element alone; only the
// faces, edges and corners are shared, and at level L those are (L+1)^3 - (L-1)^3 of
// (L+1)^3 nodes -- 37% at L=4, falling to 12% at L=16. So the overwhelming majority of the
// scatter becomes a plain write and only the macro-element skin needs reducing.
struct SSScatter {
    bool                      ready{false};
    std::vector<int>          slot;         // (e * nxe + a) -> staging slot, -1 if exclusive
    std::vector<smesh::idx_t> shared_node;  // one global node per reduction row
    std::vector<ptrdiff_t>    red_ptr;      // CRS over reduction rows
    std::vector<ptrdiff_t>    red_idx;      // staging slots feeding each row
    ptrdiff_t                 n_slots{0};
    std::vector<scalar_t>     stage;        // n_slots * N_FIELDS, for the 4-wide kernels
    std::vector<scalar_t>     stage16;      // n_slots * 16, for the block diagonal
};

inline void sscvfem_build_scatter(const SSMeshData &d, SSScatter &s) {
    SFEM_TRACE_SCOPE("sscvfem::build_scatter");
    const int       nxe = d.nxe;
    const ptrdiff_t ne  = d.nmacro;

    std::vector<int> touches((size_t)d.nnodes, 0);
    for (ptrdiff_t e = 0; e < ne; ++e)
        for (int a = 0; a < nxe; ++a) touches[(size_t)d.elems[a][e]]++;

    // Shared nodes get a reduction row; exclusive ones are written directly.
    std::vector<ptrdiff_t> row_of((size_t)d.nnodes, -1);
    s.shared_node.clear();
    for (ptrdiff_t g = 0; g < d.nnodes; ++g)
        if (touches[(size_t)g] > 1) {
            row_of[(size_t)g] = (ptrdiff_t)s.shared_node.size();
            s.shared_node.push_back((smesh::idx_t)g);
        }

    const ptrdiff_t nrows = (ptrdiff_t)s.shared_node.size();
    s.red_ptr.assign((size_t)nrows + 1, 0);
    for (ptrdiff_t g = 0; g < d.nnodes; ++g)
        if (row_of[(size_t)g] >= 0) s.red_ptr[(size_t)row_of[(size_t)g] + 1] = touches[(size_t)g];
    for (ptrdiff_t r = 0; r < nrows; ++r) s.red_ptr[(size_t)r + 1] += s.red_ptr[(size_t)r];

    s.n_slots = nrows ? s.red_ptr[(size_t)nrows] : 0;
    s.slot.assign((size_t)ne * (size_t)nxe, -1);
    s.red_idx.assign((size_t)s.n_slots, 0);

    // Fill in element order, so the gather below sums in a fixed order every run.
    std::vector<ptrdiff_t> fill(s.red_ptr.begin(), s.red_ptr.end() - (nrows ? 1 : 0));
    fill.resize((size_t)std::max<ptrdiff_t>(nrows, 0));
    for (ptrdiff_t r = 0; r < nrows; ++r) fill[(size_t)r] = s.red_ptr[(size_t)r];
    for (ptrdiff_t e = 0; e < ne; ++e)
        for (int a = 0; a < nxe; ++a) {
            const ptrdiff_t r = row_of[(size_t)d.elems[a][e]];
            if (r < 0) continue;
            const ptrdiff_t k         = fill[(size_t)r]++;
            s.slot[(size_t)e * nxe + a] = (int)k;
            s.red_idx[(size_t)k]        = k;  // slot index is its own position
        }

    s.stage.assign((size_t)s.n_slots * N_FIELDS, scalar_t(0));
    s.stage16.assign((size_t)s.n_slots * 16, scalar_t(0));
    s.ready = true;
}

// Per-element scatter: exclusive nodes straight out, shared ones staged. Templated on the
// number of values per node so the same tables serve the 4-wide kernels (Jacobian action,
// residual, block split) and the 16-wide block diagonal.
template <int W>
static SFEM_INLINE void sscvfem_scatter_element_w(const SSScatter &s, const int nxe, const ptrdiff_t e,
                                                  const smesh::idx_t *const SFEM_RESTRICT lg,
                                                  const scalar_t *const SFEM_RESTRICT     lout,
                                                  scalar_t *const SFEM_RESTRICT           dst,
                                                  scalar_t *const SFEM_RESTRICT           stage) {
    for (int a = 0; a < nxe; ++a) {
        const int sl = s.slot[(size_t)e * nxe + a];
        if (sl < 0) {
            const ptrdiff_t g = (ptrdiff_t)lg[a] * W;
            for (int c = 0; c < W; ++c) dst[g + c] += lout[(size_t)a * W + c];
        } else {
            for (int c = 0; c < W; ++c) stage[(size_t)sl * W + c] = lout[(size_t)a * W + c];
        }
    }
}

template <int W>
inline void sscvfem_reduce_shared_w(const SSScatter &s, scalar_t *const SFEM_RESTRICT dst,
                                    const scalar_t *const SFEM_RESTRICT stage) {
    const ptrdiff_t nrows = (ptrdiff_t)s.shared_node.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < nrows; ++r) {
        scalar_t acc[W] = {0};
        for (ptrdiff_t k = s.red_ptr[(size_t)r]; k < s.red_ptr[(size_t)r + 1]; ++k)
            for (int c = 0; c < W; ++c) acc[c] += stage[(size_t)s.red_idx[(size_t)k] * W + c];
        const ptrdiff_t g = (ptrdiff_t)s.shared_node[(size_t)r] * W;
        for (int c = 0; c < W; ++c) dst[g + c] += acc[c];
    }
}

static SFEM_INLINE void sscvfem_scatter_element(const SSScatter &s, const int nxe, const ptrdiff_t e,
                                                const smesh::idx_t *const SFEM_RESTRICT lg,
                                                const scalar_t *const SFEM_RESTRICT     lout,
                                                scalar_t *const SFEM_RESTRICT           jv) {
    sscvfem_scatter_element_w<N_FIELDS>(s, nxe, e, lg, lout, jv, const_cast<scalar_t *>(s.stage.data()));
}

// The same, for four separate destination arrays rather than one interleaved one. The
// nodal pressure gradient accumulates pgx, pgy, pgz and a volume weight, and it fed the
// apply, so leaving it atomic left the whole operator non-reproducible even after the
// Jacobian action's own scatter was fixed.
//
// Templated on the width because not every user wants four. The nodal gradient wants three --
// it stopped carrying a volume weight once that became cached geometry -- and a third of the
// staging, of the scatter and of the shared reduction is a third of each pass's memory
// traffic. The stage is allocated at the widest width any user needs, so a narrower pass
// simply addresses less of it; write and read must agree, which is why the width is a
// template parameter and not an argument that could differ between the two calls.
template <int W>
static SFEM_INLINE void sscvfem_scatter_element_soa_w(const SSScatter &s, const int nxe, const ptrdiff_t e,
                                                      const smesh::idx_t *const SFEM_RESTRICT lg,
                                                      const scalar_t *const SFEM_RESTRICT     lacc,
                                                      scalar_t *const                        dst[W]) {
    scalar_t *const stage = const_cast<scalar_t *>(s.stage.data());
    for (int a = 0; a < nxe; ++a) {
        const int sl = s.slot[(size_t)e * nxe + a];
        if (sl < 0) {
            const smesh::idx_t g = lg[a];
            for (int c = 0; c < W; ++c) dst[c][g] += lacc[(size_t)a * W + c];
        } else {
            for (int c = 0; c < W; ++c) stage[(size_t)sl * W + c] = lacc[(size_t)a * W + c];
        }
    }
}

template <int W>
inline void sscvfem_reduce_shared_soa_w(const SSScatter &s, scalar_t *const dst[W]) {
    const ptrdiff_t nrows = (ptrdiff_t)s.shared_node.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < nrows; ++r) {
        scalar_t acc[W] = {0};
        for (ptrdiff_t k = s.red_ptr[(size_t)r]; k < s.red_ptr[(size_t)r + 1]; ++k)
            for (int c = 0; c < W; ++c) acc[c] += s.stage[(size_t)s.red_idx[(size_t)k] * W + c];
        const smesh::idx_t g = s.shared_node[(size_t)r];
        for (int c = 0; c < W; ++c) dst[c][g] += acc[c];
    }
}

static SFEM_INLINE void sscvfem_scatter_element_soa(const SSScatter &s, const int nxe, const ptrdiff_t e,
                                                    const smesh::idx_t *const SFEM_RESTRICT lg,
                                                    const scalar_t *const SFEM_RESTRICT     lacc,
                                                    scalar_t *const                        dst[N_FIELDS]) {
    sscvfem_scatter_element_soa_w<N_FIELDS>(s, nxe, e, lg, lacc, dst);
}

inline void sscvfem_reduce_shared_soa(const SSScatter &s, scalar_t *const dst[N_FIELDS]) {
    sscvfem_reduce_shared_soa_w<N_FIELDS>(s, dst);
}

// Second pass: each shared node gathers its own contributions, in slot order.
inline void sscvfem_reduce_shared(const SSScatter &s, scalar_t *const SFEM_RESTRICT jv) {
    sscvfem_reduce_shared_w<N_FIELDS>(s, jv, s.stage.data());
}

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
    SFEM_TRACE_SCOPE("sscvfem::init");
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

// The reconstruction over an arbitrary strided nodal scalar. It is linear in that scalar
// with geometry-only weights, so applying it to a Jacobian direction q gives exactly the
// derivative of applying it to p -- the term the Rhie-Chow Jacobian was missing.
// Micro-element boundary-face mask, derived from the macro element's mask and the lattice
// position. A micro element carries a macro face only where it sits against that face of the
// lattice, so this is six tests and no storage.
//
// Level-independent by construction, which is why one macro-level mask serves every level of
// the multigrid hierarchy: nothing has to be rebuilt or transferred when a level is derefined.
// A negative macro mask means "no mask", and the coordinate test is used instead.
static SFEM_INLINE int sscvfem_micro_face_mask(const int macro, const int L, const int xi,
                                               const int yi, const int zi) {
    if (macro < 0) return -1;
    int m = 0;
    if (xi == 0)     m |= macro & 0x01;  // CVFEM face 0, x-min
    if (xi == L - 1) m |= macro & 0x02;  // face 1, x-max
    if (yi == 0)     m |= macro & 0x04;  // face 2, y-min
    if (yi == L - 1) m |= macro & 0x08;  // face 3, y-max
    if (zi == 0)     m |= macro & 0x10;  // face 4, z-min
    if (zi == L - 1) m |= macro & 0x20;  // face 5, z-max
    return m;
}

// The boundary data for one micro cell, with its face selectors projected down from the
// macro element exactly as the face and natural masks are.
//
// The values are per-sideset constants and need no projection; only the masks that say
// WHICH faces carry them do. Declared after sscvfem_micro_face_mask because it uses it, and
// after Hex8BoundaryDataT, which the boundary header defines.
static SFEM_INLINE Hex8BoundaryDataT<scalar_t> sscvfem_bd(const SSMeshData &d, const ptrdiff_t e,
                                                          const int L, const int xi, const int yi,
                                                          const int zi) {
    Hex8BoundaryDataT<scalar_t> bd;
    bd.tx    = d.bc_tx;
    bd.ty    = d.bc_ty;
    bd.tz    = d.bc_tz;
    bd.p_bar = d.bc_p;
    bd.tmask = d.macro_traction_mask.empty()
                       ? 0
                       : sscvfem_micro_face_mask((int)d.macro_traction_mask[(size_t)e], L, xi, yi, zi);
    bd.pmask = d.macro_pressure_mask.empty()
                       ? 0
                       : sscvfem_micro_face_mask((int)d.macro_pressure_mask[(size_t)e], L, xi, yi, zi);
    return bd;
}


// The reconstruction's denominator, built once.
//
// It is the sum of |det| over the micro-elements touching each node -- pure geometry, the
// same for every call on a given mesh. The sweep below used to accumulate it on every call
// alongside the gradient, which cost a fresh nnodes-sized allocation and zero-fill each time,
// a fourth field in the per-element scatter, a fourth field through the shared reduction, and
// a separate normalisation pass over every node afterwards. None of that is about the field
// being differentiated. The flat path was given this treatment and its reconstruction went
// from 4.87x slower to the fastest pass in the matvec; this is the same fix on the twin that
// did not get it.
//
// Serial and deterministic, like its flat counterpart: it runs once per mesh, so the cost is
// irrelevant beside the solve, and a thread-ordered accumulation would make the operator's
// round-off depend on the thread count for no gain.
//
// The key is (nmacro, level). Coordinates are not in it, which is a deliberate limit rather
// than an oversight: nothing in this spike moves a node after the operator is initialized,
// and a checksum over a million coordinates on every call would cost more than the pass it
// guards. A driver that ever does move points must clear grad_w_inv.
inline void sscvfem_build_grad_weight(SSMeshData &d) {
    if ((ptrdiff_t)d.grad_w_inv.size() == d.nnodes && d.grad_w_nmacro == d.nmacro &&
        d.grad_w_level == d.level)
        return;
    d.grad_w_inv.assign((size_t)d.nnodes, scalar_t(0));
    scalar_t *const SFEM_RESTRICT w = d.grad_w_inv.data();
    const int                     L = d.level;
    int                           off[8];
    sscvfem_corner_offsets(L, off);
    for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
        // Hoisted exactly as the sweep hoists it, and that is a correctness requirement
        // rather than a saving here: the denominator has to count the micro-elements the
        // numerator counted, so both must make the same degeneracy decision on the same
        // determinant.
        scalar_t ex[8], ey[8], ez[8], adj[9], det;
        for (int a = 0; a < 8; ++a) {
            const smesh::idx_t g = d.elems[off[a]][e];
            ex[a]                = (scalar_t)d.points[0][g];
            ey[a]                = (scalar_t)d.points[1][g];
            ez[a]                = (scalar_t)d.points[2][g];
        }
        sscvfem_micro_geom(ex, ey, ez, adj, &det);
        const scalar_t vol = std::fabs(det);
        if (vol < scalar_t(1e-30)) continue;
        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);
                    for (int a = 0; a < 8; ++a) w[d.elems[base + off[a]][e]] += vol;
                }
    }
    for (ptrdiff_t i = 0; i < d.nnodes; ++i)
        w[i] = w[i] > scalar_t(0) ? scalar_t(1) / w[i] : scalar_t(0);
    d.grad_w_nmacro = d.nmacro;
    d.grad_w_level  = d.level;
}

// The reconstruction over PACKED macro-elements.
//
// Same operator as the SSScatter path and the same summation structure as the flat packed
// gradient, which this mirrors deliberately: gather the field into a pack-local buffer,
// accumulate there with a plain `+=`, write the pack's owned rows straight out because no
// other pack owns them, and close the rest with the ghost reduction.
//
// What packing buys here is not the writes but WHAT COUNTS AS SHARED. The SSScatter path
// stages every macro-element face, edge and corner, because a node between two macro-elements
// cannot be written by either alone. Group those macro-elements into a pack and the node
// between two of them inside the pack is owned by the pack -- only the pack's outer boundary
// is left. At level 8 that is the difference between staging 53% of the nodes and 12%.
//
// The denominator is folded into the owned writes and into the ghost reduction rather than
// applied in a pass of its own, exactly as the flat twin does it, because the average is
// linear in its numerator: (owned + ghost) * w is owned * w + ghost * w. That removes a full
// read-modify-write over three nodal arrays from every call.
inline void sscvfem_nodal_grad_packed(SSMeshData &d, PackedData &p,
                                      const scalar_t *const SFEM_RESTRICT src, const int stride,
                                      std::vector<scalar_t> &ogx, std::vector<scalar_t> &ogy,
                                      std::vector<scalar_t> &ogz) {
    sscvfem_build_grad_weight(d);

    // The owned ranges tile [0, nnodes) exactly, so every entry is written and there is
    // nothing to pre-zero. Checked rather than assumed: a node no pack owned would otherwise
    // keep whatever the buffer held.
    const bool owns_all = p.n_packs > 0 && p.owned_nodes_ptr[0] == 0 && p.owned_nodes_ptr[p.n_packs] == d.nnodes;
    if (owns_all && (ptrdiff_t)ogx.size() == d.nnodes) {
        ogy.resize((size_t)d.nnodes);
        ogz.resize((size_t)d.nnodes);
    } else {
        ogx.assign((size_t)d.nnodes, scalar_t(0));
        ogy.assign((size_t)d.nnodes, scalar_t(0));
        ogz.assign((size_t)d.nnodes, scalar_t(0));
    }

    scalar_t *const SFEM_RESTRICT       gx_out = ogx.data();
    scalar_t *const SFEM_RESTRICT       gy_out = ogy.data();
    scalar_t *const SFEM_RESTRICT       gz_out = ogz.data();
    const scalar_t *const SFEM_RESTRICT w      = d.grad_w_inv.data();
    const ptrdiff_t node_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;

    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);
    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];

#pragma omp parallel
    {
        // Slots 7 and 8, which belong to this routine; see CVFEM_PACK_SCRATCH_SLOTS.
        scalar_t *const SFEM_RESTRICT pack_f   = thread_scratch<scalar_t>(7, (size_t)node_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(8, 3 * (size_t)node_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t e_end        = MIN(d.nmacro, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts    = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off = p.ghost_ptr[pack];

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) pack_f[k] = src[(owned + k) * stride];
            for (ptrdiff_t k = 0; k < n_ghost; ++k)
                pack_f[n_contiguous + k] = src[(ptrdiff_t)ghosts[k] * stride];
            std::memset(pack_out, 0, (size_t)n_pack_nodes * 3 * sizeof(scalar_t));

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                // One geometry per macro-element: its micro-elements are translates and share
                // a Jacobian exactly, which is what sscvfem_macro_geom has always relied on.
                scalar_t ex[8], ey[8], ez[8], adj[9], det;
                for (int a = 0; a < 8; ++a) {
                    const smesh::idx_t g = pack_local_to_global(p, pack, n_contiguous, p.elems[off[a]][e]);
                    ex[a]                = (scalar_t)px[g];
                    ey[a]                = (scalar_t)py[g];
                    ez[a]                = (scalar_t)pz[g];
                }
                sscvfem_micro_geom(ex, ey, ez, adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                // |det| times a gradient carrying 1/det: only the sign survives.
                const scalar_t sgn = det > 0 ? scalar_t(1) : scalar_t(-1);

                for (int zi = 0; zi < L; ++zi) {
                    for (int yi = 0; yi < L; ++yi) {
                        for (int xi = 0; xi < L; ++xi) {
                            const int base = sscvfem_lidx(L, xi, yi, zi);
                            scalar_t  ep[8], gx, gy, gz;
                            for (int a = 0; a < 8; ++a) ep[a] = pack_f[p.elems[base + off[a]][e]];
                            cvfem_hex8_grad_scalar(adj, sgn, ep, gx, gy, gz);
                            for (int a = 0; a < 8; ++a) {
                                scalar_t *const SFEM_RESTRICT o =
                                        pack_out + (ptrdiff_t)p.elems[base + off[a]][e] * 3;
                                o[0] += gx;
                                o[1] += gy;
                                o[2] += gz;
                            }
                        }
                    }
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t wi = w[owned + k];
                gx_out[owned + k] = pack_out[k * 3 + 0] * wi;
                gy_out[owned + k] = pack_out[k * 3 + 1] * wi;
                gz_out[owned + k] = pack_out[k * 3 + 2] * wi;
            }
            scalar_t *const SFEM_RESTRICT bx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT by = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT bz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT o = pack_out + (n_contiguous + k) * 3;
                bx[ghost_off + k]                     = o[0];
                by[ghost_off + k]                     = o[1];
                bz[ghost_off + k]                     = o[2];
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        const scalar_t *const SFEM_RESTRICT bx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
        const scalar_t *const SFEM_RESTRICT by = p.ghost_buf.data() + 1 * p.n_ghost_entries;
        const scalar_t *const SFEM_RESTRICT bz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
        scalar_t sx = 0, sy = 0, sz = 0;
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t idx = p.ghost_reduce_idx[j];
            sx += bx[idx];
            sy += by[idx];
            sz += bz[idx];
        }
        const scalar_t wi = w[dest];
        gx_out[dest] += sx * wi;
        gy_out[dest] += sy * wi;
        gz_out[dest] += sz * wi;
    }
}

inline void sscvfem_nodal_grad_strided(SSMeshData &d, const scalar_t *const SFEM_RESTRICT src,
                                       const int stride, std::vector<scalar_t> &ogx,
                                       std::vector<scalar_t> &ogy, std::vector<scalar_t> &ogz) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_grad_strided");
    // Over packed macro-elements where there is a packing, which leaves an eighth of the
    // nodes staged instead of half. Same operator either way; the summation order differs, so
    // the two agree to round-off rather than bit for bit.
    if (d.packed) {
        sscvfem_nodal_grad_packed(d, *d.packed, src, stride, ogx, ogy, ogz);
        return;
    }
    // Geometry, cached. Everything below is then about the field alone.
    sscvfem_build_grad_weight(d);
    const scalar_t *const SFEM_RESTRICT winv = d.grad_w_inv.data();

    ogx.assign((size_t)d.nnodes, 0);
    ogy.assign((size_t)d.nnodes, 0);
    ogz.assign((size_t)d.nnodes, 0);

    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;
    // Three fields where there were four. The weight used to ride along through the
    // per-element accumulator, the scatter and the shared reduction, which is a third of the
    // traffic of each spent re-deriving a quantity that does not change.
    static constexpr int NG = 3;

#pragma omp parallel
    {
        std::vector<scalar_t>     lp((size_t)d.nxe);
        std::vector<smesh::idx_t> lg((size_t)d.nxe);
        std::vector<scalar_t>     lacc((size_t)d.nxe * NG);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            if (sc) std::fill(lacc.begin(), lacc.end(), scalar_t(0));
            // Only the field. The coordinates used to be gathered for every node of the
            // macro-element -- three arrays of (L+1)^3 -- to feed a geometry computation that
            // is the same for all of them.
            for (int a = 0; a < d.nxe; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                lg[(size_t)a]        = g;
                lp[(size_t)a]        = src[(ptrdiff_t)g * stride];
            }

            // The geometry, once per macro-element rather than once per micro-element.
            //
            // A macro-element is subdivided uniformly, so its micro-elements are translates of
            // one another and share a Jacobian exactly. sscvfem_macro_geom has always relied
            // on this -- it is what the `hoisted` in apply_macro_local_hoisted means, and it
            // computes the geometry of the first micro-element and reuses it for all L^3.
            // This sweep did not, and paid L^3 geometry evaluations per macro-element where
            // one is needed: eight times too many at level 2 and sixty-four at level 4.
            scalar_t adj[9], det;
            {
                scalar_t ex[8], ey[8], ez[8];
                for (int a = 0; a < 8; ++a) {
                    const smesh::idx_t g = d.elems[off[a]][e];
                    ex[a]                = (scalar_t)d.points[0][g];
                    ey[a]                = (scalar_t)d.points[1][g];
                    ez[a]                = (scalar_t)d.points[2][g];
                }
                sscvfem_micro_geom(ex, ey, ez, adj, &det);
            }
            if (std::fabs(det) < scalar_t(1e-30)) continue;
            // |det| * grad, where grad itself carries a 1/det. The determinant cancels and
            // only its SIGN survives, so the division the gradient used to do and the
            // multiplication that undid it both disappear.
            const scalar_t sgn = det > 0 ? scalar_t(1) : scalar_t(-1);

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t  ep[8];
                        for (int a = 0; a < 8; ++a) ep[a] = lp[(size_t)(base + off[a])];
                        scalar_t gx, gy, gz;
                        cvfem_hex8_grad_scalar(adj, sgn, ep, gx, gy, gz);
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            if (sc) {
                                scalar_t *const acc = lacc.data() + (size_t)l * NG;
                                acc[0] += gx;
                                acc[1] += gy;
                                acc[2] += gz;
                            } else {
                                const smesh::idx_t id = lg[(size_t)l];
                                atomic_add(ogx.data(), id, gx);
                                atomic_add(ogy.data(), id, gy);
                                atomic_add(ogz.data(), id, gz);
                            }
                        }
                    }
                }
            }

            if (sc) {
                scalar_t *dst[NG] = {ogx.data(), ogy.data(), ogz.data()};
                sscvfem_scatter_element_soa_w<NG>(*sc, d.nxe, e, lg.data(), lacc.data(), dst);
            }
        }
    }

    if (sc) {
        scalar_t *dst[NG] = {ogx.data(), ogy.data(), ogz.data()};
        sscvfem_reduce_shared_soa_w<NG>(*sc, dst);
    }

    // The denominator, folded into one pass over the nodes instead of accumulated in the
    // sweep and divided out in another. The flat twin folds it into the pack drain and has no
    // pass at all; that needs the packed staging this path does not have, so one pass is the
    // floor here.
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t s = winv[(size_t)i];
        ogx[(size_t)i] *= s;
        ogy[(size_t)i] *= s;
        ogz[(size_t)i] *= s;
    }
}

inline void sscvfem_nodal_p_grad(SSMeshData &d) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_p_grad");
    sscvfem_nodal_grad_strided(d, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
}

// The same reconstruction applied to the Jacobian direction's pressure component.
inline void sscvfem_nodal_q_grad(SSMeshData &d, const scalar_t *const SFEM_RESTRICT dir) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_q_grad");
    sscvfem_nodal_grad_strided(d, dir + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);
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


                    const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                    const Hex8RhieChow rc{x,       y,  z,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(x, y, z, adj, &det);
                    cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,
                                                        rc, p, d.upwind_eps);
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

                        const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                        const Hex8RhieChow rc{x,       y,  z,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        scalar_t           adj[9], det;
                        sscvfem_micro_geom(x, y, z, adj, &det);
                        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,
                                                        rc, p, d.upwind_eps);
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
            // Micro-cell 0's corners, hoisted: the geometry AND the coordinates the
            // Rhie-Chow term differences.
            //
            // The lattice inside a macro element is uniform, so every micro-cell is congruent
            // to cell 0 and one adjugate serves all of them -- that is what the action does.
            // This used to hoist the adjugate but then hand the Rhie-Chow struct each cell's
            // OWN coordinates, and the two agree only to the precision the node positions are
            // stored in. smesh::geom_t is float32, so the block diagonal disagreed with the
            // action it is supposed to be the diagonal of by 4.23e-08 -- eight orders above
            // round-off, and invisible until the q-independent consistency gate looked.
            //
            // Only DIFFERENCES of these are taken (d = x_j - x_i), so cell 0's coordinates are
            // exact for the purpose, not an approximation. The boundary closure below still
            // gets each cell's real position, because it tests where the cell actually is.
            scalar_t madj[9], mdet;
            scalar_t c0x[8], c0y[8], c0z[8];
            {
                for (int a = 0; a < 8; ++a) {
                    const int l = off[a];
                    c0x[a]      = lx[(size_t)l];
                    c0y[a]      = ly[(size_t)l];
                    c0z[a]      = lz[(size_t)l];
                }
                sscvfem_micro_geom(c0x, c0y, c0z, madj, &mdet);
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

                        const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                        // The distances madj was built from: see sscvfem_residual.
                        const Hex8RhieChow rc{c0x,     c0y, c0z, pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, madj, mdet, ux, uy, uz, vx, vy, vz, q, r,
                                                             rc, p, d.upwind_eps);
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

// The time-scale configuration for this level's solve.
inline Hex8RcConfig sscvfem_rc_config(const SSMeshData &d) {
    // a0/dt through the transient diagonal's own rule, at rho = 1.
    //
    // That function already settles the history question this one has to settle, and for the
    // same reason: a coarse level built by clone_onto receives dt but never a history, so a
    // rule keyed on u_prev2 would give every level below the finest a different time scale
    // from the one it is correcting. Its comment records that this exact mistake once gave
    // every coarse level a steady Jacobian. Deriving the time scale's a0 anywhere else would
    // reintroduce it in the stabilisation instead.
    return cvfem_hex8_rc_config(d.rhie_chow_scale, sscvfem_transient_diag_weight(d, scalar_t(1)));
}

// The micro-cells of a macro element are congruent, so everything here is computed once per
// macro element and read by all L^3 of them. The Rhie-Chow time scale broke that: its
// advective branch carries the velocity, which varies cell to cell.
//
// The split keeps the hoist. Only |u|^2 is per-cell, and it enters the time scale as
// (2|u|/h)^2 = 4|u|^2/h^2, so the macro element can hold everything else --
//
//   rc_num[s]  = rc_scale * A2/Adotd     the whole geometric factor, zero where degenerate
//   rc_base[s] = (2 a0/dt)^2 + (4 nu/h^2)^2   the transient and diffusive branches
//   inv_h2[s]  = 1/|d|^2
//
// -- and a cell pays one add, one multiply, one square root and one divide rather than the
// twelve full coefficient evaluations it would otherwise need.
struct SSMacroGeom {
    scalar_t adj[9];
    scalar_t det;
    scalar_t A[3][3];
    scalar_t rc_num[CVFEM_HEX8_N_SCS];
    scalar_t rc_base[CVFEM_HEX8_N_SCS];
    scalar_t inv_h2[CVFEM_HEX8_N_SCS];
    scalar_t dvec[CVFEM_HEX8_N_SCS][3];
};

// The per-cell half of the coefficient. Identical to cvfem_hex8_rhie_chow_mdot_coeff by
// construction -- the flat-versus-semi-structured parity test is what holds the two together.
static SFEM_INLINE scalar_t sscvfem_rc_coeff(const SSMacroGeom &g, const int s, const scalar_t u2) {
    return g.rc_num[s] / std::sqrt(g.rc_base[s] + scalar_t(4) * u2 * g.inv_h2[s]);
}

inline void sscvfem_macro_geom(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                               const scalar_t rho, const scalar_t mu, const scalar_t rc_scale,
                               const Hex8RcTau &tau, SSMacroGeom &g) {
    sscvfem_micro_geom(x, y, z, g.adj, &g.det);
    cvfem_hex8_dir_areas(g.adj, g.A);
    const scalar_t nu = (mu > scalar_t(1e-30) ? mu : scalar_t(1e-30)) / (rho > scalar_t(0) ? rho : scalar_t(1));
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
        g.dvec[s][0] = x[j] - x[i];
        g.dvec[s][1] = y[j] - y[i];
        g.dvec[s][2] = z[j] - z[i];

        const scalar_t dx = g.dvec[s][0], dy = g.dvec[s][1], dz = g.dvec[s][2];
        const scalar_t ax = g.A[d][0], ay = g.A[d][1], az = g.A[d][2];
        const scalar_t h2    = dx * dx + dy * dy + dz * dz;
        const scalar_t Adotd = ax * dx + ay * dy + az * dz;
        const scalar_t A2    = ax * ax + ay * ay + az * az;
        const scalar_t lim   = scalar_t(1e-30) * (std::sqrt(A2 * h2) + scalar_t(1e-30));
        // A degenerate surface contributes nothing, exactly as the flat guard makes it:
        // a zero numerator over a one denominator, so the divide below stays finite.
        if (rc_scale == scalar_t(0) || rho == scalar_t(0) || std::fabs(Adotd) < lim) {
            g.rc_num[s]  = scalar_t(0);
            g.rc_base[s] = scalar_t(1);
            g.inv_h2[s]  = scalar_t(0);
            continue;
        }
        const scalar_t ct = scalar_t(2) * tau.inv_dt_a0;
        const scalar_t cd = scalar_t(4) * nu / h2;
        g.rc_num[s]  = rc_scale * (A2 / Adotd);
        g.rc_base[s] = ct * ct + cd * cd;
        g.inv_h2[s]  = tau.u2_scale / h2;
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
                                               const scalar_t *const SFEM_RESTRICT qgx,
                                               const scalar_t *const SFEM_RESTRICT qgy,
                                               const scalar_t *const SFEM_RESTRICT qgz,
                                               scalar_t *const SFEM_RESTRICT       r,
                                              const scalar_t ueps = scalar_t(0)) {
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

        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        const scalar_t c = sscvfem_rc_coeff(g, s, adv_x * adv_x + adv_y * adv_y + adv_z * adv_z);

        // -coeff * ((p_j - p_i) - avg(grad p) . d), with coeff and d both loop invariants.
        const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                               half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                               half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
        const scalar_t mdot_rc = -c * corr;

        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
        scalar_t amdot, sgn;
        cvfem_upwind_abs(mdot, ueps, amdot, sgn);
        const scalar_t mpos  = half * (mdot + amdot);
        const scalar_t mneg  = half * (mdot - amdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);

        // Mirror the residual's corr above: corr_q = (q_j - q_i) - avg(qg_i, qg_j) . d, which
        // contributes -c * corr_q. Keeping only c*(q_i - q_j) freezes the pressure-gradient
        // reconstruction, leaving the continuity rows ~4% wrong and capping Newton at a linear
        // rate. qgx == nullptr restores that old behaviour. See SFEM_FD_CHECK.
        const scalar_t dcorr = qgx ? (half * (qgx[i] + qgx[j]) * g.dvec[s][0] +
                                      half * (qgy[i] + qgy[j]) * g.dvec[s][1] +
                                      half * (qgz[i] + qgz[j]) * g.dvec[s][2])
                                   : scalar_t(0);
        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               c * (q[i] - q[j]) + c * dcorr;
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

    const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lvx((size_t)nxe), lvy((size_t)nxe), lvz((size_t)nxe), lq((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        // Direction gradient, gathered the same way. Empty when Rhie-Chow is off.
        const bool                has_qg = !d.qgx.empty();
        std::vector<scalar_t>     lqgx((size_t)(has_qg ? nxe : 0)), lqgy((size_t)(has_qg ? nxe : 0)),
                                  lqgz((size_t)(has_qg ? nxe : 0));
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
                if (has_qg) {
                    lqgx[(size_t)a] = d.qgx[(size_t)g];
                    lqgy[(size_t)a] = d.qgy[(size_t)g];
                    lqgz[(size_t)a] = d.qgz[(size_t)g];
                }
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
                const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rcfg.scale, rcfg.tau, mg);
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t qgx[8], qgy[8], qgz[8];
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
                            if (has_qg) {
                                qgx[a] = lqgx[(size_t)l];
                                qgy[a] = lqgy[(size_t)l];
                                qgz[a] = lqgz[(size_t)l];
                            }
                        }

                        sscvfem_action_hoisted(rho, mu, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz,
                                               has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                                               has_qg ? qgz : nullptr, r, d.upwind_eps);
                        boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                         ux, uy, uz, vx, vy, vz, q, r,
                                                         d.macro_face_mask.empty()
                                                          ? -1
                                                          : sscvfem_micro_face_mask(
                                                                    (int)d.macro_face_mask[(size_t)e],
                                                                    L, xi, yi, zi),
                                                  sscvfem_micro_face_mask(
                                                          d.macro_natural_mask.empty() ? 0
                                                              : (int)d.macro_natural_mask[(size_t)e],
                                                          L, xi, yi, zi),
                                                  sscvfem_bd(d, e, L, xi, yi, zi));

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            if (sc)
                sscvfem_scatter_element(*sc, nxe, e, lg.data(), lout.data(), jv);
            else
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = lg[(size_t)a];
                    for (int c = 0; c < N_FIELDS; ++c)
                        atomic_add(jv + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
                }
        }
    }

    if (sc) sscvfem_reduce_shared(*sc, jv);
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
// The 2x2 field-block split: (velocity, pressure) x (velocity, pressure).
//
//        | A_uu  B^T |   momentum rows
//   J =  |           |
//        | B     C   |   continuity rows
//
// Solution schemes want these separately. A Schur approximation needs B and B^T to form
// B A^-1 B^T; a segregated or projection scheme solves the momentum rows alone; the
// pressure preconditioner investigated in the standalone driver needs C by itself; and a
// Vanka or block smoother wants to address them independently. Evaluating the whole
// operator and discarding three quarters of it is the thing to avoid.
//
// Where each term lands, which is not one-to-one with the code's own structure:
//
//   viscous                     -> A_uu
//   qmid * A                    -> B^T
//   convective mpos/mneg on v   -> A_uu
//   continuity dmdot            -> split, see below
//
// The convective flux contributes to BOTH A_uu and B^T, because the mass-flux derivative
// carries a velocity part and a pressure part:
//
//   dmdot = rho/2 (v_i + v_j).A   +   c (q_i - q_j)
//           \_____ velocity _____/     \___ pressure ___/
//
// so d_pos * dmdot * u_i splits along the same line, and the continuity row splits into
// B (the velocity half) and C (the Rhie-Chow half). Getting that wrong would put the
// Rhie-Chow coupling in A_uu, where it would quietly break any Schur approximation built
// on these blocks.

enum SSBlock : int {
    SSBLOCK_UU  = 1 << 0,  // momentum rows, velocity columns
    SSBLOCK_UP  = 1 << 1,  // momentum rows, pressure column   (B^T)
    SSBLOCK_PU  = 1 << 2,  // continuity row, velocity columns (B)
    SSBLOCK_PP  = 1 << 3,  // continuity row, pressure column  (C)
    SSBLOCK_MOM = SSBLOCK_UU | SSBLOCK_UP,
    SSBLOCK_CON = SSBLOCK_PU | SSBLOCK_PP,
    SSBLOCK_ALL = SSBLOCK_MOM | SSBLOCK_CON
};

// Reference: correct by construction, and slow.
//
// A block is selected by zeroing the input components outside its columns and the output
// rows outside its rows, around the unmodified full operator. That cannot disagree with
// the operator, which is what makes it the right thing to check the fast path against --
// the fast path restates the arithmetic and could drift.
// SFEM_RC_EXACT_JAC, read once. Shared by the full apply, the block apply and the block
// reference so the three cannot drift: whatever the operator differentiates through, the
// blocks must differentiate through too, or they do not sum back to it.
inline bool sscvfem_rc_exact_jac() {
    static const int v = smesh::Env::read<int>("SFEM_RC_EXACT_JAC", 1);
    return v != 0;
}

// Whether the direction's pressure gradient has to be reconstructed for this evaluation:
// only the pressure-column blocks read it, so a velocity-column block skips the pass.
//
// d.blocks_exact_rc is how a *preconditioner* declines the term. The block apply defaults
// to being the exact restriction of the operator, because that is the contract the four
// blocks are checked against; but a preconditioner does not have to be the operator, and
// measurement on Grace says it should not pay for being one here. The term costs one nodal
// gradient reconstruction per pressure-column apply -- 0.919 ns/dof at 34,147,332 dofs on
// 72 cores, which is +140% on B^T, +217% on C and +89% on the whole block operator -- and
// changes the standalone convergence rate of the SIMPLE smoother by nothing at all: at
// 75,140 dofs the two agree to six decimals at every sweep, 0.990157 against 0.990157 at
// sweep 39. So the driver's smoother turns it off and the operator keeps it on.
inline bool sscvfem_wants_q_grad(const SSMeshData &d, const int blocks) {
    return (blocks & (SSBLOCK_UP | SSBLOCK_PP)) != 0 && d.blocks_exact_rc &&
           sscvfem_rc_exact_jac() && d.rhie_chow_scale != scalar_t(0) && !d.pgx.empty();
}

inline void sscvfem_apply_blocks_ref(SSMeshData &d, const scalar_t rho, const scalar_t mu, const int blocks,
                                     const scalar_t *const SFEM_RESTRICT dir,
                                     scalar_t *const SFEM_RESTRICT       jv) {
    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    std::vector<scalar_t> v((size_t)ndof), y((size_t)ndof);

    const bool want_ucol = (blocks & (SSBLOCK_UU | SSBLOCK_PU)) != 0;
    const bool want_pcol = (blocks & (SSBLOCK_UP | SSBLOCK_PP)) != 0;

    for (ptrdiff_t i = 0; i < ndof; ++i) jv[i] = scalar_t(0);

    const bool exact = sscvfem_rc_exact_jac() && d.rhie_chow_scale != scalar_t(0) && !d.pgx.empty();
    if (!exact) {
        d.qgx.clear();
        d.qgy.clear();
        d.qgz.clear();
    }

    for (int pass = 0; pass < 2; ++pass) {
        const bool ucol = (pass == 0);
        if (ucol && !want_ucol) continue;
        if (!ucol && !want_pcol) continue;

        for (ptrdiff_t n = 0; n < d.nnodes; ++n) {
            for (int c = 0; c < 3; ++c) v[(size_t)n * 4 + c] = ucol ? dir[(size_t)n * 4 + c] : scalar_t(0);
            v[(size_t)n * 4 + 3] = ucol ? scalar_t(0) : dir[(size_t)n * 4 + 3];
        }
        std::fill(y.begin(), y.end(), scalar_t(0));
        // Rhie-Chow's derivative reaches the kernel through d.qg, which is reconstructed
        // from the direction's pressure -- ambient state, not an argument -- so masking the
        // input vector does not mask it, and the reference would carry the whole correction
        // into every block. Reconstruct it from the masked vector instead. That is what
        // makes this a column restriction: the velocity pass sees zero pressure and so no
        // correction at all, the pressure pass sees the whole of it.
        if (exact) sscvfem_nodal_q_grad(d, v.data());
        // The default apply, named directly rather than through sscvfem_apply, which
        // is declared below this point.
        sscvfem_apply_macro_local_hoisted(d, rho, mu, v.data(), y.data());

        const int mom_bit = ucol ? SSBLOCK_UU : SSBLOCK_UP;
        const int con_bit = ucol ? SSBLOCK_PU : SSBLOCK_PP;
        for (ptrdiff_t n = 0; n < d.nnodes; ++n) {
            if (blocks & mom_bit)
                for (int c = 0; c < 3; ++c) jv[(size_t)n * 4 + c] += y[(size_t)n * 4 + c];
            if (blocks & con_bit) jv[(size_t)n * 4 + 3] += y[(size_t)n * 4 + 3];
        }
    }
}

// Fast path: the hoisted action with the unwanted terms compiled out.
template <int Blocks>
static SFEM_INLINE void sscvfem_action_blocks(const scalar_t rho, const scalar_t mu, const SSMacroGeom &g,
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
                                              const scalar_t *const SFEM_RESTRICT qgx,
                                              const scalar_t *const SFEM_RESTRICT qgy,
                                              const scalar_t *const SFEM_RESTRICT qgz,
                                              scalar_t *const SFEM_RESTRICT       r,
                                              const scalar_t ueps = scalar_t(0)) {
    constexpr bool uu = (Blocks & SSBLOCK_UU) != 0;
    constexpr bool up = (Blocks & SSBLOCK_UP) != 0;
    constexpr bool pu = (Blocks & SSBLOCK_PU) != 0;
    constexpr bool pp = (Blocks & SSBLOCK_PP) != 0;
    constexpr bool mom = uu || up;

    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    // Viscous: A_uu only. Skipped entirely for a pressure-block evaluation, which is most
    // of what makes C cheap to get on its own.
    if constexpr (uu) {
        scalar_t dgrad[9];
        cvfem_hex8_grad_sumfact(g.adj, g.det, vx, vy, vz, dgrad);
        for (int d2 = 0; d2 < 3; ++d2) {
            scalar_t tx, ty, tz;
            cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6],
                                dgrad[7], dgrad[8], g.A[d2][0], g.A[d2][1], g.A[d2][2], tx, ty, tz);
            for (int e = 0; e < 4; ++e) {
                const int i = CVFEM_HEX8_DIR_EDGES[d2][e][0];
                const int j = CVFEM_HEX8_DIR_EDGES[d2][e][1];
                r[i * 4 + 0] -= tx;
                r[i * 4 + 1] -= ty;
                r[i * 4 + 2] -= tz;
                r[j * 4 + 0] += tx;
                r[j * 4 + 1] += ty;
                r[j * 4 + 2] += tz;
            }
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      dd = s >> 2;
        const scalar_t ax = g.A[dd][0], ay = g.A[dd][1], az = g.A[dd][2];
        const scalar_t uax = half * (ux[i] + ux[j]);
        const scalar_t uay = half * (uy[i] + uy[j]);
        const scalar_t uaz = half * (uz[i] + uz[j]);
        const scalar_t c   = sscvfem_rc_coeff(g, s, uax * uax + uay * uay + uaz * uaz);

        // The upwind weights are needed only by the momentum rows. The continuity row is
        // dmdot_v + dmdot_q with no sgn in it, so for a pressure-row evaluation the whole
        // upwind computation -- the Rhie-Chow correction, the mass flux, the sign and the
        // four weights -- is dead. That is most of what makes C cheap to ask for.
        scalar_t mpos = 0, mneg = 0, d_pos = 0, d_neg = 0;
        if constexpr (mom) {
            const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                                   half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                                   half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
            const scalar_t mdot = rho * (half * (ux[i] + ux[j]) * ax + half * (uy[i] + uy[j]) * ay +
                                         half * (uz[i] + uz[j]) * az) - c * corr;
            scalar_t amdot, sgn;
            cvfem_upwind_abs(mdot, ueps, amdot, sgn);
            mpos  = half * (mdot + amdot);
            mneg  = half * (mdot - amdot);
            d_pos = half * (one + sgn);
            d_neg = half * (one - sgn);
        }

        // The two halves of the mass-flux derivative, kept apart so the blocks can be.
        const scalar_t dmdot_v = (uu || pu) ? rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay +
                                                            (vz[i] + vz[j]) * az)
                                            : scalar_t(0);
        // Rhie-Chow differentiates through the nodal pressure-gradient reconstruction, and
        // that derivative is a pressure-column term -- it is built from the gradient of the
        // *direction's* pressure -- so it belongs to B^T and C and to neither velocity-column
        // block. sscvfem_action_hoisted carries it as c * dcorr. Omitting it here did not
        // make any one block wrong in an obvious way; it made the four of them fail to sum
        // back to the operator, which is exactly the second check the bench performs and had
        // been reporting at 1.0e-01 since the exact term was introduced.
        // qgx == nullptr is the frozen-pg form, as in the hoisted kernel.
        const scalar_t dcorr = ((up || pp) && qgx) ? (half * (qgx[i] + qgx[j]) * g.dvec[s][0] +
                                                      half * (qgy[i] + qgy[j]) * g.dvec[s][1] +
                                                      half * (qgz[i] + qgz[j]) * g.dvec[s][2])
                                                   : scalar_t(0);
        const scalar_t dmdot_q = (up || pp) ? c * ((q[i] - q[j]) + dcorr) : scalar_t(0);

        if constexpr (mom) {
            scalar_t fx = 0, fy = 0, fz = 0;
            if constexpr (uu) {
                const scalar_t apos = d_pos * dmdot_v;
                const scalar_t aneg = d_neg * dmdot_v;
                fx += apos * ux[i] + mpos * vx[i] + aneg * ux[j] + mneg * vx[j];
                fy += apos * uy[i] + mpos * vy[i] + aneg * uy[j] + mneg * vy[j];
                fz += apos * uz[i] + mpos * vz[i] + aneg * uz[j] + mneg * vz[j];
            }
            if constexpr (up) {
                const scalar_t apos = d_pos * dmdot_q;
                const scalar_t aneg = d_neg * dmdot_q;
                const scalar_t qmid = half * (q[i] + q[j]);
                fx += apos * ux[i] + aneg * ux[j] + qmid * ax;
                fy += apos * uy[i] + aneg * uy[j] + qmid * ay;
                fz += apos * uz[i] + aneg * uz[j] + qmid * az;
            }
            r[i * 4 + 0] += fx;
            r[i * 4 + 1] += fy;
            r[i * 4 + 2] += fz;
            r[j * 4 + 0] -= fx;
            r[j * 4 + 1] -= fy;
            r[j * 4 + 2] -= fz;
        }

        if constexpr (pu || pp) {
            scalar_t dm = 0;
            if constexpr (pu) dm += dmdot_v;
            if constexpr (pp) dm += dmdot_q;
            r[i * 4 + 3] += dm;
            r[j * 4 + 3] -= dm;
        }
    }
}

// Macro-local sweep for a chosen set of blocks.
//
// The boundary sub-control-surface term is handled by input masking rather than by
// specialising it: it is a shared kernel that writes all four blocks, and restating it
// here to split it would be a second copy of arithmetic that already exists. Zeroing the
// direction components outside the wanted columns, and dropping the rows outside the
// wanted rows, gives its contribution to those blocks exactly. It is a boundary term, so
// it runs on a vanishing fraction of the elements and its cost does not drive this.
template <int Blocks>
inline SFEM_NOINLINE void sscvfem_apply_blocks_impl(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                    const scalar_t *const SFEM_RESTRICT dir,
                                                    scalar_t *const SFEM_RESTRICT       jv) {
    constexpr bool uu = (Blocks & SSBLOCK_UU) != 0;
    constexpr bool up = (Blocks & SSBLOCK_UP) != 0;
    constexpr bool pu = (Blocks & SSBLOCK_PU) != 0;
    constexpr bool pp = (Blocks & SSBLOCK_PP) != 0;
    constexpr bool mom = uu || up;

    const int L   = d.level;
    const int nxe = d.nxe;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lvx((size_t)nxe), lvy((size_t)nxe), lvz((size_t)nxe), lq((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        // The direction's reconstructed gradient, staged exactly as the hoisted apply stages
        // it, and only for a pressure-column block -- for A_uu and B the term is absent by
        // construction, so this gather is skipped along with the rest of the pressure work.
        const bool                has_qg = (up || pp) && !d.qgx.empty();
        std::vector<scalar_t>     lqgx((size_t)(has_qg ? nxe : 0)), lqgy((size_t)(has_qg ? nxe : 0)),
                                  lqgz((size_t)(has_qg ? nxe : 0));
        std::vector<scalar_t>     lout((size_t)nxe * N_FIELDS);

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
            // Gather only what this block reads. On Grace the gather and scatter alone are
            // 35% of the full operator, so a block that still loads all fourteen arrays
            // cannot get far below that however little arithmetic it does -- C was 46%
            // against a 35% floor.
            //
            // Coordinates and the state velocity are always needed: the first by the macro
            // geometry, the second by the boundary term, which takes ux, uy, uz whatever
            // is being masked. The state pressure and its gradient are read only by the
            // upwind switch, which lives in the momentum rows. The direction velocity is
            // read by A_uu and B; the direction pressure by B^T and C.
            constexpr bool need_state_p = mom;              // upwind correction
            constexpr bool need_dir_v   = uu || pu;
            constexpr bool need_dir_q   = up || pp;

            for (int a = 0; a < nxe; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)d.points[0][g];
                ly[(size_t)a]        = (scalar_t)d.points[1][g];
                lz[(size_t)a]        = (scalar_t)d.points[2][g];
                lux[(size_t)a]       = d.ux[(size_t)g];
                luy[(size_t)a]       = d.uy[(size_t)g];
                luz[(size_t)a]       = d.uz[(size_t)g];
                if constexpr (need_state_p) {
                    lp[(size_t)a]   = d.p[(size_t)g];
                    lpgx[(size_t)a] = d.pgx[(size_t)g];
                    lpgy[(size_t)a] = d.pgy[(size_t)g];
                    lpgz[(size_t)a] = d.pgz[(size_t)g];
                }
                if constexpr (need_dir_v) {
                    lvx[(size_t)a] = dir[(size_t)g * 4 + 0];
                    lvy[(size_t)a] = dir[(size_t)g * 4 + 1];
                    lvz[(size_t)a] = dir[(size_t)g * 4 + 2];
                }
                if constexpr (need_dir_q) lq[(size_t)a] = dir[(size_t)g * 4 + 3];
                if (has_qg) {
                    lqgx[(size_t)a] = d.qgx[(size_t)g];
                    lqgy[(size_t)a] = d.qgy[(size_t)g];
                    lqgz[(size_t)a] = d.qgz[(size_t)g];
                }
            }
            // Anything not gathered must still read as zero, since the element kernels and
            // the boundary term take all of them regardless.
            if constexpr (!need_state_p) {
                std::fill(lp.begin(), lp.end(), scalar_t(0));
                std::fill(lpgx.begin(), lpgx.end(), scalar_t(0));
                std::fill(lpgy.begin(), lpgy.end(), scalar_t(0));
                std::fill(lpgz.begin(), lpgz.end(), scalar_t(0));
            }
            if constexpr (!need_dir_v) {
                std::fill(lvx.begin(), lvx.end(), scalar_t(0));
                std::fill(lvy.begin(), lvy.end(), scalar_t(0));
                std::fill(lvz.begin(), lvz.end(), scalar_t(0));
            }
            if constexpr (!need_dir_q) std::fill(lq.begin(), lq.end(), scalar_t(0));
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
                const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rcfg.scale, rcfg.tau, mg);
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t qgx[8], qgy[8], qgz[8];
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
                            if (has_qg) {
                                qgx[a] = lqgx[(size_t)l];
                                qgy[a] = lqgy[(size_t)l];
                                qgz[a] = lqgz[(size_t)l];
                            }
                        }

                        // d.upwind_eps, not the default zero: every other call site passes it,
                        // and a block apply that smooths the upwind switch differently from
                        // the operator is not a restriction of it either.
                        sscvfem_action_blocks<Blocks>(rho, mu, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz,
                                                      has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                                                      has_qg ? qgz : nullptr, r, d.upwind_eps);

                        // Boundary term, by input masking. Two passes only when both
                        // column groups are wanted, which for the full operator is the
                        // single unmasked pass below.
                        // A row group that wants every column needs no input masking at
                        // all: run the boundary term once as it stands and keep the rows.
                        // Without this, asking for the momentum rows costs more than the
                        // whole operator, because the two masked passes outweigh the terms
                        // the specialisation removes.
                        constexpr bool all_cols_mom = uu && up;
                        constexpr bool all_cols_con = pu && pp;
                        constexpr bool no_masking =
                                (Blocks == SSBLOCK_ALL) ||
                                (all_cols_mom && !pu && !pp) || (all_cols_con && !uu && !up);

                        if constexpr (no_masking) {
                            scalar_t rb[CVFEM_HEX8_N_DOF];
                            for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                            boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                             ux, uy, uz, vx, vy, vz, q, rb);
                            for (int a = 0; a < 8; ++a) {
                                if constexpr (uu || up)
                                    for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                if constexpr (pu || pp) r[a * 4 + 3] += rb[a * 4 + 3];
                            }
                        } else {
                            scalar_t zero8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                            scalar_t rb[CVFEM_HEX8_N_DOF];
                            if constexpr (uu || pu) {
                                for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                                boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                                 ux, uy, uz, vx, vy, vz, zero8, rb);
                                for (int a = 0; a < 8; ++a) {
                                    if constexpr (uu)
                                        for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                    if constexpr (pu) r[a * 4 + 3] += rb[a * 4 + 3];
                                }
                            }
                            if constexpr (up || pp) {
                                for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                                boundary_scs_add_jacobian_action(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                                 ux, uy, uz, zero8, zero8, zero8, q, rb);
                                for (int a = 0; a < 8; ++a) {
                                    if constexpr (up)
                                        for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                    if constexpr (pp) r[a * 4 + 3] += rb[a * 4 + 3];
                                }
                            }
                        }

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            // Scatter only the rows written: a continuity-row block touches one component
            // of four, and the atomics are the expensive half of the scatter.
            // The two-pass scatter moves all four components. The rows this block
            // selection does not write are zero in lout, so they contribute nothing, and
            // the saving the component-wise atomics bought no longer applies once the
            // scatter is a plain write.
            if (sc)
                sscvfem_scatter_element(*sc, nxe, e, lg.data(), lout.data(), jv);
            else
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = lg[(size_t)a];
                    if constexpr (uu || up)
                        for (int c = 0; c < 3; ++c)
                            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
                    if constexpr (pu || pp)
                        atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, lout[(size_t)a * N_FIELDS + 3]);
                }
        }
    }

    if (sc) sscvfem_reduce_shared(*sc, jv);
}

// Runtime entry. The mask is a compile-time parameter inside, so each combination gets a
// kernel with the terms it does not need removed rather than branched over.
// Defined below, beside the body force it mirrors.
inline void     sscvfem_apply_transient_action(SSMeshData &d, const scalar_t rho,
                                               const scalar_t *const SFEM_RESTRICT dir,
                                               scalar_t *const SFEM_RESTRICT       jv);

inline void sscvfem_apply_blocks(SSMeshData &d, const scalar_t rho, const scalar_t mu, const int blocks,
                                 const scalar_t *const SFEM_RESTRICT dir,
                                 scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_blocks");
    // The block apply is meant to be the restriction of the operator to a field block, so
    // it differentiates through the same reconstruction the operator does. Only the
    // pressure-column blocks read the result, so B and A_uu skip the pass entirely.
    if (sscvfem_wants_q_grad(d, blocks)) {
        sscvfem_nodal_q_grad(d, dir);
    } else {
        d.qgx.clear();
        d.qgy.clear();
        d.qgz.clear();
    }

    switch (blocks & SSBLOCK_ALL) {
        // 0 selects no block at all: the sweep gathers the macro-element, computes
        // nothing, and scatters zeros. That is the floor any block specialisation can
        // reach, and it is worth being able to measure rather than infer.
        case 0:           sscvfem_apply_blocks_impl<0>(d, rho, mu, dir, jv);           break;
        case SSBLOCK_UU:  sscvfem_apply_blocks_impl<SSBLOCK_UU>(d, rho, mu, dir, jv);  break;
        case SSBLOCK_UP:  sscvfem_apply_blocks_impl<SSBLOCK_UP>(d, rho, mu, dir, jv);  break;
        case SSBLOCK_PU:  sscvfem_apply_blocks_impl<SSBLOCK_PU>(d, rho, mu, dir, jv);  break;
        case SSBLOCK_PP:  sscvfem_apply_blocks_impl<SSBLOCK_PP>(d, rho, mu, dir, jv);  break;
        case SSBLOCK_MOM: sscvfem_apply_blocks_impl<SSBLOCK_MOM>(d, rho, mu, dir, jv); break;
        case SSBLOCK_CON: sscvfem_apply_blocks_impl<SSBLOCK_CON>(d, rho, mu, dir, jv); break;
        case SSBLOCK_ALL: sscvfem_apply_blocks_impl<SSBLOCK_ALL>(d, rho, mu, dir, jv); break;
        default:          sscvfem_apply_blocks_ref(d, rho, mu, blocks, dir, jv);       break;
    }

    sscvfem_apply_transient_action(d, rho, dir, jv);
}

// ---------------------------------------------------------------------------
// Residual. Newton needs it, and until now the semi-structured path had only the
// Jacobian action, the block diagonal and the block split.
//
// Same two layouts as everything else, for the same reason: the naive one keeps the flat
// gather and exists to check the macro-local one against.

// Control volume per node, the semi-structured twin of build_node_volume.
//
// A micro-element's eight sub-control volumes partition it evenly, so each of its corners
// collects |det|/8. The macro geometry is affine, so one determinant serves every micro
// element of a macro element and the inner loops are pure index arithmetic.
inline void sscvfem_node_volume(SSMeshData &d, std::vector<scalar_t> &node_vol) {
    SFEM_TRACE_SCOPE("sscvfem::node_volume");
    node_vol.assign((size_t)d.nnodes, scalar_t(0));
    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
        scalar_t ex[8], ey[8], ez[8];
        for (int a = 0; a < 8; ++a) {
            const smesh::idx_t g = d.elems[off[a]][e];
            ex[a] = (scalar_t)d.points[0][g];
            ey[a] = (scalar_t)d.points[1][g];
            ez[a] = (scalar_t)d.points[2][g];
        }
        scalar_t adj[9], det;
        sscvfem_micro_geom(ex, ey, ez, adj, &det);
        const scalar_t v = std::fabs(det) / scalar_t(8);

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);
                    for (int a = 0; a < 8; ++a) {
                        const smesh::idx_t g = d.elems[base + off[a]][e];
                        atomic_add(node_vol.data(), g, v);
                    }
                }
    }
}

// Subtract the body force from the momentum rows of an interleaved residual. Mirrors
// apply_body_force in cvfem_hex8_ns_core.hpp; see the sign argument there.
inline void sscvfem_apply_body_force(SSMeshData &d, scalar_t *const SFEM_RESTRICT res) {
    if (d.fx.empty()) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t v = d.node_vol[(size_t)i];
        res[i * N_FIELDS + 0] -= d.fx[(size_t)i] * v;
        res[i * N_FIELDS + 1] -= d.fy[(size_t)i] * v;
        res[i * N_FIELDS + 2] -= d.fz[(size_t)i] * v;
    }
}

// The transient term on an interleaved residual. Mirrors apply_transient in
// cvfem_hex8_ns_core.hpp -- same coefficients, same lumped control volume, same reason for
// being a post-pass rather than a term inside the macro-element sweeps.
inline void sscvfem_apply_transient(SSMeshData &d, const scalar_t rho, scalar_t *const SFEM_RESTRICT res) {
    SFEM_TRACE_SCOPE("sscvfem::apply_transient");
    if (d.dt <= scalar_t(0)) return;
    if ((ptrdiff_t)d.u_prev.size() != 3 * d.nnodes) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
    const bool     two = d.bdf_order >= 2 && (ptrdiff_t)d.u_prev2.size() == 3 * d.nnodes;
    // Variable-step BDF2 when a previous step size has been recorded; see bdf_coeffs in the
    // flat core for the derivation and for why using the uniform coefficients after the step
    // changes is the wrong scheme rather than an approximation. w = 1 and dt_prev = 0 both
    // reduce to {3/2, -2, 1/2} exactly, so nothing that does not adapt moves.
    scalar_t a0 = two ? scalar_t(1.5) : scalar_t(1);
    scalar_t a1 = two ? scalar_t(-2) : scalar_t(-1);
    scalar_t a2 = two ? scalar_t(0.5) : scalar_t(0);
    if (two && d.dt_prev > scalar_t(0)) {
        const scalar_t w = d.dt / d.dt_prev;
        if (w != scalar_t(1)) {
            const scalar_t den = scalar_t(1) + w;
            a0 = (scalar_t(1) + scalar_t(2) * w) / den;
            a1 = -den;
            a2 = w * w / den;
        }
    }
    const scalar_t inv = scalar_t(1) / d.dt;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w   = rho * d.node_vol[(size_t)i] * inv;
        const size_t   k   = (size_t)i * 3;
        const scalar_t u[3] = {d.ux[(size_t)i], d.uy[(size_t)i], d.uz[(size_t)i]};
        for (int c = 0; c < 3; ++c) {
            const scalar_t prev2 = two ? d.u_prev2[k + (size_t)c] : scalar_t(0);
            res[i * N_FIELDS + c] += w * (a0 * u[c] + a1 * d.u_prev[k + (size_t)c] + a2 * prev2);
        }
    }
}

// The weight the transient term puts on each velocity diagonal entry: rho V a0 / dt.
inline scalar_t sscvfem_transient_diag_weight(const SSMeshData &d, const scalar_t rho) {
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
    if ((ptrdiff_t)d.u_prev.size() == 3 * d.nnodes) {
        const bool two = d.bdf_order >= 2 && (ptrdiff_t)d.u_prev2.size() == 3 * d.nnodes;
        // a0 must track the residual's, or a variable step leaves the Jacobian diagonal
        // differentiating a scheme the residual is no longer running.
        scalar_t a0 = two ? scalar_t(1.5) : scalar_t(1);
        if (two && d.dt_prev > scalar_t(0)) {
            const scalar_t w = d.dt / d.dt_prev;
            if (w != scalar_t(1)) a0 = (scalar_t(1) + scalar_t(2) * w) / (scalar_t(1) + w);
        }
        return a0 * rho / d.dt;
    }
    return (d.bdf_order >= 2 ? scalar_t(1.5) : scalar_t(1)) * rho / d.dt;
}

inline void sscvfem_apply_transient_action(SSMeshData &d, const scalar_t rho,
                                           const scalar_t *const SFEM_RESTRICT dir,
                                           scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("sscvfem::apply_transient_action");
    const scalar_t a = sscvfem_transient_diag_weight(d, rho);
    if (a == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t w = a * d.node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) jv[i * N_FIELDS + c] += w * dir[i * N_FIELDS + c];
    }
}

inline SFEM_NOINLINE void sscvfem_residual_naive(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                 scalar_t *const SFEM_RESTRICT res) {
    SFEM_TRACE_SCOPE("sscvfem::residual_naive");
    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    for (ptrdiff_t i = 0; i < ndof; ++i) res[i] = scalar_t(0);

    const int L = d.level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
        for (int zi = 0; zi < L; ++zi) {
            for (int yi = 0; yi < L; ++yi) {
                for (int xi = 0; xi < L; ++xi) {
                    const int    base = sscvfem_lidx(L, xi, yi, zi);
                    smesh::idx_t g[8];
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
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
                        pgx[a] = d.pgx[(size_t)g[a]];
                        pgy[a] = d.pgy[(size_t)g[a]];
                        pgz[a] = d.pgz[(size_t)g[a]];
                    }
                    const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                    const Hex8RhieChow rc{x,       y,  z,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(x, y, z, adj, &det);
                    cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r, rc,
                                                         d.upwind_eps);
                    boundary_scs_add_residual(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, p, r);
                    for (int a = 0; a < 8; ++a)
                        for (int c = 0; c < N_FIELDS; ++c)
                            atomic_add(res + (ptrdiff_t)g[a] * N_FIELDS + c, 0, r[a * 4 + c]);
                }
            }
        }
    }
    sscvfem_apply_body_force(d, res);
    sscvfem_apply_transient(d, rho, res);
}

// zero_first=false accumulates, which is what sfem::Op::gradient needs: Function runs
// every operator over one shared output buffer without clearing between them.
// The nodal velocity gradient the deferred correction extrapolates with, semi-structured.
// Three passes of sscvfem_nodal_grad_strided -- the same reconstruction the pressure uses and
// the same one CVFEMNavierStokes::nodal_velocity_gradient publishes -- interleaved into the
// [i*9 + r*3 + c] layout the correction kernel reads.
inline void sscvfem_assemble_nodal_u_grad(SSMeshData &d) {
    SFEM_TRACE_SCOPE("sscvfem::assemble_nodal_u_grad");
    std::vector<scalar_t> gx, gy, gz;
    d.ugrad.assign((size_t)d.nnodes * 9, scalar_t(0));
    const scalar_t *const src[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
    for (int r = 0; r < 3; ++r) {
        sscvfem_nodal_grad_strided(d, src[r], 1, gx, gy, gz);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.ugrad[(size_t)i * 9 + (size_t)r * 3 + 0] = gx[(size_t)i];
            d.ugrad[(size_t)i * 9 + (size_t)r * 3 + 1] = gy[(size_t)i];
            d.ugrad[(size_t)i * 9 + (size_t)r * 3 + 2] = gz[(size_t)i];
        }
    }
}

inline SFEM_NOINLINE void sscvfem_residual(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                           scalar_t *const SFEM_RESTRICT res, const bool zero_first = true) {
    SFEM_TRACE_SCOPE("sscvfem::residual");
    // Deferred-correction convection. Off by default and then bit-for-bit the scheme every
    // recorded number here was measured with; the gradient is built once per residual, which
    // is once per Newton step, because the correction is lagged by construction.
    d.conv_ho      = smesh::Env::read<int>("SFEM_CONV_HO", 0);
    d.conv_limiter = smesh::Env::read<int>("SFEM_CONV_LIMITER", 1);
    if (d.conv_ho) sscvfem_assemble_nodal_u_grad(d);
    else d.ugrad.clear();

    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    if (zero_first)
        for (ptrdiff_t i = 0; i < ndof; ++i) res[i] = scalar_t(0);

    const int L   = d.level;
    const int nxe = d.nxe;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        // Nine per node when the correction is on, empty otherwise -- one allocation that
        // costs nothing to a run that has not asked for it.
        std::vector<scalar_t>     lug(d.conv_ho ? (size_t)nxe * 9 : 0);
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
                lpgx[(size_t)a]      = d.pgx[(size_t)g];
                lpgy[(size_t)a]      = d.pgy[(size_t)g];
                lpgz[(size_t)a]      = d.pgz[(size_t)g];
                if (!lug.empty())
                    for (int k = 0; k < 9; ++k) lug[(size_t)a * 9 + (size_t)k] = d.ugrad[(size_t)g * 9 + (size_t)k];
            }
            std::fill(lout.begin(), lout.end(), scalar_t(0));

            SSMacroGeom mg;
            // The corners mg is built from outlive the block below, because the Rhie-Chow
            // term takes its node distances from them -- as the Jacobian action takes them
            // from mg.dvec, which is built from the same corners. Each micro cell's own
            // coordinates agree with those only on an affine macro element. On a curved one
            // they did not, and the residual and its Jacobian action disagreed in every
            // continuity row: measured on the FDA nozzle by SFEM_FD_CHECK, 6.0e-02 at macro
            // core 2 / L 2, 3.2e-02 at L 4 and 1.2e-02 at macro core 4 / L 2, and exact with
            // Rhie-Chow off.
            scalar_t ex[8], ey[8], ez[8];
            {
                for (int a = 0; a < 8; ++a) {
                    const int l = off[a];
                    ex[a]       = lx[(size_t)l];
                    ey[a]       = ly[(size_t)l];
                    ez[a]       = lz[(size_t)l];
                }
                const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rcfg.scale, rcfg.tau, mg);
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t  x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                        scalar_t g8[CVFEM_HEX8_N_NODES * 9];
                        scalar_t  r[CVFEM_HEX8_N_DOF];
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
                            if (!lug.empty())
                                for (int k = 0; k < 9; ++k) g8[a * 9 + k] = lug[(size_t)l * 9 + (size_t)k];
                        }
                        const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                        const Hex8RhieChow rc{ex,      ey, ez, pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        // Deferred-correction convection, on the path the production solver
                        // actually runs: FGMRES preconditioned by multigrid needs this lattice,
                        // so a correction that existed only on the flat mesh could not be used
                        // for anything at scale. Null when off, which is the arithmetic this
                        // call did before.
                        const bool ho = !lug.empty();
                        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, mg.adj, mg.det, ux, uy, uz, p, r,
                                                             rc, d.upwind_eps,
                                                             ho ? g8 : nullptr,
                                                             ho ? x : nullptr, ho ? y : nullptr,
                                                             ho ? z : nullptr, d.conv_limiter);
                        boundary_scs_add_residual(rho, mu, 0, mg.adj, mg.det, d.Lx, d.Ly, d.Lz, x, y, z,
                                                  ux, uy, uz, p, r,
                                                  d.macro_face_mask.empty()
                                                          ? -1
                                                          : sscvfem_micro_face_mask(
                                                                    (int)d.macro_face_mask[(size_t)e],
                                                                    L, xi, yi, zi),
                                                  sscvfem_micro_face_mask(
                                                          d.macro_natural_mask.empty() ? 0
                                                              : (int)d.macro_natural_mask[(size_t)e],
                                                          L, xi, yi, zi),
                                                  sscvfem_bd(d, e, L, xi, yi, zi));
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < N_FIELDS; ++c) lout[(size_t)l * N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            if (sc)
                sscvfem_scatter_element(*sc, nxe, e, lg.data(), lout.data(), res);
            else
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = lg[(size_t)a];
                    for (int c = 0; c < N_FIELDS; ++c)
                        atomic_add(res + (ptrdiff_t)g * N_FIELDS + c, 0, lout[(size_t)a * N_FIELDS + c]);
                }
        }
    }

    if (sc) sscvfem_reduce_shared(*sc, res);
    sscvfem_apply_body_force(d, res);
    sscvfem_apply_transient(d, rho, res);
}

// ---------------------------------------------------------------------------
// Block diagonal: the 4x4 block per node, which is what a block-Jacobi smoother inverts.
//
// The multigrid path needs one of these at every level, so it is not the once-per-Newton
// cost it looked like when only a single-level solve existed.
//
// Unlike the flat path, this can use the slot mask. cvfem_hex8_ns_upwind_jacobian_add_slots
// writes exclusively through cvfem_hex8_bsr_acc, which drops a negative slot, so passing
// -1 everywhere off the diagonal makes the full element kernel produce the block diagonal
// with none of the off-diagonal write traffic. The flat assemble_block_diag cannot do that:
// its affine path runs the SymPy kernel, whose 768 writes go straight to values[...] with
// no guard, so a negative slot there is an out-of-bounds write and it has to assemble the
// whole element into a 64-block scratch and throw away seven eighths of it.
//
// Both variants below use the same kernel as the apply, so the diagonal and the operator
// cannot drift apart.

// Control: the flat gather, one masked element assembly per micro-element, atomics to a
// node-indexed destination.
inline SFEM_NOINLINE void sscvfem_block_diag_naive(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                                   std::vector<scalar_t> &diag) {
    SFEM_TRACE_SCOPE("sscvfem::block_diag_naive");
    diag.assign((size_t)d.nnodes * 16, scalar_t(0));
    scalar_t *const SFEM_RESTRICT out = diag.data();

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
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                    for (int a = 0; a < 8; ++a) {
                        g[a]   = d.elems[base + off[a]][e];
                        x[a]   = (scalar_t)d.points[0][g[a]];
                        y[a]   = (scalar_t)d.points[1][g[a]];
                        z[a]   = (scalar_t)d.points[2][g[a]];
                        ux[a]  = d.ux[(size_t)g[a]];
                        uy[a]  = d.uy[(size_t)g[a]];
                        uz[a]  = d.uz[(size_t)g[a]];
                        p[a]   = d.p[(size_t)g[a]];
                        pgx[a] = d.pgx[(size_t)g[a]];
                        pgy[a] = d.pgy[(size_t)g[a]];
                        pgz[a] = d.pgz[(size_t)g[a]];
                    }

                    // Diagonal slots address the destination by node; everything else is
                    // dropped by the guard in cvfem_hex8_bsr_acc.
                    // count_t, not ptrdiff_t: boundary_scs_add_jacobian takes count_t
                    // slots. It is signed, so -1 still means "drop this block".
                    smesh::count_t sl[64];
                    for (int a = 0; a < 8; ++a) {
                        for (int b = 0; b < 8; ++b) sl[a * 8 + b] = -1;
                        sl[a * 8 + a] = (smesh::count_t)g[a];
                    }

                    const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                    const Hex8RhieChow rc{x,       y,  z,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(x, y, z, adj, &det);
                    cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj, det, ux, uy, uz, sl, out, rc, p);
                    boundary_scs_add_jacobian<true>(rho, mu, 0, adj, det, d.Lx, d.Ly, d.Lz, x, y, z, ux, uy, uz, sl, out);
                }
            }
        }
    }
}

// The default: gather the macro-element once, accumulate into a macro-local destination
// addressed by local node so the element assembly needs no atomics at all, and scatter
// once per macro node at the end. Geometry is lifted out of the loop as in the apply.
inline SFEM_NOINLINE void sscvfem_block_diag(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                             std::vector<scalar_t> &diag) {
    SFEM_TRACE_SCOPE("sscvfem::block_diag");
    diag.assign((size_t)d.nnodes * 16, scalar_t(0));
    scalar_t *const SFEM_RESTRICT out = diag.data();

    const int L   = d.level;
    const int nxe = d.nxe;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSScatter *const sc = d.scatter ? d.scatter.get() : nullptr;

#pragma omp parallel
    {
        std::vector<smesh::idx_t> lg((size_t)nxe);
        std::vector<scalar_t>     lx((size_t)nxe), ly((size_t)nxe), lz((size_t)nxe);
        std::vector<scalar_t>     lux((size_t)nxe), luy((size_t)nxe), luz((size_t)nxe), lp((size_t)nxe);
        std::vector<scalar_t>     lpgx((size_t)nxe), lpgy((size_t)nxe), lpgz((size_t)nxe);
        std::vector<scalar_t>     lout((size_t)nxe * 16);

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
                lpgx[(size_t)a]      = d.pgx[(size_t)g];
                lpgy[(size_t)a]      = d.pgy[(size_t)g];
                lpgz[(size_t)a]      = d.pgz[(size_t)g];
            }
            std::fill(lout.begin(), lout.end(), scalar_t(0));

            // Micro-cell 0's corners, hoisted: the geometry AND the coordinates the
            // Rhie-Chow term differences.
            //
            // The lattice inside a macro element is uniform, so every micro-cell is congruent
            // to cell 0 and one adjugate serves all of them -- that is what the action does.
            // This used to hoist the adjugate but then hand the Rhie-Chow struct each cell's
            // OWN coordinates, and the two agree only to the precision the node positions are
            // stored in. smesh::geom_t is float32, so the block diagonal disagreed with the
            // action it is supposed to be the diagonal of by 4.23e-08 -- eight orders above
            // round-off, and invisible until the q-independent consistency gate looked.
            //
            // Only DIFFERENCES of these are taken (d = x_j - x_i), so cell 0's coordinates are
            // exact for the purpose, not an approximation. The boundary closure below still
            // gets each cell's real position, because it tests where the cell actually is.
            scalar_t madj[9], mdet;
            scalar_t c0x[8], c0y[8], c0z[8];
            {
                for (int a = 0; a < 8; ++a) {
                    const int l = off[a];
                    c0x[a]      = lx[(size_t)l];
                    c0y[a]      = ly[(size_t)l];
                    c0z[a]      = lz[(size_t)l];
                }
                sscvfem_micro_geom(c0x, c0y, c0z, madj, &mdet);
            }

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                        smesh::count_t sl[64];
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
                            for (int b = 0; b < 8; ++b) sl[a * 8 + b] = -1;
                        }
                        // Local node index: the destination is this macro-element's own
                        // buffer, so no thread can be writing the same entry.
                        for (int a = 0; a < 8; ++a) sl[a * 8 + a] = (smesh::count_t)(base + off[a]);

                        const Hex8RcConfig rcfg = sscvfem_rc_config(d);
                        const Hex8RhieChow rc{c0x,     c0y, c0z, pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux,  uy,  uz,  rcfg.tau};
                        cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, madj, mdet, ux, uy, uz, sl,
                                                                       lout.data(), rc, p);
                        boundary_scs_add_jacobian<false>(rho, mu, 0, madj, mdet, d.Lx, d.Ly, d.Lz, x, y, z,
                                                         ux, uy, uz, sl, lout.data(),
                                                         d.macro_face_mask.empty()
                                                          ? -1
                                                          : sscvfem_micro_face_mask(
                                                                    (int)d.macro_face_mask[(size_t)e],
                                                                    L, xi, yi, zi),
                                                  sscvfem_micro_face_mask(
                                                          d.macro_natural_mask.empty() ? 0
                                                              : (int)d.macro_natural_mask[(size_t)e],
                                                          L, xi, yi, zi),
                                                  sscvfem_bd(d, e, L, xi, yi, zi));
                    }
                }
            }

            if (sc)
                sscvfem_scatter_element_w<16>(*sc, nxe, e, lg.data(), lout.data(), out,
                                              const_cast<scalar_t *>(sc->stage16.data()));
            else
                for (int a = 0; a < nxe; ++a) {
                    const smesh::idx_t g = lg[(size_t)a];
                    for (int k = 0; k < 16; ++k)
                        atomic_add(out + (ptrdiff_t)g * 16 + k, 0, lout[(size_t)a * 16 + k]);
                }
        }
    }

    if (sc) sscvfem_reduce_shared_w<16>(*sc, out, sc->stage16.data());

    // The transient term's diagonal: rho V a0 / dt on each velocity component, nothing on
    // pressure. Added here rather than in the macro-element sweeps for the same reason
    // sscvfem_apply_transient is a post-pass, so the two stay consistent by construction.
    {
        const scalar_t a = sscvfem_transient_diag_weight(d, rho);
        if (a != scalar_t(0)) {
            if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const scalar_t w = a * d.node_vol[(size_t)i];
                for (int c = 0; c < 3; ++c) diag[(size_t)i * 16 + (size_t)c * 4 + (size_t)c] += w;
            }
        }
    }
}

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
    SFEM_TRACE_SCOPE("sscvfem::apply");
    // Rhie-Chow differentiates through the nodal pressure-gradient reconstruction, so the
    // direction's own reconstructed gradient is needed for the Jacobian action to be exact.
    // One extra pass per apply, the same shape as the one already done for p.
    // SFEM_RC_EXACT_JAC=0 restores the frozen-pg Jacobian, for A/B against this fix. The
    // preconditioner is still built from the assembled Jacobian, which keeps the frozen form,
    // so making the action exact also makes the two disagree -- that is what the A/B measures.
    if (sscvfem_rc_exact_jac() && d.rhie_chow_scale != scalar_t(0) && !d.pgx.empty()) {
        sscvfem_nodal_q_grad(d, dir);
    } else {
        d.qgx.clear();
        d.qgy.clear();
        d.qgz.clear();
    }
    sscvfem_apply_macro_local_hoisted(d, rho, mu, dir, jv);
    // The transient term's contribution to the Jacobian action, rho V a0 / dt on each
    // velocity component. The flat path does this in
    // apply_jacobian_action_accumulate and this line was simply missing, so the
    // semi-structured residual carried the BDF term while the Jacobian the Krylov solver
    // applied did not.
    //
    // It costs nothing at dt <= 0, which is why every steady result was unaffected and the
    // omission survived: the ss pump solves the steady problem to 4.7e-15 and stalls its
    // transient at 1.6 of a Re=20 target. For a small timestep rho V a0 / dt is the
    // DOMINANT diagonal, so leaving it out does not perturb the Newton direction, it
    // replaces it.
    sscvfem_apply_transient_action(d, rho, dir, jv);
}
