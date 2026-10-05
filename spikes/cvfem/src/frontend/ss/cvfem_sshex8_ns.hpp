#pragma once

// HEX8 CVFEM Navier-Stokes on a semi-structured (sshex8) mesh: THE STAGING SIDE.
//
// SSMeshData, SSScatter, the tables they cache, the environment the options come from, and one
// launcher per sweep. The sweeps themselves are in kernels/semistructured/cvfem_sshex8_ns.hpp
// and take only what they read; this file is what turns a mesh and a set of options into those
// arguments. The split follows DESIGN.md: "No user level option flags are propgated down here
// (like now), they are handled outside in the front-end."
//

#include "frontend/op/cvfem_hex8_ns_core.hpp"
#include "packed_elements.hpp"   // packed_elements_matmul_nonsym: BLAS gemm, loop fallback
#include "smesh_exchange.hpp"    // the nodal reconstruction is completed across ranks
#include "smesh_mesh.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <type_traits>
#include "kernels/semistructured/cvfem_sshex8_ns.hpp"
#include "kernels/semistructured/affine/cvfem_sshex8_ns_affine.hpp"
#include "kernels/semistructured/isoparametric/cvfem_sshex8_ns_isoparam.hpp"

// ---------------------------------------------------------------------------

struct SSMeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    int                          level{0};
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nmacro{0};
    int                          nxe{0};  // (L+1)^3, nodes per macro-element
    idx_t               **elems{nullptr};
    geom_t              **points{nullptr};
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
    // The cell-Peclet blend, resolved once per residual and carried as data for the reason
    // the flat MeshData carries it: the kernels that read it are SFEM_HOST_DEVICE.
    // Limiter freezing: the deferred correction evaluated once per continuation stage and held
    // fixed through that stage's Newton iterations, rather than re-limited at every residual.
    //
    // It is the remedy named in the literature for exactly this failure, and it is ORTHOGONAL
    // to Venkatakrishnan's eps^2, which is why both exist. eps^2 buys convergence by relaxing
    // the bound -- measured, and it relaxes it everywhere, not only where the field is smooth.
    // Freezing buys convergence by removing the fixed point the lagged source has to reach,
    // and costs nothing in boundedness at all: the correction it holds is a correction the
    // unfrozen arm would also have produced.
    // Non-null turns on the limiter's boundedness counting (src/venkata). Diagnostic only:
    // the deferred correction is unreachable without a nodal gradient, so the default path
    // never evaluates the null test this adds.
    Hex8LimiterStats     *limiter_stats{nullptr};
    int                        conv_freeze{0};
    std::vector<scalar_t>      conv_frozen;  // empty until the stage's correction is built

    // Venkatakrishnan's eps^2 coefficient, already carrying the units: eps^2 = this times the
    // local edge length cubed. Zero unless SFEM_VENKAT_K is set, and zero is bit-for-bit the
    // limiter as it behaved before the deactivation term existed.
    scalar_t                   conv_venkat_c{0};
    Hex8PecletConfig<scalar_t> conv_peclet{};
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
    // WORK BUFFERS, HELD RATHER THAN DECLARED AT EACH CALL.
    //
    // Three front-end paths used to open with std::vector locals: the nodal velocity gradient's
    // three component arrays, and the two residuals the frozen deferred correction differences.
    // Both run inside the Newton loop, so those were allocations per step rather than per mesh.
    // They sit here with ugrad and conv_frozen, which are the same kind of thing.
    std::vector<scalar_t> ugrad_work[3];
    std::vector<scalar_t> conv_work[2];
    // The reference block apply's own pair. Not conv_work: nothing calls it from inside the
    // frozen-correction branch today, and a shared buffer that is only safe because of that
    // is a trap for whatever calls it next.
    std::vector<scalar_t> blocks_ref_work[2];
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

    // 1 where a macro element is curved (its trilinear map has cross terms), and every
    // geometric quantity of its micro cells must then come from the cell's own corners
    // rather than be hoisted; empty when no macro element is. See sscvfem_classify_macros.
    //
    // Last, and deliberately: inserted before macro_face_mask it moved every later field
    // by 24 bytes, and the block diagonal -- which reads those fields per micro cell and
    // whose own source was unchanged -- compiled differently and ran 7-9% slower.
    std::vector<uint8_t> macro_curved;

    // Completes the nodal gradient reconstruction across ranks; null on a serial mesh.
    //
    // Built once and cached rather than per apply, because the Jacobian action reconstructs
    // on every Krylov iteration and Exchange::create_nodal is collective. Keyed the same way
    // grad_w_inv is, on the node count it was built for.
    std::shared_ptr<smesh::Exchange> grad_exchange;
    ptrdiff_t                        grad_exchange_nnodes{-1};

    // THE CURVATURE PARTITION: macro-element indices with the straight ones first and the curved
    // ones after, and n_straight the boundary between them. Empty means the identity order, so a
    // sweep indexes e == i directly and a mesh with no curved macro element pays nothing.
    //
    // This is what lets the geometry split be two sweeps over two ranges instead of one sweep
    // branching per macro element. Whether a macro element is curved is mesh data, not a
    // configuration -- one mesh has both kinds side by side -- so it cannot become a template
    // parameter the way the flat layouts' geometry did. But it does not change between applies
    // either, so the partition is setup work, like the pack ordering and the element colouring
    // already are. Rebuilt with macro_curved, once per level.
    //
    // Both parts stay in ascending index order, which is why a mesh with nothing curved gets the
    // identity and every fingerprint is unchanged.
    //
    // At the very END of the struct, and deliberately: see the note on macro_curved, where
    // inserting a field earlier moved the fields the block diagonal reads per micro cell and
    // cost 7-9% with its own source untouched.
    std::vector<ptrdiff_t> macro_order;
    ptrdiff_t              n_straight{0};
};
// Defined below, next to the macro-element geometry it configures; the element sweeps that
// need it sit above.
inline scalar_t sscvfem_transient_diag_weight(const SSMeshData &d, const scalar_t rho);
inline __attribute__((always_inline)) Hex8RcConfig sscvfem_rc_config(const SSMeshData &d);
inline void         sscvfem_classify_macros(SSMeshData &d);

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
    std::vector<idx_t> shared_node;  // one global node per reduction row
    std::vector<ptrdiff_t>    red_ptr;      // CRS over reduction rows
    std::vector<ptrdiff_t>    red_idx;      // staging slots feeding each row
    ptrdiff_t                 n_slots{0};
    std::vector<scalar_t>     stage;        // n_slots * CVFEM_HEX8_N_FIELDS, for the 4-wide kernels
    std::vector<scalar_t>     stage16;      // n_slots * 16, for the block diagonal
    // BUILD-TIME SCRATCH, HELD RATHER THAN DECLARED LOCALLY.
    //
    // sscvfem_build_scatter's three working arrays: the per-node incidence count, the node to
    // reduction-row map, and the running fill position per row. They are only meaningful while
    // the tables above are being built, and they are here because no std::vector local belongs
    // in this file -- not because anything reads them afterwards. `fill` is sized by rows, the
    // other two by nodes.
    std::vector<int>       build_touches;
    std::vector<ptrdiff_t> build_row_of;
    std::vector<ptrdiff_t> build_fill;
};

inline void sscvfem_build_scatter(const SSMeshData &d, SSScatter &s) {
    SFEM_TRACE_SCOPE("sscvfem::build_scatter");
    const int       nxe = d.nxe;
    const ptrdiff_t ne  = d.nmacro;

    std::vector<int> &touches = s.build_touches;
    touches.assign((size_t)d.nnodes, 0);
    for (ptrdiff_t e = 0; e < ne; ++e)
        for (int a = 0; a < nxe; ++a) touches[(size_t)d.elems[a][e]]++;

    // Shared nodes get a reduction row; exclusive ones are written directly.
    std::vector<ptrdiff_t> &row_of = s.build_row_of;
    row_of.assign((size_t)d.nnodes, -1);
    s.shared_node.clear();
    for (ptrdiff_t g = 0; g < d.nnodes; ++g)
        if (touches[(size_t)g] > 1) {
            row_of[(size_t)g] = (ptrdiff_t)s.shared_node.size();
            s.shared_node.push_back((idx_t)g);
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
    std::vector<ptrdiff_t> &fill = s.build_fill;
    fill.assign((size_t)std::max<ptrdiff_t>(nrows, 0), 0);
    for (ptrdiff_t r = 0; r < nrows; ++r) fill[(size_t)r] = s.red_ptr[(size_t)r];
    for (ptrdiff_t e = 0; e < ne; ++e)
        for (int a = 0; a < nxe; ++a) {
            const ptrdiff_t r = row_of[(size_t)d.elems[a][e]];
            if (r < 0) continue;
            const ptrdiff_t k         = fill[(size_t)r]++;
            s.slot[(size_t)e * nxe + a] = (int)k;
            s.red_idx[(size_t)k]        = k;  // slot index is its own position
        }

    s.stage.assign((size_t)s.n_slots * CVFEM_HEX8_N_FIELDS, scalar_t(0));
    s.stage16.assign((size_t)s.n_slots * 16, scalar_t(0));
    s.ready = true;
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

    sscvfem_classify_macros(d);
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

// Flag the curved macro elements. A trilinear map x(xi) = c0 + c_x xi + c_y eta + c_z zeta +
// c_xy xi eta + c_yz eta zeta + c_xz xi zeta + c_xyz xi eta zeta is affine exactly when the four
// cross coefficients vanish; the test is relative to the linear ones, at 1e-5, which clears the
// float32 storage of the coordinates by two orders.
//
// The corners are not the whole story once a mesh places its micro nodes on the geometry
// instead of on the chords between the corners -- smesh::Mesh::warp_semistructured_hex8_nozzle
// does exactly that. A macro element whose corners happen to be affine can then carry a curved
// lattice, and hoisting would silently put its micro cells back on the chords. So a macro element
// is also curved where any lattice node is off the trilinear map of its corners, at the same
// relative tolerance.
// THE TWO RANGES A SPLIT SWEEP PAIR IS LAUNCHED OVER, and how a partial element range is
// restricted to each half of the curvature partition.
//
// A sweep pair covers macro elements [begin, end) -- usually the whole mesh, but the nodal
// gradient hands the scatter sweep the tail the packs do not reach on a distributed mesh. The
// sweeps iterate POSITIONS in macro_order, so what each needs is the positions whose element
// falls in that range.
//
// That set is a contiguous run of positions, and only because each half of the partition is in
// ascending element order -- which is what sscvfem_classify_macros guarantees and
// cvfem_ss_curvature_partition_test asserts. So a binary search finds it, and no sweep has to
// filter per element.
//
// With no order array -- a mesh with nothing curved -- the straight half is the element range
// itself and the curved half is empty, so the box pays nothing for any of this.
// The partition itself, or null where there is none.
inline const ptrdiff_t *sscvfem_order(const SSMeshData &d) {
    return d.macro_order.empty() ? nullptr : d.macro_order.data();
}

inline cvfem_range sscvfem_order_positions(const SSMeshData &d, const bool curved,
                                           const ptrdiff_t begin, const ptrdiff_t end) {
    // The search itself is sscvfem_order_run, in the kernel layer, because the packed
    // nodal-gradient sweep needs it per pack and cannot reach into SSMeshData.
    return sscvfem_order_run(sscvfem_order(d), curved ? d.n_straight : 0,
                             curved ? d.nmacro : d.n_straight, begin, end);
}

// This thread's slice of one half. Called inside the parallel region, like every other
// cvfem_range_split in this file.
inline cvfem_range sscvfem_affine_range(const SSMeshData &d, const ptrdiff_t begin, const ptrdiff_t end) {
    const cvfem_range p = sscvfem_order_positions(d, false, begin, end);
    return cvfem_range_split(p.begin, p.end, 1, cvfem_thread_index(), cvfem_n_threads());
}

inline cvfem_range sscvfem_isoparam_range(const SSMeshData &d, const ptrdiff_t begin, const ptrdiff_t end) {
    const cvfem_range p = sscvfem_order_positions(d, true, begin, end);
    return cvfem_range_split(p.begin, p.end, 1, cvfem_thread_index(), cvfem_n_threads());
}

inline void sscvfem_classify_macros(SSMeshData &d) {
    int ext[8];
    sscvfem_macro_corner_offsets(d.level, ext);
    d.macro_curved.assign((size_t)d.nmacro, 0);
    ptrdiff_t n_curved = 0;
    for (ptrdiff_t e = 0; e < d.nmacro; ++e) {
        double c[8][3];
        for (int a = 0; a < 8; ++a) {
            const idx_t gn = d.elems[ext[a]][e];
            for (int k = 0; k < 3; ++k) c[a][k] = (double)d.points[k][gn];
        }
        double lin = 0, cross = 0;
        for (int k = 0; k < 3; ++k) {
            const double lx  = c[1][k] - c[0][k], ly = c[3][k] - c[0][k], lz = c[4][k] - c[0][k];
            const double cxy = c[0][k] - c[1][k] + c[2][k] - c[3][k];
            const double cyz = c[0][k] - c[3][k] + c[7][k] - c[4][k];
            const double cxz = c[0][k] - c[1][k] + c[5][k] - c[4][k];
            const double cxyz = -c[0][k] + c[1][k] - c[2][k] + c[3][k] + c[4][k] - c[5][k] + c[6][k] - c[7][k];
            lin += lx * lx + ly * ly + lz * lz;
            cross += cxy * cxy + cyz * cyz + cxz * cxz + cxyz * cxyz;
        }
        bool curved = cross > 1e-10 * lin;
        if (!curved) {
            const int L = d.level;
            double    dev = 0;
            for (int zi = 0; zi <= L && dev <= 1e-10 * lin; ++zi)
                for (int yi = 0; yi <= L; ++yi)
                    for (int xi = 0; xi <= L; ++xi) {
                        const double          r[3] = {(double)xi / L, (double)yi / L, (double)zi / L};
                        const idx_t gn   = d.elems[sscvfem_lidx(L, xi, yi, zi)][e];
                        for (int k = 0; k < 3; ++k) {
                            double t = 0;
                            for (int a = 0; a < 8; ++a) {
                                double N = 1;
                                for (int q = 0; q < 3; ++q)
                                    N *= CVFEM_HEX8_REF_XI[a][q] > 0.5 ? r[q] : 1.0 - r[q];
                                t += N * c[a][k];
                            }
                            const double dd = (double)d.points[k][gn] - t;
                            dev             = std::max(dev, dd * dd);
                        }
                    }
            curved = dev > 1e-10 * lin;
        }
        if (curved) {
            d.macro_curved[(size_t)e] = 1;
            ++n_curved;
        }
    }
    if (n_curved == 0) {
        d.macro_curved.clear();
        d.macro_order.clear();
        d.n_straight = d.nmacro;
    } else {
        d.macro_order.resize((size_t)d.nmacro);
        // The two fill positions: straight from the front, curved from where the straight ones end.
        ptrdiff_t ns = 0, nc = d.nmacro - n_curved;
        for (ptrdiff_t e = 0; e < d.nmacro; ++e)
            if (d.macro_curved[(size_t)e]) d.macro_order[(size_t)nc++] = e;
            else d.macro_order[(size_t)ns++] = e;
        d.n_straight = ns;
        std::printf("sscvfem: %td of %td macro elements are curved -- their micro cells get their own geometry\n",
                    n_curved, d.nmacro);
    }
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
            const idx_t g = d.elems[off[a]][e];
            ex[a]                = (scalar_t)d.points[0][g];
            ey[a]                = (scalar_t)d.points[1][g];
            ez[a]                = (scalar_t)d.points[2][g];
        }
        sscvfem_micro_geom(ex, ey, ez, adj, &det);
        const bool     curved_e = sscvfem_macro_curved(d.macro_curved.empty() ? nullptr : d.macro_curved.data(), e);
        const scalar_t vol      = std::fabs(det);
        if (!curved_e && vol < scalar_t(1e-30)) continue;
        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);
                    scalar_t  v    = vol;
                    if (curved_e) {
                        scalar_t cx[8], cy[8], cz[8], cadj[9], cdet;
                        sscvfem_cell_corners(d.elems, d.points, e, base, off, cx, cy, cz);
                        sscvfem_micro_geom(cx, cy, cz, cadj, &cdet);
                        v = std::fabs(cdet);
                        if (v < scalar_t(1e-30)) continue;
                    }
                    for (int a = 0; a < 8; ++a) w[d.elems[base + off[a]][e]] += v;
                }
    }
    for (ptrdiff_t i = 0; i < d.nnodes; ++i)
        w[i] = w[i] > scalar_t(0) ? scalar_t(1) / w[i] : scalar_t(0);
    d.grad_w_nmacro = d.nmacro;
    d.grad_w_level  = d.level;
}

// The packed nodal gradient, end to end. The front-end side: the cached reconstruction weight,
// the allocation, and the decision about whether the outputs need pre-zeroing at all.
inline void sscvfem_nodal_grad_packed(SSMeshData &d, PackedData &p,
                                      const scalar_t *const SFEM_RESTRICT src, const int stride,
                                      std::vector<scalar_t> &ogx, std::vector<scalar_t> &ogy,
                                      std::vector<scalar_t> &ogz, const bool apply_weight = true) {
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

    #pragma omp parallel
        sscvfem_nodal_grad_packed_sweep(
                cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),d.grad_w_inv.data(), d.level, sscvfem_order(d), d.n_straight, d.nmacro, d.points, p.elems, const_cast<scalar_t *>(p.ghost_buf.data()), p.ghost_idx, p.ghost_ptr, p.ghost_reduce_dest, p.ghost_reduce_idx, p.ghost_reduce_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.n_ghost_reduce_rows, p.n_packed_elements, p.n_packs, p.owned_nodes_ptr, src, stride, ogx.data(), ogy.data(), ogz.data(),
                                    apply_weight);
    // The ghost reduction, the pack layout's second and independent loop: over the ghost
    // reduce rows rather than the packs, so it gets its own range and runs after the pack
    // pass's threads have joined. Width three with the reconstruction's weight folded in,
    // which is what the scaled instantiation is for -- the ghost rows have to be
    // normalised the same way the drain normalises the owned ones.
    if (p.n_ghost_reduce_rows > 0) {
        scalar_t *const g3[3] = {ogx.data(), ogy.data(), ogz.data()};
    #pragma omp parallel
        cvfem_hex8_ghost_reduce_soa_range<3, /*SCALED=*/true>(
                cvfem_range_split(0, p.n_ghost_reduce_rows, 1, cvfem_thread_index(),
                                  cvfem_n_threads()),
                p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
                p.n_ghost_entries, p.ghost_buf.data(),
                apply_weight ? d.grad_w_inv.data() : nullptr, g3);
    }
}

// Whether the packing spans the whole mesh. On a distributed mesh it does not.
//
// smesh restricts packing to the owned-not-shared prefix there, and has to: a node's
// position inside its owner's owned block is a number the neighbouring ranks have already
// recorded in their ghost indices, so permuting anything else silently redirects their
// gathers. The consequence for this file is that the packs cover only a prefix of the macro
// elements.
//
// That is not merely a missing contribution. sscvfem_nodal_grad_packed bounds its element
// loop with MIN(d.nmacro, (pack + 1) * n_elements_per_pack) -- by d.nmacro, not by the
// packed element count -- while p.elems is sized to the packed range. Once the two differ,
// the last pack reads that array past its end.
//
// The condition is the one the format already answers: sscvfem_nodal_grad_packed computes
// the same owned_nodes_ptr tiling test as `owns_all` to decide whether it may skip
// pre-zeroing. Asking it here rather than inventing a second notion of coverage keeps one
// definition of what a complete packing is.
inline bool sscvfem_pack_covers_all_elements(const SSMeshData &d, const PackedData &p) {
    return p.n_packs > 0 && p.owned_nodes_ptr[0] == 0 &&
           p.owned_nodes_ptr[p.n_packs] == d.nnodes;
}

// The denominator, folded into one pass over the nodes instead of accumulated in the
// sweep and divided out in another. The flat twin folds it into the pack drain and has no
// pass at all; that needs the packed staging this path does not have, so one pass is the
// floor here.
//
// Its own function now because two different sweeps can feed it: whichever combination of
// passes produced the raw sums, the division happens once, here, at the end.
inline void sscvfem_nodal_grad_normalize(SSMeshData &d, std::vector<scalar_t> &ogx,
                                         std::vector<scalar_t> &ogy, std::vector<scalar_t> &ogz) {
    const scalar_t *const SFEM_RESTRICT winv = d.grad_w_inv.data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t s = winv[(size_t)i];
        ogx[(size_t)i] *= s;
        ogy[(size_t)i] *= s;
        ogz[(size_t)i] *= s;
    }

    // Complete the reconstruction on the nodes this rank does not own.
    //
    // The sweep that consumes this field reads it at EVERY local node, including ghosts and
    // aura, because an element touching an owned node also touches nodes owned elsewhere.
    // The accumulation above runs over local elements only, so those entries hold a partial
    // numerator over a partial denominator -- the errors do not cancel, and the sweep then
    // carries them into owned rows. That makes A*x depend on how the domain was cut, which
    // is why the Krylov iteration count stopped being decomposition invariant: measured on
    // the cavity with no multigrid and block-Jacobi, 812 linear iterations at one rank
    // became 4765 at four and 52186 at eight, while the same case with the reconstruction
    // disabled (SFEM_RC_EXACT_JAC=0) stayed flat at 6056 / 6067 / 6067 / 6051.
    //
    // A gather is sufficient and a scatter_add is NOT needed, which is a statement about
    // the aura rather than about this loop: every element touching an owned node is local,
    // so the OWNER's entry is already complete and only the copies need filling. That was
    // measured rather than assumed -- summed over the owned range and keyed by global id,
    // the first reconstruction agrees to 3.1e-16 between one, two and four ranks, which is
    // the reassociation the packed and two-pass paths differ by anyway.
    //
    // GhostsAndAura, not GhostsOnly: the sweep reaches aura nodes, so a ghosts-only gather
    // would leave exactly the slots this exists to fill. Serial returns above without
    // touching this, so the byte-compared verification matrix is unaffected.
    if (d.mesh && d.mesh->is_distributed() && d.mesh->comm() && d.mesh->comm()->size() > 1) {
        if (!d.grad_exchange || d.grad_exchange_nnodes != d.nnodes) {
            d.grad_exchange = smesh::Exchange::create_nodal(
                    d.mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
            d.grad_exchange_nnodes = d.nnodes;
        }
        if (d.grad_exchange) {
            d.grad_exchange->gather(ogx.data(), 1);
            d.grad_exchange->gather(ogy.data(), 1);
            d.grad_exchange->gather(ogz.data(), 1);
        }
    }

    // SFEM_GRAD_OWNED_SUM=1: is this reconstruction complete on the nodes this rank OWNS?
    //
    // The reconstruction accumulates over local elements and divides by a weight built the
    // same way, and nothing exchanges the result, so ghost and aura nodes hold a partial
    // numerator over a partial denominator. That much is certain from the code. What is NOT
    // written down anywhere is whether the OWNER's entry is complete -- it is complete only
    // if every element touching an owned node is local, which is what the aura is supposed
    // to guarantee but which no comment in smesh or ARCHITECTURE.md actually states.
    //
    // The answer decides the fix and the two possibilities are not interchangeable:
    //
    //   owned sums match serial  -> the owner is authoritative, and a plain gather of the
    //                               finished field into ghost and aura slots is correct
    //   owned sums differ        -> the owner's own accumulation is short, and the fix must
    //                               scatter_add the raw sums back to their owners BEFORE
    //                               normalising, then gather
    //
    // Keyed by global id and summed over the owned range only, because local indices are
    // incomparable across partitions and the ghost entries each rank also stores belong to
    // somebody else. Gated, so an unset variable leaves the fast path exactly as it was.
    if (smesh::Env::read<int>("SFEM_GRAD_OWNED_SUM", 0) && d.mesh) {
        const bool dist = d.mesh->is_distributed() && d.mesh->comm() && d.mesh->comm()->size() > 1;
        const ptrdiff_t n_owned =
                dist ? d.mesh->distributed()->n_nodes_owned() : d.nnodes;

        long double plain = 0, gidw = 0;
        for (ptrdiff_t i = 0; i < n_owned; ++i) {
            const long double g =
                    dist ? (long double)d.mesh->distributed()->node_mapping()->data()[i] : (long double)i;
            const long double v = (long double)ogx[(size_t)i] + (long double)ogy[(size_t)i] +
                                  (long double)ogz[(size_t)i];
            plain += v;
            gidw += v * g;
        }

        double gp = (double)plain, gw = (double)gidw, gn = (double)n_owned;
        int    rank = 0;
        if (dist) {
            gp   = d.mesh->comm()->sum(gp);
            gw   = d.mesh->comm()->sum(gw);
            gn   = d.mesh->comm()->sum(gn);
            rank = d.mesh->comm()->rank();
        }
        if (rank == 0)
            std::printf("gradsum: owned_nodes %.0f  sum %.17g  gidsum %.17g\n", gn, gp, gw);
    }
}

inline void sscvfem_nodal_grad_strided(SSMeshData &d, const scalar_t *const SFEM_RESTRICT src,
                                       const int stride, std::vector<scalar_t> &ogx,
                                       std::vector<scalar_t> &ogy, std::vector<scalar_t> &ogz) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_grad_strided");
    // Over packed macro-elements where there is a packing, which leaves an eighth of the
    // nodes staged instead of half. Same operator either way; the summation order differs, so
    // the two agree to round-off rather than bit for bit.
    //
    // Guarded on the packing actually covering the mesh. A serial mesh always satisfies that,
    // so this path is entered exactly as before and the fast case is unchanged bit for bit.
    // A distributed mesh does not, and takes the two-pass path below.
    if (d.packed && sscvfem_pack_covers_all_elements(d, *d.packed)) {
        sscvfem_nodal_grad_packed(d, *d.packed, src, stride, ogx, ogy, ogz);
        return;
    }

    // Two passes where the packing covers only part of the mesh, rather than abandoning the
    // packed path for all of it.
    //
    // The packs span the owned-not-shared element prefix on a distributed mesh, so falling
    // back to the scatter sweep for everything threw away the packed reconstruction on the
    // large majority of elements to handle the minority the packs do not reach. Instead:
    // the packed sweep takes [0, n_packed), the scatter sweep takes [n_packed, nmacro), and
    // one normalisation finishes both.
    //
    // The two are composable because the ranges are DISJOINT and every write is additive:
    // sums over disjoint element sets add, and the reduce accumulates into dst rather than
    // assigning. Both passes therefore leave raw, unnormalised sums, and the single pass at
    // the end divides once -- which is also why the packed pass is asked not to fold the
    // weight into its drain here.
    //
    // Serial never reaches this: it satisfies covers_all and returns above, on the untouched
    // fast path. So the byte-compared verification matrix is unaffected by any of it.
    if (d.packed) {
        PackedData     &p        = *d.packed;
        const ptrdiff_t n_packed = p.n_packed_elements > 0 ? p.n_packed_elements : 0;
        // Zeroes the outputs itself, since it cannot own all of them here.
        sscvfem_nodal_grad_packed(d, p, src, stride, ogx, ogy, ogz, /*apply_weight=*/false);
        {
            // A PARTIAL range needs the staging buffer pre-zeroed: a slot a pack already
            // wrote is read by the reduction, and a slot this range never writes would
            // otherwise carry whatever the buffer held. The full range writes every slot
            // before reading it, so it skips the fill -- which is why this is the caller's
            // and not the sweep's: it is shared work that must happen once, not per thread.
            const ptrdiff_t _b = n_packed, _e = d.nmacro;
            if (d.scatter && d.scatter->ready && _b > 0 && d.scatter->n_slots > 0) {
                const SSScatter &sc = *d.scatter;
                scalar_t *const st = const_cast<scalar_t *>(sc.stage.data());
                std::fill(st, st + (size_t)sc.n_slots * 3, scalar_t(0));
            }
        #pragma omp parallel
        {
                sscvfem_nodal_grad_scatter_affine(d.elems, d.level, d.nxe, d.points, d.scatter ? d.scatter->n_slots : 0, d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, src, stride, ogx.data(), ogy.data(), ogz.data(),
                        sscvfem_affine_range(d, _b, _e), sscvfem_order(d));
                sscvfem_nodal_grad_scatter_isoparam(d.elems, d.level, d.nxe, d.points, d.scatter ? d.scatter->n_slots : 0, d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, src, stride, ogx.data(), ogy.data(), ogz.data(),
                        sscvfem_isoparam_range(d, _b, _e), sscvfem_order(d));
        }

            if (d.scatter && d.scatter->ready) {
                const SSScatter &sc = *d.scatter;
                scalar_t *dst[3] = {ogx.data(), ogy.data(), ogz.data()};
                const ptrdiff_t nrows = (ptrdiff_t)sc.shared_node.size();
        #pragma omp parallel
                sscvfem_reduce_shared_soa_w<3>(
                        cvfem_range_split(0, nrows, 1, cvfem_thread_index(), cvfem_n_threads()),
                        sc.red_idx.data(), sc.red_ptr.data(), sc.shared_node.data(),
                        const_cast<scalar_t *>(sc.stage.data()), dst);
            }
        }
        sscvfem_nodal_grad_normalize(d, ogx, ogy, ogz);
        return;
    }

    // Geometry, cached. Everything below is then about the field alone.
    sscvfem_build_grad_weight(d);

    ogx.assign((size_t)d.nnodes, 0);
    ogy.assign((size_t)d.nnodes, 0);
    ogz.assign((size_t)d.nnodes, 0);

    {
        // A PARTIAL range needs the staging buffer pre-zeroed: a slot a pack already
        // wrote is read by the reduction, and a slot this range never writes would
        // otherwise carry whatever the buffer held. The full range writes every slot
        // before reading it, so it skips the fill -- which is why this is the caller's
        // and not the sweep's: it is shared work that must happen once, not per thread.
        const ptrdiff_t _b = 0, _e = d.nmacro;
        if (d.scatter && d.scatter->ready && _b > 0 && d.scatter->n_slots > 0) {
            const SSScatter &sc = *d.scatter;
            scalar_t *const st = const_cast<scalar_t *>(sc.stage.data());
            std::fill(st, st + (size_t)sc.n_slots * 3, scalar_t(0));
        }
    #pragma omp parallel
    {
            sscvfem_nodal_grad_scatter_affine(d.elems, d.level, d.nxe, d.points, d.scatter ? d.scatter->n_slots : 0, d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, src, stride, ogx.data(), ogy.data(), ogz.data(),
                    sscvfem_affine_range(d, _b, _e), sscvfem_order(d));
            sscvfem_nodal_grad_scatter_isoparam(d.elems, d.level, d.nxe, d.points, d.scatter ? d.scatter->n_slots : 0, d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, src, stride, ogx.data(), ogy.data(), ogz.data(),
                    sscvfem_isoparam_range(d, _b, _e), sscvfem_order(d));
    }

        if (d.scatter && d.scatter->ready) {
            const SSScatter &sc = *d.scatter;
            scalar_t *dst[3] = {ogx.data(), ogy.data(), ogz.data()};
            const ptrdiff_t nrows = (ptrdiff_t)sc.shared_node.size();
    #pragma omp parallel
            sscvfem_reduce_shared_soa_w<3>(
                    cvfem_range_split(0, nrows, 1, cvfem_thread_index(), cvfem_n_threads()),
                    sc.red_idx.data(), sc.red_ptr.data(), sc.shared_node.data(),
                    const_cast<scalar_t *>(sc.stage.data()), dst);
        }
    }
    sscvfem_nodal_grad_normalize(d, ogx, ogy, ogz);
}

inline void sscvfem_nodal_p_grad(SSMeshData &d) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_p_grad");
    sscvfem_nodal_grad_strided(d, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
}

// The same reconstruction applied to the Jacobian direction's pressure component.
inline void sscvfem_nodal_q_grad(SSMeshData &d, const scalar_t *const SFEM_RESTRICT dir) {
    SFEM_TRACE_SCOPE("sscvfem::nodal_q_grad");
    sscvfem_nodal_grad_strided(d, dir + 3, CVFEM_HEX8_N_FIELDS, d.qgx, d.qgy, d.qgz);
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
//
// Forced inline because it is called once per micro cell in the hottest sweeps, and plain
// `inline` left the decision to GCC's unit-wide budget: code added elsewhere in this header for
// curved macro elements tipped it into an out-of-line copy, and the block diagonal on a box --
// whose own source had not changed -- ran 9% slower from the call.
inline __attribute__((always_inline)) Hex8RcConfig sscvfem_rc_config(const SSMeshData &d) {
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

// The shared reduction, for a launcher that has just run an element pass.
//
// It is a second, independent loop -- over the reduction rows rather than the macro elements --
// so DESIGN.md gives it its own entry point, and it runs after the element pass's threads have
// joined, which is also the barrier it needs: a row sums staging slots other macro elements
// wrote. Null or unbuilt scatter means the atomic path, which has nothing to reduce.
//
// THIS is where the row loop's parallel region lives, and the only place: the reduction kernel
// takes a range like every other sweep. Getting that wrong once cost the operator-consistency
// tests -- while the kernel still owned a region, wrapping it in another gave each outer thread
// a one-thread inner team, so every thread ran the whole row loop and each shared node was
// accumulated once per thread.
//
// Templated on the width because the two widths read different staging buffers: the 4-wide
// passes stage into SSScatter::stage and the block diagonal into stage16, at 16 values per slot.
// One helper, so a width cannot be paired with the wrong buffer.
template <int W = CVFEM_HEX8_N_FIELDS>
inline void sscvfem_drain_shared(SSMeshData &d, scalar_t *const SFEM_RESTRICT dst) {
    if (!d.scatter || !d.scatter->ready) return;
    const SSScatter &sc    = *d.scatter;
    const scalar_t *const stage = (W == 16) ? sc.stage16.data() : sc.stage.data();
    const ptrdiff_t       nrows = (ptrdiff_t)sc.shared_node.size();
#pragma omp parallel
    sscvfem_reduce_shared_w<W>(cvfem_range_split(0, nrows, 1, cvfem_thread_index(), cvfem_n_threads()),
                               sc.red_idx.data(), sc.red_ptr.data(), sc.shared_node.data(), dst,
                               stage);
}

inline void sscvfem_apply_blocks_ref(SSMeshData &d, const scalar_t rho, const scalar_t mu, const int blocks,
                                     const scalar_t *const SFEM_RESTRICT dir,
                                     scalar_t *const SFEM_RESTRICT       jv) {
    const ptrdiff_t ndof = d.nnodes * CVFEM_HEX8_N_FIELDS;
    // The reference implementation's two work vectors, held with the others.
    std::vector<scalar_t> &v = d.blocks_ref_work[0], &y = d.blocks_ref_work[1];
    v.assign((size_t)ndof, scalar_t(0));
    y.assign((size_t)ndof, scalar_t(0));

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
        {
            CVFEM_TRACE_SCOPE("sscvfem::apply_macro_local_hoisted");
            #pragma omp parallel
            {
                    sscvfem_apply_macro_hoisted_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, v.data(), y.data());
                    sscvfem_apply_macro_hoisted_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, v.data(), y.data());
            }
        }
        sscvfem_drain_shared(d, y.data());

        const int mom_bit = ucol ? SSBLOCK_UU : SSBLOCK_UP;
        const int con_bit = ucol ? SSBLOCK_PU : SSBLOCK_PP;
        for (ptrdiff_t n = 0; n < d.nnodes; ++n) {
            if (blocks & mom_bit)
                for (int c = 0; c < 3; ++c) jv[(size_t)n * 4 + c] += y[(size_t)n * 4 + c];
            if (blocks & con_bit) jv[(size_t)n * 4 + 3] += y[(size_t)n * 4 + 3];
        }
    }
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
        case 0:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<0>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<0>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_UU:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_UU>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_UU>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_UP:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_UP>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_UP>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_PU:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_PU>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_PU>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_PP:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_PP>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_PP>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_MOM:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_MOM>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_MOM>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_CON:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_CON>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_CON>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        case SSBLOCK_ALL:
            #pragma omp parallel
            {
                    sscvfem_apply_blocks_affine<SSBLOCK_ALL>(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                    sscvfem_apply_blocks_isoparam<SSBLOCK_ALL>(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
            }
            break;
        default:          sscvfem_apply_blocks_ref(d, rho, mu, blocks, dir, jv);       break;
    }

    sscvfem_drain_shared(d, jv);
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
            const idx_t g = d.elems[off[a]][e];
            ex[a] = (scalar_t)d.points[0][g];
            ey[a] = (scalar_t)d.points[1][g];
            ez[a] = (scalar_t)d.points[2][g];
        }
        scalar_t adj[9], det;
        sscvfem_micro_geom(ex, ey, ez, adj, &det);
        const scalar_t v        = std::fabs(det) / scalar_t(8);
        const bool     curved_e = sscvfem_macro_curved(d.macro_curved.empty() ? nullptr : d.macro_curved.data(), e);

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);
                    scalar_t  vc   = v;
                    if (curved_e) {
                        scalar_t cx[8], cy[8], cz[8], cadj[9], cdet;
                        sscvfem_cell_corners(d.elems, d.points, e, base, off, cx, cy, cz);
                        sscvfem_micro_geom(cx, cy, cz, cadj, &cdet);
                        vc = std::fabs(cdet) / scalar_t(8);
                    }
                    for (int a = 0; a < 8; ++a) {
                        const idx_t g = d.elems[base + off[a]][e];
                        atomic_add(node_vol.data(), g, vc);
                    }
                }
    }
}

// Whether there is a body force at all, and whether the control volume it is weighted by has
// been built, are staging questions; the sweep above answers neither.
inline void sscvfem_apply_body_force(SSMeshData &d, scalar_t *const SFEM_RESTRICT res) {
    if (d.fx.empty()) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
    #pragma omp parallel
        sscvfem_apply_body_force_sweep(cvfem_range_split(0, d.nnodes, 1, cvfem_thread_index(), cvfem_n_threads()),d.fx.data(), d.fy.data(), d.fz.data(), d.node_vol.data(), res);
}

// Whether the term is live -- a positive step and a history of the right length -- and whether
// the control volume has been built. The sweep divides by dt unconditionally, so these two
// guards are what make that safe, and they are the caller's to answer.
inline void sscvfem_apply_transient(SSMeshData &d, const scalar_t rho, scalar_t *const SFEM_RESTRICT res) {
    if (d.dt <= scalar_t(0)) return;
    if ((ptrdiff_t)d.u_prev.size() != 3 * d.nnodes) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
    // The BDF rule is kernels/cvfem_bdf.hpp's; which members hold the step sizes and whether
    // the history has two levels is what this side knows. The sweep reads u_prev2 only when
    // the order it is handed says two levels are there, so resolving the coefficients here is
    // also what keeps that read in bounds.
    {  // its own scope: ScopedEvent's variable name is fixed, so two trace scopes in one block collide
        CVFEM_TRACE_SCOPE("sscvfem::apply_transient_sweep");
        #pragma omp parallel
            sscvfem_apply_transient_sweep(cvfem_range_split(0, d.nnodes, 1, cvfem_thread_index(), cvfem_n_threads()),d.dt, d.node_vol.data(), d.u_prev.data(), d.u_prev2.data(), d.ux.data(), d.uy.data(), d.uz.data(), rho,
                                      cvfem_bdf_coeffs(d.bdf_order, d.dt, d.dt_prev,
                                                       (ptrdiff_t)d.u_prev2.size() == 3 * d.nnodes),
                                      res);
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

// As for the residual's transient term: a zero weight means there is nothing to add, and the
// control volume has to exist before the sweep reads it.
inline void sscvfem_apply_transient_action(SSMeshData &d, const scalar_t rho,
                                           const scalar_t *const SFEM_RESTRICT dir,
                                           scalar_t *const SFEM_RESTRICT       jv) {
    if (sscvfem_transient_diag_weight(d, rho) == scalar_t(0)) return;
    if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
    {  // its own scope: ScopedEvent's variable name is fixed, so two trace scopes in one block collide
        CVFEM_TRACE_SCOPE("sscvfem::apply_transient_action_sweep");
        #pragma omp parallel
            sscvfem_apply_transient_action_sweep(cvfem_range_split(0, d.nnodes, 1, cvfem_thread_index(), cvfem_n_threads()), d.node_vol.data(), sscvfem_transient_diag_weight(d, rho), rho, dir, jv);
    }
}

// The control residual, end to end. The body force and the transient term are post-passes over
// nodes with their own guards and their own cached control volume, which is front-end work; the
// sweep is the element pass.
inline void sscvfem_residual_naive(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                   scalar_t *const SFEM_RESTRICT res) {
    // Zeroed here, not in the sweep: the sweep is range-driven and runs once per thread.
    const ptrdiff_t ndof = d.nnodes * CVFEM_HEX8_N_FIELDS;
    for (ptrdiff_t i = 0; i < ndof; ++i) res[i] = scalar_t(0);
    {  // its own scope: ScopedEvent's variable name is fixed, so two trace scopes in one block collide
        CVFEM_TRACE_SCOPE("sscvfem::residual_naive_sweep");
        #pragma omp parallel
        {
                sscvfem_residual_naive_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.conv_peclet, d.elems, d.level, d.nnodes, d.p.empty() ? nullptr : d.p.data(), d.pgx.empty() ? nullptr : d.pgx.data(), d.pgy.empty() ? nullptr : d.pgy.data(), d.pgz.empty() ? nullptr : d.pgz.data(), d.points, d.upwind_eps, d.ux.empty() ? nullptr : d.ux.data(), d.uy.empty() ? nullptr : d.uy.data(), d.uz.empty() ? nullptr : d.uz.data(), sscvfem_rc_config(d), rho, mu, res);
                sscvfem_residual_naive_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.conv_peclet, d.elems, d.level, d.nnodes, d.p.empty() ? nullptr : d.p.data(), d.pgx.empty() ? nullptr : d.pgx.data(), d.pgy.empty() ? nullptr : d.pgy.data(), d.pgz.empty() ? nullptr : d.pgz.data(), d.points, d.upwind_eps, d.ux.empty() ? nullptr : d.ux.data(), d.uy.empty() ? nullptr : d.uy.data(), d.uz.empty() ? nullptr : d.uz.data(), sscvfem_rc_config(d), rho, mu, res);
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
    std::vector<scalar_t> &gx = d.ugrad_work[0], &gy = d.ugrad_work[1], &gz = d.ugrad_work[2];
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
                                           scalar_t *const SFEM_RESTRICT res, const bool zero_first = true,
                                           const int ho_override = -1) {
    SFEM_TRACE_SCOPE("sscvfem::residual");
    // Deferred-correction convection. Off by default and then bit-for-bit the scheme every
    // recorded number here was measured with; the gradient is built once per residual, which
    // is once per Newton step, because the correction is lagged by construction.
    d.conv_ho      = (ho_override >= 0) ? ho_override : smesh::Env::read<int>("SFEM_CONV_HO", 0);
    // DEFAULT 0, BECAUSE 0 IS WHAT CONVERGES. Measured on the backward-facing step at
    // Re = 40, the deferred correction reaches the target in 24 Newton steps unlimited
    // and does not converge at all with either limiter -- the bounded-face clip stalls
    // at Re 14.04 after 399 steps, Venkatakrishnan's smooth form at Re 21.92 after 411.
    // The limiter is the cause, not the reconstruction. Both remain reachable, because
    // a bounded scheme is still wanted and reproducing the failure is a legitimate need.
    d.conv_limiter = smesh::Env::read<int>("SFEM_CONV_LIMITER", 0);
    // SFEM_VENKAT_K: the dimensionless K in Venkatakrishnan's eps^2 = (K dx)^3, which is his
    // deactivation threshold -- below it the limiter switches itself off and the increment
    // passes through untouched. DEFAULT 0, which is the zero-severity control: it recovers
    // the arm measured above exactly, so a sweep over K starts from a known point rather than
    // from a different scheme. K is dimensionless and the scales are restored here, from the
    // reference velocity and the box length, because the kernel has neither.
    {
        const scalar_t k = smesh::Env::read<scalar_t>("SFEM_VENKAT_K", 0);
        const scalar_t u = smesh::Env::read<scalar_t>("SFEM_U", 1);
        d.conv_venkat_c  = cvfem_venkata_eps2_coeff(k, u, d.Lx);
    }
    {
        d.limiter_stats = cvfem_limiter_stats_sink();
    }

    d.conv_peclet = cvfem_hex8_peclet_config<scalar_t>();
    if (d.conv_ho && d.conv_peclet.form) {
        // Same refusal as the flat path, and it must be here too: this reader is independent,
        // so a check in only one of them leaves the other running the undefined combination.
        std::fprintf(stderr,
                     "SFEM_PECLET_BLEND with SFEM_CONV_HO is not defined: the deferred "
                     "correction's donor split comes from the unblended flux. Run one or the "
                     "other.\n");
        std::abort();
    }
    // SFEM_CONV_FREEZE: hold the deferred correction fixed through a continuation stage.
    //
    // The correction is a LAGGED source, so the Newton loop is also a fixed-point iteration on
    // it, and an arm whose limiter changes which branch it takes between iterations makes that
    // iteration chatter instead of settle. Freezing removes the fixed point outright: the
    // stage's correction is built once from the stage's opening state and then it is a
    // constant, so Newton sees a smooth first-order problem with a source term.
    //
    // Built as the DIFFERENCE of two residuals on the same state rather than by a new sweep.
    // R(ho) - R(lo) is exactly the correction's contribution -- every other term is identical
    // between them -- so this reuses the residual the tree already has instead of adding a
    // second path that computes the same quantity. It costs two extra residual evaluations per
    // stage, against the tens to hundreds of Newton steps the arm takes.
    //
    // AND IT MAKES NEWTON CONSISTENT, which is the part worth predicting before measuring.
    // Frozen, the residual is R_lo(u) plus a constant, so its exact Jacobian is J_lo(u) --
    // which is precisely the first-order Jacobian this path already assembles, because the
    // deferred correction deliberately keeps the tangent first order. Unfrozen, the
    // correction moves with u and that assembled Jacobian is NOT the residual's derivative,
    // so Newton is inexact and the outer iteration is really a fixed point on the lagged
    // source. Freezing removes the inexactness rather than damping it, so the expectation is
    // convergence at roughly the first-order arm's step count -- and that is the prediction
    // the measurement should be read against.
    //
    // The recursion terminates because the two inner calls pass ho_override, and the branch
    // fires only for the outer, environment-driven call.
    if (ho_override < 0) {
    // DEFAULT 1, BECAUSE 1 IS WHAT WORKS. Measured on Grace, backward-facing step at Re = 40
    // and the manufactured solution at Re = 100:
    //
    //                          Newton steps          u L2 rate
    //                       7,060      47,268        (8/16/32)
    //   unfrozen, K = 0       103         303          2.120
    //   unfrozen, unlimited    24    NO CONVERGENCE    2.227
    //   FROZEN,   K = 0        15          12          2.272
    //
    // Frozen is better on every axis measured: fewer Newton steps, 3.3x fewer linear
    // iterations, final residuals of 1e-9 against 5e-7, a HIGHER fitted order of accuracy,
    // and it converges where the unlimited arm does not. It also keeps the bound fully
    // intact, which Venkatakrishnan's eps^2 buys its convergence by giving up.
    //
    // It is an approximation and the default should say so: frozen, the converged state
    // solves R_lo(u) + frozen = 0 rather than R_ho(u) = 0, with the correction taken at the
    // stage's opening state. The accuracy ladder is what licenses the default -- if the
    // correction were too stale the order would fall toward 1, and it rises instead.
    //
    // SFEM_CONV_FREEZE=0 restores the unfrozen scheme. This changes nothing when
    // SFEM_CONV_HO is off, which is still the overall default: the branch is guarded on it.
        d.conv_freeze = smesh::Env::read<int>("SFEM_CONV_FREEZE", 1);
        if (d.conv_freeze && d.conv_ho) {
            const ptrdiff_t n = d.nnodes * CVFEM_HEX8_N_FIELDS;
            if ((ptrdiff_t)d.conv_frozen.size() != n) {
                std::vector<scalar_t> &r_ho = d.conv_work[0], &r_lo = d.conv_work[1];
                r_ho.assign((size_t)n, scalar_t(0));
                r_lo.assign((size_t)n, scalar_t(0));
                sscvfem_residual(d, rho, mu, r_ho.data(), true, 1);
                sscvfem_residual(d, rho, mu, r_lo.data(), true, 0);
                d.conv_frozen.resize((size_t)n);
                for (ptrdiff_t i = 0; i < n; ++i) d.conv_frozen[(size_t)i] = r_ho[(size_t)i] - r_lo[(size_t)i];
            }
            sscvfem_residual(d, rho, mu, res, zero_first, 0);
            for (ptrdiff_t i = 0; i < n; ++i) res[i] += d.conv_frozen[(size_t)i];
            return;
        }
        // Not frozen this call: drop any correction held from an earlier configuration so a
        // run that turns freezing off mid-flight cannot keep adding a stale source.
        if (!d.conv_freeze && !d.conv_frozen.empty()) d.conv_frozen.clear();
    }

    if (d.conv_ho) sscvfem_assemble_nodal_u_grad(d);
    else d.ugrad.clear();

    const ptrdiff_t ndof = d.nnodes * CVFEM_HEX8_N_FIELDS;
    if (zero_first)
        for (ptrdiff_t i = 0; i < ndof; ++i) res[i] = scalar_t(0);

    {
        CVFEM_TRACE_SCOPE("sscvfem::residual_sweep");
        #pragma omp parallel
        {
                sscvfem_residual_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.conv_ho, d.conv_limiter, d.conv_peclet, d.conv_venkat_c, d.elems, d.level, d.limiter_stats, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.empty() ? nullptr : d.p.data(), d.pgx.empty() ? nullptr : d.pgx.data(), d.pgy.empty() ? nullptr : d.pgy.data(), d.pgz.empty() ? nullptr : d.pgz.data(), d.points, d.ugrad.empty() ? nullptr : d.ugrad.data(), d.upwind_eps, d.ux.empty() ? nullptr : d.ux.data(), d.uy.empty() ? nullptr : d.uy.data(), d.uz.empty() ? nullptr : d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, res);
                sscvfem_residual_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.conv_ho, d.conv_limiter, d.conv_peclet, d.conv_venkat_c, d.elems, d.level, d.limiter_stats, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.empty() ? nullptr : d.p.data(), d.pgx.empty() ? nullptr : d.pgx.data(), d.pgy.empty() ? nullptr : d.pgy.data(), d.pgz.empty() ? nullptr : d.pgz.data(), d.points, d.ugrad.empty() ? nullptr : d.ugrad.data(), d.upwind_eps, d.ux.empty() ? nullptr : d.ux.data(), d.uy.empty() ? nullptr : d.uy.data(), d.uz.empty() ? nullptr : d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, res);
        }
    }
    sscvfem_drain_shared(d, res);
    sscvfem_apply_body_force(d, res);
    sscvfem_apply_transient(d, rho, res);
}

// The control's allocation, which is the caller's. The sweep writes a buffer.
inline void sscvfem_block_diag_naive(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                                     std::vector<scalar_t> &diag) {
    diag.assign((size_t)d.nnodes * 16, scalar_t(0));
    {  // its own scope: ScopedEvent's variable name is fixed, so two trace scopes in one block collide
        CVFEM_TRACE_SCOPE("sscvfem::block_diag_naive_sweep");
        #pragma omp parallel
        {
            sscvfem_block_diag_naive_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.elems, d.level,
                                       d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(),
                                       d.pgz.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(),
                                       sscvfem_rc_config(d), rho, mu, diag.data());
            sscvfem_block_diag_naive_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.elems, d.level,
                                       d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(),
                                       d.pgz.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(),
                                       sscvfem_rc_config(d), rho, mu, diag.data());
        }

    }
}

// The block diagonal, end to end. This is the front-end side: it owns the allocation, makes
// sure the cached node volume the transient pass reads has been built, and runs the two passes
// in order. Neither sweep allocates or decides anything.
inline void sscvfem_block_diag(SSMeshData &d, const scalar_t rho, const scalar_t mu,
                               std::vector<scalar_t> &diag) {
    diag.assign((size_t)d.nnodes * 16, scalar_t(0));
    {  // its own scope: ScopedEvent's variable name is fixed, so two trace scopes in one block collide
        CVFEM_TRACE_SCOPE("sscvfem::block_diag_sweep");
        #pragma omp parallel
        {
                sscvfem_block_diag_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage16.empty() ? nullptr : d.scatter->stage16.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, diag.data());
                sscvfem_block_diag_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage16.empty() ? nullptr : d.scatter->stage16.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, diag.data());
        }
    }
    // The element pass has joined, so the staging slots are all written: now each shared node
    // sums the ones that belong to it, in slot order.
    sscvfem_drain_shared<16>(d, diag.data());
    // The node volume is built only when the transient pass will read it, which is what the
    // guard inside the sweep used to do: on a steady solve the weight is zero and
    // sscvfem_node_volume is a full sweep over the macro elements for nothing.
    if (sscvfem_transient_diag_weight(d, rho) != scalar_t(0)) {
        if ((ptrdiff_t)d.node_vol.size() != d.nnodes) sscvfem_node_volume(d, d.node_vol);
        #pragma omp parallel
            sscvfem_block_diag_transient(cvfem_range_split(0, d.nnodes, 1, cvfem_thread_index(), cvfem_n_threads()), d.node_vol.data(), sscvfem_transient_diag_weight(d, rho), rho, diag.data());
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
// The alternatives are kept rather than deleted. The naive apply -- now
// sscvfem_apply_naive_affine over the straight macro elements and _isoparam over the curved
// ones -- is the correctness control, and is what the benchmark checks every other variant
// against.
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
    {
        CVFEM_TRACE_SCOPE("sscvfem::apply_macro_local_hoisted");
        #pragma omp parallel
        {
                sscvfem_apply_macro_hoisted_affine(sscvfem_affine_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
                sscvfem_apply_macro_hoisted_isoparam(sscvfem_isoparam_range(d, 0, d.nmacro), sscvfem_order(d), d.Lx, d.Ly, d.Lz, d.bc_p, d.bc_tx, d.bc_ty, d.bc_tz, d.elems, d.level, d.macro_face_mask.empty() ? nullptr : d.macro_face_mask.data(), d.macro_natural_mask.empty() ? nullptr : d.macro_natural_mask.data(), d.macro_pressure_mask.empty() ? nullptr : d.macro_pressure_mask.data(), d.macro_traction_mask.empty() ? nullptr : d.macro_traction_mask.data(), d.nxe, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.empty() ? nullptr : d.qgx.data(), d.qgy.data(), d.qgz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), sscvfem_rc_config(d), d.scatter ? d.scatter->red_idx.empty() ? nullptr : d.scatter->red_idx.data() : nullptr, d.scatter ? d.scatter->red_ptr.empty() ? nullptr : d.scatter->red_ptr.data() : nullptr, d.scatter ? d.scatter->shared_node.empty() ? nullptr : d.scatter->shared_node.data() : nullptr, d.scatter ? d.scatter->slot.empty() ? nullptr : d.scatter->slot.data() : nullptr, d.scatter ? d.scatter->stage.empty() ? nullptr : d.scatter->stage.data() : nullptr, d.scatter ? (ptrdiff_t)d.scatter->shared_node.size() : 0, rho, mu, dir, jv);
        }
    }
    sscvfem_drain_shared(d, jv);
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
