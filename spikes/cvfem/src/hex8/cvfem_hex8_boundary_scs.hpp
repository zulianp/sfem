#ifndef CVFEM_HEX8_BOUNDARY_SCS_HPP
#define CVFEM_HEX8_BOUNDARY_SCS_HPP

// Boundary sub-control-surface terms for the CVFEM HEX8 Navier-Stokes operators.
//
// Lifted verbatim out of cvfem_hex8_ns_steady.cpp so the CUDA kernels can call the same
// code the solver does, rather than a second copy of it. Templated on the scalar type
// and marked device-callable, exactly as the volume kernels were.
//
// Not self-contained: the includer must already provide scalar_t, SFEM_RESTRICT and the
// CVFEM HEX8 volume kernels (cvfem_hex8_grad_sumfact, cvfem_hex8_dir_areas, ...).

#include "cvfem_portability.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// Prescribed boundary data, carried alongside the face masks.
//
// A default-constructed instance means "nothing prescribed", which reproduces the existing
// behaviour on every face exactly -- so this is a trailing default argument on the routines
// below and no call site had to change. That is the shape Hex8RhieChowT already uses to
// make the Rhie-Chow term optional, and it is preferred here over widening the face masks
// to two bits per face: the values a boundary condition needs are per-sideset constants,
// not per-face bits, and re-encoding the masks would have touched all three signatures and
// about a dozen call sites for no gain.
//
// Two conditions, each reducing to something already supported:
//
//   traction   (pI - tau).n = t on the faces tmask selects, which must be a subset of
//              nmask -- a traction condition IS the natural condition, with a value. t = 0,
//              or a face in nmask but not tmask, is the do-nothing outflow bit for bit:
//              the term added vanishes and its square root is not even evaluated.
//
//              tmask exists rather than the value simply applying to all of nmask because a
//              single run routinely has both -- an outlet that is genuinely traction-free
//              and a surface that is pushed -- and one scalar triple covering every natural
//              face cannot express that. It would instead apply the pushed surface's
//              traction to the outlet as well, silently. This mirrors the pmask/p_bar pair
//              below: one per-face selector, one per-sideset constant.
//   pressure   p = p_bar on the faces pmask selects, with the viscous traction still taken
//              from the interior state. This is the closed-face flux with the nodal
//              pressure replaced by a prescribed one, which is what a port held at a
//              pressure is.
//
// nmask wins where both select the same face: a face cannot be both traction-free and
// pressure-prescribed, and silently applying both would be worse than picking one and
// saying so.
template <typename scalar_t>
struct Hex8BoundaryDataT {
    scalar_t tx{0}, ty{0}, tz{0};  // prescribed traction, on the faces tmask selects
    int      tmask{0};             // faces carrying it; 0 means every natural face is free
    int      pmask{0};             // faces carrying a prescribed pressure
    scalar_t p_bar{0};             // and its value
};

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE bool on_plane(const scalar_t c, const scalar_t value, const scalar_t L) {
    const scalar_t tol = scalar_t(1e-8) * std::max(L, scalar_t(1));
    return std::fabs(c - value) <= tol;
}

#define CVFEM_HEX8_BFACE_NODES_INIT {{0, 3, 7, 4}, \
                                                     {1, 2, 6, 5}, \
                                                     {0, 1, 5, 4}, \
                                                     {3, 2, 6, 7}, \
                                                     {0, 1, 2, 3}, \
                                                     {4, 5, 6, 7}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const int (&cvfem_hex8_bface_nodes_tbl())[6][4] {
    static constexpr int t[6][4] = CVFEM_HEX8_BFACE_NODES_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_NODES cvfem_hex8_bface_nodes_tbl()
#else
static constexpr int CVFEM_HEX8_BFACE_NODES[6][4] = CVFEM_HEX8_BFACE_NODES_INIT;
#endif
#define CVFEM_HEX8_BFACE_AXIS_INIT {0, 0, 1, 1, 2, 2}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const int (&cvfem_hex8_bface_axis_tbl())[6] {
    static constexpr int t[6] = CVFEM_HEX8_BFACE_AXIS_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_AXIS cvfem_hex8_bface_axis_tbl()
#else
static constexpr int CVFEM_HEX8_BFACE_AXIS[6] = CVFEM_HEX8_BFACE_AXIS_INIT;
#endif
#define CVFEM_HEX8_BFACE_OUT_INIT {-1, 1, -1, 1, -1, 1}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_bface_out_tbl())[6] {
    static constexpr double t[6] = CVFEM_HEX8_BFACE_OUT_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_OUT cvfem_hex8_bface_out_tbl()
#else
static constexpr double CVFEM_HEX8_BFACE_OUT[6] = CVFEM_HEX8_BFACE_OUT_INIT;
#endif
#define CVFEM_HEX8_BFACE_XI_INIT { \
        {{0, double(0.25), double(0.25)}, \
         {0, double(0.75), double(0.25)}, \
         {0, double(0.75), double(0.75)}, \
         {0, double(0.25), double(0.75)}}, \
        {{1, double(0.25), double(0.25)}, \
         {1, double(0.75), double(0.25)}, \
         {1, double(0.75), double(0.75)}, \
         {1, double(0.25), double(0.75)}}, \
        {{double(0.25), 0, double(0.25)}, \
         {double(0.75), 0, double(0.25)}, \
         {double(0.75), 0, double(0.75)}, \
         {double(0.25), 0, double(0.75)}}, \
        {{double(0.25), 1, double(0.25)}, \
         {double(0.75), 1, double(0.25)}, \
         {double(0.75), 1, double(0.75)}, \
         {double(0.25), 1, double(0.75)}}, \
        {{double(0.25), double(0.25), 0}, \
         {double(0.75), double(0.25), 0}, \
         {double(0.75), double(0.75), 0}, \
         {double(0.25), double(0.75), 0}}, \
        {{double(0.25), double(0.25), 1}, \
         {double(0.75), double(0.25), 1}, \
         {double(0.75), double(0.75), 1}, \
         {double(0.25), double(0.75), 1}}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_bface_xi_tbl())[6][4][3] {
    static constexpr double t[6][4][3] = CVFEM_HEX8_BFACE_XI_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_XI cvfem_hex8_bface_xi_tbl()
#else
static constexpr double CVFEM_HEX8_BFACE_XI[6][4][3] = CVFEM_HEX8_BFACE_XI_INIT;
#endif

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE int hex8_face_on_domain(const int f, const scalar_t *const SFEM_RESTRICT x,
                                           const scalar_t *const SFEM_RESTRICT y, const scalar_t *const SFEM_RESTRICT z,
                                           const scalar_t Lx, const scalar_t Ly, const scalar_t Lz) {
    const int      axis  = CVFEM_HEX8_BFACE_AXIS[f];
    const scalar_t L     = axis == 0 ? Lx : (axis == 1 ? Ly : Lz);
    const scalar_t plane = CVFEM_HEX8_BFACE_OUT[f] < 0 ? scalar_t(0) : L;
    for (int k = 0; k < 4; ++k) {
        const int      a = CVFEM_HEX8_BFACE_NODES[f][k];
        const scalar_t c = axis == 0 ? x[a] : (axis == 1 ? y[a] : z[a]);
        if (!on_plane(c, plane, L)) return 0;
    }
    return 1;
}

// The elements that actually have a boundary face, listed rather than filtered.
//
// Testing the mask inside a full sweep is not enough, and the reason is worth recording
// because it is invisible in the work count. The boundary elements are the shell of the
// mesh, so under `schedule(static)` they land in a few threads' chunks: skipping the
// interior cut the WORK by an order of magnitude and left the WALL TIME unchanged, because
// the pass is bound by whichever thread owns the shell. Measured on Grace at 1,853,572 dof:
// 1058 us/call before the skip, 1070 us/call after -- no change at all, for ~91% less work.
//
// Iterating a compacted list restores the balance: every thread gets an equal share of the
// elements that do something, and the interior is not visited at all.
template <typename MeshT>
static void cvfem_hex8_compact_boundary_elems(MeshT &d) {
    d.bnd_elems.clear();
    // The gather map is indexed by position in bnd_elems, so it describes the list that
    // was current when it was built and nothing else.
    d.bnd_gather_valid = false;
    for (ptrdiff_t e = 0; e < d.nelements; ++e)
        if (d.face_mask_eff[(size_t)e]) d.bnd_elems.push_back(e);
}

// ------------------------------------------------- the shell's node gather map
//
// A node-indexed CSR over the boundary shell, so the closure can be summed in an order
// fixed by the index array instead of by thread timing.
//
// The two boundary passes used to scatter their element contribution into the shared node
// arrays with `atomic_add` under `schedule(static)`. Static scheduling fixes WHICH thread
// owns a face, but not the order in which two threads holding faces that meet at a node
// commit to it, and floating-point addition is not associative -- so the operator was not
// bit-reproducible with itself across threads. Measured on one case at 33,124 dof: five
// runs at one thread all took 922 linear iterations and printed the same residual to every
// digit at iteration 100, while five at 72 threads took 900, 839, 1595, 943 and 742.
//
// The map turns the scatter into a gather. Each entry is a slot `i * 8 + a` naming the
// local node `a` of the i-th boundary element, grouped by the global node it lands on, so
// a pass can stage its per-element contributions and then have each node sum the slots
// that belong to it, alone and in a fixed order. That is deterministic for any thread
// count, and the summation order is the same one a serial run would use.
//
// Built serially. It is O(boundary shell), which is a surface rather than a volume, and a
// parallel build would have to be sorted afterwards to be reproducible anyway -- which is
// the property the whole map exists to provide.
template <typename MeshT>
static void cvfem_hex8_build_bnd_gather(MeshT &d) {
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
    if (d.bnd_gather_valid && d.bnd_gather_n_bnd == n_bnd) return;

    std::vector<std::pair<smesh::idx_t, int32_t>> pairs;
    pairs.reserve((size_t)n_bnd * CVFEM_HEX8_N_NODES);
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e = d.bnd_elems[(size_t)i];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
            pairs.emplace_back(d.elems[a][e], (int32_t)(i * CVFEM_HEX8_N_NODES + a));
    }
    // By node first, then by slot, so the summation order within a node is the order the
    // boundary list gives -- the order a serial sweep would have produced.
    std::sort(pairs.begin(), pairs.end());

    d.bnd_gather_dest.clear();
    d.bnd_gather_ptr.clear();
    d.bnd_gather_slot.clear();
    d.bnd_gather_slot.reserve(pairs.size());
    d.bnd_gather_ptr.push_back(0);
    for (size_t j = 0; j < pairs.size();) {
        const smesh::idx_t node = pairs[j].first;
        d.bnd_gather_dest.push_back(node);
        for (; j < pairs.size() && pairs[j].first == node; ++j) d.bnd_gather_slot.push_back(pairs[j].second);
        d.bnd_gather_ptr.push_back((ptrdiff_t)d.bnd_gather_slot.size());
    }
    d.bnd_r.assign((size_t)n_bnd * CVFEM_HEX8_N_DOF, scalar_t(0));
    d.bnd_gather_n_bnd = n_bnd;
    d.bnd_gather_valid = true;
}

// Commit one boundary element's contribution.
//
// Into the stage, where cvfem_hex8_bnd_gather_* will sum it per node in a fixed order, or
// straight into the node arrays when the atomic path is forced for measurement. The stage
// is written rather than accumulated, because it is reused across calls and the gather
// reads exactly the slots this sweep wrote.
//
// The all-zero early-out is kept on the atomic path only. There it skips four atomics; on
// the staged path there is nothing to skip, and skipping the write would leave the previous
// call's value in the slot for the gather to read.
template <typename MeshT>
static SFEM_INLINE void cvfem_hex8_bnd_commit(MeshT &d, const ptrdiff_t i, const ptrdiff_t e,
                                              const scalar_t *const SFEM_RESTRICT r, const int force_atomic,
                                              scalar_t *const SFEM_RESTRICT fx, scalar_t *const SFEM_RESTRICT fy,
                                              scalar_t *const SFEM_RESTRICT fz, scalar_t *const SFEM_RESTRICT fc) {
    if (!force_atomic) {
        scalar_t *const SFEM_RESTRICT stage = d.bnd_r.data() + i * CVFEM_HEX8_N_DOF;
        for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) stage[k] = r[k];
        return;
    }
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        if (r[a * 4 + 0] == scalar_t(0) && r[a * 4 + 1] == scalar_t(0) && r[a * 4 + 2] == scalar_t(0) &&
            r[a * 4 + 3] == scalar_t(0))
            continue;
        const smesh::idx_t g = d.elems[a][e];
        CVFEM_ATOMIC_ADD(fx[g], r[a * 4 + 0]);
        CVFEM_ATOMIC_ADD(fy[g], r[a * 4 + 1]);
        CVFEM_ATOMIC_ADD(fz[g], r[a * 4 + 2]);
        CVFEM_ATOMIC_ADD(fc[g], r[a * 4 + 3]);
    }
}

// The same, for a destination that interleaves the four fields per node.
template <typename MeshT>
static SFEM_INLINE void cvfem_hex8_bnd_commit_interleaved(MeshT &d, const ptrdiff_t i, const ptrdiff_t e,
                                                          const scalar_t *const SFEM_RESTRICT r,
                                                          const int                           force_atomic,
                                                          scalar_t *const SFEM_RESTRICT       jv) {
    if (!force_atomic) {
        scalar_t *const SFEM_RESTRICT stage = d.bnd_r.data() + i * CVFEM_HEX8_N_DOF;
        for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) stage[k] = r[k];
        return;
    }
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        if (r[a * 4 + 0] == scalar_t(0) && r[a * 4 + 1] == scalar_t(0) && r[a * 4 + 2] == scalar_t(0) &&
            r[a * 4 + 3] == scalar_t(0))
            continue;
        const ptrdiff_t g = (ptrdiff_t)d.elems[a][e] * 4;
        CVFEM_ATOMIC_ADD(jv[g + 0], r[a * 4 + 0]);
        CVFEM_ATOMIC_ADD(jv[g + 1], r[a * 4 + 1]);
        CVFEM_ATOMIC_ADD(jv[g + 2], r[a * 4 + 2]);
        CVFEM_ATOMIC_ADD(jv[g + 3], r[a * 4 + 3]);
    }
}

// Sum the staged contributions into the node arrays, one node per iteration.
//
// Each iteration owns its destination outright and reads its slots in the order the map
// lists them, so the result does not depend on the thread count or on which thread got
// which node -- which is the whole point. The sum is accumulated in locals and added to
// the destination once, because the interior sweep has already written there.
template <typename MeshT>
static void cvfem_hex8_bnd_gather_soa(MeshT &d, scalar_t *const SFEM_RESTRICT fx, scalar_t *const SFEM_RESTRICT fy,
                                      scalar_t *const SFEM_RESTRICT fz, scalar_t *const SFEM_RESTRICT fc) {
    const ptrdiff_t                     n     = (ptrdiff_t)d.bnd_gather_dest.size();
    const scalar_t *const SFEM_RESTRICT stage = d.bnd_r.data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t k = 0; k < n; ++k) {
        scalar_t sx = scalar_t(0), sy = scalar_t(0), sz = scalar_t(0), sc = scalar_t(0);
        for (ptrdiff_t j = d.bnd_gather_ptr[(size_t)k]; j < d.bnd_gather_ptr[(size_t)k + 1]; ++j) {
            const scalar_t *const SFEM_RESTRICT v = stage + (ptrdiff_t)d.bnd_gather_slot[(size_t)j] * 4;
            sx += v[0];
            sy += v[1];
            sz += v[2];
            sc += v[3];
        }
        const smesh::idx_t g = d.bnd_gather_dest[(size_t)k];
        fx[g] += sx;
        fy[g] += sy;
        fz[g] += sz;
        fc[g] += sc;
    }
}

// The same, for a destination that interleaves the four fields per node.
template <typename MeshT>
static void cvfem_hex8_bnd_gather_interleaved(MeshT &d, scalar_t *const SFEM_RESTRICT jv) {
    const ptrdiff_t                     n     = (ptrdiff_t)d.bnd_gather_dest.size();
    const scalar_t *const SFEM_RESTRICT stage = d.bnd_r.data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t k = 0; k < n; ++k) {
        scalar_t s0 = scalar_t(0), s1 = scalar_t(0), s2 = scalar_t(0), s3 = scalar_t(0);
        for (ptrdiff_t j = d.bnd_gather_ptr[(size_t)k]; j < d.bnd_gather_ptr[(size_t)k + 1]; ++j) {
            const scalar_t *const SFEM_RESTRICT v = stage + (ptrdiff_t)d.bnd_gather_slot[(size_t)j] * 4;
            s0 += v[0];
            s1 += v[1];
            s2 += v[2];
            s3 += v[3];
        }
        const ptrdiff_t g = (ptrdiff_t)d.bnd_gather_dest[(size_t)k] * 4;
        jv[g + 0] += s0;
        jv[g + 1] += s1;
        jv[g + 2] += s2;
        jv[g + 3] += s3;
    }
}

// ------------------------------------------------- effective boundary face mask
//
// Which of an element's six faces lie on the domain boundary, as a single per-element
// bitfield: the mask the operator compiled from a sideset where there is one, and
// otherwise the same bounding-box test the kernels above would run face by face -- which
// is the default, since d.face_mask is only built under SFEM_BOUNDARY_MASK=1.
//
// Evaluating it here is not about the test, which is cheap. It is about letting the caller
// skip the element outright. The boundary closure is a second sweep over the WHOLE mesh
// that gathers coordinates, fields and, for the Jacobian, the direction -- 88 doubles per
// element -- in order to do work on the boundary layer alone. At n=140 that is 2% of the
// elements paying for 100% of the gathers.
//
// The skip is exact, not an approximation: fmask is read in exactly one place in each of
// the three kernels above, the per-face inclusion test, so an element whose effective mask
// is zero contributes nothing at all. tests/cvfem_boundary_mask_test.cpp asserts precisely
// that ("fmask 0 contributes nothing"), and it also asserts that the six single-face
// contributions sum to the all-face one, which is what makes a per-face precomputation
// equivalent to the per-face test it replaces.
template <typename MeshT>
static void cvfem_hex8_build_face_mask_eff(MeshT &d) {
    // The bounding-box branch below reads Lx/Ly/Lz, so they are part of the key: a
    // MeshData reused for a different domain would otherwise keep a mask describing the
    // old one, and a wrong face mask is silent -- it leaves a control volume open and the
    // solve converges to the wrong answer rather than failing.
    if (d.face_mask_eff_valid && (ptrdiff_t)d.face_mask_eff.size() == d.nelements &&
        d.face_mask_eff_lx == d.Lx && d.face_mask_eff_ly == d.Ly && d.face_mask_eff_lz == d.Lz)
        return;
    d.face_mask_eff.assign((size_t)d.nelements, 0);
    d.face_mask_eff_valid = true;
    d.face_mask_eff_lx    = d.Lx;
    d.face_mask_eff_ly    = d.Ly;
    d.face_mask_eff_lz    = d.Lz;
    if (!d.face_mask.empty()) {
        for (ptrdiff_t e = 0; e < d.nelements; ++e) d.face_mask_eff[(size_t)e] = d.face_mask[(size_t)e];
        cvfem_hex8_compact_boundary_elems(d);
        return;
    }
    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const auto g = d.elems[a][e];
            x[a]         = scalar_t(px[g]);
            y[a]         = scalar_t(py[g]);
            z[a]         = scalar_t(pz[g]);
        }
        int m = 0;
        for (int f = 0; f < 6; ++f)
            if (hex8_face_on_domain(f, x, y, z, d.Lx, d.Ly, d.Lz)) m |= 1 << f;
        d.face_mask_eff[(size_t)e] = (uint8_t)m;
    }
    cvfem_hex8_compact_boundary_elems(d);
}

// The mask to hand the kernels for element e: the effective one when it has been built,
// and otherwise the original convention, -1 for "decide from the bounding box".
template <typename MeshT>
static SFEM_INLINE int cvfem_hex8_face_mask_of(const MeshT &d, const ptrdiff_t e) {
    if (!d.face_mask_eff.empty()) return (int)d.face_mask_eff[(size_t)e];
    return d.face_mask.empty() ? -1 : (int)d.face_mask[(size_t)e];
}

// Templated on the slot type rather than fixed to smesh::count_t, because the block
// diagonal addresses its destination with ptrdiff_t node indices and relies on -1 meaning
// "drop this write". Fixing the type here would force that array into count_t, whose
// signedness is a build option (SMESH_COUNT_TYPE) -- and an unsigned -1 is an
// out-of-bounds write rather than a dropped one.
template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void hex8_visc_jac_row(const scalar_t mu, const scalar_t ax, const scalar_t ay, const scalar_t az,
                                          const scalar_t w[][3], const int row, const Slot *const SFEM_RESTRICT slots,
                                          scalar_t *const SFEM_RESTRICT values) {
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
        const scalar_t wx  = w[k][0];
        const scalar_t wy  = w[k][1];
        const scalar_t wz  = w[k][2];
        const scalar_t d00 = -(scalar_t(2) * wx * ax + wy * ay + wz * az) * mu;
        const scalar_t d01 = -(wx * ay) * mu;
        const scalar_t d02 = -(wx * az) * mu;
        const scalar_t d10 = -(wy * ax) * mu;
        const scalar_t d11 = -(wx * ax + scalar_t(2) * wy * ay + wz * az) * mu;
        const scalar_t d12 = -(wy * az) * mu;
        const scalar_t d20 = -(wz * ax) * mu;
        const scalar_t d21 = -(wz * ay) * mu;
        const scalar_t d22 = -(wx * ax + wy * ay + scalar_t(2) * wz * az) * mu;
        cvfem_hex8_bsr_acc_mom<Atomic>(values, slots[row * 8 + k], d00, d01, d02, d10, d11, d12, d20, d21, d22);
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_residual(const scalar_t rho, const scalar_t mu, const int isoparam, const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                  const scalar_t Lx, const scalar_t Ly, const scalar_t Lz,
                                                  const scalar_t *const SFEM_RESTRICT x, const scalar_t *const SFEM_RESTRICT y,
                                                  const scalar_t *const SFEM_RESTRICT z, const scalar_t *const SFEM_RESTRICT ux,
                                                  const scalar_t *const SFEM_RESTRICT uy, const scalar_t *const SFEM_RESTRICT uz,
                                                  const scalar_t *const SFEM_RESTRICT p, scalar_t *const SFEM_RESTRICT r,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
    scalar_t grad_el[9];
    // Hoisted: the area magnitude a prescribed traction needs costs a square root per node
    // per face, and t = 0 is the overwhelmingly common case.
    const int have_traction =
            bd.tmask != 0 && (bd.tx != scalar_t(0) || bd.ty != scalar_t(0) || bd.tz != scalar_t(0));
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, ux, uy, uz, grad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az, grad[9];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                cvfem_hex8_grad_at(adj, det, dN, ux, uy, uz, grad);
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int c = 0; c < 9; ++c) grad[c] = grad_el[c];
            }
            scalar_t tau_x, tau_y, tau_z;
            cvfem_hex8_traction(mu, grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], ax, ay, az,
                                tau_x, tau_y, tau_z);
            const scalar_t mdot = rho * (ux[i] * ax + uy[i] * ay + uz[i] * az);
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: the closed-face flux with p_bar in place of the
                // nodal pressure. The viscous traction still comes from the interior state,
                // so this prescribes the pressure and not the whole normal traction.
                r[i * 4 + 0] += mdot * ux[i] + bd.p_bar * ax - tau_x;
                r[i * 4 + 1] += mdot * uy[i] + bd.p_bar * ay - tau_y;
                r[i * 4 + 2] += mdot * uz[i] + bd.p_bar * az - tau_z;
                r[i * 4 + 3] += mdot;
            } else if ((nmask >> f) & 1) {
                // Do-nothing (natural) outflow: (p I - tau) . n = 0, so the pressure and
                // viscous traction are prescribed rather than evaluated. Dropping them is
                // what makes this a genuine outflow condition and what removes the constant-
                // pressure nullspace -- with p_i * a retained, a uniform pressure shift
                // integrates to zero over every closed control volume and the gauge stays
                // undetermined, so the solve needs a pin and behaves badly with one.
                //
                // Backflow guard: the convective term uses max(mdot, 0). The interior kernel
                // has an upwind switch (mpos/mneg) and this one does not, so on a face where
                // mdot < 0 the unguarded form would convect the *downwind* value into the
                // domain -- the classic finite-volume backflow instability, and a
                // recirculating outlet is where it bites.
                const scalar_t mup = mdot > scalar_t(0) ? mdot : scalar_t(0);
                scalar_t dS = scalar_t(0);
                if (have_traction && ((bd.tmask >> f) & 1)) dS = std::sqrt(ax * ax + ay * ay + az * az);
                r[i * 4 + 0] += mup * ux[i] + bd.tx * dS;
                r[i * 4 + 1] += mup * uy[i] + bd.ty * dS;
                r[i * 4 + 2] += mup * uz[i] + bd.tz * dS;
                // Continuity carries the true flux, not the guarded one: clipping it would
                // destroy global mass conservation, which is the property being verified.
                r[i * 4 + 3] += mdot;
            } else {
                r[i * 4 + 0] += mdot * ux[i] + p[i] * ax - tau_x;
                r[i * 4 + 1] += mdot * uy[i] + p[i] * ay - tau_y;
                r[i * 4 + 2] += mdot * uz[i] + p[i] * az - tau_z;
                r[i * 4 + 3] += mdot;
            }
        }
    }
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_jacobian(const scalar_t rho, const scalar_t mu, const int isoparam, const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                 const scalar_t Lx, const scalar_t Ly, const scalar_t Lz,
                                                 const scalar_t *const SFEM_RESTRICT x, const scalar_t *const SFEM_RESTRICT y,
                                                 const scalar_t *const SFEM_RESTRICT z, const scalar_t *const SFEM_RESTRICT ux,
                                                 const scalar_t *const SFEM_RESTRICT uy, const scalar_t *const SFEM_RESTRICT uz,
                                                 const Slot *const SFEM_RESTRICT slots, scalar_t *const SFEM_RESTRICT values,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
    scalar_t A[3][3];
    scalar_t w_el[CVFEM_HEX8_N_NODES][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_dir_areas(adj, A);
        const scalar_t inv_det = scalar_t(1) / det;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            cvfem_hex8_pushforward(adj, inv_det, CVFEM_HEX8_DN_REF[a][0], CVFEM_HEX8_DN_REF[a][1], CVFEM_HEX8_DN_REF[a][2],
                                   w_el[a][0], w_el[a][1], w_el[a][2]);
        }
    }

    for (int f = 0; f < 6; ++f) {
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az;
            scalar_t  w[CVFEM_HEX8_N_NODES][3];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                const scalar_t inv_det = scalar_t(1) / det;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    cvfem_hex8_pushforward(adj, inv_det, dN[a][0], dN[a][1], dN[a][2], w[a][0], w[a][1], w[a][2]);
                }
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    w[a][0] = w_el[a][0];
                    w[a][1] = w_el[a][1];
                    w[a][2] = w_el[a][2];
                }
            }

            const scalar_t un   = ux[i] * ax + uy[i] * ay + uz[i] * az;
            const scalar_t mdot = rho * un;
            const Slot sii = slots[i * 8 + i];

            // The natural-outflow branch, matching boundary_scs_add_residual and
            // boundary_scs_add_jacobian_action. This function used to take nmask and
            // ignore it, so on a do-nothing face the assembled matrix kept the pressure
            // column and the viscous row that the residual drops -- it was not the
            // derivative of anything the solver evaluates. The matrix-free path was
            // unaffected (the action above has always branched), but the assembled
            // matrix feeds the coarse-grid operators, the Vanka patch solves and the
            // block-diagonal preconditioner, so the inconsistency reached the multigrid
            // for every case with an open outlet.
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: the closed-face block without its pressure column.
                // p_bar is data, not an unknown, so d(residual)/dp is zero on this face;
                // the viscous row stays because tau still depends on the velocity.
                hex8_visc_jac_row<Atomic>(mu, ax, ay, az, w, i, slots, values);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
                continue;
            }
            if ((nmask >> f) & 1) {
                // d/du of max(mdot, 0) * u_i: the same velocity block as the closed face
                // where mdot > 0, and nothing where it is not. No pressure column, because
                // the residual has no p_i * a term here; no viscous row, for the same
                // reason. Continuity still carries the true flux and so keeps its row.
                if (mdot > scalar_t(0)) {
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
                }
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
                continue;
            }

            hex8_visc_jac_row<Atomic>(mu, ax, ay, az, w, i, slots, values);

            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 3, ax);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 3, ay);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 3, az);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_jacobian_action(const scalar_t rho, const scalar_t mu, const int isoparam,
                                                         const scalar_t *const SFEM_RESTRICT adj, const scalar_t det, const scalar_t Lx, const scalar_t Ly,
                                                         const scalar_t Lz, const scalar_t *const SFEM_RESTRICT x,
                                                         const scalar_t *const SFEM_RESTRICT y, const scalar_t *const SFEM_RESTRICT z,
                                                         const scalar_t *const SFEM_RESTRICT ux, const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz, const scalar_t *const SFEM_RESTRICT vx,
                                                         const scalar_t *const SFEM_RESTRICT vy, const scalar_t *const SFEM_RESTRICT vz,
                                                         const scalar_t *const SFEM_RESTRICT q, scalar_t *const SFEM_RESTRICT r,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
    scalar_t dgrad_el[9];
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, vx, vy, vz, dgrad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az, dgrad[9];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                cvfem_hex8_grad_at(adj, det, dN, vx, vy, vz, dgrad);
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int c = 0; c < 9; ++c) dgrad[c] = dgrad_el[c];
            }
            scalar_t dtx, dty, dtz;
            cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6], dgrad[7], dgrad[8], ax,
                                ay, az, dtx, dty, dtz);
            const scalar_t mdot  = rho * (ux[i] * ax + uy[i] * ay + uz[i] * az);
            const scalar_t dmdot = rho * (vx[i] * ax + vy[i] * ay + vz[i] * az);
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: as the closed face but without the q[i] * a term,
                // since the pressure on this face is data and the direction cannot move it.
                r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i] - dtx;
                r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i] - dty;
                r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i] - dtz;
                r[i * 4 + 3] += dmdot;
            } else if ((nmask >> f) & 1) {
                // Exact derivative of the natural-outflow residual above. The max(mdot, 0)
                // guard is piecewise linear, so its derivative is dmdot*u_i + mdot*v_i where
                // mdot > 0 and zero where it is not. The kink at mdot == 0 is the same class
                // of non-differentiability the interior upwind switch already has.
                if (mdot > scalar_t(0)) {
                    r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i];
                    r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i];
                    r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i];
                }
                r[i * 4 + 3] += dmdot;
            } else {
                r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i] + q[i] * ax - dtx;
                r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i] + q[i] * ay - dty;
                r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i] + q[i] * az - dtz;
                r[i * 4 + 3] += dmdot;
            }
        }
    }
}

// Nodal pressure gradient, used by the Rhie-Chow mass-flux interpolation. Lives here
// rather than in the solver so the device kernels call the same code.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_grad_scalar(const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                               const scalar_t *const SFEM_RESTRICT p, scalar_t &gx, scalar_t &gy,
                                               scalar_t &gz) {
    scalar_t dr, ds, dt;
    cvfem_hex8_face_diff(p, dr, ds, dt);
    cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, gx, gy, gz);
}

// ---------------------------------------------------------------------------
// Block-Jacobi preconditioner block.
//
// The 4x4 diagonal block is singular for incompressible flow -- the pressure-pressure
// entry is zero, which is the saddle-point structure -- so a plain 4x4 inverse is the
// wrong operation. This mirrors build_block_jacobi in cvfem_hex8_ns_steady.cpp: invert
// the 3x3 velocity sub-block, take the reciprocal of the pressure diagonal, and leave
// the velocity-pressure coupling out.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE bool cvfem_hex8_invert3_vel(
        const scalar_t *const SFEM_RESTRICT a, scalar_t *const SFEM_RESTRICT inv) {
    const scalar_t a00 = a[0], a01 = a[1], a02 = a[2];
    const scalar_t a10 = a[4], a11 = a[5], a12 = a[6];
    const scalar_t a20 = a[8], a21 = a[9], a22 = a[10];
    const scalar_t x0 = a11 * a22, x1 = a12 * a21, x2 = a01 * a12;
    const scalar_t x3 = a01 * a22, x4 = a02 * a11;
    const scalar_t det = a00 * (x0 - x1) + a02 * a10 * a21 - a10 * x3 + a20 * x2 - a20 * x4;
    // Magnitude bounds rather than isfinite(): a classification call can be folded away
    // by fast-math, and this kernel is compiled with -use_fast_math.
    const scalar_t ad = det < scalar_t(0) ? -det : det;
    if (!(ad > scalar_t(1e-30)) || !(ad < scalar_t(1e300))) return false;
    const scalar_t s = scalar_t(1) / det;
    inv[0]  = s * (x0 - x1);
    inv[1]  = s * (a02 * a21 - x3);
    inv[2]  = s * (x2 - x4);
    inv[4]  = s * (-a10 * a22 + a12 * a20);
    inv[5]  = s * (a00 * a22 - a02 * a20);
    inv[6]  = s * (-a00 * a12 + a02 * a10);
    inv[8]  = s * (a10 * a21 - a11 * a20);
    inv[9]  = s * (-a00 * a21 + a01 * a20);
    inv[10] = s * (a00 * a11 - a01 * a10);
    return true;
}

// One node's preconditioner block. `constrained` is the 4 per-field Dirichlet flags.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_block_jacobi_block(
        const scalar_t *const SFEM_RESTRICT blk,
        const unsigned char *const SFEM_RESTRICT constrained,
        scalar_t *const SFEM_RESTRICT inv) {
    for (int i = 0; i < 16; ++i) inv[i] = scalar_t(0);
    const int c0 = constrained ? constrained[0] : 0;
    const int c1 = constrained ? constrained[1] : 0;
    const int c2 = constrained ? constrained[2] : 0;
    const int c3 = constrained ? constrained[3] : 0;

    // Scale of the block, for relative floors below. An absolute threshold cannot express
    // "small compared with this block": a diagonal of 1e-20 passes |d| > 1e-30 and yields an
    // inverse of 1e20, which the smoother then applies to the residual every sweep.
    scalar_t blk_scale = scalar_t(0);
    for (int k = 0; k < 16; ++k) {
        const scalar_t a = blk[k] < scalar_t(0) ? -blk[k] : blk[k];
        if (a > blk_scale) blk_scale = a;
    }
    const scalar_t blk_floor = blk_scale * scalar_t(1e-14);

    if (!(c0 | c1 | c2) && cvfem_hex8_invert3_vel(blk, inv)) {
        // velocity 3x3 inverse written above
    } else {
        for (int f = 0; f < 3; ++f) {
            if (constrained && constrained[f]) {
                inv[f * 4 + f] = scalar_t(1);
            } else {
                const scalar_t d  = blk[f * 4 + f];
                const scalar_t ad = d < scalar_t(0) ? -d : d;
                // Falls back to 1, not to a huge inverse: a degenerate diagonal means this
                // dof gets an unscaled (weak) update rather than an explosive one.
                inv[f * 4 + f] =
                        (ad > blk_floor && ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
            }
        }
    }
    if (c3) {
        inv[15] = scalar_t(1);
    } else {
        const scalar_t d  = blk[15];
        const scalar_t ad = d < scalar_t(0) ? -d : d;
        inv[15] = (ad > blk_floor && ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
    }
}

#endif  // CVFEM_HEX8_BOUNDARY_SCS_HPP
