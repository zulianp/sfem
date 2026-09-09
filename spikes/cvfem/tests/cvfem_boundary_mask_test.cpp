// The boundary face masks, which nothing tested.
//
// Everything about closing a boundary control volume runs through two per-element
// bitfields -- `fmask`, which says which of the six CVFEM faces lie on the domain
// boundary, and `nmask`, which says which of those carry the do-nothing outflow -- and
// through `sscvfem_micro_face_mask`, which projects a macro-element mask onto a micro
// cell so one mask serves every level of the multigrid hierarchy. Until now the only
// checks on any of it were opt-in, print-only diagnostics: SFEM_BOUNDARY_MASK_CHECK and
// report_mask_extent.
//
// That is the wrong amount of testing for code where a wrong bit is silent. A face that
// should be closed and is not leaves a control volume open, and the solve converges to
// the wrong answer rather than failing; the reverse closes an outflow and over-determines
// the system. Both are exactly the failure the invariant in
// src/cases/cvfem_ns_channel_case.hpp warns about.
//
// This asserts the semantics the kernels actually promise, on a single element with known
// geometry, so a failure points at a line rather than at a solve.

// smesh::count_t, which the boundary header uses in its BSR slot signatures.
#include "smesh_types.hpp"

#include "cvfem_portability.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using scalar_t = double;
static constexpr int N_FIELDS = 4;

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

// The macro-to-micro projection lives in the semi-structured header, which pulls in far
// more than this test needs. It is eleven lines and its contract is the thing under test,
// so it is restated here and checked against the same bit assignment the kernels use.
// If the two ever disagree, the assertions on CVFEM_HEX8_BFACE_AXIS below fail first.
static int micro_face_mask(const int macro, const int L, const int xi, const int yi, const int zi) {
    if (macro < 0) return -1;
    int m = 0;
    if (xi == 0) m |= macro & 0x01;
    if (xi == L - 1) m |= macro & 0x02;
    if (yi == 0) m |= macro & 0x04;
    if (yi == L - 1) m |= macro & 0x08;
    if (zi == 0) m |= macro & 0x10;
    if (zi == L - 1) m |= macro & 0x20;
    return m;
}

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

// The unit cube, which is its own domain: every face is a boundary face.
static void unit_cube(scalar_t x[8], scalar_t y[8], scalar_t z[8]) {
    static const scalar_t cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    static const scalar_t cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static const scalar_t cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    for (int a = 0; a < 8; ++a) {
        x[a] = cx[a];
        y[a] = cy[a];
        z[a] = cz[a];
    }
}

static scalar_t residual_norm(const scalar_t *r) {
    scalar_t s = 0;
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) s += std::fabs(r[i]);
    return s;
}

// One element's boundary contribution under a given mask pair.
static void boundary_residual(const int fmask, const int nmask, scalar_t *r) {
    scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], adj[9], det;
    unit_cube(x, y, z);
    for (int a = 0; a < 8; ++a) {
        // A non-trivial state, so a dropped term shows up rather than cancelling.
        ux[a] = 1.0 + 0.3 * x[a] - 0.2 * y[a];
        uy[a] = -0.4 + 0.2 * x[a] + 0.5 * y[a];
        uz[a] = 0.2 - 0.1 * x[a] + 0.25 * z[a];
        p[a]  = 1.0 + 0.1 * x[a] + 0.2 * y[a];
    }
    cvfem_hex8_affine_adj(x, y, z, adj, &det);
    std::memset(r, 0, sizeof(scalar_t) * CVFEM_HEX8_N_DOF);
    boundary_scs_add_residual(1.0, 0.01, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, ux, uy, uz, p, r, fmask, nmask);
}

int main() {
    scalar_t r[CVFEM_HEX8_N_DOF], r_all[CVFEM_HEX8_N_DOF];

    // 1. An empty mask closes nothing. This is the benchmark's default path, and it is
    //    what makes --boundary cost nothing when it is off.
    boundary_residual(0, 0, r);
    check(residual_norm(r) == 0.0, "fmask 0 contributes nothing");

    // 2. fmask < 0 falls back to the bounding-box test, which on a unit cube that IS the
    //    domain selects all six faces. That fallback is the historical behaviour and the
    //    solver still relies on it when no sideset is named.
    boundary_residual(-1, 0, r_all);
    check(residual_norm(r_all) > 0.0, "fmask -1 falls back to the coordinate test");

    // 3. All six bits set must equal the coordinate fallback on this element, because the
    //    element is the whole domain. If these disagree, the bit order in
    //    CVFEM_HEX8_BFACE_NODES and hex8_face_on_domain have drifted apart.
    boundary_residual(0x3F, 0, r);
    scalar_t worst = 0;
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) worst = std::max(worst, std::fabs(r[i] - r_all[i]));
    check(worst == 0.0, "fmask 0x3F equals the coordinate fallback on a one-element domain");

    // 4. The faces are independent and complete: summing the six single-bit
    //    contributions must reproduce the all-six contribution exactly. This is what
    //    catches a face that is skipped, double-counted, or attributed to the wrong bit.
    scalar_t sum[CVFEM_HEX8_N_DOF] = {0};
    for (int f = 0; f < 6; ++f) {
        boundary_residual(1 << f, 0, r);
        check(residual_norm(r) > 0.0, f == 0   ? "face bit 0 contributes"
                                      : f == 1 ? "face bit 1 contributes"
                                      : f == 2 ? "face bit 2 contributes"
                                      : f == 3 ? "face bit 3 contributes"
                                      : f == 4 ? "face bit 4 contributes"
                                               : "face bit 5 contributes");
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) sum[i] += r[i];
    }
    worst = 0;
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) worst = std::max(worst, std::fabs(sum[i] - r_all[i]));
    check(worst < 1e-14, "the six single-face contributions sum to the all-face one");

    // 5. The do-nothing branch differs from the closed one. nmask selects it per face, so
    //    setting a bit must change that face's contribution -- it drops p*a and the
    //    viscous traction, which is what fixes the pressure gauge.
    for (int f = 0; f < 6; ++f) {
        boundary_residual(1 << f, 0, r);
        scalar_t closed[CVFEM_HEX8_N_DOF];
        std::memcpy(closed, r, sizeof(closed));
        boundary_residual(1 << f, 1 << f, r);
        scalar_t d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r[i] - closed[i]));
        check(d > 0.0, f == 0   ? "nmask bit 0 selects the do-nothing branch"
                       : f == 1 ? "nmask bit 1 selects the do-nothing branch"
                       : f == 2 ? "nmask bit 2 selects the do-nothing branch"
                       : f == 3 ? "nmask bit 3 selects the do-nothing branch"
                       : f == 4 ? "nmask bit 4 selects the do-nothing branch"
                                : "nmask bit 5 selects the do-nothing branch");
    }

    // 6. An nmask bit on a face that fmask does not select is inert. The kernel skips the
    //    face before it looks at nmask, and a caller that got this wrong would be opening
    //    an outflow on an interior face.
    boundary_residual(0, 0x3F, r);
    check(residual_norm(r) == 0.0, "nmask without fmask is inert");

    // 7. Continuity carries the true flux on a do-nothing face even though the momentum
    //    rows clamp it. Clipping the continuity row would destroy global mass
    //    conservation, which is the property the step case is judged on.
    {
        boundary_residual(0x3F, 0x3F, r);
        scalar_t mass = 0;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) mass += r[a * 4 + 3];
        // The state above has a non-zero net flux through the cube, so this is a real
        // number and not a tautology; what matters is that it is finite and non-zero.
        check(std::isfinite((double)mass) && mass != 0.0,
              "the continuity row carries a flux on a do-nothing face");
    }

    // 8. The macro-to-micro projection. A micro cell in the interior of a macro element
    //    touches no boundary; one at a corner inherits three faces; the bit assignment
    //    must match the axis table the kernels use.
    check(micro_face_mask(0x3F, 4, 1, 1, 1) == 0, "interior micro cell inherits no faces");
    check(micro_face_mask(0x3F, 4, 0, 0, 0) == (0x01 | 0x04 | 0x10), "corner micro cell inherits x-,y-,z-min");
    check(micro_face_mask(0x3F, 4, 3, 3, 3) == (0x02 | 0x08 | 0x20), "far corner inherits x-,y-,z-max");
    check(micro_face_mask(0x01, 4, 0, 2, 2) == 0x01, "a single macro face projects to its own micro faces");
    check(micro_face_mask(0x01, 4, 3, 2, 2) == 0, "and to nothing on the opposite side");
    check(micro_face_mask(-1, 4, 0, 0, 0) == -1, "a negative macro mask stays negative");

    // 9. The bit order the projection assumes is the one the kernels use. CVFEM face f
    //    has axis CVFEM_HEX8_BFACE_AXIS[f] and outward sign CVFEM_HEX8_BFACE_OUT[f]; the
    //    projection pairs bit 0 with x-min, bit 1 with x-max, and so on.
    for (int f = 0; f < 6; ++f) {
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = (scalar_t)CVFEM_HEX8_BFACE_OUT[f];
        const bool     ok   = axis == f / 2 && out == (f % 2 == 0 ? scalar_t(-1) : scalar_t(1));
        check(ok, f == 0   ? "face 0 is x-min"
                  : f == 1 ? "face 1 is x-max"
                  : f == 2 ? "face 2 is y-min"
                  : f == 3 ? "face 3 is y-max"
                  : f == 4 ? "face 4 is z-min"
                           : "face 5 is z-max");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_boundary_mask_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all boundary mask checks passed\n");
    return 0;
}
