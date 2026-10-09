// THE LANE-BLOCKED PACKS INSTANTIATE AT SINGLE PRECISION, AND GET TWICE THE LANES.
//
// DESIGN.md's correction: "the kernels should be templated as well. They should support
// different types for the computation, template scalar_t, geom_t, idx_t, etc... (in a short
// time we would like to try single precision kernels as well)."
//
// The lane width was the obstacle, and it is the thing this test exists to hold. It used to be
// `VEC_BYTES / sizeof(scalar_t)` at NAMESPACE scope -- one number per build -- so a sweep
// instantiated at `float` would have kept the width computed for `double`: sixteen lanes' worth
// of work in an eight-lane pack, reading past the end of every staged array. Nothing would have
// failed to compile. The width now travels with the type as
// `cvfem_hex8_vec_size<S>`, and this checks that it does.
//
// A COMPILE GATE, because what it protects is an instantiation. The leaf element kernels have
// been templated on the scalar for a long time and the CUDA smoke test instantiates them at
// both widths; the packs are what the SWEEPS carry, and until they were templates the sweeps
// could not be. Keeping an f32 instantiation in the default build is what stops the templating
// from decaying back into decoration -- a `CVFEM_HEX8_VEC_SIZE` written into a pack by habit
// compiles fine and silently re-binds that pack to the build's scalar.
//
// It also asserts the RATIO rather than the two numbers alone: 32 and 16 are facts about
// VEC_BYTES=128, but "f32 gets twice the lanes" is the property that makes single precision
// worth trying, and it holds whatever VEC_BYTES is set to.
#include "smesh_types.hpp"
#include "kernels/cvfem_portability.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "support/cvfem_default_types.hpp"
using idx_t   = smesh::idx_t;
using count_t = smesh::count_t;

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"

static_assert(cvfem_hex8_vec_size<float> == 2 * cvfem_hex8_vec_size<double>,
              "a single-precision lane group must hold twice as many elements");
static_assert(cvfem_hex8_vec_size<double> >= 1, "invalid double-precision lane width");

// ONE LANE GROUP IS THE SAME NUMBER OF BYTES AT EITHER PRECISION. That is what lane blocking
// means here -- a fixed byte width per group, not a fixed element count -- and it is the
// property an f32 sweep is for: the same traffic carries twice the elements.
static_assert(sizeof(Hex8InputPackT<float>) == sizeof(Hex8InputPackT<double>),
              "a lane group changed size with the scalar type");
static_assert(sizeof(Hex8ResidualPackT<float>) == sizeof(Hex8ResidualPackT<double>), "");
static_assert(sizeof(Hex8CoordPackT<float>) == sizeof(Hex8CoordPackT<double>), "");

// Every lane-blocked pack, instantiated. A pack that still spells CVFEM_HEX8_VEC_SIZE compiles
// but binds to the build's scalar, so its f32 instantiation would come out the f64 size and the
// assertions above would fail.
template <typename S>
static int instantiate() {
    Hex8InputPackT<S>    in{};
    Hex8ResidualPackT<S> out{};
    Hex8CoordPackT<S>    xyz{};
    Hex8UGradPackT<S>    hop{};
    Hex8RhieChowPackT<S> rcp{};
    // Touch one element of each, so none of them is optimised away before it has been checked.
    return (in.ux[0][0] == S(0) && out.rx[0][0] == S(0) && xyz.x[0][0] == S(0) &&
            hop.g[0][0][0] == S(0) && rcp.pgx[0][0] == S(0))
                   ? 0
                   : 1;
}

// AND THE LANE-BLOCKED KERNELS THAT TAKE THEM, COMPARED ACROSS THE TWO PRECISIONS.
//
// The packs being templates is necessary but not sufficient: a kernel still spelling the plain
// pack alias in its signature binds to the build's scalar and cannot be called from an f32
// sweep, and a kernel still INDEXING with the build's CVFEM_HEX8_VEC_SIZE compiles and then
// walks a 16-element stride through a 32-lane pack. Neither is a compile error.
//
// WHY A CROSS-PRECISION COMPARISON rather than a check that every lane was written. The weaker
// check was tried and is too weak: a lane is written by several kernels in sequence, so a short
// stride in one of them -- the viscous term, say -- is covered up by the convective term that
// writes the same lane afterwards. Comparing the f32 answer against the f64 one catches a short
// stride wherever it is, because the lanes the broken kernel skipped then carry a different
// value rather than no value.
//
// The input is LANE-INVARIANT: every lane of every pack holds the same element. That makes the
// two precisions directly comparable despite having different lane counts, and it adds a second
// property worth having -- every lane of the output must agree with lane 0, so a kernel that
// mixes lanes is caught too.
//
// The tolerance is 1e-5 relative, which is loose for f32's ~1e-7 and is deliberately not a
// claim about the accuracy of single precision for this operator. What that costs on a real
// mesh is a question for a verification case; this is a structural check.
template <typename S>
static void run_kernels(S out_rx[CVFEM_HEX8_N_NODES], S out_rc[CVFEM_HEX8_N_NODES], int &lane_mix) {
    constexpr int W = cvfem_hex8_vec_size<S>;
    alignas(ALIGN_BYTES) S cof[9][W] = {};
    alignas(ALIGN_BYTES) S det[W]    = {};
    Hex8InputPackT<S>    in{}, du{};
    Hex8ResidualPackT<S> out{};
    for (int l = 0; l < W; ++l) {
        det[l]    = S(1);
        cof[0][l] = cof[4][l] = cof[8][l] = S(1);
    }
    // Lane-invariant, and off the upwind switch: a sheared velocity with a linear pressure. A
    // zero state sits exactly on the switch's non-differentiable point and returns NaN at both
    // precisions, which is correct behaviour and a poor oracle.
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
        for (int l = 0; l < W; ++l) {
            const S t   = S(1) + S(a) / S(16);
            in.ux[a][l] = t;
            in.uy[a][l] = S(0.5) * t;
            in.uz[a][l] = S(0.25) * t;
            in.p[a][l]  = S(0.1) * S(a);
            du.ux[a][l] = S(0.3) * t;
            du.p[a][l]  = S(0.02) * S(a);
        }

    const S POISON = S(-12345);
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
        for (int l = 0; l < W; ++l) out.rx[a][l] = out.rc[a][l] = POISON;

    cvfem_hex8_ns_upwind_residual_sumfact_simd<S>(
            S(1), S(1), cof[0], cof[1], cof[2], cof[3], cof[4], cof[5], cof[6], cof[7], cof[8],
            det, in, out);

    lane_mix = 0;
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        out_rx[a] = out.rx[a][0];
        out_rc[a] = out.rc[a][0];
        for (int l = 1; l < W; ++l)
            if (out.rx[a][l] != out.rx[a][0] || out.rc[a][l] != out.rc[a][0]) lane_mix = l;
    }
}

static int compare_precisions() {
    double rx64[CVFEM_HEX8_N_NODES], rc64[CVFEM_HEX8_N_NODES];
    float  rx32[CVFEM_HEX8_N_NODES], rc32[CVFEM_HEX8_N_NODES];
    int    mix64 = 0, mix32 = 0;
    run_kernels<double>(rx64, rc64, mix64);
    run_kernels<float>(rx32, rc32, mix32);

    int bad = 0;
    if (mix64 || mix32) {
        std::fprintf(stderr,
                     "  lanes of one node disagree (f64 at lane %d of %d, f32 at lane %d of %d) "
                     "-- the input is lane-invariant, so a kernel is crossing lanes or reading "
                     "past its stride\n",
                     mix64, cvfem_hex8_vec_size<double>, mix32, cvfem_hex8_vec_size<float>);
        bad += 1;
    }
    double worst = 0;
    int    worst_node = -1;
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
        for (int c = 0; c < 2; ++c) {
            const double r = c ? rc64[a] : rx64[a];
            const double m = c ? (double)rc32[a] : (double)rx32[a];
            const double d = std::fabs(m - r) / (std::fabs(r) > 1e-30 ? std::fabs(r) : 1.0);
            if (d > worst) { worst = d; worst_node = a; }
        }
    if (!(worst < 1.0e-5)) {
        std::fprintf(stderr,
                     "  f32 and f64 disagree by %.3e relative at node %d (bound 1e-5) -- at this "
                     "tolerance that is a structural fault, not round-off\n",
                     worst, worst_node);
        bad += 1;
    } else {
        std::printf("cvfem_mixed_precision_packs: f32 agrees with f64 to %.2e relative "
                    "over %d and %d lanes\n",
                    worst, cvfem_hex8_vec_size<float>, cvfem_hex8_vec_size<double>);
    }
    return bad;
}

int main() {
    const int bad = instantiate<float>() + instantiate<double>() + compare_precisions();
    std::printf("cvfem_mixed_precision_packs: f32 lanes=%d  f64 lanes=%d  "
                "lane group=%zu bytes at both\n",
                cvfem_hex8_vec_size<float>,
                cvfem_hex8_vec_size<double>,
                sizeof(Hex8InputPackT<float>) / CVFEM_HEX8_N_NODES / 4);
    if (bad) std::fprintf(stderr, "cvfem_mixed_precision_packs: %d check(s) failed\n", bad);
    return bad;
}
