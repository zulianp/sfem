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

int main() {
    const int bad = instantiate<float>() + instantiate<double>();
    std::printf("cvfem_mixed_precision_packs: f32 lanes=%d  f64 lanes=%d  "
                "lane group=%zu bytes at both\n",
                cvfem_hex8_vec_size<float>,
                cvfem_hex8_vec_size<double>,
                sizeof(Hex8InputPackT<float>) / CVFEM_HEX8_N_NODES / 4);
    if (bad) std::fprintf(stderr, "a pack did not zero-initialise\n");
    return bad;
}
