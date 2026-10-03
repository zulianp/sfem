// THE LANE LOOP MUST VECTORISE, AND THIS IS WHAT SAYS SO.
//
// Every lane-blocked kernel in this spike is built on one shape: a loop over
// CVFEM_HEX8_VEC_SIZE lanes whose body is the flux for one sub-control surface. The whole
// design rests on that loop becoming vector instructions, and the history of this tree is
// largely the history of it quietly not doing so -- a call the compiler declined to inline, a
// ternary inside a ternary, a guard on the wrong operand. Each cost a measured factor and each
// was invisible at the source level.
//
// WHY THIS IS NOT A COMPILER REMARK. The obvious gate is
// -Rpass-missed=loop-vectorize with -Werror=pass-failed, and it does not work here. The lane
// loop has a COMPILE-TIME CONSTANT trip count, so the loop vectoriser never engages: the loop
// is fully unrolled and the straight-line body is then vectorised by the SLP pass instead.
// Measured on this machine with a loop of exactly this shape, -Rpass-missed=loop-vectorize is
// silent whether the result is scalar or vector, so a gate built on it can never fire. What
// does report is -Rpass=slp-vectorizer, and what is unambiguous is the emitted code.
//
// So the gate reads the object: each kernel below is called through a noinline wrapper with
// external linkage, which gives it its own symbol, and the script beside this file checks that
// the instructions between that symbol and the next contain vector registers. That is the same
// thing the generator's own notes record doing by hand -- "measured on the emitted object, the
// Venkatakrishnan kernel held 6000 instructions and not one NEON register".
// scalar_t is smesh's, but kernels/ is required to carry no library dependency and in fact
// includes nothing of smesh -- it uses the name and leaves it to the includer. Spelling it here
// rather than pulling in smesh_config.hpp is therefore part of what this gate checks: that the
// lane-blocked kernels compile against nothing but the standard library.
using scalar_t = double;

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"

// NOTHING HERE MAY BE KNOWN TO THE COMPILER, AND ONLY ONE KERNEL IS IN THE OBJECT.
//
// Two things a first version of this file got wrong, both of which made the measurement read
// something other than the kernel. The packs were zero-initialised globals, and the entry stub
// came out as "movi d0, #0" -- constant propagation had eaten the body and the gate was
// inspecting the folded remains. So every input now arrives as a parameter of an
// external-linkage wrapper, which is opaque at this translation unit's boundary.
//
// And all three wrappers in one object were tail-merged into a single shared body reached by
// three stubs, so a scalar kernel would have hidden behind its neighbours' vector instructions.
// CVFEM_VEC_GATE_KERNEL therefore selects exactly one wrapper per compilation, and the build
// compiles this file once per kernel. One kernel per object is what gives the check its
// resolution.
#ifndef CVFEM_VEC_GATE_KERNEL
#error "define CVFEM_VEC_GATE_KERNEL to the kernel under test"
#endif

#define CVFEM_VEC_GATE_HEAD                                                                   \
    const scalar_t rho, const scalar_t mu, const scalar_t *SFEM_RESTRICT c0,                  \
            const scalar_t *SFEM_RESTRICT c1, const scalar_t *SFEM_RESTRICT c2,               \
            const scalar_t *SFEM_RESTRICT c3, const scalar_t *SFEM_RESTRICT c4,               \
            const scalar_t *SFEM_RESTRICT c5, const scalar_t *SFEM_RESTRICT c6,               \
            const scalar_t *SFEM_RESTRICT c7, const scalar_t *SFEM_RESTRICT c8,               \
            const scalar_t *SFEM_RESTRICT det, const Hex8InputPack &in, Hex8ResidualPack &out

#define CVFEM_VEC_GATE_ARGS rho, mu, c0, c1, c2, c3, c4, c5, c6, c7, c8, det, in, out

#define CVFEM_VEC_GATE_ENTRY extern "C" __attribute__((noinline)) void cvfem_vecgate_kernel

// 1 -- the first-order residual, the kernel the standing performance gate measures.
#if CVFEM_VEC_GATE_KERNEL == 1
CVFEM_VEC_GATE_ENTRY(CVFEM_VEC_GATE_HEAD) {
    cvfem_hex8_ns_upwind_residual_sumfact_simd(CVFEM_VEC_GATE_ARGS);
}

// 2 -- the same with Rhie-Chow, which this discretisation never runs without.
#elif CVFEM_VEC_GATE_KERNEL == 2
CVFEM_VEC_GATE_ENTRY(CVFEM_VEC_GATE_HEAD, const Hex8RhieChowPack *rc) {
    cvfem_hex8_ns_upwind_residual_sumfact_simd(CVFEM_VEC_GATE_ARGS, rc, scalar_t(1));
}

// 3 -- the higher-order arm. This is where a limiter has stopped the loop before: the generator's
// own notes record the Venkatakrishnan kernel emitting 6000 instructions and not one NEON
// register.
#elif CVFEM_VEC_GATE_KERNEL == 3
CVFEM_VEC_GATE_ENTRY(CVFEM_VEC_GATE_HEAD, const Hex8RhieChowPack *rc, const Hex8UGradPack *ho) {
    cvfem_hex8_ns_upwind_residual_sumfact_simd(CVFEM_VEC_GATE_ARGS, rc, scalar_t(1), scalar_t(0),
                                               ho);
}

// 10..13 -- THE LANE KERNEL ITSELF, ONE LIMITER PER OBJECT.
//
// Kernel 3 above enters through the sweep, which dispatches on the runtime ho->limiter and so
// inlines all four limiters into one object. That dilutes the measurement: a limiter whose lane
// loop went scalar would be averaged against three that did not. These entries bind LIM as a
// template argument and call the lane kernel directly, which is both the sharpest resolution
// available and the loop DESIGN.md is actually talking about.
#elif CVFEM_VEC_GATE_KERNEL >= 10 && CVFEM_VEC_GATE_KERNEL <= 13
CVFEM_VEC_GATE_ENTRY(const scalar_t rho,
                     const scalar_t                      mu,
                     const scalar_t                      rc_scale,
                     const scalar_t                      half,
                     const scalar_t *const SFEM_RESTRICT Ax0,
                     const scalar_t *const SFEM_RESTRICT Ay0,
                     const scalar_t *const SFEM_RESTRICT Az0,
                     const scalar_t *const SFEM_RESTRICT Ax1,
                     const scalar_t *const SFEM_RESTRICT Ay1,
                     const scalar_t *const SFEM_RESTRICT Az1,
                     const scalar_t *const SFEM_RESTRICT Ax2,
                     const scalar_t *const SFEM_RESTRICT Ay2,
                     const scalar_t *const SFEM_RESTRICT Az2,
                     const Hex8InputPack                &in,
                     const Hex8RhieChowPack             *rc,
                     Hex8ResidualPack                   &out,
                     const scalar_t                      ueps,
                     const Hex8UGradPack *const          ho,
                     const scalar_t *const SFEM_RESTRICT cenx,
                     const scalar_t *const SFEM_RESTRICT ceny,
                     const scalar_t *const SFEM_RESTRICT cenz,
                     const scalar_t *const SFEM_RESTRICT edx,
                     const scalar_t *const SFEM_RESTRICT edy,
                     const scalar_t *const SFEM_RESTRICT edz) {
    cvfem_hex8_conv_all_simd<true, false, true, CVFEM_VEC_GATE_KERNEL - 10>(rho,
                                                                           mu,
                                                                           rc_scale,
                                                                           half,
                                                                           Ax0,
                                                                           Ay0,
                                                                           Az0,
                                                                           Ax1,
                                                                           Ay1,
                                                                           Az1,
                                                                           Ax2,
                                                                           Ay2,
                                                                           Az2,
                                                                           in,
                                                                           rc,
                                                                           out,
                                                                           ueps,
                                                                           ho,
                                                                           cenx,
                                                                           ceny,
                                                                           cenz,
                                                                           edx,
                                                                           edy,
                                                                           edz);
}

#else
#error "unknown CVFEM_VEC_GATE_KERNEL"
#endif
