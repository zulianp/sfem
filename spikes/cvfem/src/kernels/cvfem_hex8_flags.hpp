#ifndef CVFEM_HEX8_FLAGS_HPP
#define CVFEM_HEX8_FLAGS_HPP

// THE DECISIONS A SWEEP IS HANDED, AND NOTHING THAT MAKES THEM.
//
// Hex8RcConfig and Hex8Extras are what a sweep reads to know which optional terms are on and
// what time scale the Rhie-Chow coefficient was built with. They are plain values -- three ints
// and two scalars -- and they appeared in kernel signatures, which is how a staging header
// ended up named there: Hex8Extras had a constructor taking MeshData, so the type could not be
// mentioned without the mesh.
//
// The split is the one DESIGN.md asks for. The values live here, in src/kernels/, because the
// kernels consume them. Resolving them from a mesh is a front-end job and stays in the staging
// headers, as cvfem_hex8_rc_config_for and cvfem_hex8_extras_of -- which is also the honest
// shape, since both are resolved once per solve and neither is a per-element quantity.
//
// Hex8RcTau comes from the microkernel header, which owns it because the face kernel reads it.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"

// The Rhie-Chow time scale's configuration, resolved once per solve.
//
// One definition for every path -- flat, packed, semi-structured, benchmark -- so none of them
// can end up evaluating a different time scale than the others. Each caller supplies a0/dt from
// its own transient history, which is the only part that differs.
struct Hex8RcConfig {
    Hex8RcTau tau;
    scalar_t  scale{0};
};

// Which optional terms this sweep is running.
//
// Both Rhie-Chow and the boundary closure are off by default, and the default path must stay
// exactly as fast as it was -- the throughput gate compares against numbers measured without
// them. So the gathers they need sit behind a flag hoisted out of the element loop rather than
// being done unconditionally.
//
// With both off, the Rhie-Chow pack keeps its null pointers, cvfem_hex8_rhie_chow_active() is
// false and the element kernel's branch folds away, fmask stays 0 and the caller skips the
// boundary term entirely. Nothing is gathered and nothing is written.
struct Hex8Extras {
    int with_rc{0};
    int with_bnd{0};
    // The exact Rhie-Chow Jacobian, which differentiates through the nodal gradient
    // reconstruction as well. Only the Jacobian action fills qgx/qgy/qgz, so this is off
    // wherever they are empty and the kernel falls back to the frozen-gradient form.
    int with_qg{0};
    // What the coefficient table was built with. Resolved once, not per element.
    Hex8RcConfig rcfg{};
};

#endif  // CVFEM_HEX8_FLAGS_HPP
