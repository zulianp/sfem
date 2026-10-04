#ifndef CVFEM_KERNEL_KIND_HPP
#define CVFEM_KERNEL_KIND_HPP

// WHICH MICRO-KERNEL A SWEEP RUNS, AS A TEMPLATE ARGUMENT.
//
// This is --kernel, a user-level option, and DESIGN.md says such a flag is "handled outside in
// the front-end" and not propagated into src/kernels/. It was propagated: the sweeps took a
// KernelKind and tested it INSIDE their element and pack loops, six ways in two of them. That is
// the guard shape this tree's own notes record costing 1.83x, and the vectorisation gate refuses
// it outright one loop further in.
//
// The enum itself belongs here, because what a sweep is templated on has to be nameable here --
// the same arrangement as the ISO geometry flag and the LIM limiter. What stays in the front end
// is everything that turns a string into one of these values: parse_kernel, kernel_is_valid and
// the predicates beside them.
//
// Dispatch follows the pattern the higher-order lane kernels already use for LIM: a switch in
// the caller selecting among template instantiations, listing only the values that sweep
// supports, so the 13 values here do not become 13 instantiations of every body.
enum class KernelKind {
    Current,
    Fd,
    Sumfact,
    Sympy,
    SympyBlock,
    SympyRow,
    SympyFace,
    // Assembly only: rebuild just the velocity-dependent terms, reusing a viscous part
    // assembled once. See assemble_jacobian_atomic_{linear,nonlinear}.
    Split,
    // Jacobian action only. Generated, and differing from each other solely in the scope
    // one sp.cse call was given: all 32 outputs at once, the four dofs of a node, or one
    // component across the eight nodes. The action had no generated form at all until
    // these, so this axis has never been measured for it.
    SympyAction,
    SympyActionNode,
    SympyActionComp,
    // The finest cut: one sub-control surface per scope. Face-wise lost badly as an
    // ASSEMBLY arrangement, but for a reason that does not exist here -- it issued
    // 2016 atomic adds against flat's 768, and the action accumulates into a local.
    SympyActionFace,
    // Two-level: the geometry-only subexpressions factored in their own pass, then the
    // field algebra with the geometry reduced to atoms. Emitted flat and face-wise so
    // the hoist can be read on its own and on top of the best arrangement -- cutting
    // the scope loses cross-scope reuse, and hoisting is what gives the shared
    // adjugate back.
    SympyActionGeom,
    SympyActionGeomFace
};

// Which variants are the generated residual. constexpr, because the sweeps test it on their
// template argument -- `if constexpr (kernel_uses_sympy_residual(K))` -- rather than per pack.
static constexpr bool kernel_uses_sympy_residual(const KernelKind k) {
    return k == KernelKind::Sympy || k == KernelKind::SympyBlock || k == KernelKind::SympyRow || k == KernelKind::SympyFace;
}

#endif  // CVFEM_KERNEL_KIND_HPP
