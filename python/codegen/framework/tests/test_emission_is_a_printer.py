"""How far emission still is from being a printer.

The prescribed architecture says L6 prints what the layers above it decided.
The honest measure of that is not how many lines the emitters have -- they
grew through the whole IR migration, because a node constructor is longer
than the string literal it replaces -- but how many *decisions* they still
make.

A decision is a branch whose test reads planning-layer input: the lowered
form's dependency set, the element specialization, the system, a rule, or a
plan.  Those branches are emission choosing what to emit rather than how to
spell it, and each one that leaves is a decision that moved to where it
belongs.

The count only falls by relocating a decision, not by tidying.  When
``local_kernel_stream_plans`` took over the kernel signature it fell; when
the call arguments started reading the same plan instead of re-deriving the
identical conditions it fell again.  Neither made the emitters smaller.
"""

import ast
import collections
import os
import unittest


EMITTERS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emitters"
)

#: Names that identify planning-layer input inside a branch test.
PLAN_INPUTS = (
    "dependencies",
    "system",
    "specialization",
    "rule",
    "plan",
    "basis_family",
    "geometry_family",
    "form",
    "layout",
    "formats",
)

#: Decisions still made in the emission layer.  Shrink-only: lower it when a
#: decision moves, and never raise it to make a change fit.
#:
#: Raised once, 298 -> 412, to correct the measure rather than to accommodate a
#: change.  The first matcher looked for ``'dependencies'`` quoted inside the
#: AST dump, so every qualified name missed: ``state_dependencies`` is a
#: dependency set, ``quadrature_rule`` is a rule and ``matrix_format_plan`` is a
#: plan, and all 130 such branches read as not-a-decision.  Nothing moved into
#: emission to cause the rise.
#:
#: 199 -> 194: five copies of one ternary deciding a kernel's gather order left
#: `emitters/residual_codegen.py` for `plans.layout.gather_shape_order`.  They
#: were five copies of the same four lines and four of them were wrong in the
#: same way -- they asked whether the element was a Cartesian *hex* and never
#: whether it was a Cartesian quad -- which is the argument for the rule this
#: budget enforces: a decision duplicated across emission sites is a decision
#: that can disagree with itself.
#:
#: 194 -> 189: the same shape again, and the largest duplicated decision left in
#: `energy_codegen.py`.  Five sites spelled `"+=" if form.output_mode ==
#: "accumulate" else "="` and a sixth spelled its traffic consequence -- a
#: kernel that accumulates reads its output before writing it -- separately.
#: `plans.form_emission.output_assignment` and `output_is_accumulated` state it
#: once, so the body and the diagnostics record cannot disagree about whether
#: the kernel reads its output.
#:
#: 189 -> 183: `writes_per_shape(form)` was the largest remaining cluster, and
#: seven of its sixteen sites were one question -- what a mesh kernel writes.
#: `plans.form_emission.mesh_output_shape` states it: the element stride a
#: scalar output does not carry, the staging buffer's name, and its extents
#: inside the lane.  The parameters, the buffer, the zero fill and the call
#: argument read that record instead of asking again.
#:
#: The zero fill is the one worth naming.  It was two texts -- a lane loop for
#: the scalar, a stream loop around a lane loop for the per-shape output -- and
#: it is now one text at two depths, because the plan says how many extents
#: there are and emission opens a loop per extent.  An empty sequence emits the
#: inner loop alone, which is what the scalar branch used to spell.
#:
#: What it also found: the six sites shaping the buffer asked
#: `writes_per_shape`, while the one naming the streams asked a wider question,
#: and for a graph-lowered form of non-zero order with a single output the two
#: disagree.  `mesh_output_streams` keeps that behaviour and says so rather than
#: collapsing it, because collapsing it would change what is emitted.
#:
#: Measured afterwards rather than argued: instrumenting that site across a full
#: regeneration of all eight materials logged eighty-four calls, every one of
#: them carrying a weak form.  Repeating the probe under this whole suite, which
#: builds configurations the generators do not, found no disagreement either.
#: The wider question's second answer was never once computed, so it could only
#: ever have diverged silently.  The two questions are one question now and the
#: branch is gone.  The budget does not move for it, because the branch was in
#: `plans/` rather than in an emitter -- which is the one thing this ratchet
#: cannot see, and the reason the collapse had to be gated on byte-identity
#: instead.
#:
#: 183 -> 179: the other question `writes_per_shape` conflates.  A 0-form
#: weights its energy density into one accumulator per element; a 1- or 2-form
#: forms the loperand and contracts it onto the test functions.  Two body
#: shapes, named as `plans.form_emission.FormAccumulation` beside the
#: `FormContraction` that already said *how* the contraction is reached.
#:
#: Four sites map their spelling to it instead of branching: the block
#: signature's output parameter, the per-point loperand buffer, the per-component
#: loperand scalars, and -- the one that matters most -- which expressions the
#: diagnostics cost model counts.  That last one has to be the expressions the
#: body actually evaluates, and it was asking the question separately from the
#: body emitters, so a record counting the wrong half would have been wrong in a
#: way nothing compiles against.
#:
#: Three sites remain in this cluster and are not table lookups: each selects
#: the whole remainder of a body emitter, so moving them means splitting three
#: functions rather than mapping a value.
#:
#: 179 -> 176: those three, split.  Each was an early return in the middle of a
#: weak-form body emitter -- the 0-form accumulated its density and stopped, the
#: rest of the function was the 1- and 2-form's loperand and contraction -- so
#: the choice was made where it could not be named.  Each emitter now has two
#: tails with one signature and a table keyed on `form_accumulation`, which is
#: the shape `_BLOCK_FUNCTION_BY_CONTRACTION` had already established beside it.
#:
#: The split is what showed how much the tails close over: the tensor-product
#: one needs nothing beyond the step's own arguments, the constant-P1 one needs
#: two more and the simplex one three.  That was invisible while they were
#: inline, and it is the measure of how far each body emitter is from being a
#: step that could be moved.
#:
#: 176 -> 172, and `writes_per_shape` is down to **zero** decision branches from
#: the sixteen it began with.  The last four: the mesh operator's output scatter
#: split into the two operations it always was -- a scalar adds its lane into
#: the element's slot, a per-shape output scatters through the connectivity --
#: and two pointer arrays became sequences a scalar leaves empty.
#:
#: The fourth is the one that had hidden furthest.  A stepped objective exists
#: only for a 0-form carrying a weak form, and the emitter said so as a guard
#: returning an empty list five hundred lines above the end of the function it
#: guarded.  `plans.form_emission.publishes_objective_steps` says it in this
#: layer's own words -- the body shape and the contraction, both already named
#: here -- and the emitter looks it up, the way `_KERNEL_BY_APPLICABILITY` does
#: in `inexact_apply_codegen.py`.
#:
#: What remains of `writes_per_shape` in that emitter is a comprehension filter,
#: one table lookup and three pass-throughs: no branch reads it any more.
#:
#: 172 -> 168, and the first of these to come out of `residual_codegen.py`.
#: `plans/geometry_quantities` records in its own docstring that its subject was
#: "asked again as `if dependencies.uses_adjugate:` at some twenty sites"; four
#: were left and these are two of them, plus the two that built the affine
#: geometry stream list.
#:
#: Those last two were the same nested ternary written twice -- the cached
#: metric, or the adjugate when the form needs one, then the determinant -- and
#: it is the sequence `mesh_geometry_streams` already published at the ABI.
#: What separated them was only the frozen `g_` prefix, so the plan now states
#: the sequence once and applies the prefix on top of it.
#:
#: 168 -> 160: which of a field row's coefficients are live.  The gradient half
#: was already a comprehension filter at four sites -- an absent coefficient
#: simply not appearing -- while the value half was an `if` beside it saying the
#: same thing a different way, and one site said it a third way with a
#: `continue`.  `plans.dependencies.live_test_coefficients` is the sequence;
#: what a coefficient is multiplied by, and what a test function's declaration
#: looks like, stay with the bodies that spell them differently.
#:
#: One of the eight is not an absence and did not become one: every row writes
#: into the quadrature buffer whether or not it has a value coefficient, so a
#: row without one writes a zero rather than writing nothing.  That is a table
#: of two terms, not an empty sequence, and the difference is real.
#:
#: 160 -> 157, and none of it needed a new plan function.  `plans/dependencies`
#: already had `contracted_gradient_components` -- "the same question
#: `uses_test_gradients` answers, phrased as a range so that emission can
#: iterate it instead of branching on it" -- and `contracted_test_quantities`
#: beside it.  Three sites in `residual_codegen.py` were still asking the
#: boolean: the reference-gradient buffer, the per-point contraction, and which
#: of the two `tensor_integrate` entry points the kernel calls.
#:
#: The third is why they belong together.  A kernel that declared the gradient
#: buffer and called the entry point that does not take it, or the reverse,
#: would not compile -- but nothing said the two were one decision.  Both read
#: the same plan answer now.
#:
#: 157 -> 153: `stream_layout == "contiguous"`, to zero.  `plans/streams` was
#: already turning that word into a `DataStreamLayout` -- AOS for a lane-major
#: tile the kernel indexes directly, SOA for an array of pointers -- and four
#: sites compared the word themselves to pick a spelling: a helper-name suffix,
#: two call arguments and a C parameter declaration.
#:
#: The word is the caller's and the layout is the plan's, so `field_stream_layout`
#: states the mapping once and `local_kernel_stream_plans` reads it too.  What
#: stays in emission is what an offset into a tile looks like against a pointer
#: array, which is C.
#:
#: 153 -> 147, and the plan function had named these sites itself.
#: `form_contraction`'s docstring reads "asked twenty-three times in the energy
#: emitter as `form.weak_form is None` or `is not None`, eight of them in one
#: function" -- and six of that eight were still there, in
#: `_sfem_soa_mesh_operator_function`.
#:
#: They are one structure: whether the block call sits inside a quadrature loop.
#: A pointwise form is contracted at each point, so this file opens the loop,
#: indents the call and closes the brace, and passes `q` where a deferred-flux
#: form passes the geometry stride.  Six separate `if`s said that; six tables
#: keyed on the same plan answer say it now, and the buffer extent cannot
#: disagree with the loop that indexes it.
#:
#: 147 -> 144: the Laplacian special cases, removed.  Three specialised
#: generators in `residual_codegen.py` -- 579 lines -- were reached by a
#: nine-term conjunction on the material's *name*, `prefix ==
#: "laplace_proteus_hex8"` and two twins, plus three `#include` branches naming
#: SFEM's hand-written headers.  Emission pattern-matching a material is what
#: "no `if kind == ...` in a backend" forbids outright.
#:
#: They were dead: laplace is written as an energy now and reaches this emitter
#: for nothing, so the tree is byte-identical without them and no measured
#: performance moved.  What they knew is recorded as ARCHITECTURE.html OP 26 --
#: five of their six properties are already in the general path, and the sixth,
#: specialising a unit material constant out at compile time, is the kernel form
#: to aim at.
#:
#: 144 -> 138: which field streams a form's element API carries.  Five sites
#: asked the current half and one the direction half, which is the shape
#: `plans/streams.live_field_roles` describes for the mesh boundary -- "emission
#: asks this as an unrolled loop ... wherever a buffer is declared, a gather is
#: emitted, a scratch slot is taken or a stream argument is named".
#:
#: `element_api_field_roles` is the element API's version of that sequence, and
#: separate because the defaults differ: a form carrying no lowered dependencies
#: still has an element API, and the answer there is that the current state is
#: present and the direction follows the form's own `has_direction`.  That
#: default was the whole of the difference and was restated at every site.
#:
#: 138 -> 132, in two parts.
#:
#: Three predicates about a form -- does it read the state, does it read a
#: direction, which material constants does it name -- lived in
#: `emitters/energy_codegen.py`, each carrying its own "a form not lowered
#: through a dependency set" fallback.  They are pure questions about a form and
#: they are `plans/form_emission`'s now.  The fallback is the part that is not
#: `getattr`: what such a form reads is the caller's declaration, and saying so
#: three times in emission was saying it in the wrong layer.
#:
#: And `form == "jacobian_action"` at three sites, which is the question of
#: which forms publish a packed-mesh kernel.  Only the Jacobian action does, and
#: `plans/geometry_variants` now says why: packing removes the scatter from a
#: matrix-free apply, and a residual scatters once per Newton step while an
#: objective scatters nothing, so neither has the cost packing exists to remove.
#: A sequence, for the reason `packed_mesh_layouts` beside it is one.
#:
#: 132 -> 128: the packed matrix-assembly kernel's two state roles.
#:
#: Two of the four were not a decision about anything.  The packed call built
#: its argument list by finding each state argument's position and assigning the
#: *same value* back over it -- `args[args.index(x)] = x`, three times, guarded
#: by the two role flags -- so the whole search-and-replace left the list as
#: `list(call_args)` found it.  Only the output argument was actually
#: substituted.  Presumably the packed kernel once took differently named
#: arguments and the names later converged; what was left behind was the shell
#: of a substitution that no longer substitutes.
#:
#: The other two are the pack gather, which wrote the same eleven-line block
#: twice -- once for `pk_current` reading `u` through `current_stride`, once for
#: `pk_previous` reading `u_old` through `previous_stride`.  `plans/streams`
#: already names that sequence, and `live_field_roles`' own docstring already
#: listed "a gather is emitted" among the sites re-asking it; `MeshFieldRole`
#: carries the suffix, so `role.field_pointer(field.name)` is the difference
#: between the two copies.
#:
#: 128 -> 125: whether the test function's gradient is contracted, asked by the
#: code that declares the buffer holding it, by the code that fills that buffer
#: inside the quadrature loop, and by the code that calls the contraction which
#: reads it.  One fact about what a body stages, and the three sites that have
#: to agree about a buffer's existence were each deciding it for themselves.
#:
#: The middle one needed nothing new: `contracted_gradient_components` is that
#: question already phrased as a range, and its docstring says it exists so
#: emission can iterate instead of branching.  The other two read
#: `plans.dependencies.staged_test_quantities`, which is deliberately *not*
#: `contracted_test_quantities` beside it -- the value buffer is staged even
#: when no coefficient multiplies the value, because the contraction reads it
#: either way and a staged zero is cheaper than a second kernel shape.  The
#: emitter's own `_TENSOR_INTEGRATE_BY_QUANTITIES` had recorded that in a
#: comment; now a plan says it, and both tables key on the same sequence.
#:
#: 125 -> 118, and this one was not duplication *within* an emitter but between
#: the two of them.  `_packed_crs_passes` existed in both files with
#: byte-identical bodies, and `_matrix_formats_from_plan` in one was
#: `_matrix_format_values` in the other -- the same code differing only in what
#: it called a local list.
#:
#: Both are questions about a `MatrixFormatPlan`, and what the emitters were
#: really doing was lowering its enums to the words the ABI carries with
#: `getattr(f, "value", str(f)).lower()` -- a fallback for not being sure what
#: they had been handed.  In `plans/matrix_formats` that uncertainty is not
#: available: `MatrixAssemblyVariantPlan.__post_init__` coerces every field
#: through its enum, so a format *is* a `MatrixFormat`.
#:
#: One filter went with them rather than moving.  Their version skipped a
#: variant whose pass was `none`, and a packed variant cannot have one --
#: `__post_init__` rejects `PACKED` with `NONE` outright, which is the stronger
#: statement and the one the plan's docstring keeps.
#:
#: Worth recording because it nearly went wrong: the plan function
#: `packed_crs_passes` collides with a local of that name in two emitter
#: functions and with a parameter of that name in three more.  Renaming only
#: the assignment would have left `if packed_crs_passes:` resolving to the
#: imported function object, which is always truthy.  Scope was checked with an
#: AST pass rather than with grep, and byte-identity was the gate that would
#: have caught it regardless.
#:
#: 118 -> 112: who decides that an optional plan was not supplied.
#:
#: A third copy of one function, and this one across layers.
#: `_validate_diagnostics_plan_names` lived in `symbolic/core.py` -- the bottom
#: of the stack, which has no business knowing that diagnostics plans exist --
#: and was `plans.diagnostics.validate_diagnostics_plan_names` minus its type
#: check.  `energy_codegen.py` imported the first and `residual_codegen.py` the
#: second, so each emitter had its own.  The symbolic copy had one user and is
#: gone.
#:
#: The six guards around the two validators were redundant in a more
#: interesting way.  `validate_diagnostics_plan_names` already opens with `if
#: plan is None: return None`, so three `if diagnostics_plan is not None:` in
#: emission were re-deciding what the validator had decided.  Its sibling
#: `validate_reference_data_plan` raised `TypeError` on None instead, which
#: forced its three callers to guard.  Every emitter defaults both parameters to
#: None, so validating an unsupplied plan is vacuous; the two siblings answer
#: that the same way now, and the callers just call.
#:
#: What is deliberately *not* counted down: four `emission_plan is None`
#: branches remain, and they are precondition raises at emitter entry points.
#: They choose nothing about what to emit -- they refuse to emit -- so this
#: measure counts them only because `emission_plan` is a plan input.  Excluding
#: a branch whose body is solely a `raise` would lower the number without
#: improving anything, which is adjusting the scorer rather than the code.  Both
#: the guards and the measure stay as they are; the note is here so the next
#: reader knows the remainder is not all emission logic.
#:
#: 112 -> 110: whether a family is the tensor-product one.  The number is small
#: because most of the sites were already calling a plan function; what was
#: wrong was the function.
#:
#: `plans/layout._is_tensor_product_family(rule, basis_family)` never read its
#: first parameter.  Nineteen call sites passed a `rule` or a `cell_rule` that
#: the body ignored, which made it read as a question about the quadrature rule
#: when it is a question about the family alone -- and six of those sites pass
#: `geometry_family` rather than `basis_family`, which is the tell.
#:
#: It was private by name and imported across modules anyway, by
#: `residual_codegen.py`.  `energy_codegen.py` and `energy.py` wrote
#: `str(basis_family) == "tensor_product"` out at seven sites instead of
#: reaching for something whose underscore said not to, and two of those seven
#: re-derived the `basis_family is None` check that already lived inside the
#: function.
#:
#: The unified predicate raises on None where three of the converted sites used
#: to return False for it.  That widening was not assumed to be safe: a full
#: regeneration of all eight materials raised nothing and moved no byte, and the
#: suite -- which builds configurations the generators do not, including the
#: CUDA and HIP source builders that define `emits_tensor_product_header`
#: separately -- reported the baseline set unchanged.  A family is never absent
#: on any path either reaches.
#:
#: 110 -> 109: one branch, and the smallest step here so far, but the one worth
#: reading.  Three emitters computed the two reference-traffic numbers every
#: `KernelDiagnostics` record carries -- `reference_scalars` and
#: `quadrature_weight_scalars` -- three different ways: `energy_codegen.py`
#: branched on the basis family, `residual_codegen.py` split
#: `sfem_reference_data(rule)` on the `q_weight` prefix, and
#: `inexact_apply_codegen.py` writes 0 and the point count.
#:
#: What was wrong in the first of those is that it asked the *basis family* a
#: question the *rule* answers about itself: `sfem_reference_data` branches on
#: `rule.is_tensor_product` internally and returns the 1D tables for exactly
#: those elements.  That the two agree where they overlap was measured -- (8, 2)
#: for HEX8, QUAD4 and PROTEUS_HEX8, (18, 3) for HEX27 -- rather than argued.
#:
#: The two measures were deliberately *not* merged.  An energy kernel is charged
#: only for the reference arrays its own signature takes, so one that never
#: reads `shape` is not billed for it; the residual recipe counts the rule's
#: full data.  Collapsing them would have started charging kernels for tables
#: they do not read, and the numbers feed `tools/flops_audit.py`, which is how
#: the roofline is generated rather than written.
#:
#: 109 -> 107: the adjugate alias, which had its own answer already in hand.
#:
#: Both sites called `plans.geometry_quantities.local_geometry_streams` and
#: discarded the roles on the very next line -- `for name, _role in ...` --
#: before re-deriving what those roles had just said, as
#: `dependencies.uses_adjugate and not uses_cached_affine_metric`.  Whether a
#: kernel aliases an adjugate *is* whether the plan put an adjugate role in the
#: sequence: a cached metric carries the adjugate's work and takes its place,
#: and a form that reads no adjugate never gets one.
#:
#: The extent went with it.  It was `dim * dim` and is now the length of the
#: sequence being aliased, so the array cannot disagree with the streams it
#: points at.
BUDGET = 107


def _tested_names(test):
    """Every identifier a branch test reads, bare names and attributes alike."""
    for node in ast.walk(test):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr


def _is_plan_input(identifier):
    """Whether an identifier names planning-layer input.

    Matched on the identifier itself rather than on a substring of the dump,
    because the qualified names are most of the population.  A name qualified
    with an underscore counts; a word that merely ends in the same letters
    (``transform``, ``platform``) does not.
    """
    return any(
        identifier == word or identifier.endswith("_" + word)
        for word in PLAN_INPUTS
    )


def _decision_branches():
    counts = collections.Counter()
    for name in sorted(os.listdir(EMITTERS)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(EMITTERS, name)
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp)):
                continue
            if any(map(_is_plan_input, _tested_names(node.test))):
                counts[name] += 1
    return counts


class EmissionIsBecomingAPrinterTest(unittest.TestCase):
    maxDiff = None

    def test_decisions_in_emission_only_shrink(self):
        counts = _decision_branches()
        total = sum(counts.values())
        self.assertLessEqual(
            total,
            BUDGET,
            "emission gained %d decision branches (now %d, budget %d).  A "
            "branch testing the lowered form or a plan is emission choosing "
            "what to emit; put the choice in plans/ and let the emitter spell "
            "the result.\n%s"
            % (
                total - BUDGET,
                total,
                BUDGET,
                "\n".join("  %-26s %d" % kv for kv in sorted(counts.items())),
            ),
        )
        self.assertEqual(
            total,
            BUDGET,
            "emission is down to %d decision branches and the budget still "
            "says %d -- lower it to lock the improvement in, and say in the "
            "commit which decision moved" % (total, BUDGET),
        )

    def test_the_two_big_emitters_hold_almost_all_of_it(self):
        """Where the work is, so the number is not just a score."""
        counts = _decision_branches()
        big = counts["residual_codegen.py"] + counts["energy_codegen.py"]
        self.assertGreater(
            big,
            0.9 * sum(counts.values()),
            "decision logic has spread out of the two big emitters; the "
            "budget was calibrated on the assumption it is concentrated there",
        )


if __name__ == "__main__":
    unittest.main()
