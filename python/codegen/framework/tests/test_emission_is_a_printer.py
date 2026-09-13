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
BUDGET = 160


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
