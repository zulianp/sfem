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
BUDGET = 199


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
