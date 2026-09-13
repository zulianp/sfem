"""A generated kernel contains the computation and not much else.

The standing rule for this tree is that emitted kernels stay lean: no
validation inside the element loop, no temporaries nothing reads, no wrappers
that publish a name for a call the caller can spell.  Three departures from it
were found by surveying the shipped tree, and this gates each so that none can
come back quietly.

`tools/lean_audit.py` is the measurement; the numbers below are the budgets.
They are ratchets: a change that lowers one should lower the number here too,
and a change that raises one has to say why.
"""

import os
import unittest

from codegen.framework.tools.lean_audit import survey


def _generated_tree():
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "frontend",
            "ops",
            "generated",
        )
    )


#: Assignments whose result nothing reads, transitively.
#:
#: This was 1557 before the emitters started asking, per field rather than per
#: system, which of a field's value and gradient a form reads -- and before
#: `plans/dependencies.py` kept the symbols it was already computing and then
#: discarding in an `any(...)`.  Then the geometry preamble stopped being
#: declared unconditionally by four sites that could have asked
#: `local_geometry_quantities`, and the test value stopped being declared by a
#: form that contracts only test gradients: 375 -> 127.
#:
#: What is left is three paths, each wanting the same kind of fix: the
#: inexact-apply tangent gathers connectivity it never scatters through (29),
#: the mixed local bodies map a physical gradient per field rather than per
#: component read (56), and their geometry preamble is still unconditional
#: (42).  A ratchet to drive down, not a target that has been met.
#:
#: 127 -> 87 when the `(void)name;` discards went: the scan counts a mention as
#: a read, so every discard was keeping its own subject alive.
#:
#: 87 -> 66 when the inexact-apply tangent stopped forming the element's full
#: reference gradients.  It left this list entirely: the 29 it contributed are
#: gone, and what remains is 58 in the mixed local bodies and 8 elsewhere.
DEAD_ASSIGNMENT_BUDGET = 66

#: Runs of back-to-back single-statement `#pragma omp simd` lane loops.  A run
#: longer than one is N loops and N pragmas where one loop with N statements
#: does the same stores.  There were 186 runs of nine, 156 of four and 130 of
#: three; the Jacobian zero-fill and accumulation account for almost all of
#: them and are now fused, leaving 7 nines and 7 fours in paths that build the
#: same values a different way.  A ratchet, not a target that has been met.
LONGEST_LANE_LOOP_RUN = 9
LANE_LOOP_RUNS_LONGER_THAN_ONE = 32

#: `(void)name;` statements, and the declarations that made them necessary.
#:
#: A discard exists to stop `-Wextra -Werror` -- which `SFEM_ENABLE_DEV_MODE`
#: turns on -- complaining about a name nothing reads, so its presence says the
#: declaration was the mistake.  There were 463 discards and 648 constants that
#: no kernel read: `ND` in every kernel handed an adjugate it never
#: differentiates, `NQ1`/`NS1` wherever the point count is a literal rather than
#: an `integer_root`, `N_FIELD_STREAMS` in every local body, and `nnodes`, which
#: no kernel in the tree has ever read.
#:
#: Both are zero and must stay zero: the emitters now compose a prologue from
#: the names their own body mentions, and a parameter the body ignores is
#: emitted without a name -- which is what the boundary emitter always did.
VOID_DISCARD_BUDGET = 0
UNUSED_CONSTANT_BUDGET = 0

#: Named parameters no body reads, with no discard to excuse them.  This is
#: what `-Wextra` rejects and it is unchanged by the work that cleared the two
#: budgets above -- these were never marked, so nothing pointed at them.  Almost
#: all are one shape: the matrix-assembly kernel takes every sparse format's
#: parameters and emits only the selected format's branch, so `diag_offsets`,
#: `ndiag` and the five `coo_*` arrays go unread in 15 kernels each.
#:
#: 179 -> 74 once those were marked: the assembly signature carries every
#: format's arrays so the dispatch can call it whatever the format is, and the
#: resolver checks each against the body the kernel ends up with, so the ones a
#: given format does read keep their names.  What is left is the simplex local
#: kernels, whose reference-basis streams go unread when the kernel is handed a
#: cached metric instead -- the fix there is for the plan to stop putting the
#: stream on the boundary, not for emission to unname it.  A ratchet to drive
#: down, not a target that has been met.
#:
#: 74 -> 56 with the same change: a tangent that reads no field no longer names
#: the one-dimensional basis tables its contraction would have used.  All 56
#: that remain are the simplex local kernels described above.
UNUSED_PARAMETER_BUDGET = 56

#: Node-ordering permutations built inside a kernel.
#:
#: A micro-kernel is written against the lexicographic basis, so an element
#: whose mesh numbers its nodes otherwise reconciles the two in a forwarding
#: wrapper -- `QUAD4` and `HEX8` always did, `HEX27` now does too -- and the
#: kernel itself reorders nothing.  There were 81, all in `two_phase_flow`,
#: `navier_stokes` and `stokes`.
#:
#: A selection is not a permutation and is not counted: the pressure of a
#: lexicographic HEX27_HEX8 pair lives on cell nodes 0, 2, 6, 8, 18, 20, 24 and
#: 26, which says which nodes carry the space rather than reordering it.  The
#: same array was a permutation before this work and is a selection after, so
#: the measure tells them apart by their indices rather than by their name.
KERNEL_PERMUTATION_BUDGET = 0

#: Lane-indexed reads whose base is fixed for the whole loop were 2016 and are
#: 35.  The 35 are the `test`/`integrate` reductions in
#: `tensor_product_kernels.hpp`, whose base moves with the reduction variable;
#: naming it needs the loop nest interchanged, which was measured on Grace and
#: costs up to 14% (see the note at the head of
#: `emitters/tensor_product_kernels.py`).  There is no budget for them here
#: because they are not the defect this file gates -- they are the shape a
#: strided reduction has.

#: `extern "C"` wrappers around a `KernelDiagnostics` free function.  There
#: were 1400 of them and nothing referenced any.
WRAPPED_HELPER_BUDGET = 0


class KernelsAreLeanTest(unittest.TestCase):
    def setUp(self):
        self.survey = survey(_generated_tree())

    def test_no_kernel_computes_a_value_nothing_reads(self):
        dead = self.survey["dead"]
        self.assertLessEqual(
            len(dead),
            DEAD_ASSIGNMENT_BUDGET,
            "these assignments are never read:\n%s"
            % "\n".join(
                "  %s: %s: %s" % (row[0], row[1], row[3][:100]) for row in dead[:20]
            ),
        )

    def test_lane_loops_are_not_opened_one_per_component(self):
        runs = self.survey["lane_loop_runs"]
        self.assertTrue(runs, "no lane loops were found; the survey is not reading the tree")
        longest = max(runs)
        self.assertLessEqual(longest, LONGEST_LANE_LOOP_RUN)
        # The count matters more than the maximum: one long run left in a
        # corner is a smaller problem than many.
        long_runs = sum(count for length, count in runs.items() if length > 1)
        self.assertLessEqual(
            long_runs,
            LANE_LOOP_RUNS_LONGER_THAN_ONE,
            "%d places open a fresh lane loop per component; one loop with that "
            "many statements does the same stores in one SIMD region" % long_runs,
        )

    def test_nothing_is_declared_that_nothing_reads(self):
        discards = self.survey["void_discards"]
        self.assertLessEqual(
            len(discards),
            VOID_DISCARD_BUDGET,
            "these kernels discard a name instead of not declaring it:\n%s"
            % "\n".join("  %s: (void)%s;" % row for row in discards[:20]),
        )
        constants = self.survey["unused_constants"]
        self.assertLessEqual(
            len(constants),
            UNUSED_CONSTANT_BUDGET,
            "these constants are declared and never read:\n%s"
            % "\n".join("  %s:%d: %s" % (row[0], row[2], row[1]) for row in constants[:20]),
        )

    def test_no_kernel_names_a_parameter_it_ignores(self):
        parameters = self.survey["unused_parameters"]
        self.assertLessEqual(
            len(parameters),
            UNUSED_PARAMETER_BUDGET,
            "these parameters are named and never read:\n%s"
            % "\n".join("  %s: %s" % row for row in parameters[:20]),
        )

    def test_no_kernel_reorders_its_own_nodes(self):
        permutations = self.survey["kernel_permutations"]
        self.assertLessEqual(
            len(permutations),
            KERNEL_PERMUTATION_BUDGET,
            "these kernels permute their own connectivity instead of being "
            "handed it in the order they are written for:\n%s"
            % "\n".join("  %s: %s[%d]" % row for row in permutations[:20]),
        )

    def test_no_kernel_wraps_a_shared_diagnostics_helper(self):
        wrappers = self.survey["wrapped_helpers"]
        self.assertLessEqual(
            len(wrappers),
            WRAPPED_HELPER_BUDGET,
            "these entry points only wrap a KernelDiagnostics helper: %s"
            % ", ".join(name for _path, name in wrappers[:10]),
        )


if __name__ == "__main__":
    unittest.main()
