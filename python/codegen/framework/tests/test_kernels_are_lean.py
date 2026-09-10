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
DEAD_ASSIGNMENT_BUDGET = 127

#: Runs of back-to-back single-statement `#pragma omp simd` lane loops.  A run
#: longer than one is N loops and N pragmas where one loop with N statements
#: does the same stores.  There were 186 runs of nine, 156 of four and 130 of
#: three; the Jacobian zero-fill and accumulation account for almost all of
#: them and are now fused, leaving 7 nines and 7 fours in paths that build the
#: same values a different way.  A ratchet, not a target that has been met.
LONGEST_LANE_LOOP_RUN = 9
LANE_LOOP_RUNS_LONGER_THAN_ONE = 32

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
