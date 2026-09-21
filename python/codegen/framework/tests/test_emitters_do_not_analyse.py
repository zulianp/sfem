"""Emitters must not decide what to emit by analysing symbols.

Working out which data a kernel reads -- which field roles it touches, whether
it needs values or gradients, which parameters survive, which coefficients are
structurally zero -- is planning.  It fixes how much memory traffic an element
costs, which is the largest single lever on a matrix-free kernel.  The emission
layer's job is to spell the answer, not to compute it.

The tell is ``free_symbols``.  An emitter reaching for it is re-deriving
something the planning layer either already knows or should.  This module pins
the remaining occurrences, in the same shrink-only style as the layering
ratchet: the count may fall and the list may lose entries, but nothing may be
added without a step that removes it again.

The two survivors are argument validation, not planning -- they reject a form
the emitter cannot handle rather than deciding what it emits.  They are listed
so that the distinction is a recorded judgement rather than an oversight.
"""

import ast
import os
import unittest


EMITTERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emitters"
)

# module -> number of permitted `free_symbols` references, with why.
#
#   boundary_codegen  2  both inside validation guards:
#                        _validate_boundary_coefficients rejects forbidden
#                        symbols, _validate_boundary_metadata checks the
#                        declared dependency set covers what is used.
#   energy_codegen    2  the energy path still derives its own weak-form
#                        substitutions; migrating it is the remaining half of
#                        the decision move that residual_codegen has completed.
ALLOWED_SYMBOL_ANALYSIS = {
    "boundary_codegen.py": 2,
    "energy_codegen.py": 2,
}


def _symbol_analysis_counts():
    counts = {}
    for name in sorted(os.listdir(EMITTERS_DIR)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(EMITTERS_DIR, name)
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        hits = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "free_symbols"
        )
        if hits:
            counts[name] = hits
    return counts


class EmittersDoNotAnalyseTest(unittest.TestCase):
    def test_no_emitter_gains_symbol_analysis(self):
        counts = _symbol_analysis_counts()
        offenders = []
        for name, hits in sorted(counts.items()):
            allowed = ALLOWED_SYMBOL_ANALYSIS.get(name, 0)
            if hits > allowed:
                offenders.append(
                    "%s: %d references, %d allowed" % (name, hits, allowed)
                )
        self.assertEqual(
            offenders,
            [],
            "emitter(s) analyse symbols to decide what to emit:\n  "
            + "\n  ".join(offenders)
            + "\n\ndeciding what a kernel reads belongs in plans/; the emitter "
            "should be handed the answer",
        )

    def test_allowance_shrinks_and_never_goes_stale(self):
        """A module that stops analysing must lose its entry."""
        counts = _symbol_analysis_counts()
        stale = []
        for name, allowed in sorted(ALLOWED_SYMBOL_ANALYSIS.items()):
            actual = counts.get(name, 0)
            if actual < allowed:
                stale.append(
                    "%s: allowance %d, actual %d -- lower it" % (name, allowed, actual)
                )
        self.assertEqual(stale, [], "\n  ".join(stale))

    def test_the_residual_emitter_analyses_nothing(self):
        """The largest emitter is fully free of it, and must stay that way."""
        counts = _symbol_analysis_counts()
        self.assertEqual(
            counts.get("residual_codegen.py", 0),
            0,
            "residual_codegen.py analyses free symbols again; its dependency "
            "pruning lives in plans/dependencies.py and its stream selection in "
            "plans/streams.py",
        )


if __name__ == "__main__":
    unittest.main()
