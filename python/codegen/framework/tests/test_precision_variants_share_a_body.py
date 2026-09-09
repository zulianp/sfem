"""One entry point per kernel, which selects its scalar at run time.

Every kernel used to be published twice: a `double` symbol and a `_float`
twin, each restating the whole parameter list, each forward-declared in the
dispatch source, and each reached through a `switch (resolved_real_type)` in
every element case.  Counting the second signature, the second prototype and
the inner switch, that layer was about 2.1 MB -- 18% of the tree -- for a
boundary whose only caller is the dispatch beside it.

The entry point now takes the scalar's width and instantiates the template
itself.  What this pins:

  * the tree really does publish runtime-typed entry points, so the checks
    below are not passing over an empty set;
  * no `_float` twin is published outside the boundary sources, which are the
    one emitter still spelling its entry points from whole-function templates;
  * an entry point forwards rather than computes -- no loop, no parallel
    pragma -- because a body that computes is the arithmetic existing twice,
    which is what the original version of this test was written to stop;
  * the dispatch proves, where both spellings are visible, that
    `smesh::PrimitiveType` numbers its scalars by width.

`tools/reproducibility.py` drives the dimension-generic entry points, so a
`float` body that drifted from its `double` twin moved no digest: the two would
simply have started computing different things. That is why the property is
gated here rather than left to the answers.
"""

import os
import re
import unittest


GENERATED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "frontend", "ops", "generated",
)

_DEFINITION = re.compile(r'extern "C"[^;{()]*?([A-Za-z_][A-Za-z0-9_]*)\s*\(')

#: What a body contains when it is computing rather than forwarding.
_WORK = ("for (", "#pragma")

#: The two emitters that still publish a precision pair, and the files they
#: write.  Both spell their entry points from whole-function templates rather
#: than from a parameter list, so neither was converted with the rest;
#: `package/op_wrappers.py` keeps the merge that turns such a pair into one
#: runtime-typed dispatch entry, and it is reached only by these.
_UNCONVERTED = (
    "boundary_operator.cpp",
    "inexact_apply_inline.hpp",
    "inexact_apply_operator.cpp",
)


def _definitions(text):
    """`(name, body)` for each `extern "C"` definition, braces matched."""
    for match in _DEFINITION.finditer(text):
        open_brace = text.find("{", match.end())
        if open_brace < 0:
            continue
        semicolon = text.find(";", match.end())
        if 0 <= semicolon < open_brace:
            continue  # a declaration, not a definition
        depth, index = 0, open_brace
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    yield match.group(1), text[open_brace:index + 1]
                    break
            index += 1


class PrecisionVariantsShareABodyTest(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir(GENERATED):
            self.skipTest("generated tree not present")
        self.entry_points = {}
        self.sources = {}
        for base, _dirs, files in os.walk(GENERATED):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
                self.sources[path] = text
                for symbol, body in _definitions(text):
                    self.entry_points[symbol] = (path, body)

    def test_the_tree_publishes_runtime_typed_entry_points(self):
        """A guard over an empty set passes for the wrong reason."""
        typed = [
            symbol
            for symbol, (_path, body) in self.entry_points.items()
            if "switch (scalar_bytes)" in body
        ]
        self.assertGreater(len(typed), 400)

    def test_no_kernel_publishes_a_precision_twin(self):
        offenders = [
            "%s (%s)" % (symbol, os.path.relpath(path, GENERATED))
            for symbol, (path, _body) in sorted(self.entry_points.items())
            if symbol.endswith("_float")
            and not any(path.endswith(tail) for tail in _UNCONVERTED)
            and not path.endswith("_dispatch.cpp")
        ]
        self.assertEqual(
            offenders[:10],
            [],
            "%d kernels still publish a second symbol per precision" % len(offenders),
        )

    def test_a_runtime_typed_entry_point_forwards_rather_than_computes(self):
        offenders = []
        for symbol, (path, body) in sorted(self.entry_points.items()):
            if "switch (scalar_bytes)" not in body:
                continue
            if any(marker in body for marker in _WORK):
                offenders.append("%s (%s)" % (symbol, os.path.relpath(path, GENERATED)))
        self.assertEqual(
            offenders[:10],
            [],
            "%d entry points compute rather than forward, so the arithmetic "
            "exists once per precision" % len(offenders),
        )

    def test_the_dispatch_proves_the_width_correspondence(self):
        """The kernels select on a width; the ABI names an enum.

        Only the dispatch sources see both, so that is where the two are
        proved equal rather than assumed.
        """
        # The diagnostics dispatch returns a record and calls no kernel, so it
        # never spells a width and has nothing to prove.
        dispatches = [
            path
            for path in self.sources
            if path.endswith("_dispatch.cpp")
            and not path.endswith("_diagnostics_dispatch.cpp")
        ]
        self.assertGreater(len(dispatches), 8)
        missing = [
            os.path.relpath(path, GENERATED)
            for path in sorted(dispatches)
            if "static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double)"
            not in self.sources[path]
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
