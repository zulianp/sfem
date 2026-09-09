"""A precision variant is an entry point, not a second copy of the kernel.

Every kernel that is published at more than one scalar precision is emitted as
one `template <typename s_t> ... _impl` with a thin `extern "C"` forwarder per
precision.  That is how most of the emitter has always worked.  The packed
traversal did not: it wrote the whole kernel out once per precision, opening
each copy with `using s_t = double;` or `using s_t = float;` and repeating the
identical text below.  100 kernels across the four energy materials carried
748,549 bytes of duplicated arithmetic that way -- 6% of the tree, and two
copies of an expression that nothing kept in step.

Nothing would have caught a drift between them either.  `tools/reproducibility.py`
drives the `real_t` entry points, so a `float` body that changed on its own moves
no digest; the two would simply have started computing different things.

The test is therefore about shape, not about text: a precision-suffixed entry
point may forward and may adapt its arguments, but it may not contain a loop or
a parallel pragma, because those mean it is doing the work rather than handing
it on.
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
        for base, _dirs, files in os.walk(GENERATED):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for symbol, body in _definitions(handle.read()):
                        self.entry_points[symbol] = (path, body)

    def test_the_tree_publishes_precision_variants_at_all(self):
        """A guard over an empty set passes for the wrong reason."""
        floats = [s for s in self.entry_points if s.endswith("_float")]
        self.assertGreater(len(floats), 500)

    def test_no_precision_variant_carries_a_kernel_body(self):
        offenders = []
        for symbol, (path, body) in sorted(self.entry_points.items()):
            if not symbol.endswith("_float"):
                continue
            if any(marker in body for marker in _WORK):
                offenders.append("%s (%s)" % (symbol, os.path.relpath(path, GENERATED)))
        self.assertEqual(
            offenders[:10],
            [],
            "%d precision variants compute rather than forward, so the "
            "arithmetic exists twice" % len(offenders),
        )

    def test_a_precision_variant_has_a_template_to_forward_to(self):
        """Forwarding is only cheap because the template is shared.

        Checked on the packed kernels specifically, since they are the ones that
        just acquired it and the ones a future change is most likely to undo.
        """
        packed = [s for s in self.entry_points
                  if s.endswith("_float") and "_packed" in s and "_impl" not in s]
        self.assertGreater(len(packed), 50)
        missing = [s for s in packed
                   if "%s_impl" % s[:-len("_float")] not in self.entry_points
                   and "_impl<" not in self.entry_points[s][1]
                   and "_float(" not in self.entry_points[s][1]]
        self.assertEqual(missing[:10], [], "%d have nothing to forward to" % len(missing))


if __name__ == "__main__":
    unittest.main()
