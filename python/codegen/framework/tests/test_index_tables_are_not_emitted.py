"""A generated kernel computes its indices; it does not look them up.

The block matrix scatter used to publish four `constexpr` tables --
`ROW_COMPONENT`, `ROW_SHAPE`, `COL_COMPONENT`, `COL_SHAPE`, each `NC * NS`
entries -- walk a single flattened `row_stream`, and read the two indices back
out of them.  For a three-component HEX8 that is 96 `int`s per scatter function
holding nothing but `i / 8` and `i % 8`, with the column pair byte-identical to
the row pair because the block is square.  The element matrix is component-major
by construction, so both indices were closed-form in the position all along --
the tables were the stored inverse of a flattening the loop nest never had to
perform.

The fix was the loop nest, and it is not the point of this file.  The point is
that the *style* has no place in the generator: an emitter that flattens a loop
and then recovers the parts is writing data where it should be writing loops,
and it costs more on a device than a host, where the table is constant-memory
traffic or register pressure for arithmetic the hardware does for free.

So this walks the tree the build compiles and rejects any integer table whose
contents are a closed form in the index.  A genuine permutation is data and
stays: `SHAPE_ORDER` carries the mesh's node numbering against the Cartesian
tensor-product order (`0, 1, 3, 2` on a quad), which no expression produces.
The test names the closed form it recognised, so a new instance arrives with its
own replacement already written in the failure message.
"""

import os
import re
import unittest


GENERATED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "frontend", "ops", "generated",
)

_TABLE = re.compile(
    r'static constexpr \w+ ([A-Za-z_]\w*)\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}', re.S
)

#: Sources a generated kernel is entitled to carry as data.
#:
#: One entry, and it earns it: the mesh's node order against the Cartesian
#: tensor-product order is a convention, not a function of the index.
DATA_TABLES = ("SHAPE_ORDER",)


def closed_form(values):
    """The expression this table is, in the index, or `None` if it is data."""
    n = len(values)
    if n < 2:
        return None
    if values == list(range(n)):
        return "the index itself"
    if len(set(values)) == 1:
        return "the constant %d" % values[0]
    for divisor in range(2, n + 1):
        if n % divisor:
            continue
        if values == [i // (n // divisor) for i in range(n)]:
            return "i / %d" % (n // divisor)
        if values == [i % divisor for i in range(n)]:
            return "i %% %d" % divisor
    step = values[1] - values[0]
    if step and values == [values[0] + step * i for i in range(n)]:
        return "%d + %d * i" % (values[0], step)
    return None


def _integer_tables(root):
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            if not name.endswith((".cpp", ".hpp", ".cu", ".cuh")):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="replace") as handle:
                source = handle.read()
            for match in _TABLE.finditer(source):
                table = match.group(1)
                body = match.group(2).replace("\n", " ")
                try:
                    values = [int(v) for v in body.split(",") if v.strip()]
                except ValueError:
                    continue
                if values:
                    yield os.path.relpath(path, root), table, values


class IndexTablesAreNotEmittedTest(unittest.TestCase):
    def setUp(self):
        self.tables = tuple(_integer_tables(GENERATED))

    def test_the_tree_is_walked_at_all(self):
        """A survey that found nothing would pass every assertion below."""
        self.assertGreater(len(self.tables), 0)

    def test_no_table_holds_a_closed_form_in_its_index(self):
        offenders = []
        for path, table, values in self.tables:
            if table in DATA_TABLES:
                continue
            form = closed_form(values)
            if form is not None:
                offenders.append(
                    "%s: %s[%d] is %s -- write the loop, not the table"
                    % (path, table, len(values), form)
                )
        self.assertEqual(sorted(set(offenders))[:10], [], "\n  ".join([""] + sorted(set(offenders))[:10]))

    def test_the_tables_that_remain_are_data(self):
        """And the exemption stays narrow enough to be worth having."""
        for path, table, values in self.tables:
            if table not in DATA_TABLES:
                continue
            self.assertIsNone(
                closed_form(values),
                "%s: %s is exempt but is a closed form; the exemption is hiding "
                "the defect it was meant to make room for" % (path, table),
            )
            self.assertEqual(
                sorted(values),
                list(range(len(values))),
                "%s: %s is exempt as a permutation but is not one" % (path, table),
            )

    def test_no_kernel_recovers_a_loop_variable_by_dividing_it(self):
        """The same style one step less bad: flatten, then divide it back out.

        A loop over `NC * NS` that computes `i / NS` and `i % NS` in its body is
        the table rewritten as arithmetic.  It is cheaper and still the wrong
        shape -- the two loops it is standing in for say what it means.
        """
        loop = re.compile(r'for \(int (\w+) = 0; \1 < ([^;]+); \+\+\1\)')
        recover = re.compile(r'\b(\w+)\s*[/%]\s*(NS|NC|ND|NQ|NDOFS|N_\w+)\b')
        offenders = []
        for base, _dirs, names in os.walk(GENERATED):
            for name in sorted(names):
                if not name.endswith((".cpp", ".hpp", ".cu", ".cuh")):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    source = handle.read()
                variables = set(match.group(1) for match in loop.finditer(source))
                for match in recover.finditer(source):
                    if match.group(1) in variables:
                        offenders.append(
                            "%s: %s" % (os.path.relpath(path, GENERATED), match.group(0))
                        )
        self.assertEqual(sorted(set(offenders))[:10], [])


if __name__ == "__main__":
    unittest.main()
