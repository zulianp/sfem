"""Every published C ABI name is one the framework knows how to place.

The public entry-point grammar has two halves that must agree.  The emitters
*compose* these names; `package/op_wrappers.py` *parses* them, because L7 is
handed kernel sources as text and has no other source for what a kernel is.  A
rename that moves only one half is the failure this file exists for: the parser
stops recognising anything, `_dispatch_mapping` returns `None` for every kernel,
the caller treats that as "wants no dispatch", the wrapper emits its fallback
paths -- and generation reports success.  That is not a hypothetical.  It is
what a previous attempt at this rename actually produced: 2,023 names in the new
form, 3,514 in the old, and a clean exit.

The cure is that declining a name must be a decision rather than an accident.
`conventions.classify_abi_name` puts every published name into a named category,
`ABI_NON_MESH_TAILS` lists the categories that legitimately have no
dimension-generic entry point, and anything outside both is a hard error at
generation time.  The tests below assert the property that makes the guard worth
having: the classification is *total* over the tree the build compiles, so the
guard can never be vacuous.
"""

import collections
import os
import re
import unittest

from codegen.framework.plans import conventions
from codegen.framework.package import op_wrappers


GENERATED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "frontend", "ops", "generated",
)

# The name is the last identifier before the parameter list, whatever the return
# type spells -- `int`, `const sfem::codegen::KernelDiagnostics *`, `double`.
_DECLARATION = re.compile(r'extern "C"([^;{()]*?)([A-Za-z_][A-Za-z0-9_]*)\s*\(')

#: What `_dispatch_groups` considers.  The rate and diagnostics accessors are
#: excluded there by return type rather than by name, which is why some of them
#: carry a geometry marker without ever being offered a dispatch.
_DISPATCHABLE_RETURN = " int "


def _published_names(dispatchable_only=False):
    names = set()
    for base, _dirs, files in os.walk(GENERATED):
        for name in files:
            if not name.endswith((".cpp", ".hpp")):
                continue
            with open(os.path.join(base, name), encoding="utf-8", errors="replace") as handle:
                for returns, symbol in _DECLARATION.findall(handle.read()):
                    if dispatchable_only and returns != _DISPATCHABLE_RETURN:
                        continue
                    names.add(symbol)
    return names


class ClassificationIsTotalTest(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir(GENERATED):
            self.skipTest("generated tree not present")
        self.names = _published_names()

    def test_the_tree_is_large_enough_for_this_to_mean_something(self):
        """A guard over an empty set passes for the wrong reason."""
        self.assertGreater(len(self.names), 3000)

    def test_every_published_name_is_classified(self):
        unclassified = sorted(n for n in self.names if conventions.classify_abi_name(n) is None)
        self.assertEqual(
            unclassified[:10],
            [],
            "%d published names match no category; either an emitter invented a "
            "spelling or conventions.py has fallen behind it" % len(unclassified),
        )

    def test_every_category_is_populated(self):
        """An unreachable category is a rule nobody is following any more."""
        kinds = collections.Counter(
            conventions.classify_abi_name(n)[0] for n in self.names
        )
        expected = {"mesh", "diagnostics", "query", "local", "boundary"}
        self.assertEqual(set(kinds) , expected)
        for kind in expected:
            self.assertGreater(kinds[kind], 0, kind)

    def test_every_mesh_kernel_can_be_given_a_dimension(self):
        """`mesh` is exactly the set that gets a dimension-generic dispatch.

        `_insert_dispatch_dimension` and `classify_abi_name` read the same
        markers, so this asserts they stay read from the same table rather than
        drifting into two tables that happen to agree today.
        """
        dispatchable = _published_names(dispatchable_only=True)
        self.assertGreater(len(dispatchable), 1000)
        failed = []
        for name in sorted(dispatchable):
            kind, marker = conventions.classify_abi_name(name)
            spliced = op_wrappers._insert_dispatch_dimension(name, 3)
            if kind == "mesh":
                if spliced is None or "_3d%s" % marker not in spliced:
                    failed.append(name)
            elif spliced is not None:
                # A non-mesh name that nonetheless carries a marker would be
                # given a dispatch it has no signature for.  The `void` rate
                # accessors do exactly this and are safe only because the
                # dispatch loop filters on return type; among the `int` kernels
                # the two classifications have to agree outright.
                failed.append(name)
        self.assertEqual(failed[:10], [], "%d names disagree" % len(failed))


class MarkerTableTest(unittest.TestCase):
    def test_markers_are_ordered_longest_first(self):
        """`_affine_mesh_soa` is a prefix of `_affine_mesh_soa_aos_unit`."""
        lengths = [len(marker) for marker in conventions.dimension_markers()]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_markers_are_derived_from_the_token_tables(self):
        markers = set(conventions.dimension_markers())
        self.assertEqual(
            len(markers),
            len(conventions.ABI_GEOMETRY_TOKENS)
            * len(conventions.ABI_GEOMETRY_QUALIFIERS)
            * len(conventions.ABI_LAYOUT_TAILS),
        )
        for marker in markers:
            self.assertTrue(
                any(marker.startswith("_%s_" % g) for g in conventions.ABI_GEOMETRY_TOKENS),
                marker,
            )

    def test_a_moved_token_stops_classifying_the_current_tree(self):
        """The guard is live, not vacuous.

        Renaming a geometry token here without moving the emitters must make the
        tree unclassifiable -- that is the whole mechanism.  Asserting it keeps
        someone from "fixing" a future failure by widening the tables until
        everything matches again.
        """
        if not os.path.isdir(GENERATED):
            self.skipTest("generated tree not present")
        names = _published_names()
        original = conventions.ABI_GEOMETRY_TOKENS
        try:
            conventions.ABI_GEOMETRY_TOKENS = ("aff",) + original[1:]
            broken = [n for n in names if conventions.classify_abi_name(n) is None]
        finally:
            conventions.ABI_GEOMETRY_TOKENS = original
        self.assertGreater(len(broken), 100)
        self.assertEqual([n for n in names if conventions.classify_abi_name(n) is None], [])


class DeclineReasonsTest(unittest.TestCase):
    """The two ways a name gets no dispatch are not interchangeable.

    Collapsing them is how the guard would be quietly disarmed: a future
    mixed-order material adds a sub-block, someone widens the survivable case,
    and a broken parse rides in under the same exemption.
    """

    ELEMENTS = {"tet10_tet4": ("TET10_TET4", 3), "tet10": ("TET10_TET4", 3)}

    def test_a_sub_space_element_is_declined_for_want_of_an_element(self):
        """Taylor-Hood names its pressure block after `tet4`, which the op does
        not mesh, so it cannot be keyed by `smesh::ElemType`."""
        self.assertIs(
            op_wrappers._dispatch_mapping(
                "poro",
                "poro_form_2_p_p_tet4_jacobian_action_a_msoa",
                self.ELEMENTS,
            ),
            op_wrappers._DISPATCH_NO_ELEMENT,
        )

    def test_a_missing_geometry_token_is_declined_as_a_broken_parse(self):
        """The element is found, so the name parsed -- but it carries no marker.

        The name below is the *pre-rename* long form, which is exactly what a
        half-finished rename leaves behind: emitters still spelling
        `affine_mesh_soa` while this layer has moved to `a_msoa`.
        """
        self.assertIs(
            op_wrappers._dispatch_mapping(
                "poro",
                "poro_tet10_apply_affine_mesh_soa",
                self.ELEMENTS,
            ),
            op_wrappers._DISPATCH_NO_MARKER,
        )

    def test_a_well_formed_name_maps(self):
        mapped = op_wrappers._dispatch_mapping(
            "poro", "poro_tet10_apply_a_msoa", self.ELEMENTS
        )
        self.assertEqual(mapped[0], "poro_apply_3d_a_msoa")


if __name__ == "__main__":
    unittest.main()
