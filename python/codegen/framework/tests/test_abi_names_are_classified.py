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
        # The thresholds moved down when the tree stopped publishing a
        # symbol per precision and a wrapper per diagnostics helper.  They
        # are here so the checks cannot pass over an empty set, not as a
        # claim about the right size, so they track what the tree has.
        self.assertGreater(len(self.names), 1200)

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
        self.assertGreater(len(dispatchable), 800)
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


class QualifierSlotTest(unittest.TestCase):
    """The slot between the verb and the geometry has to parse, not be sniffed.

    `_dispatch_source_kind` decides which translation unit a dispatch entry point
    is written into, and it used to decide by looking for `_packed_` in the name.
    A search cannot tell "this kernel is not packed" from "the token for packed
    has moved and I no longer recognise it"; the second reads as the first, the
    packed entry points join the plain ones, and two translation units silently
    become one.  Parsing the slot makes the two distinguishable.
    """

    def setUp(self):
        if not os.path.isdir(GENERATED):
            self.skipTest("generated tree not present")
        self.mesh = [
            n for n in _published_names()
            if (conventions.classify_abi_name(n) or ("",))[0] == "mesh"
        ]

    def test_every_mesh_name_has_a_verb_and_a_known_qualifier(self):
        self.assertGreater(len(self.mesh), 700)
        unparsed = []
        for name in sorted(self.mesh):
            try:
                conventions.abi_qualifier(name)
            except ValueError as error:
                unparsed.append("%s: %s" % (name, error))
        self.assertEqual(unparsed[:5], [], "%d names do not parse" % len(unparsed))

    def test_the_qualifier_slot_is_actually_used(self):
        """A total parse over an always-empty slot proves nothing."""
        seen = collections.Counter(conventions.abi_qualifier(n) for n in self.mesh)
        self.assertGreater(seen[""], 0)
        self.assertGreater(seen["packed"], 0)
        self.assertGreater(seen["packed_two_pass"], 0)

    def test_the_qualifier_table_covers_every_plan_value(self):
        """The table must come from the plans, not from a sample of the output.

        The first version of it was read off the shipped tree, which does not
        emit `packed_one_pass` -- only `tests/test_m11_matrix_formats.py` does.
        Generation refused that name the first time a test reached it.  Reading
        the plans here is what makes the omission visible without waiting for a
        material to exercise the combination.

        `plans/conventions.py` cannot import these enums: `apply_variants` already
        imports the naming table, and the dependency only points one way.  So the
        agreement is asserted rather than derived.
        """
        from codegen.framework.plans.apply_variants import MeshTraversal
        from codegen.framework.plans.matrix_formats import PackedAssemblyPass

        known = {short for _, short in conventions.ABI_TRAVERSAL_SPELLING}
        # `MeshTraversal` is a plain class of string constants; `PackedAssemblyPass`
        # is an Enum.  Read both the way each is written.
        expected = {
            value
            for name, value in vars(MeshTraversal).items()
            if not name.startswith("_") and isinstance(value, str)
            and value != MeshTraversal.STANDARD
        }
        expected |= {
            "packed_%s" % assembly.value
            for assembly in PackedAssemblyPass
            if assembly.value != "none"
        }
        self.assertEqual(
            expected - known,
            set(),
            "a plan value has no spelling in the naming table",
        )
        for qualifier in known:
            self.assertIn(
                qualifier,
                conventions.ABI_TRAVERSAL_UNIT,
                "%s has a spelling but no translation unit" % qualifier,
            )

    def test_a_moved_traversal_token_is_refused(self):
        """The guard is live: move the spelling and the tree stops parsing.

        Asserting this keeps a future failure from being "fixed" by widening the
        table until everything matches again.
        """
        original = conventions.ABI_TRAVERSAL_SPELLING
        try:
            conventions.ABI_TRAVERSAL_SPELLING = (
                ("packed_two_pass", "2p"), ("packed", "pk"),
            )
            refused = 0
            for name in self.mesh:
                try:
                    conventions.abi_qualifier(name)
                except ValueError:
                    refused += 1
        finally:
            conventions.ABI_TRAVERSAL_SPELLING = original
        self.assertGreater(refused, 100)
        for name in self.mesh:
            conventions.abi_qualifier(name)  # and it parses again afterwards

    def test_the_file_split_on_disk_matches_the_parse(self):
        """Each dispatch translation unit holds exactly the kinds it is named for.

        This is the property the sniff used to break without saying so: the file
        was still written, just with the wrong contents in it.
        """
        mismatched = []
        inspected = []
        for base, _dirs, files in os.walk(GENERATED):
            if os.path.basename(base) != "op":
                continue
            for name in files:
                if not name.endswith("_dispatch.cpp") or "diagnostics" in name:
                    continue
                # `sfem_GeneratedLaplace_packed_affine_dispatch.cpp` -- the kind
                # is what follows the operator name.
                stem = name[: -len("_dispatch.cpp")]
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    text = handle.read()
                # The entry points carry the export macro; the plain
                # `extern "C"` lines above them are the private per-element
                # declarations this unit calls into.
                for symbol in re.findall(
                    r'SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int (\w+)\(', text
                ):
                    if (conventions.classify_abi_name(symbol) or ("",))[0] != "mesh":
                        continue
                    inspected.append(symbol)
                    got = op_wrappers._dispatch_source_kind(symbol)
                    if not stem.endswith(got):
                        mismatched.append("%s in %s (parsed %s)" % (symbol, name, got))
        # The check is only worth having if it reached the entry points at all.
        self.assertGreater(len(inspected), 100)
        self.assertEqual(mismatched[:5], [], "%d misfiled" % len(mismatched))


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
