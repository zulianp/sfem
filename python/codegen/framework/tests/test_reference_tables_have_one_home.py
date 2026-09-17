"""Reference tables are keyed by basis and rule, not by material.

Shape functions, shape gradients and quadrature weights are emitted inline into
every per-element source today, as a struct named after the material:
`laplace_tet10_affine_reference_data` and `neohookean_ogden_tet10_affine_reference_data`
are byte-identical bodies under different names.  Across the shipped tree that
is 174 struct definitions holding 20 distinct tables, and 32 of the structs are
defined in two files each -- the element header and the operator source, which
does not include it.  Identical definitions across translation units are
ODR-legal, so it links; it stays legal only while two independent emission paths
keep producing the same bytes, and nothing checks that they do.

This file pins the identity the tables will be shared under.  The key is what
makes sharing safe, so it is what needs testing:

  - the split is lossless -- a basis's tables plus its rule's weights are exactly
    what the single combined table used to be, so nothing changes by separating
    them;
  - the key distinguishes what must be distinguished, and merges what may be
    merged.  Those are two different failure modes and both are cheap to state.
"""

import os
import re
import unittest

from codegen.framework.emitters import quadrature_codegen
from codegen.framework.fem.reference import (
    sfem_basis_reference_data,
    sfem_element_quadrature_rule,
    sfem_quadrature_reference_data,
    sfem_reference_data,
)
from codegen.framework.plans.reference_data import ReferenceBasisDataPlan


def _tables(entries):
    return tuple((entry.name, entry.values) for entry in entries)


class TheSplitIsLosslessTest(unittest.TestCase):
    """Separating the basis from the rule must not change a single number."""

    def test_basis_plus_rule_is_the_combined_table(self):
        for element_type, order in (
            ("TET4", 1), ("TET4", 4), ("TET10", 4), ("TRI3", 1), ("TRI6", 4),
            ("QUAD4", None), ("HEX8", None), ("PROTEUS_HEX8", None),
        ):
            with self.subTest(element=element_type, order=order):
                rule = sfem_element_quadrature_rule(element_type, order)
                combined = _tables(sfem_reference_data(rule))
                split = _tables(
                    sfem_basis_reference_data(element_type, rule)
                    + sfem_quadrature_reference_data(rule)
                )
                self.assertEqual(split, combined)


class TheKeyIsTheIdentityTest(unittest.TestCase):
    def _basis(self, element_type, rule, family, n_shape_1d=0, n_qp_1d=0):
        shape = "shape_1d" if family == "tensor_product" else "shape"
        return ReferenceBasisDataPlan(
            "cell", element_type, "", family, rule.n_shape, rule.n_qp,
            n_shape_1d, n_qp_1d, shape, (),
            sfem_basis_reference_data(element_type, rule),
        )

    def test_the_same_basis_at_a_different_rule_is_a_different_key(self):
        """The HEX8 basis is sampled at two points per direction under its own
        cell rule and at three under a HEX27 one.  Same polynomial basis, two
        tables -- so `line_p1_q2` and `line_p1_q3` must not collide."""
        rule = sfem_element_quadrature_rule("HEX8", None)
        two = self._basis("HEX8", rule, "tensor_product", n_shape_1d=2, n_qp_1d=2)
        three = self._basis("HEX8", rule, "tensor_product", n_shape_1d=2, n_qp_1d=3)
        self.assertNotEqual(two.key, three.key)
        self.assertEqual(two.key, "line_p1_q2")
        self.assertEqual(three.key, "line_p1_q3")

    def test_a_tensor_product_basis_does_not_key_on_its_element(self):
        """The tables are 1-D, so PROTEUS_HEX8 and PROTEUS_QUAD4 share them.

        Keying on the element instead would keep four copies of one 424-byte
        body -- which is what the tree holds today, 28 times over.
        """
        hexa = sfem_element_quadrature_rule("PROTEUS_HEX8", None)
        quad = sfem_element_quadrature_rule("PROTEUS_QUAD4", None)
        a = self._basis("PROTEUS_HEX8", hexa, "tensor_product", n_shape_1d=2, n_qp_1d=2)
        b = self._basis("PROTEUS_QUAD4", quad, "tensor_product", n_shape_1d=2, n_qp_1d=2)
        self.assertEqual(a.key, b.key)
        self.assertEqual(_tables(a.tables), _tables(b.tables))

    def test_a_field_basis_is_the_same_table_as_that_element_owns(self):
        """This is what lets a Taylor-Hood pair decompose into headers a
        pure-element material already uses: the TET4 basis under a TET10 cell
        rule *is* the TET4 basis at that rule."""
        cell = sfem_element_quadrature_rule("TET10", 4)
        own = sfem_element_quadrature_rule("TET4", 4)
        self.assertEqual(cell.n_qp, own.n_qp)
        self.assertEqual(
            _tables(sfem_basis_reference_data("TET4", cell)),
            _tables(sfem_basis_reference_data("TET4", own)),
        )


class TheHeaderIsSelfContainedTest(unittest.TestCase):
    def setUp(self):
        rule = sfem_element_quadrature_rule("TET4", 1)
        self.basis = ReferenceBasisDataPlan(
            "cell", "TET4", "", "simplex", 4, 1, 0, 0, "shape",
            ("grad_ref_x", "grad_ref_y", "grad_ref_z"),
            sfem_basis_reference_data("TET4", rule),
        )
        self.source = quadrature_codegen.reference_basis_header_source(
            self.basis, "quad_tet_q1"
        )

    def test_the_path_carries_the_key(self):
        self.assertEqual(
            quadrature_codegen.reference_header_path(self.basis.key),
            "reference/tet4_q1.hpp",
        )

    def test_the_header_guards_itself_and_names_its_rule(self):
        self.assertIn("#ifndef SFEM_CODEGEN_REFERENCE_TET4_Q1_HPP", self.source)
        # A sibling include: the weights have one owner, and the basis header
        # is the thing that says which rule it belongs to.
        self.assertIn('#include "quad_tet_q1.hpp"', self.source)
        self.assertIn("namespace sfem {", self.source)
        self.assertIn("struct ref_tet4_q1 {", self.source)

    def test_the_header_carries_no_weights(self):
        """The weights are the rule's, not the basis's."""
        self.assertNotIn("q_weight", self.source)


GENERATED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "frontend", "ops", "generated",
)

_TABLE = re.compile(r"    static const s_t data\[\d+\] = \{s_t")
_CALL = re.compile(r"sfem::codegen::((?:ref_|quad_)[a-z0-9_]+)<[a-z_]+>::([a-z0-9_]+)\(\)")


class TheTablesLiveInOnePlaceTest(unittest.TestCase):
    """A table is emitted once and forwarded to, never copied.

    The failure this guards is not subtle to state and was easy to reach: an
    emitter prints the numbers inline again, and the tree quietly grows a second
    copy that nothing keeps in step with the first.
    """

    def setUp(self):
        if not os.path.isdir(GENERATED):
            self.skipTest("generated tree not present")
        self.shared, self.elsewhere = [], []
        for base, _dirs, files in os.walk(GENERATED):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
                if not _TABLE.search(text):
                    continue
                relative = os.path.relpath(path, GENERATED)
                (self.shared if relative.startswith("reference" + os.sep)
                 else self.elsewhere).append(relative)

    def test_the_shared_headers_exist_and_hold_tables(self):
        self.assertGreater(len(self.shared), 10)

    def test_no_table_is_written_outside_them(self):
        self.assertEqual(
            sorted(self.elsewhere)[:10],
            [],
            "%d sources carry reference tables of their own" % len(self.elsewhere),
        )

    def test_every_call_names_a_header_that_exists(self):
        """A call into a struct nobody emits is a compile error, so this cannot
        fail silently -- it fails early instead, with the name."""
        # Walked rather than listed: `reference/` carries a `cuda/` folder with
        # the device spelling of the same tables, the way every other directory
        # in the repository keeps its device sources.
        structs = set()
        for base, _dirs, names in os.walk(os.path.join(GENERATED, "reference")):
            for name in names:
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    structs.update(re.findall(r"struct ([a-z0-9_]+) \{", handle.read()))
        self.assertGreater(len(structs), 10)
        unknown, seen = set(), 0
        for base, _dirs, files in os.walk(GENERATED):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                with open(os.path.join(base, name), encoding="utf-8", errors="replace") as handle:
                    for match in _CALL.finditer(handle.read()):
                        seen += 1
                        if match.group(1) not in structs:
                            unknown.add(match.group(1))
        self.assertGreater(seen, 2000)
        self.assertEqual(sorted(unknown), [])

    def test_no_kernel_keeps_a_reference_struct_of_its_own(self):
        """The per-kernel struct is gone, and its name was never an identity.

        `navier_stokes_form_1_p_affine_reference_data` was defined twice in one
        program -- a 6-point triangle rule in the 2-D translation unit, an
        11-point tetrahedron rule in the 3-D one -- because the mixed path names
        its struct from the bare material prefix, with no element in it.  Two
        bodies, one mangled symbol.  Naming a table after the rule it belongs to
        cannot collide that way, and this checks the old names have not returned.

        The boundary kernels are the exception: `emitters/boundary_codegen.py`
        emits its own structs and does not go through `plans/reference_data.py`
        at all.
        """
        offenders = []
        for base, _dirs, files in os.walk(GENERATED):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                relative = os.path.relpath(os.path.join(base, name), GENERATED)
                if "boundary" in relative:
                    continue
                with open(os.path.join(base, name), encoding="utf-8", errors="replace") as handle:
                    if "_reference_data" in handle.read():
                        offenders.append(relative)
        self.assertEqual(sorted(offenders)[:8], [])
