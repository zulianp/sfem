"""The stored-tangent apply is opt-in, and what it publishes when it is opted into.

A material that asks for it publishes, per element, the material's tangent at
one point bound to that element (`<unit>_<element>_inexact.hpp`), and the
element's own tables (`reference/inexact_<element>_q<n>.hpp`); its generated Op
overrides the `Op` stored-tangent methods and hands everything else to the
library (`src/frontend/ops/inexact/`).  These tests pin both halves: that
asking nothing changes nothing, and that asking produces exactly that.

The numerical comparison is `tests/test_inexact_point_tangent.py` (against the
directly integrated action) and the library's Newton cases.
"""

import dataclasses
import importlib
import os
import sys
import unittest


MATERIALS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
)


def _material(name):
    if MATERIALS not in sys.path:
        sys.path.insert(0, MATERIALS)
    return importlib.import_module(name).material


class InexactApplyGenerationTest(unittest.TestCase):
    def test_it_is_off_by_default(self):
        """The mechanism refuses unless asked, whatever the shipped materials ask for."""
        from sfem import gen

        defaults = {f.name: f.default for f in dataclasses.fields(gen.CodeGenerator)}
        self.assertFalse(defaults["inexact_apply"])
        self.assertFalse(_material("laplace").inexact_apply)

    def test_nothing_is_emitted_unless_asked(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=False)
        self.assertFalse([name for name in files if "inexact" in name])

    def test_opting_in_publishes_the_unit_and_the_element(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        emitted = sorted(name for name in files if "inexact" in name)
        self.assertEqual(
            emitted,
            [
                "d3/tet4/linear_elasticity_tet4_inexact.hpp",
                "reference/inexact_contract_d3.hpp",
                "reference/inexact_tet4_q1.hpp",
            ],
        )

    def test_the_point_tangent_reads_only_its_arguments(self):
        """The material at one point: no lanes, no connectivity, no stored data.

        Everything the set-up reads besides the material -- the gather, the
        gradients, the geometry -- is the element's or the library's, which is
        what lets one container serve every material.
        """
        files = self._generate("neohookean_ogden", "HEX8", opt_in=True)
        source = files["d3/hex8/neohookean_ogden_hex8_inexact.hpp"]
        body = source[source.index("void point("):]
        for absent in ("lane", "elements", "stride", "evb"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, body)
        for expected in ("gu_ref", "adjugate", "determinant", "parameters", "tangent["):
            with self.subTest(expected=expected):
                self.assertIn(expected, body)
        self.assertIn("using element = inexact_hex8_q8;", source)
        self.assertIn("static constexpr int NC = 45;", source)

    def test_the_element_header_is_material_independent(self):
        """Two materials on one element emit the same element tables, byte for byte."""
        first = self._generate("linear_elasticity", "TET10", opt_in=True)
        second = self._generate("neohookean_ogden", "TET10", opt_in=True)
        for name in ("reference/inexact_tet10_q11.hpp", "reference/inexact_contract_d3.hpp"):
            with self.subTest(header=name):
                self.assertEqual(first[name], second[name])

    @staticmethod
    def _generate(name, element, opt_in):
        import tempfile

        from sfem import gen

        material = dataclasses.replace(_material(name), inexact_apply=opt_in)
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(material, out_dir, elements=(element,), clean=True)
            files = {}
            for path in result.sources:
                relative = os.path.relpath(path, out_dir)
                # The shared reference headers are hoisted to the tree root.
                relative = relative[3:] if relative.startswith("../") else relative
                with open(path, encoding="utf-8") as handle:
                    files[relative] = handle.read()
            return files


class InexactApplyOpWrapperTest(InexactApplyGenerationTest):
    """The generated Op has to offer the split, or SFEM cannot reach it."""

    def _op_files(self, opt_in):
        files = self._generate("linear_elasticity", "TET4", opt_in=opt_in)
        header = files["op/sfem_GeneratedLinearElasticity.hpp"]
        source = files["op/sfem_GeneratedLinearElasticity.cpp"]
        return header, source

    def test_the_op_declares_the_stored_tangent_methods(self):
        header, source = self._op_files(opt_in=True)
        for declaration in (
            "bool inexact_supported() const override;",
            "bool tangent_is_symmetric() const override;",
            "void set_inexact_options(const InexactOptions &options) override;",
            "int inexact_update(const real_t *const x) override;",
            "int inexact_apply(const real_t *const h, real_t *const out) override;",
            "std::size_t inexact_bytes() const override;",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, header)
        self.assertIn("inexact::create_store<codegen::linear_elasticity_tet4_inexact>", source)

    def test_the_apply_prefers_the_packed_mesh_when_there_is_one(self):
        _header, source = self._op_files(opt_in=True)
        body = source[source.index("::inexact_apply("):]
        self.assertIn("has_packed_mesh()", body)
        self.assertIn("packed_view(", body)

    def test_an_op_without_the_split_declares_nothing(self):
        """The default `inexact_supported() == false` has to survive."""
        header, source = self._op_files(opt_in=False)
        self.assertNotIn("inexact_supported", header)
        self.assertNotIn("inexact_update", header)
        self.assertNotIn("InexactStore", source)

    def test_the_op_refuses_a_stale_apply_and_takes_no_state(self):
        _header, source = self._op_files(opt_in=True)
        start = source.index("::inexact_apply(")
        body = source[start:source.index("\n  }", start)]
        self.assertIn("requires inexact_update first", body)
        for absent in ("require_real_value", "input.u", "jacobian"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, body)


if __name__ == "__main__":
    unittest.main()
