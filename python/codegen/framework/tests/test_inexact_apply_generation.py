"""The inexact apply is opt-in, and what it publishes when it is opted into.

The kernel computes a different operator from the exact apply on anything but
an affine simplex, so it is never generated unless a material asks.  These
tests pin both halves: that asking nothing changes nothing, and that asking
produces one additional entry point with the same arguments the exact apply
takes -- which is what lets the two be driven side by side.

The numerical comparison lives in `spikes/inexact_apply_compare`, because it
needs a compiler and a mesh; it reports agreement to 3.2e-16 on 1029 degrees
of freedom.
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
        """The mechanism refuses unless asked, whatever the shipped materials ask for.

        This used to read `linear_elasticity`'s own flag, which stopped being a
        statement about the default the moment that material started shipping
        the split.  The default belongs to `CodeGenerator`.
        """
        from sfem import gen

        defaults = {f.name: f.default for f in dataclasses.fields(gen.CodeGenerator)}
        self.assertFalse(defaults["inexact_apply"])
        self.assertFalse(_material("laplace").inexact_apply)

    def test_nothing_is_emitted_unless_asked(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=False)
        self.assertFalse([name for name in files if "inexact" in name])

    def test_opting_in_publishes_one_header_per_element(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        emitted = [name for name in files if "inexact" in name]
        self.assertEqual(
            emitted,
            [
                "d3/tet4/linear_elasticity_tet4_inexact_apply_inline.hpp",
                "d3/tet4/linear_elasticity_tet4_inexact_apply_operator.cpp",
            ],
        )

    def test_it_publishes_the_split_pair(self):
        """Assembly and apply, and the apply free of everything but the store.

        The point of the split is that `Sbar` absorbs the geometry, the state
        and the material, so the apply reads only the tangent and the vector.
        That is what makes it cheaper than the exact apply, so it is worth
        pinning rather than leaving to inspection.
        """
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        source = files["d3/tet4/linear_elasticity_tet4_inexact_apply_inline.hpp"]

        assembly = "linear_elasticity_tet4_inexact_apply_tangent_affine_mesh_soa_impl"
        stored = "linear_elasticity_tet4_inexact_apply_stored_affine_mesh_soa_impl"
        compressed = "linear_elasticity_tet4_inexact_apply_compressed_affine_mesh_soa_impl"
        for name in (assembly, stored, compressed):
            with self.subTest(kernel=name):
                self.assertIn(name, source)

        # The assembly takes the geometry, the material and the state, and
        # writes the tangent.
        body = source[source.index(assembly):source.index(stored)]
        for expected in (
            "g_jacobian_adjugate8",
            "g_jacobian_determinant0",
            "const scalar_t mu",
            "const scalar_t lmbda",
            "tangent_t *const SFEM_RESTRICT tangent",
        ):
            with self.subTest(assembly=expected):
                self.assertIn(expected, body)

        # The apply takes neither, and no state either.
        body = source[source.index(stored):source.index(compressed)]
        for absent in (
            "g_jacobian_adjugate",
            "g_jacobian_determinant",
            "const scalar_t mu",
            "const scalar_t lmbda",
            "u_stride",
        ):
            with self.subTest(absent_from_apply=absent):
                self.assertNotIn(absent, body)
        for expected in (
            "const tangent_t *const SFEM_RESTRICT tangent",
            "const ptrdiff_t h_stride",
            "scalar_t *const SFEM_RESTRICT outz",
        ):
            with self.subTest(apply=expected):
                self.assertIn(expected, body)

    def test_the_fused_kernel_is_gone(self):
        """The fused apply was scaffolding and is not emitted.

        It rebuilt the tangent on every apply, which is strictly more work than
        the exact apply it was compared against.  Its job was to be driven with
        the exact kernel's arguments while the split was built; the stored apply
        gates correctness now.
        """
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        source = files["d3/tet4/linear_elasticity_tet4_inexact_apply_inline.hpp"]
        self.assertNotIn("apply_inexact_affine_mesh_soa", source)

    @staticmethod
    def _generate(name, element, opt_in):
        import tempfile

        from sfem import gen

        material = dataclasses.replace(_material(name), inexact_apply=opt_in)
        with tempfile.TemporaryDirectory() as out_dir:
            result = gen.generate(
                material, out_dir, elements=(element,), clean=True
            )
            files = {}
            for path in result.sources:
                with open(path, encoding="utf-8") as handle:
                    files[os.path.relpath(path, out_dir)] = handle.read()
            return files


if __name__ == "__main__":
    unittest.main()


class InexactApplyAbiTest(InexactApplyGenerationTest):
    """The split has to be reachable from SFEM, not only from a spike.

    The templated kernels live in the header; the symbols the library links
    against live in the operator source.  Each is emitted once per scalar type,
    because the dispatch layer collapses a `<name>`/`<name>_float` pair into one
    public entry point taking `smesh::PrimitiveType` -- which is the shape the
    rest of SFEM's C ABI already has.
    """

    def _operator_source(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        return files["d3/tet4/linear_elasticity_tet4_inexact_apply_operator.cpp"]

    def test_each_kernel_crosses_the_abi_at_both_precisions(self):
        source = self._operator_source()
        for kernel in ("tangent", "stored", "compressed"):
            for suffix in ("", "_float"):
                name = ("linear_elasticity_tet4_inexact_apply_%s"
                        "_affine_mesh_soa%s" % (kernel, suffix))
                with self.subTest(kernel=kernel, precision=suffix or "double"):
                    self.assertIn('extern "C" int %s(' % name, source)

    def test_the_store_uses_sfem_types(self):
        """`metric_tensor_t` and `compressed_t` are what the library stores."""
        source = self._operator_source()
        self.assertIn("metric_tensor_t *const SFEM_RESTRICT tangent", source)
        self.assertIn("const compressed_t *const SFEM_RESTRICT tangent", source)
        self.assertIn("const scaling_t *const SFEM_RESTRICT scaling", source)

    def test_the_apply_still_takes_no_geometry_or_state(self):
        """The ABI must not reintroduce what the split exists to remove."""
        source = self._operator_source()
        start = source.index("linear_elasticity_tet4_inexact_apply_stored_affine_mesh_soa(")
        body = source[start:source.index("}", start)]
        for absent in ("g_jacobian", "u_stride", "mu", "lmbda"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, body)


class InexactApplyOpWrapperTest(InexactApplyGenerationTest):
    """The generated Op has to offer the split, or SFEM cannot reach it.

    `Op` declares `inexact_supported`, `inexact_update` and `inexact_apply` with
    defaults that refuse, so an Op that generates nothing overrides nothing and
    every existing caller is unaffected.  These pin the other half: that a
    material which asks for the split says so, owns the store, and cannot apply
    from one that was never assembled.
    """

    def _op_files(self, opt_in):
        files = self._generate("linear_elasticity", "TET4", opt_in=opt_in)
        header = files["op/sfem_GeneratedLinearElasticity.hpp"]
        source = files["op/sfem_GeneratedLinearElasticity.cpp"]
        return header, source

    def test_the_op_declares_the_inexact_methods(self):
        header, _source = self._op_files(opt_in=True)
        self.assertIn("bool inexact_supported() const override { return true; }", header)
        self.assertIn("int inexact_update(const real_t *const x) override;", header)
        self.assertIn(
            "int inexact_apply(const real_t *const h, real_t *const out) override;", header
        )

    def test_an_op_without_the_split_declares_nothing(self):
        """The default `inexact_supported() == false` has to survive."""
        header, source = self._op_files(opt_in=False)
        self.assertNotIn("inexact_supported", header)
        self.assertNotIn("inexact_update", header)
        self.assertNotIn("inexact_tangent", source)

    def test_the_op_owns_the_store_and_refuses_a_stale_apply(self):
        _header, source = self._op_files(opt_in=True)
        self.assertIn("SharedBuffer<metric_tensor_t> inexact_tangent;", source)
        # sized once, refilled in place
        self.assertIn("if (!cache->inexact_tangent) {", source)
        # and an apply before an update is an error, not an empty answer
        start = source.index("::inexact_apply(")
        body = source[start:source.index("\n    }", start)]
        self.assertIn("requires inexact_update first", body)

    def test_the_op_apply_takes_no_state(self):
        """The whole point, restated where SFEM will see it."""
        _header, source = self._op_files(opt_in=True)
        start = source.index("::inexact_apply(")
        body = source[start:source.index("\n    }", start)]
        for absent in ("jacobian_soa", "require_real_value", "x + 0"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, body)
