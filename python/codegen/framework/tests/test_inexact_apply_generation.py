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
        self.assertFalse(_material("linear_elasticity").inexact_apply)

    def test_nothing_is_emitted_unless_asked(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=False)
        self.assertFalse([name for name in files if "inexact" in name])

    def test_opting_in_publishes_one_header_per_element(self):
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        emitted = [name for name in files if "inexact" in name]
        self.assertEqual(
            emitted, ["d3/tet4/linear_elasticity_tet4_inexact_apply_inline.hpp"]
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

        material = _material(name)
        if opt_in:
            material = dataclasses.replace(material, inexact_apply=True)
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
