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

    def test_it_takes_the_arguments_the_exact_apply_takes(self):
        """Same arguments, plus the state the exact linear kernel does not need.

        A linear material's tangent does not depend on the state, so its exact
        apply takes only the increment.  The inexact kernel builds its tangent
        from the state, so it always takes one -- and passing zero recovers the
        exact operator, which is how the comparison in the spike is set up.
        """
        files = self._generate("linear_elasticity", "TET4", opt_in=True)
        source = files["d3/tet4/linear_elasticity_tet4_inexact_apply_inline.hpp"]
        self.assertIn("linear_elasticity_tet4_apply_inexact_affine_mesh_soa_impl", source)
        for expected in (
            "const ptrdiff_t nelements",
            "idx_t **const SFEM_RESTRICT elements",
            "g_jacobian_adjugate8",
            "g_jacobian_determinant0",
            "const scalar_t mu",
            "const scalar_t lmbda",
            "const ptrdiff_t u_stride",
            "const ptrdiff_t h_stride",
            "scalar_t *const SFEM_RESTRICT outz",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)
        # No quadrature loop and no reference tables: that is the whole point.
        self.assertNotIn("N_QP", source)
        self.assertNotIn("q_weight", source)
        self.assertNotIn("grad_ref", source)

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
