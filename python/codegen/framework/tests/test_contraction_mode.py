"""Which contraction a material gets, and what actually decides it.

The energy/residual split was read as duplication -- two emitters building the
same capabilities twice.  Phase 1b found it is not.  The two paths carry
different mathematical objects and contract them differently:

    residual 1-form   weak     free symbols include u_test_grad_0..2
    energy   1-form   strong   free symbols are the differentiation variables
                               F[0..8]; no test function appears

The energy path contracts its strong form against the test functions later, in
``_append_transformed_loperand_lines`` and ``tensor_test``, and that contraction
is sum-factorised: ``TensorProductResidualOps`` works through one-dimensional
``shape_1d`` and ``grad_1d`` operators with ``integer_root(N_QP)`` per
dimension.  The residual path has already contracted symbolically by the time
emission sees it, so there is nothing left to factorise.

The consequence is visible in the generated tree and is the reason this test
exists: **only the energy materials reach the sum-factorised contraction.**
Laplace, the simplest and most performance-critical operator in the framework,
is written as a residual and therefore does not -- which is the context for the
hand-written low-order fast path recorded as OP 18.

So the split is not energy against residual.  It is strong-contracted against
weak-contracted, and the front end a material happens to be written in silently
decides which one it gets.  Unifying by making the energy path produce weak
forms would remove sum factorisation from the materials that have it; the
direction is the other way, with contraction mode becoming an explicit
form-layer property that either front end can carry.

This pins the fact so the finding cannot quietly stop being true.
"""

import importlib
import unittest

from codegen.framework.symbolic.forms import FormOrder

#: (material, whether its 1-form carries test-function symbols)
#:
#: `laplace` used to be the scalar entry here and is written as an energy now,
#: so `two_phase_flow` stands in for a residual-formulated scalar system.  The
#: fact being pinned is about the two contraction modes, not about any one
#: material, and it stops being pinned if the table holds only one kind.
MATERIALS = (
    ("two_phase_flow", True),
    ("stokes", True),
    ("linear_elasticity", False),
    ("neohookean_ogden", False),
)


def _collection(name, dim=3):
    material = importlib.import_module(
        "codegen.framework.materials.%s" % name
    ).material
    system = material.systems.for_dim(dim)
    return system.form_collection(system.equations[0])


class ContractionModeTest(unittest.TestCase):
    maxDiff = None

    def test_the_two_kinds_carry_different_objects_at_one_form(self):
        """A weak form names its test function; a strong one does not."""
        for name, weak in MATERIALS:
            with self.subTest(material=name):
                collection = _collection(name)
                expression = collection.form(FormOrder.ONE).expression
                symbols = {str(symbol) for symbol in expression.free_symbols}
                has_test = any("test" in symbol for symbol in symbols)
                self.assertEqual(
                    has_test,
                    weak,
                    "%s is a %s formulation and its 1-form %s test-function "
                    "symbols; if that has changed, the contraction mode has "
                    "moved and OP 20 needs revisiting"
                    % (
                        name,
                        collection.kind.value,
                        "should carry" if weak else "should not carry",
                    ),
                )

    def test_weakness_tracks_the_form_kind_today(self):
        """The thing to break: contraction mode is implied by the front end.

        Nothing states the contraction mode; it follows from whether the
        material was written as an energy or as a residual.  When that stops
        being true -- when a residual formulation can ask for the strong,
        sum-factorised contraction -- this test is what should fail.
        """
        for name, weak in MATERIALS:
            with self.subTest(material=name):
                collection = _collection(name)
                self.assertEqual(
                    collection.kind.value == "residual",
                    weak,
                    "contraction mode still tracks the form kind exactly",
                )


if __name__ == "__main__":
    unittest.main()
