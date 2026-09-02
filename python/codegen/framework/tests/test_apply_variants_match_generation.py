"""The apply-variant plan is checked against what generation actually emits.

``plans/apply_variants.py`` states which matrix-free apply kernels each
formulation and element gets.  That rule was recovered from a full regeneration
and matched 181 of 181 apply kernels -- but a one-off match in a session is not
a guarantee, and the 44 MB reference tree the check used is not committed.

This regenerates a representative kernel per rule dimension and requires the
emitted apply symbols to be exactly the set the plan predicts.  Not a subset and
not a superset: an emitter that quietly gains or loses a variant fails here.

The materials are chosen to cover every dimension of the rule rather than for
breadth:

    laplace / HEX8    packed traversal, curved element -- all three packed
                      shapes, plus the equal-order AoS variant
    laplace / TET4    packed traversal, affine-equivalent linear simplex --
                      packed affine only
    two_phase_flow    equal-order, no packed support -- AoS but no packed
    stokes            mixed order (Taylor-Hood) -- no AoS at all

Generation is not cheap, so this stays deliberately small.  The exhaustive
check is a full snapshot regeneration.
"""

import re
import unittest

from codegen.framework.plans.apply_variants import apply_variant_plan


#: (material, element, plan inputs) covering each dimension of the rule.
CASES = (
    ("laplace", "HEX8", dict(mixed_order=False, supports_packed=True,
                             affine_equivalent_element=False)),
    ("laplace", "TET4", dict(mixed_order=False, supports_packed=True,
                             affine_equivalent_element=True)),
    ("two_phase_flow", "TRI3", dict(mixed_order=False, supports_packed=False,
                                    affine_equivalent_element=True)),
    ("stokes", None, dict(mixed_order=True, supports_packed=False,
                          affine_equivalent_element=False)),
)

#: Matches an emitted apply entry point and pulls its variant suffix out.
APPLY_SYMBOL = re.compile(
    r"\bint\s+[a-z0-9_]+?_(?:jacobian_action|residual)_"
    r"((?:packed_two_pass_|packed_)?(?:affine|isoparametric)_mesh_(?:soa|aos)(?:_float)?)\s*\("
)


def _generate(material_name, element):
    import importlib

    from sfem import gen

    material = importlib.import_module(
        "codegen.framework.materials.%s" % material_name
    ).material
    elements = (element,) if element else tuple(getattr(material, "elements", ()) or ())[:1]
    user_input = gen.UserInputStage.create(material, elements, 8, None)
    form_evaluation = gen._evaluate_forms(user_input)
    codegen_plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
    return gen.CodeGenerationStage(user_input, codegen_plan).run()


def _emitted_suffixes(files, form):
    """Variant suffixes of every emitted apply symbol for ``form``."""
    pattern = re.compile(
        r"\bint\s+[a-z0-9_]+?_%s_"
        r"((?:packed_two_pass_|packed_)?(?:affine|isoparametric)_mesh_(?:soa|aos)(?:_float)?)\s*\("
        % form
    )
    found = set()
    for source in files.values():
        found.update(pattern.findall(source))
    return found


class ApplyVariantsMatchGenerationTest(unittest.TestCase):
    maxDiff = None

    def _check(self, material_name, element, form, plan_inputs):
        files = _generate(material_name, element)
        emitted = _emitted_suffixes(files, form)
        if not emitted:
            self.skipTest("%s emits no %s apply kernels" % (material_name, form))
        plan = apply_variant_plan(
            is_jacobian_action=(form == "jacobian_action"), **plan_inputs
        )
        expected = set(plan.suffixes())
        self.assertEqual(
            emitted,
            expected,
            "%s/%s %s: emitted variants do not match the plan\n"
            "  only emitted: %s\n  only planned: %s"
            % (
                material_name,
                element or "default",
                form,
                sorted(emitted - expected),
                sorted(expected - emitted),
            ),
        )

    def test_jacobian_action_variants_match_the_plan(self):
        for material_name, element, plan_inputs in CASES:
            with self.subTest(material=material_name, element=element):
                self._check(material_name, element, "jacobian_action", plan_inputs)

    def test_residual_variants_match_the_plan(self):
        for material_name, element, plan_inputs in CASES:
            with self.subTest(material=material_name, element=element):
                self._check(material_name, element, "residual", plan_inputs)


if __name__ == "__main__":
    unittest.main()
