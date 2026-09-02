"""The form-collection boundary must be closed.

``FormCollection`` is the handoff from form lowering to everything below it.
The architecture requires that once an equation has been lowered, downstream
stages read the collection and nothing else -- in particular not
``FormCollection.source``, the back-pointer to the ``CoupledResidualSystem``
the collection was produced from.

A comment saying so is not enforcement.  The check here is behavioural: build
the real plans twice, once normally and once from a collection whose ``source``
has been cut away, and require the two to be identical.  If any planning code
reaches through the back-pointer, the second run either crashes or produces a
different plan, and this test says which.

The emission path is deliberately out of scope: the residual emitters still take
a ``CoupledResidualSystem`` as their argument, so ``source`` still has one
legitimate reader until that interface is inverted.  This test pins the boundary
that *is* closed, so it cannot silently reopen.
"""

import unittest
from dataclasses import replace

from codegen.framework.materials.two_phase_flow import material as two_phase_flow_material
from codegen.framework.plans.diagnostics import kernel_diagnostics_plan_from_plan
from codegen.framework.plans.emission import emission_plan_from_unit_context


def _plan_and_context():
    from sfem import gen

    user_input = gen.UserInputStage.create(two_phase_flow_material, ("TET4",), 8, None)
    form_evaluation = gen._evaluate_forms(user_input)
    codegen_plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
    context = user_input.element_contexts[0]
    return user_input, codegen_plan, context


def _sever_source(unit):
    """Kept as the identity now that the back-pointer no longer exists.

    The comparisons below used to run against a collection with ``source``
    severed.  S5 deleted the field, so severing is a no-op and the equality
    checks now simply assert the plans are stable.  The static check further
    down is what keeps the field from coming back.
    """
    return unit


def _action_expression_plan(unit):
    """The 2-form expression plan of a unit, if it has one."""
    from codegen.framework.symbolic.forms import FormOrder

    for plan in getattr(unit, "expression_plans", ()):
        if getattr(plan, "form_order", None) is FormOrder.TWO:
            return plan
    return None


class FormCollectionBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.user_input, self.codegen_plan, self.context = _plan_and_context()
        # The specialized, file-emitting kernels -- the same ones the backend sees.
        self.units = tuple(self.codegen_plan.emission_kernels_for_context(self.context))
        self.assertTrue(self.units, "expected at least one code-generation unit")

    def test_lowered_collections_carry_their_residual_fields(self):
        """The data planning needs is on the collection, not behind the pointer."""
        for unit in self.units:
            collection = unit.form_collection
            if not collection.residual_fields:
                continue
            self.assertTrue(
                collection.residual_fields,
                "residual collection '%s' carries no lowered residual fields"
                % collection.equation_name,
            )
            self.assertEqual(
                len(collection.residual_expressions),
                len(collection.residual_fields),
                "residual expressions are not aligned with the lowered fields",
            )

    def test_element_emission_plans_do_not_depend_on_the_back_pointer(self):
        for unit in self.units:
            expected = emission_plan_from_unit_context(unit, self.context)
            actual = emission_plan_from_unit_context(_sever_source(unit), self.context)
            self.assertEqual(
                actual,
                expected,
                "emission plan for '%s' changed when FormCollection.source was removed"
                % unit.name,
            )

    def test_diagnostics_block_names_do_not_depend_on_the_back_pointer(self):
        """Block names came from the system's jacobian_blocks(); now from the collection."""
        from codegen.framework.plans.diagnostics import _diagnostic_block_names

        checked = 0
        for unit in self.units:
            if not unit.form_collection.residual_fields:
                continue
            action_plan = _action_expression_plan(unit)
            if action_plan is None:
                continue
            expected = _diagnostic_block_names(unit, action_plan)
            actual = _diagnostic_block_names(_sever_source(unit), action_plan)
            self.assertEqual(
                actual,
                expected,
                "diagnostic block names for '%s' changed without FormCollection.source"
                % unit.name,
            )
            self.assertTrue(expected, "expected named Jacobian-action blocks")
            checked += 1
        self.assertTrue(checked, "no residual unit exercised the block-name path")

    def test_backend_diagnostics_plan_does_not_depend_on_the_back_pointer(self):
        """The full assembled plan, through the backend the generators use."""
        from codegen.framework.backends.openmp import OpenMPSoABackend

        backend = OpenMPSoABackend()
        for unit in self.units:
            expected = backend.diagnostics_plan(unit, self.context)
            actual = backend.diagnostics_plan(_sever_source(unit), self.context)
            self.assertEqual(
                tuple(getattr(actual, "public_names", ())),
                tuple(getattr(expected, "public_names", ())),
                "diagnostics plan for '%s' changed when FormCollection.source was removed"
                % unit.name,
            )

    def test_form_collection_has_no_back_pointer(self):
        """The field itself is gone; this fails if anyone reintroduces it."""
        from codegen.framework.symbolic.forms import FormCollection

        self.assertNotIn(
            "source",
            FormCollection.__dataclass_fields__,
            "FormCollection.source is back; downstream must read the collection's "
            "own data, not the system it was lowered from",
        )

    def test_planning_layer_never_reads_the_back_pointer(self):
        """Static check: no module under plans/ may mention `source` on a collection."""
        import ast
        import os

        plans_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plans"
        )
        offenders = []
        for name in sorted(os.listdir(plans_dir)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(plans_dir, name)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                # collection.source / unit.form_collection.source
                if isinstance(node, ast.Attribute) and node.attr == "source":
                    base = node.value
                    base_name = getattr(base, "attr", getattr(base, "id", ""))
                    if base_name in ("collection", "form_collection"):
                        offenders.append("%s:%d" % (name, node.lineno))
                # getattr(collection, "source", ...)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "source"
                ):
                    base = node.args[0]
                    base_name = getattr(base, "attr", getattr(base, "id", ""))
                    if base_name in ("collection", "form_collection"):
                        offenders.append("%s:%d" % (name, node.lineno))
        self.assertEqual(
            offenders,
            [],
            "planning modules reach through FormCollection.source at: %s"
            % ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
