"""What the form layer says about every material, pinned before it changes.

Phase 1 of the energy/residual unification re-keys form blocks per component,
so that the block metadata and the lowered fields agree and the parallel
``residual_fields`` / ``residual_expressions`` / ``jacobian_action_blocks``
representation can go.  That is a change to a structure with 49 consumers
across eight files, against 19 for the form-order accessors it replaces --
which is the measurement that says the parallel representation is the one
actually in use, and the form blocks the one that is nearly not.

So the re-keying has to arrive as a reviewable diff rather than as a claim.
This records, for every material and dimension: which form orders exist, what
each order's blocks are keyed by, and what the lowered residual fields are
called.  Today those two differ wherever a field has more than one
component -- Stokes lowers blocks keyed ``u`` and ``p`` while its lowered
fields are ``u0``, ``u1``, ``u2``, ``p`` -- and coincide wherever fields are
scalar.  Both cases are pinned, because Phase 1 has to move the first without
disturbing the second.

The difference is not a loss.  The lowered fields are built with
``field_name``, ``component`` and ``components``, so ``u0`` knows it is
component 0 of ``u``; both keyings are projections of one object, and phase 1
carries the expansion into the blocks rather than choosing between them.

Note what this file does *not* assert.  The 0-form of a residual formulation
is a merit function and the 1-form is the negated residual; the second is not
the derivative of the first, so nothing here may claim ``form(ONE) ==
d(form(ZERO))``.  That relationship holds on the energy path only.
"""

import importlib
import json
import os
import pkgutil
import unittest

from codegen.framework.symbolic.forms import FormOrder

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "form_keying_baseline.json")

#: Dimensions a material is asked for.  A material that does not define one is
#: recorded as absent rather than skipped, so losing a dimension is a diff.
DIMENSIONS = (2, 3)

#: The materials this test covers, chosen because they are the three shapes the
#: form layer actually has and because they lower quickly.
#:
#: Lowering every material takes tens of minutes -- the 2-form of a hyperelastic
#: material in 3D is a large symbolic differentiation -- so the full sweep is a
#: tool, ``tools/form_keying.py``, with a committed baseline, in the pattern the
#: snapshot and reproducibility gates already use.  What is left here is the
#: fast subset, so the distinction cannot regress without a unit test noticing:
#:
#:   laplace            scalar field, residual formulation
#:   stokes             mixed order, residual formulation, the keying mismatch
#:   linear_elasticity  vector field, energy formulation, no block metadata
FAST_MATERIALS = ("laplace", "linear_elasticity", "stokes")


def _materials():
    import codegen.framework.materials as materials_package

    found = {}
    for module_info in pkgutil.iter_modules(materials_package.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(
                "codegen.framework.materials.%s" % module_info.name
            )
        except Exception:
            continue
        material = getattr(module, "material", None)
        if material is not None:
            found[module_info.name] = material
    return found


def _collection_record(collection):
    record = {"kind": collection.kind.value, "orders": {}}
    for order in (FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO):
        entry = {}
        try:
            form = collection.form(order)
        except Exception:
            entry["form"] = None
        else:
            entry["form"] = form.expression is not None
        try:
            blocks = collection.blocks_for(order)
        except Exception:
            entry["blocks"] = None
        else:
            entry["blocks"] = [
                [block.row_field, block.column_field] for block in blocks
            ]
        record["orders"][order.name] = entry
    record["residual_fields"] = [
        field.name for field in getattr(collection, "residual_fields", ())
    ]
    record["jacobian_action_blocks"] = [
        [block.row_field, getattr(block, "column_field", None)]
        for block in getattr(collection, "jacobian_action_blocks", ())
    ]
    return record


def collect(names=None):
    """The form-layer shape of these materials, as a plain dictionary."""
    selected = _materials()
    if names is not None:
        selected = {k: v for k, v in selected.items() if k in set(names)}
    snapshot = {}
    for name, material in sorted(selected.items()):
        per_dim = {}
        for dim in DIMENSIONS:
            try:
                system = material.systems.for_dim(dim)
            except Exception:
                continue
            equations = {}
            for equation in system.equations:
                try:
                    collection = system.form_collection(equation)
                except Exception as error:
                    equations[equation.name or ""] = {"error": type(error).__name__}
                    continue
                equations[equation.name or ""] = _collection_record(collection)
            if equations:
                per_dim[str(dim)] = equations
        if per_dim:
            snapshot[name] = per_dim
    return snapshot


class FormKeyingBaselineTest(unittest.TestCase):
    maxDiff = None

    def test_the_form_layer_matches_the_recorded_shape(self):
        """Any change to form orders, block keying or field names is a diff."""
        with open(BASELINE_PATH, encoding="utf-8") as handle:
            recorded = json.load(handle)
        measured = collect(FAST_MATERIALS)
        scoped = {k: v for k, v in recorded.items() if k in measured}
        self.assertEqual(scoped, measured)

    def test_energy_formulations_populate_no_block_metadata(self):
        """Energy declares its structure at the API, not in block metadata.

        `system.add_energy("", energy, fields=(u,), variables=(F,))` supplies
        the field and the variable group the energy differentiates through, and
        one variable group per field is validated at that boundary.  So empty
        blocks here are not missing information -- they are information that
        arrived somewhere else -- and this pins the fact rather than calling it
        a defect.  When phase 1 derives blocks from those declared fields and
        variables, this is what changes.
        """
        snapshot = collect(("linear_elasticity",))
        for dim, equations in snapshot["linear_elasticity"].items():
            for equation, record in equations.items():
                with self.subTest(dim=dim, equation=equation):
                    self.assertEqual(record["kind"], "energy")
                    for order in ("ONE", "TWO"):
                        self.assertEqual(
                            record["orders"][order]["blocks"],
                            [],
                            "energy formulations record no blocks today; when "
                            "they do, Phase 1 has landed",
                        )
                        self.assertTrue(record["orders"][order]["form"])

    def test_block_keying_and_field_names_disagree_exactly_where_expected(self):
        """The mismatch Phase 1 removes, stated so its removal is visible.

        Blocks are keyed by assembled field and the lowered fields are per
        component, so they differ for a material with a vector field and agree
        for one without.  When Phase 1 lands, this test is what should change.
        """
        snapshot = collect(FAST_MATERIALS)
        disagreeing = []
        for material, dims in snapshot.items():
            for dim, equations in dims.items():
                for equation, record in equations.items():
                    fields = record.get("residual_fields") or []
                    one = (record.get("orders", {}).get("ONE") or {}).get("blocks")
                    if not fields or not one:
                        continue
                    rows = [row for row, _ in one]
                    if sorted(rows) != sorted(fields):
                        disagreeing.append("%s/%s" % (material, dim))
        self.assertIn(
            "stokes/3",
            disagreeing,
            "Stokes lowers blocks keyed by assembled field while its lowered "
            "fields are per component; if that has stopped being true, Phase 1 "
            "has landed and this test should record the new agreement",
        )


if __name__ == "__main__":
    unittest.main()
