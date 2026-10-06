"""An assembly plan must name the streams its kernels actually use.

Connecting the last four assembly plans found that two of them did not.
``DIAAssemblyPlan.diagonal_offsets`` said ``"diagonal_offsets"`` while every
generated DIA scatter reads a parameter called ``diag_offsets``, and
``COOAssemblyPlan`` said ``"rowidx"``/``"colidx"`` where the kernels write
``rows`` and ``cols``.  Nothing was wrong with the generated code -- the plans
were simply describing something else, and they could, because no emitter read
them.  A plan nobody consults is free to be wrong about the thing it exists to
define.

So this pins the agreement rather than trusting it.  Two assertions, because
the weaker one alone is satisfied by an emitter that ignores its plan and
happens to hardcode the same name: the names a plan holds must appear in the
emitted scatter, *and* renaming one in the plan must change the kernel.

It is a weaker guarantee than deriving every occurrence from the plan.  The
ABI names are still repeated across roughly a hundred literal sites in the two
emitters -- the mesh-operator signatures and the call sites that pass these
buffers in -- so changing a plan renames the scatter's own parameter and not
its callers.  Centralizing those is open work; until then this is what keeps
the plan's description honest.
"""

import unittest

from codegen.framework.emitters import energy_codegen, residual_codegen
from codegen.framework.plans.matrix_formats import (
    BlockDiagSymAssemblyPlan,
    BSRAssemblyPlan,
    CRSAssemblyPlan,
)


class Scatter:
    """One format's plan, the emitter that spells it, and what it must name."""

    def __init__(self, label, plan_type, emit, fields, has_reduction_policy=True):
        self.label = label
        self.plan_type = plan_type
        self._emit = emit
        self.fields = fields
        self.has_reduction_policy = has_reduction_policy

    def text(self, assembly=None):
        return "\n".join(self._emit(assembly))


SCATTERS = (
    Scatter(
        "CRS",
        CRSAssemblyPlan,
        lambda a: residual_codegen._scalar_crs_matrix_scatter_lines(
            "probe", 4, assembly=a
        ),
        ("row_pointer", "column_index", "value_stream"),
    ),
    Scatter(
        "BSR",
        BSRAssemblyPlan,
        lambda a: energy_codegen._sfem_soa_hessian_scatter_bsr_lines(
            "probe", 3, 4, assembly=a
        ),
        ("row_pointer", "column_index", "value_stream"),
    ),
    Scatter(
        "block-diagonal-symmetric",
        BlockDiagSymAssemblyPlan,
        lambda a: energy_codegen._sfem_soa_hessian_scatter_block_diag_sym_lines(
            "probe", 3, 4, assembly=a
        ),
        ("value_stream",),
    ),
)


class AssemblyPlansNameTheirKernelsTest(unittest.TestCase):
    maxDiff = None

    def test_default_names_appear_in_the_emitted_scatter(self):
        """The names the plan holds are the names the kernel is written with."""
        for scatter in SCATTERS:
            with self.subTest(format=scatter.label):
                plan = scatter.plan_type()
                text = scatter.text(plan)
                for field in scatter.fields:
                    name = getattr(plan, field)
                    self.assertIn(
                        name,
                        text,
                        "%s plan's %s is '%s', which appears nowhere in the "
                        "emitted scatter -- the plan is describing a kernel "
                        "that is not the one being generated"
                        % (scatter.label, field, name),
                    )

    def test_a_renamed_stream_reaches_the_kernel(self):
        """Proof the name is read, not coincidentally equal to a literal."""
        for scatter in SCATTERS:
            for field in scatter.fields:
                with self.subTest(format=scatter.label, field=field):
                    renamed = scatter.plan_type(**{field: "probe_stream_name"})
                    self.assertIn(
                        "probe_stream_name",
                        scatter.text(renamed),
                        "%s scatter ignores the plan's %s: renaming it left "
                        "the emitted kernel unchanged" % (scatter.label, field),
                    )

    def test_an_unspellable_reduction_stops_generation(self):
        """A policy the emitter cannot spell must fail, not silently degrade."""
        for scatter in SCATTERS:
            if not scatter.has_reduction_policy:
                continue
            with self.subTest(format=scatter.label):
                plan = scatter.plan_type(reduction_policy="serial_add")
                with self.assertRaises(ValueError):
                    scatter.text(plan)


if __name__ == "__main__":
    unittest.main()
