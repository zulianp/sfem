"""The target is chosen above emission, not inside it.

``targets/`` (L5) defines OpenMP, AVX512, ARM SVE/SME, CUDA and HIP targets,
and ``pipeline/driver.py`` maps a ``KernelTarget`` onto a backend built with
one of them.  That decision was then discarded: three emitter modules each
defined a private ``_target()`` returning ``OpenMPTarget()``, so the residual
path printed OpenMP no matter which backend was driving it.  The backend's
target reached a language check and the energy emitter, and nothing else.

``targets/context.py`` closes that: a backend binds its target for the
duration of ``emit``, and the emitters read it.  These tests pin both halves
-- that the binding works and is properly scoped, and that it actually
reaches the generated text through the real backend.

The last test is a ratchet on what is *not* fixed.  Sixty-six OpenMP pragmas
were written as string literals in the emitters, bypassing the target
entirely; twenty-four of them, the ones with an exact accessor, are now
converted.  The rest are counted here so the number can only fall.
"""

import ast
import os
import unittest

from codegen.framework.targets import (
    AVX512Target,
    CUDATarget,
    OpenMPTarget,
    current_target,
    use_target,
)
from codegen.framework.targets.targets import ARMSVETarget, HIPTarget


EMITTERS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emitters"
)


def _target_constructions(tree):
    """Yield ``(call, allowed)`` for every concrete target constructed in ``tree``.

    A construction is allowed when the caller can still override it: as a
    default in a function signature, as a dataclass field default in a class
    body, or as the fallback half of ``X() if arg is None else arg``.
    """
    injectable = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for default in list(args.defaults) + [d for d in args.kw_defaults if d]:
                injectable.update(id(c) for c in ast.walk(default))
        elif isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value:
                    injectable.update(id(c) for c in ast.walk(statement.value))
        elif isinstance(node, ast.IfExp):
            injectable.update(id(c) for c in ast.walk(node))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.endswith("Target")
        ):
            yield node, id(node) in injectable


class TargetBindingTest(unittest.TestCase):
    def test_defaults_to_openmp(self):
        """What every caller got when the emitters hardcoded it."""
        self.assertIsInstance(current_target(), OpenMPTarget)

    def test_binds_and_restores(self):
        with use_target(CUDATarget()):
            self.assertIsInstance(current_target(), CUDATarget)
        self.assertIsInstance(current_target(), OpenMPTarget)

    def test_nests(self):
        with use_target(CUDATarget()):
            with use_target(AVX512Target()):
                self.assertIsInstance(current_target(), AVX512Target)
            self.assertIsInstance(current_target(), CUDATarget)

    def test_none_leaves_the_outer_binding_alone(self):
        """A backend built without an explicit target must not clobber one."""
        with use_target(CUDATarget()):
            with use_target(None):
                self.assertIsInstance(current_target(), CUDATarget)

    def test_restores_even_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with use_target(CUDATarget()):
                raise RuntimeError("boom")
        self.assertIsInstance(current_target(), OpenMPTarget)


class EmittersFollowTheBindingTest(unittest.TestCase):
    """All three emitter modules read the binding rather than choosing."""

    def test_residual_emitter_follows(self):
        from codegen.framework.emitters import residual_codegen

        self.assertEqual(residual_codegen._function_qualifier(), "static SFEM_INLINE")
        self.assertEqual(residual_codegen._vectorize_pragma(), "#pragma omp simd")
        with use_target(CUDATarget()):
            self.assertEqual(
                residual_codegen._function_qualifier(),
                "__host__ __device__ __forceinline__",
            )
            self.assertIsNone(residual_codegen._vectorize_pragma())

    def test_boundary_emitter_follows(self):
        from codegen.framework.emitters import boundary_codegen

        with use_target(CUDATarget()):
            self.assertEqual(
                boundary_codegen._function_qualifier(),
                "__host__ __device__ __forceinline__",
            )

    def test_tensor_product_geometry_follows(self):
        from codegen.framework.emitters import tensor_product_geometry

        self.assertEqual(tensor_product_geometry._target_work_item_index(), "lane")
        with use_target(CUDATarget()):
            self.assertEqual(tensor_product_geometry._target_work_item_index(), "0")

    def test_no_emitter_chooses_a_target(self):
        """The defect itself, stated precisely.

        Naming a concrete target is not automatically wrong: a dataclass field
        default and a ``target=None`` parameter fallback both leave the choice
        with the caller, and that is the pattern the residual path is moving
        toward -- ``emitters/energy.py`` and ``emitters/kernel_codegen.py``
        already use it.  What is wrong is constructing one where no caller can
        override it, which is a target decision made at L6.  This test allows
        the first and rejects the second.
        """
        offenders = []
        for name in sorted(os.listdir(EMITTERS)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(EMITTERS, name)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node, allowed in _target_constructions(tree):
                if not allowed:
                    offenders.append(
                        "%s:%d %s()" % (name, node.lineno, node.func.id)
                    )
        self.assertEqual(
            offenders,
            [],
            "an emitter constructs a target no caller can override; read the "
            "binding with current_target() instead",
        )


class BackendBindsItsTargetTest(unittest.TestCase):
    """The half that matters: does the binding reach the generated text?"""

    def test_backend_target_reaches_the_residual_emitters(self):
        from sfem import gen
        from codegen.framework.backends.openmp import OpenMPSoABackend
        from codegen.framework.materials.laplace import material

        class ProbeTarget(OpenMPTarget):
            """Distinguishable only by its vectorize pragma."""

            def vectorize_pragma(self):
                return "#pragma probe simd"

        def build():
            user_input = gen.UserInputStage.create(material, ("TET4",), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(
                user_input, form_evaluation
            ).run()
            return gen.CodeGenerationStage(user_input, plan).run()

        baseline = build()
        saved = gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP]
        gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP] = OpenMPSoABackend(
            target=ProbeTarget()
        )
        try:
            probed = build()
        finally:
            gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP] = saved

        baseline_text = "\n".join(baseline.values())
        probed_text = "\n".join(probed.values())
        self.assertEqual(baseline_text.count("#pragma probe simd"), 0)
        self.assertGreater(
            probed_text.count("#pragma probe simd"),
            0,
            "the backend's target never reached the emitters",
        )
        self.assertLess(
            probed_text.count("#pragma omp simd"),
            baseline_text.count("#pragma omp simd"),
            "the probe target replaced no OpenMP pragma",
        )


class HardcodedPragmaRatchetTest(unittest.TestCase):
    """What still bypasses the target.  Shrink-only.

    A literal pragma in an emitter is a target decision made at L6 by writing
    it down.  ``residual_codegen`` had 41 and now has 17: the twenty-four with
    an exact accessor were converted.  ``energy_codegen`` had 25 and now has
    14, its eleven leaf pragmas moved onto the ``source_builder`` it already
    receives.

    The fourteen that remain are not oversights, and three separate reasons
    keep them:

    ``#pragma omp parallel`` (4) and ``#pragma omp for schedule(static)`` (3)
    come in pairs that open a parallel region with its own brace and put a
    work-sharing construct inside it.  They are *structure*, not spelling.
    An accessor returning nothing for CUDA would leave a bare block around a
    serial loop -- code that compiles and is quietly wrong -- which hides the
    portability gap instead of closing it.  Restructuring is the real fix.

    Four ``#pragma omp atomic update`` sit in the ``_sfem_soa_hessian_scatter_*``
    helpers for CRS, DIA and COO.  These were recorded as blocked on threading
    a ``source_builder`` through seven functions; the target binding removed
    that blocker, since a helper reads ``current_target()`` directly.  Three
    are already gone -- ``block_diag_sym``, ``patch`` and ``bsr`` are built
    from IR nodes and take their atomic from the target through
    ``ScatterNode(atomic=True)``.

    The three that remain are the non-BSR formats, which are not the target
    for vector problems, so they are left as strings rather than migrated for
    parity's sake.

    One ``#pragma omp parallel`` is in ``_sfem_packed_thread_scratch_header_source``,
    which preallocates thread-local scratch.  That is OpenMP-specific
    infrastructure rather than a pragma spelling, and porting it is a design
    question about what per-thread scratch means on a GPU.
    """

    #: file -> literal pragmas remaining.  Lower these; never raise them.
    #: 15 -> 9 in the residual emitter when the three Laplacian special cases
    #: went: each carried its own `#pragma omp parallel` and `#pragma omp for`,
    #: written out rather than asked of the target, because each was a
    #: hand-shaped kernel rather than something the emitters build.  Removing a
    #: path that bypassed the target binding is the cheapest way to improve this
    #: budget, and the least informative -- the six that left were never going
    #: to be ported, they were going to be deleted.
    BUDGET = {
        "residual_codegen.py": 9,
        "energy_codegen.py": 9,
    }

    def _literal_pragmas(self, name):
        path = os.path.join(EMITTERS, name)
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("#pragma omp")
        )

    def test_literal_pragma_budget_only_shrinks(self):
        for name, budget in sorted(self.BUDGET.items()):
            with self.subTest(module=name):
                found = self._literal_pragmas(name)
                self.assertLessEqual(
                    found,
                    budget,
                    "%s gained a hardcoded pragma; route it through the target"
                    % name,
                )
                self.assertEqual(
                    found,
                    budget,
                    "%s now has %d literal pragmas, budget says %d -- lower the "
                    "budget to lock the improvement in" % (name, found, budget),
                )

    def test_no_other_emitter_hardcodes_pragmas(self):
        for name in sorted(os.listdir(EMITTERS)):
            if not name.endswith(".py") or name in self.BUDGET:
                continue
            with self.subTest(module=name):
                self.assertEqual(
                    self._literal_pragmas(name),
                    0,
                    "%s hardcodes a pragma; route it through the target" % name,
                )


class WorkItemAccessorTest(unittest.TestCase):
    """What each target answers for the work item, spelled out.

    Worth pinning explicitly rather than inferring from generated output,
    because the generated tree is OpenMP only: every SIMT answer below is
    otherwise never evaluated by anything, and would first be exercised by
    whoever eventually runs a CUDA build of the residual or inexact families.
    """

    def test_cpu_answers_are_the_literals_the_emitters_write_today(self):
        for target in (OpenMPTarget(), AVX512Target()):
            with self.subTest(target=target.name):
                self.assertEqual(target.work_item_index(), "lane")
                self.assertEqual(target.work_item_subscript(), "[lane]")
                self.assertEqual(target.work_item_offset("q", "VS"), "q * VS + lane")
                self.assertEqual(
                    target.work_item_offset("q", "geometry_stride"),
                    "q * geometry_stride + lane",
                )
                self.assertEqual(target.element_index(), "evb + lane")
                self.assertEqual(target.work_item_name("adj", 0), "adj_lane0")
                self.assertEqual(
                    target.work_item_prologue_lines("  "),
                    ("  const int lane = 0;",),
                )

    def test_sve_carries_its_index_type_into_the_prologue(self):
        """The one policy field that differs on a shipping CPU target."""
        self.assertEqual(
            ARMSVETarget().work_item_prologue_lines(),
            ("const ptrdiff_t lane = 0;",),
        )

    def test_simt_answers_drop_the_work_item_rather_than_name_it(self):
        for target in (CUDATarget(), HIPTarget()):
            with self.subTest(target=target.name):
                self.assertEqual(target.work_item_index(), "0")
                self.assertEqual(target.work_item_subscript(), "[0]")
                # The stride survives; only the `+ lane` term goes.  Four of the
                # seven offset sites stride by `geometry_stride`, a runtime mesh
                # parameter, so dropping the stride would be wrong arithmetic.
                self.assertEqual(target.work_item_offset("q", "VS"), "q * VS")
                self.assertEqual(
                    target.work_item_offset("q", "geometry_stride"),
                    "q * geometry_stride",
                )
                self.assertEqual(target.element_index(), "evb")
                self.assertEqual(target.work_item_prologue_lines("  "), ())

    def test_the_scope_is_a_loop_on_a_lane_target_and_a_block_on_a_thread_one(self):
        from codegen.framework.ir.kernel_ast import BlockNode, LoopKind, LoopNode

        cpu = OpenMPTarget().work_item_scope_node(())
        self.assertIsInstance(cpu, LoopNode)
        self.assertIs(cpu.loop_kind, LoopKind.SIMD)
        self.assertEqual(cpu.iterator.symbol.name, "lane")
        self.assertEqual(cpu.iterator.index_type.name, "int")
        self.assertTrue(cpu.vectorized)

        self.assertIsInstance(CUDATarget().work_item_scope_node(()), BlockNode)

    def test_the_serial_scope_is_never_vectorized(self):
        """A scatter two work items can collide in must not get a simd pragma.

        The packed scatters are serial on purpose -- two lanes of one block can
        land on the same node -- so this is the one accessor where getting it
        wrong produces a data race rather than a diff, and `check-tree` would
        pass while the answer became nondeterministic.
        """
        from codegen.framework.ir.kernel_ast import BlockNode, LoopNode

        for target in (OpenMPTarget(), AVX512Target(), ARMSVETarget()):
            with self.subTest(target=target.name):
                node = target.serial_work_item_scope_node(())
                self.assertIsInstance(node, LoopNode)
                self.assertFalse(node.vectorized)
                self.assertTrue(target.work_item_scope_node(()).vectorized)
        self.assertIsInstance(CUDATarget().serial_work_item_scope_node(()), BlockNode)


class InexactFamilyUnderACudaTargetTest(unittest.TestCase):
    """The narrow, true claim about CUDA -- and it is worth stating narrowly.

    `backends/cuda.py` refuses an emitted file that contains the substring
    `lane`, and `CUDASoABackend.emit_inexact` returned nothing because of it.
    That rule is what this conversion is about, so it is what gets asserted:
    emitted under a bound `CUDATarget`, the inexact family no longer spells a
    work item or an OpenMP pragma.

    It does *not* assert that the backend accepts the family, and it should not.
    `_validate_cuda_source_contract` also wants a file whose name ends
    `_operator.cu` containing `__global__ void`, and this emitter writes
    `%s_operator.cpp` with host `extern "C"` functions around an OpenMP mesh
    loop.  Reaching that needs the mesh-loop lowering -- `.cu` naming, a
    grid-stride kernel, a host launcher, atomic scatters -- which the energy
    family already has and this one does not.

    This is also the only thing in the repository that evaluates the SIMT arm of
    the work-item accessors through a real emitter rather than in isolation.
    """

    def test_it_spells_no_work_item_and_no_openmp_pragma(self):
        import dataclasses

        from sfem import gen
        from codegen.framework.emitters.inexact_apply_codegen import inexact_apply_files
        from codegen.framework.materials.linear_elasticity import material

        unit_material = dataclasses.replace(material, inexact_apply=True)
        user_input = gen.UserInputStage.create(unit_material, ("TET4",), 8, None)
        form_evaluation = gen._evaluate_forms(user_input)
        plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
        context = user_input.element_contexts[0]
        unit = list(plan.emission_kernels_for_context(context))[0]

        with use_target(CUDATarget()):
            emitted = dict(inexact_apply_files(unit_material, unit, context))

        self.assertTrue(emitted)
        for path, source in sorted(emitted.items()):
            with self.subTest(generated=os.path.basename(path)):
                self.assertNotIn("lane", source)
                self.assertNotIn("#pragma omp", source)

    def test_the_same_emitter_still_spells_lanes_for_a_cpu_target(self):
        """The conversion is a lowering, not a deletion."""
        import dataclasses

        from sfem import gen
        from codegen.framework.emitters.inexact_apply_codegen import inexact_apply_files
        from codegen.framework.materials.linear_elasticity import material

        unit_material = dataclasses.replace(material, inexact_apply=True)
        user_input = gen.UserInputStage.create(unit_material, ("TET4",), 8, None)
        form_evaluation = gen._evaluate_forms(user_input)
        plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
        context = user_input.element_contexts[0]
        unit = list(plan.emission_kernels_for_context(context))[0]

        with use_target(OpenMPTarget()):
            emitted = dict(inexact_apply_files(unit_material, unit, context))

        self.assertTrue(any("lane" in source for source in emitted.values()))


class WorkItemLoweringRatchetTest(unittest.TestCase):
    """How much of the emitted text still spells the work item by hand.

    The conversion this counts cannot be measured by `codegen_snapshot
    check-tree`.  The shipped tree is generated for OpenMP only, and OpenMP's
    answer for the work item *is* the literal the emitters write today, so a
    correct conversion moves no byte and byte-identity stays true by
    construction at every step.  That makes it the right regression gate -- it
    is byte-exact over 387 files and catches the dropped space in a `%s` splice
    that is the realistic failure mode -- and no kind of progress gate.

    So generate twice: once normally, once against a target identical to OpenMP
    but for the name it gives the work item.  A site that asks the target
    follows the rename; a site that writes `lane` does not.  What survives is
    exactly the text that would still say `lane` on a GPU, which is what
    `backends/cuda.py` rejects and what keeps four of five targets from
    shipping.

    Unlike a literal count over the Python sources this cannot be gamed by
    tidying, and it is also where the work really is:
    `inexact_apply_codegen.py` held 19 of the 124 literal `lane` strings in the
    emitters and produced 398 of the tokens, because its literals sit inside
    loops over nodes and components.

    **Two materials, because one was not enough.**  This started measuring
    linear elasticity alone, which reaches the energy and inexact emitters and
    never touches `residual_codegen.py` -- so it reported the defect as 470
    tokens when the residual path alone carries 1195 more.  A single-material
    counter is the same mistake as a single-size benchmark: it is not wrong, it
    is narrow, and it made the largest surface invisible.  TRI3 keeps the
    residual material cheap; TET4 is the smallest element that exercises the
    inexact, energy and operator paths at once.
    """

    #: material -> element -> generated file -> tokens that ignore the target.
    #: Only ever down.
    #:
    #: linear_elasticity 470 -> 72 when `inexact_apply_codegen.py` converted:
    #: its 398 went to zero in one pass and the file left the set entirely,
    #: which is why the set of files is asserted and not just the counts.
    #: Then 72 -> 0 when `energy_codegen.py` followed, so the energy and inexact
    #: paths are done and everything that remains is `residual_codegen.py`.
    #: An empty entry is kept rather than deleted: it is what says this material
    #: is converted, and it fails loudly if a hand-spelled work item comes back.
    BUDGET = {
        ("linear_elasticity", "TET4"): {},
        ("two_phase_flow", "TRI3"): {
            "two_phase_flow_d2_simplex_local.hpp": 400,
            "two_phase_flow_form_2_p_c_p_c_d2_simplex_local.hpp": 132,
            "two_phase_flow_form_2_p_w_p_w_d2_simplex_local.hpp": 132,
            "two_phase_flow_form_1_p_c_d2_simplex_local.hpp": 124,
            "two_phase_flow_form_1_p_w_d2_simplex_local.hpp": 124,
            "two_phase_flow_form_2_p_c_p_w_d2_simplex_local.hpp": 108,
            "two_phase_flow_form_2_p_w_p_c_d2_simplex_local.hpp": 108,
            "two_phase_flow_tri3_operator.cpp": 13,
            "two_phase_flow_form_1_p_c_tri3_operator.cpp": 9,
            "two_phase_flow_form_1_p_w_tri3_operator.cpp": 9,
            "two_phase_flow_form_2_p_c_p_c_tri3_operator.cpp": 9,
            "two_phase_flow_form_2_p_c_p_w_tri3_operator.cpp": 9,
            "two_phase_flow_form_2_p_w_p_c_tri3_operator.cpp": 9,
            "two_phase_flow_form_2_p_w_p_w_tri3_operator.cpp": 9,
        },
    }

    def _probe(self, material_name, element):
        import dataclasses
        import importlib

        from sfem import gen
        from codegen.framework.backends.openmp import OpenMPSoABackend

        class LaneProbe(OpenMPTarget):
            """OpenMP in every respect but the name it gives the work item."""

            def loop_lowering_policy(self):
                policy = super().loop_lowering_policy()
                return dataclasses.replace(policy, lane_index="wi")

        module = importlib.import_module(
            "codegen.framework.materials.%s" % material_name
        )
        material = module.material
        if material_name == "linear_elasticity":
            material = dataclasses.replace(material, inexact_apply=True)

        saved = gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP]
        gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP] = OpenMPSoABackend(
            target=LaneProbe()
        )
        try:
            user_input = gen.UserInputStage.create(material, (element,), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
            return gen.CodeGenerationStage(user_input, plan).run()
        finally:
            gen.BACKENDS_BY_TARGET[gen.KernelTarget.OPENMP] = saved

    def test_work_item_budget_only_shrinks(self):
        import re

        word = re.compile(r"\blane\b")
        for (material, element), budget in sorted(self.BUDGET.items()):
            probed = self._probe(material, element)
            found = {}
            for path, source in probed.items():
                count = len(word.findall(source))
                if count:
                    found[os.path.basename(path)] = count

            with self.subTest(material=material, element=element):
                self.assertEqual(
                    sorted(found),
                    sorted(budget),
                    "the set of files with hand-spelled work items changed for "
                    "%s; update the budget and say which file moved" % material,
                )
                for name in sorted(budget):
                    self.assertLessEqual(
                        found[name],
                        budget[name],
                        "%s gained %d work-item tokens that ignore the target; "
                        "ask the target for the index instead of writing `lane`"
                        % (name, found[name] - budget[name]),
                    )
                    self.assertEqual(
                        found[name],
                        budget[name],
                        "%s is down to %d hand-spelled work items and the budget "
                        "still says %d -- lower it to lock the improvement in"
                        % (name, found[name], budget[name]),
                    )

    def test_the_probe_would_notice_a_conversion(self):
        """The counter is only meaningful if the rename reaches anything at all.

        Most of the tree already follows the target.  Pinning that the probe has
        an effect is what stops the budget above from passing because the target
        binding broke rather than because the emitters improved.
        """
        import re

        probed = self._probe("linear_elasticity", "TET4")
        renamed = sum(len(re.findall(r"\bwi\b", source)) for source in probed.values())
        self.assertGreater(
            renamed,
            1000,
            "renaming the work item changed almost nothing -- the target "
            "binding is broken, not the emitters",
        )


if __name__ == "__main__":
    unittest.main()
