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
    it down.  ``residual_codegen`` had 41 and now has 17 -- the twenty-four
    with an exact accessor were converted.  ``energy_codegen``'s 25 are a
    different shape: that module already receives target syntax through a
    ``source_builder`` parameter, which is the pattern the residual path
    lacks, so its literals should move to that object rather than to a
    module-level accessor.
    """

    #: file -> literal pragmas remaining.  Lower these; never raise them.
    BUDGET = {
        "residual_codegen.py": 17,
        "energy_codegen.py": 25,
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


if __name__ == "__main__":
    unittest.main()
