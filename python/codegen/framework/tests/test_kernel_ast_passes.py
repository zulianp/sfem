"""The IR is inspected, not only printed.

Before this, ``KernelASTPass.apply`` returned its input unchanged and had no
production consumer.  A tree was built and handed straight to the printer, so
the layer meant to hold vectorisation, unrolling and blocking held nothing --
the third of the three gaps in the architecture assessment.

``VectorizationContractPass`` closes it with an analysis rather than a
transform.  The contract it checks is not new: ``test_gen_api`` already
enforces it by reading generated text, finding a lane-loop line, walking
backwards for a pragma and counting braces to find the body.  Every one of
those steps recovers structure the tree already has, so this is the same
contract asserted where it is true -- and asserted during emission, so a
kernel that breaks it never becomes a file.
"""

import unittest

from codegen.framework.emitters.ast_printer import (
    DEFAULT_PASSES,
    render_kernel_ast_lines,
)
from codegen.framework.ir.kernel_ast import (
    BufferDeclNode,
    IfNode,
    LoopKind,
    LoopNode,
    ScatterNode,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)
from codegen.framework.ir.passes import (
    VectorizationContractPass,
    VectorizationContractViolation,
)


def _lane_loop(body, vectorized=True):
    lane = iterator("lane", "int")
    return LoopNode(
        LoopKind.SIMD,
        lane,
        iteration_range(0, expr_ref("nelems")),
        pre_increment(lane),
        body=tuple(body),
        vectorized=vectorized,
    )


class VectorizationContractTest(unittest.TestCase):
    def test_a_clean_lane_loop_passes(self):
        loop = _lane_loop([BufferDeclNode("const scalar_t", "x", (), expr_ref("1"))])
        self.assertEqual(len(render_kernel_ast_lines("k", (loop,))), 3)

    def test_a_nested_loop_is_rejected(self):
        inner = iterator("j", "int")
        loop = _lane_loop(
            [
                LoopNode(
                    LoopKind.SHAPE,
                    inner,
                    iteration_range(0, expr_ref("N")),
                    pre_increment(inner),
                    body=(BufferDeclNode("int", "y", (), expr_ref("0")),),
                )
            ]
        )
        with self.assertRaises(VectorizationContractViolation) as caught:
            render_kernel_ast_lines("kernel_with_nest", (loop,))
        self.assertIn("kernel_with_nest", str(caught.exception))
        self.assertIn("lane", str(caught.exception))

    def test_an_atomic_is_rejected(self):
        loop = _lane_loop(
            [ScatterNode(expr_ref("o[0]"), expr_ref("v"), "+=", atomic=True)]
        )
        with self.assertRaises(VectorizationContractViolation):
            render_kernel_ast_lines("k", (loop,))

    def test_it_looks_through_conditionals(self):
        """A violation hidden in a branch is still a violation."""
        loop = _lane_loop(
            [
                IfNode(
                    expr_ref("ok"),
                    body=(
                        ScatterNode(
                            expr_ref("o[0]"), expr_ref("v"), "+=", atomic=True
                        ),
                    ),
                )
            ]
        )
        with self.assertRaises(VectorizationContractViolation):
            render_kernel_ast_lines("k", (loop,))

    def test_an_unvectorised_loop_may_nest_and_scatter_atomically(self):
        """The constraint is about vectorised lanes, not loops in general.

        The scatter helpers nest four deep and accumulate atomically; none of
        those loops is a vector lane, and forbidding them would be wrong.
        """
        inner = iterator("j", "int")
        loop = _lane_loop(
            [
                LoopNode(
                    LoopKind.SHAPE,
                    inner,
                    iteration_range(0, expr_ref("N")),
                    pre_increment(inner),
                    body=(
                        ScatterNode(
                            expr_ref("o[0]"), expr_ref("v"), "+=", atomic=True
                        ),
                    ),
                )
            ],
            vectorized=False,
        )
        self.assertTrue(render_kernel_ast_lines("k", (loop,)))


class ThePipelineIsWiredInTest(unittest.TestCase):
    def test_the_default_pipeline_carries_the_contract_pass(self):
        names = [p.name for p in DEFAULT_PASSES.passes]
        self.assertIn("vectorization-contract", names)

    def test_a_pass_declares_whether_it_changes_answers(self):
        """An analysis must not claim to be performance-changing."""
        contract = VectorizationContractPass()
        self.assertTrue(contract.parity_preserving)
        self.assertFalse(contract.performance_changing)

    def test_passes_can_be_skipped_for_deliberately_invalid_trees(self):
        loop = _lane_loop(
            [ScatterNode(expr_ref("o[0]"), expr_ref("v"), "+=", atomic=True)]
        )
        self.assertTrue(render_kernel_ast_lines("k", (loop,), passes=None))

    def test_every_generated_kernel_passes_the_contract(self):
        """The real corpus, not a constructed tree."""
        from sfem import gen

        from codegen.framework.materials.laplace import material

        user_input = gen.UserInputStage.create(material, ("HEX8",), 8, None)
        form_evaluation = gen._evaluate_forms(user_input)
        plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
        files = gen.CodeGenerationStage(user_input, plan).run()
        self.assertTrue(files, "generation produced nothing to check")


if __name__ == "__main__":
    unittest.main()
