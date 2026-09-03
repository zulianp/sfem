"""One kernel's structure lives in the IR, and must stay there.

``ir/kernel_ast.py`` has been a complete, well-formed kernel IR with no
production consumer: the residual path built zero nodes, and the energy path
used it only as a line-fragment generator -- six call sites, each emitting a
loop header or a single statement, with the surrounding braces hand-written.

``_simplex_gradient_metric_body`` is the first kernel whose whole body is a
tree the printer walks.  It is the constant-P1 simplex gradient-metric kernel:
a quadrature loop over a vector-lane loop over a statement sequence, which is
the smallest complete kernel the framework emits and the shape
``EvaluationStatement.hoist_scope`` already describes.

These tests pin two separate things.  That the loop nest really is built from
IR nodes -- not strings that happen to look the same -- and that the text it
prints still satisfies the lane-loop contract the vectorisation tests depend
on.  Byte-identity of the generated tree is checked by the snapshot gate; what
byte-identity cannot tell you is *which* code produced the bytes, which is the
whole point of this step.
"""

import unittest

from codegen.framework.emitters import residual_codegen
from codegen.framework.emitters.ast_printer import CLikeKernelASTPrinter
from codegen.framework.ir.kernel_ast import (
    BufferDeclNode,
    LoopKind,
    LoopNode,
    ScatterNode,
    expr_ref,
)


class QuadratureLaneKernelTest(unittest.TestCase):
    """The helper that turns lane-scoped statements into a printed loop nest."""

    def _render(self, body):
        return residual_codegen._quadrature_lane_kernel_lines(body, indent="    ")

    def test_emits_the_expected_loop_nest(self):
        lines = self._render(
            [BufferDeclNode("const scalar_t", "x", (), expr_ref("current[0][lane]"))]
        )
        self.assertEqual(
            lines,
            [
                "    for (int q = 0; q < N_QP; ++q) {",
                "        #pragma omp simd",
                "        for (int lane = 0; lane < nelems; ++lane) {",
                "            const scalar_t x = current[0][lane];",
                "        }",
                "    }",
            ],
        )

    def test_the_lane_loop_satisfies_the_vectorisation_contract(self):
        """`test_gen_api` requires this exact spelling, pragma and nesting."""
        lines = self._render(
            [ScatterNode(expr_ref("output[0][lane]"), expr_ref("value"), "+=")]
        )
        lane = [i for i, l in enumerate(lines) if "for (int lane" in l]
        self.assertEqual(len(lane), 1, "expected exactly one lane loop")
        index = lane[0]
        self.assertEqual(lines[index].strip(), "for (int lane = 0; lane < nelems; ++lane) {")
        self.assertEqual(lines[index - 1].strip(), "#pragma omp simd")
        body = lines[index + 1 : -2]
        self.assertFalse(
            any("for (" in line for line in body),
            "a nested loop inside the lane loop breaks vectorisation",
        )
        self.assertFalse(any("#pragma omp atomic" in line for line in body))

    def test_braces_are_closed(self):
        """A LoopNode with an empty body prints no closing brace; ours must never be empty."""
        lines = self._render([ScatterNode(expr_ref("o[0][lane]"), expr_ref("v"), "+=")])
        self.assertEqual(sum(l.count("{") for l in lines), sum(l.count("}") for l in lines))


class KernelBodyIsBuiltFromNodesTest(unittest.TestCase):
    def test_the_body_generator_constructs_ir_nodes(self):
        """The proof that this is an AST and not a string builder.

        Every statement reaching the printer must be an IR node.  If the body
        generator regressed to assembling strings, the nodes captured here would
        not appear and the emitted text could still be identical -- which is why
        this is checked at the node level rather than in the output.
        """
        captured = []
        original = residual_codegen._quadrature_lane_kernel_lines

        def capture(lane_body, indent="    "):
            captured.extend(lane_body)
            return original(lane_body, indent=indent)

        residual_codegen._quadrature_lane_kernel_lines = capture
        try:
            from sfem import gen

            from codegen.framework.materials.laplace import material

            user_input = gen.UserInputStage.create(material, ("TET4",), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
            gen.CodeGenerationStage(user_input, plan).run()
        finally:
            residual_codegen._quadrature_lane_kernel_lines = original

        self.assertTrue(captured, "the gradient-metric kernel body was never generated")
        for node in captured:
            self.assertIsInstance(
                node,
                (BufferDeclNode, ScatterNode),
                "the kernel body must reach the printer as IR nodes",
            )
        self.assertTrue(
            any(isinstance(node, ScatterNode) for node in captured),
            "the kernel must accumulate into its output through a ScatterNode",
        )
        self.assertTrue(
            any(isinstance(node, BufferDeclNode) for node in captured),
            "the kernel's temporaries must be declared through BufferDeclNode",
        )


class PrinterHandlesRealBodiesTest(unittest.TestCase):
    def test_nested_loops_with_bodies_close_their_braces(self):
        """Guards the trap: the printer omits `}` only for an empty body.

        Six existing call sites rely on that, using body-less loops as header
        generators and closing the brace themselves.  This kernel relies on the
        opposite, so both behaviours have to keep working.
        """
        printer = CLikeKernelASTPrinter(vectorize_pragma="#pragma omp simd")
        from codegen.framework.ir.kernel_ast import (
            iteration_range,
            iterator,
            pre_increment,
        )

        empty = iterator("i", "int")
        header_only = printer.print_node(
            LoopNode(LoopKind.SIMD, empty, iteration_range(0, expr_ref("n")), pre_increment(empty))
        )
        self.assertNotIn("}", "".join(header_only))

        outer = iterator("q", "int")
        with_body = printer.print_node(
            LoopNode(
                LoopKind.QUADRATURE,
                outer,
                iteration_range(0, expr_ref("N_QP")),
                pre_increment(outer),
                body=(ScatterNode(expr_ref("o[0]"), expr_ref("v"), "+="),),
            )
        )
        self.assertEqual("".join(with_body).count("{"), "".join(with_body).count("}"))


if __name__ == "__main__":
    unittest.main()
