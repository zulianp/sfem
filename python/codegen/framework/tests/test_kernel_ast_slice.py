"""Two kernels' structure lives in the IR, and must stay there.

``ir/kernel_ast.py`` has been a complete, well-formed kernel IR with no
production consumer: the residual path built zero nodes, and the energy path
used it only as a line-fragment generator -- six call sites, each emitting a
loop header or a single statement, with the surrounding braces hand-written.

``_simplex_gradient_metric_body`` is the first kernel whose whole body is a
tree the printer walks.  It is the constant-P1 simplex gradient-metric kernel:
a quadrature loop over a vector-lane loop over a statement sequence, which is
the smallest complete kernel the framework emits and the shape
``EvaluationStatement.hoist_scope`` already describes.

``_constant_p1_gradient_expanded_body`` is the second, and the one that shows
the first was not a special case.  It is the same nest fed by five statement
sources rather than two, so the statement helpers themselves --
``_physical_gradient_nodes`` and ``_coefficient_evaluation_nodes`` -- now
produce nodes, and their ``_lines`` counterparts are those nodes printed.  The
tests below pin that equivalence, because a helper that drifted from its
printed view would reintroduce string building underneath an IR-shaped API.

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


#: The formulation that reaches ``_constant_p1_gradient_expanded_body``.
#: The gradient-metric branch is tried first and wins wherever it applies, so
#: this kernel is only reached by a form that is gradient-only on a linear
#: simplex yet has no gradient-metric transformation -- among the shipped
#: materials, only this one.  A sweep of all fifteen found it on TET4 and TRI3
#: and nowhere else, which is why the test below names it explicitly rather
#: than searching.
EXPANDED_MATERIAL = "mooney_rivlin_kelvin_voigt_newmark"
EXPANDED_ELEMENT = "TET4"


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


class StatementHelpersProduceNodesTest(unittest.TestCase):
    """The statement sources hand back IR, and their printed views agree.

    Each helper exists in two forms: a ``_nodes`` builder and a ``_lines``
    printer.  If the printed view were ever reimplemented as string formatting
    the outputs could stay identical while the IR quietly lost a consumer, so
    the equivalence is asserted directly rather than inferred from the
    generated code.
    """

    def test_physical_gradient_nodes_are_declarations(self):
        nodes = residual_codegen._physical_gradient_nodes("u", 3)
        self.assertEqual(len(nodes), 3, "one declaration per physical direction")
        for node in nodes:
            self.assertIsInstance(node, BufferDeclNode)
            self.assertEqual(node.extents, ())

    def test_physical_gradient_lines_are_those_nodes_printed(self):
        nodes = residual_codegen._physical_gradient_nodes("u", 3)
        self.assertEqual(
            residual_codegen._physical_gradient_lines("u", 3, "        "),
            residual_codegen._print_statement_nodes(nodes, "        "),
        )

    def test_the_printed_view_still_matches_the_original_spelling(self):
        """Byte-identity of the whole tree depends on this exact text."""
        self.assertEqual(
            residual_codegen._physical_gradient_lines("u", 2, "    "),
            [
                "    const scalar_t u_grad_0 = (u_grad_0_ref * adj0 + u_grad_1_ref * adj2) / det;",
                "    const scalar_t u_grad_1 = (u_grad_0_ref * adj1 + u_grad_1_ref * adj3) / det;",
            ],
        )

    def test_print_statement_nodes_honours_indent(self):
        node = BufferDeclNode("const scalar_t", "x", (), expr_ref("y"))
        self.assertEqual(
            residual_codegen._print_statement_nodes([node], "      "),
            ["      const scalar_t x = y;"],
        )


class BothKernelBodiesReachThePrinterAsNodesTest(unittest.TestCase):
    """Two kernels now, distinguished by the AST name each one registers."""

    def _capture(self, material_name, element):
        captured = {}
        original = residual_codegen._quadrature_lane_kernel_lines

        def capture(lane_body, indent="    ", name="quadrature_lane_body"):
            captured.setdefault(name, []).extend(lane_body)
            return original(lane_body, indent=indent, name=name)

        residual_codegen._quadrature_lane_kernel_lines = capture
        try:
            import importlib

            from sfem import gen

            material = importlib.import_module(
                "codegen.framework.materials.%s" % material_name
            ).material
            user_input = gen.UserInputStage.create(material, (element,), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
            gen.CodeGenerationStage(user_input, plan).run()
        finally:
            residual_codegen._quadrature_lane_kernel_lines = original
        return captured

    def _assert_all_nodes(self, statements):
        self.assertTrue(statements)
        for node in statements:
            self.assertIsInstance(
                node,
                (BufferDeclNode, ScatterNode),
                "the kernel body must reach the printer as IR nodes",
            )
        self.assertTrue(any(isinstance(n, ScatterNode) for n in statements))
        self.assertTrue(any(isinstance(n, BufferDeclNode) for n in statements))

    def test_gradient_metric_body_is_nodes(self):
        captured = self._capture("laplace", "TET4")
        self.assertIn("simplex_gradient_metric_body", captured)
        self._assert_all_nodes(captured["simplex_gradient_metric_body"])

    def test_constant_p1_expanded_body_is_nodes(self):
        captured = self._capture(EXPANDED_MATERIAL, EXPANDED_ELEMENT)
        self.assertIn(
            "constant_p1_gradient_expanded_body",
            captured,
            "%s/%s no longer exercises the expanded constant-P1 kernel; pick a "
            "material that does, or this test silently stops proving anything"
            % (EXPANDED_MATERIAL, EXPANDED_ELEMENT),
        )
        self._assert_all_nodes(captured["constant_p1_gradient_expanded_body"])
