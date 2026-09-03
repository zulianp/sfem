"""A kernel is one tree: signature and body together.

Before ``FunctionDefNode`` a kernel was half a tree.  Two bodies had migrated
to the IR, but every signature -- the template line, the qualifier, the
parameter list, the closing brace -- was string concatenation in
``_local_function``.  No pass could see a kernel as a unit, and the qualifier
could not come from the target because nothing asked it.

Now ``_local_function`` returns a printed ``FunctionDefNode``.  Bodies that
have not migrated ride along inside it as ``RawLinesNode``, which is a
deliberately visible escape hatch: it makes every kernel a tree immediately
instead of only the two whose bodies are nodes, and it marks precisely what is
left.  The ratchet at the bottom counts those and only lets the number fall.
"""

import unittest

from codegen.framework.emitters import residual_codegen
from codegen.framework.emitters.ast_printer import CLikeKernelASTPrinter
from codegen.framework.ir.kernel_ast import (
    BufferDeclNode,
    FunctionDefNode,
    LoopKind,
    LoopNode,
    RawLinesNode,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)


class FunctionDefPrintingTest(unittest.TestCase):
    """The spelling byte-identity depends on."""

    def test_prints_the_signature_the_emitter_used_to_build(self):
        node = FunctionDefNode(
            "laplace_block",
            params=("const int nelems", "scalar_t output[1][VECTOR_SIZE]"),
            body=(BufferDeclNode("static constexpr int", "DIM", (), expr_ref("3")),),
            qualifier="static SFEM_INLINE",
            template_params=("typename scalar_t", "int N_QP"),
        )
        self.assertEqual(
            CLikeKernelASTPrinter().print_node(node),
            (
                "template <typename scalar_t, int N_QP>",
                "static SFEM_INLINE void laplace_block(",
                "        const int nelems,",
                "        scalar_t output[1][VECTOR_SIZE]",
                ") {",
                "    static constexpr int DIM = 3;",
                "}",
            ),
        )

    def test_a_body_node_is_indented_inside_the_function(self):
        lane = iterator("lane", "int")
        node = FunctionDefNode(
            "k",
            body=(
                LoopNode(
                    LoopKind.SIMD,
                    lane,
                    iteration_range(0, expr_ref("nelems")),
                    pre_increment(lane),
                    body=(BufferDeclNode("const scalar_t", "x", (), expr_ref("1")),),
                ),
            ),
        )
        lines = CLikeKernelASTPrinter().print_node(node)
        self.assertIn("    for (int lane = 0; lane < nelems; ++lane) {", lines)
        self.assertIn("        const scalar_t x = 1;", lines)
        self.assertEqual(
            "".join(lines).count("{"), "".join(lines).count("}"), "unbalanced braces"
        )

    def test_raw_lines_print_verbatim_and_ignore_the_indent(self):
        """The property that makes RawLinesNode an escape hatch rather than a node.

        Its text was written with absolute indentation baked in, so indenting
        it again would shift already-correct code.  This asymmetry is the
        reason the node exists and the reason it should disappear.
        """
        node = RawLinesNode(("    already indented;",), reason="not yet IR")
        self.assertEqual(
            CLikeKernelASTPrinter().print_node(node, "        "),
            ("    already indented;",),
        )

    def test_the_qualifier_comes_from_the_target(self):
        from codegen.framework.targets import CUDATarget, use_target

        def build():
            return FunctionDefNode("k", qualifier=residual_codegen._function_qualifier())

        self.assertEqual(build().qualifier, "static SFEM_INLINE")
        with use_target(CUDATarget()):
            self.assertEqual(build().qualifier, "__host__ __device__ __forceinline__")


class EveryLocalKernelIsATreeTest(unittest.TestCase):
    def test_local_function_returns_a_printed_function_node(self):
        """Captured at the node, because the printed text cannot show its origin."""
        seen = []
        original = residual_codegen._print_kernel_function

        def capture(node):
            seen.append(node)
            return original(node)

        residual_codegen._print_kernel_function = capture
        try:
            from sfem import gen

            from codegen.framework.materials.laplace import material

            user_input = gen.UserInputStage.create(material, ("TET4",), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
            gen.CodeGenerationStage(user_input, plan).run()
        finally:
            residual_codegen._print_kernel_function = original

        self.assertTrue(seen, "no local kernel was emitted through the IR")
        for node in seen:
            self.assertIsInstance(node, FunctionDefNode)
            self.assertTrue(node.name)
            self.assertTrue(node.body, "%s has an empty body" % node.name)
            self.assertIn("scalar_t", " ".join(node.template_params))

    def test_migrated_bodies_carry_no_raw_lines(self):
        """laplace/TET4 takes the gradient-metric path, which is fully IR."""
        seen = []
        original = residual_codegen._print_kernel_function

        def capture(node):
            seen.append(node)
            return original(node)

        residual_codegen._print_kernel_function = capture
        try:
            from sfem import gen

            from codegen.framework.materials.laplace import material

            user_input = gen.UserInputStage.create(material, ("TET4",), 8, None)
            form_evaluation = gen._evaluate_forms(user_input)
            plan = gen.SpecializedFormManipulationStage(user_input, form_evaluation).run()
            gen.CodeGenerationStage(user_input, plan).run()
        finally:
            residual_codegen._print_kernel_function = original

        fully_ir = [
            node
            for node in seen
            if not any(isinstance(child, RawLinesNode) for child in node.body)
        ]
        self.assertTrue(
            fully_ir,
            "no kernel reached the printer as a tree with no un-migrated body",
        )


class RawLinesRatchetTest(unittest.TestCase):
    """How much of a kernel is still text.  Shrink-only.

    Three body generators still hand back pre-rendered lines: the generic
    simplex body, the tensor-product body, and the loop nest for targets whose
    lowering policy opens a bare work-item block instead of a lane loop, which
    the IR has no node for.  Each is a place a ``KernelASTPass`` cannot reach.
    """

    #: Construction sites of RawLinesNode in the residual emitter.
    BUDGET = 3

    def _raw_lines_sites(self):
        import ast
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "emitters",
            "residual_codegen.py",
        )
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RawLinesNode"
        ]

    def test_only_shrinks(self):
        sites = self._raw_lines_sites()
        self.assertLessEqual(
            len(sites),
            self.BUDGET,
            "a new un-migrated kernel body appeared; build it from IR nodes",
        )
        self.assertEqual(
            len(sites),
            self.BUDGET,
            "RawLinesNode is down to %d sites, budget says %d -- lower the budget"
            % (len(sites), self.BUDGET),
        )

    def test_every_escape_hatch_says_what_it_is_waiting_on(self):
        """An anonymous RawLinesNode is untracked debt."""
        for site in self._raw_lines_sites():
            reasons = [kw for kw in site.keywords if kw.arg == "reason"]
            self.assertTrue(
                reasons,
                "RawLinesNode at line %d has no reason= explaining what is left"
                % site.lineno,
            )


if __name__ == "__main__":
    unittest.main()
