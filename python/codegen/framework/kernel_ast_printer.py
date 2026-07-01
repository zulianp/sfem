from dataclasses import dataclass

try:
    from .kernel_ast import (
        AssignmentNode,
        BufferDeclNode,
        CallNode,
        GatherNode,
        KernelAST,
        LoopNode,
        OpaqueStatementNode,
        ScatterNode,
    )
except ImportError:
    from kernel_ast import (
        AssignmentNode,
        BufferDeclNode,
        CallNode,
        GatherNode,
        KernelAST,
        LoopNode,
        OpaqueStatementNode,
        ScatterNode,
    )


@dataclass(frozen=True)
class CLikeKernelASTPrinter:
    indent_unit: str = "    "
    vectorize_pragma: str = ""
    atomic_update_pragma: str = ""

    def print_ast(self, ast):
        if not isinstance(ast, KernelAST):
            raise TypeError("print_ast expects KernelAST")
        lines = []
        for node in ast.nodes:
            lines.extend(self.print_node(node))
        return tuple(lines)

    def print_node(self, node, indent=""):
        if isinstance(node, OpaqueStatementNode):
            return (indent + node.statement,)
        if isinstance(node, LoopNode):
            lines = []
            if node.vectorized and self.vectorize_pragma:
                lines.append("%s%s" % (indent, self.vectorize_pragma))
            lines.append(
                "%sfor (%s %s = %s; %s < %s; %s) {"
                % (
                    indent,
                    node.index_type,
                    node.index,
                    node.begin,
                    node.index,
                    node.end,
                    node.increment,
                )
            )
            if node.body:
                for body_node in node.body:
                    lines.extend(self.print_node(body_node, indent + self.indent_unit))
                lines.append("%s}" % indent)
            return tuple(lines)
        if isinstance(node, AssignmentNode):
            return ("%s%s %s %s;" % (indent, node.lhs, node.operator, node.rhs),)
        if isinstance(node, BufferDeclNode):
            extents = "".join("[%s]" % extent for extent in node.extents)
            initializer = " = %s" % node.initializer if node.initializer else ""
            return ("%s%s %s%s%s;" % (indent, node.scalar_type, node.name, extents, initializer),)
        if isinstance(node, CallNode):
            templates = (
                "<%s>" % ", ".join(node.template_arguments)
                if node.template_arguments
                else ""
            )
            return ("%s%s%s(%s);" % (indent, node.callee, templates, ", ".join(node.arguments)),)
        if isinstance(node, GatherNode):
            return ("%s%s = %s[%s];" % (indent, node.target, node.source, node.index),)
        if isinstance(node, ScatterNode):
            lines = []
            if node.atomic and self.atomic_update_pragma:
                lines.append("%s%s" % (indent, self.atomic_update_pragma))
            lines.append("%s%s %s %s;" % (indent, node.target, node.operator, node.value))
            return tuple(lines)
        raise TypeError("unsupported Kernel AST node %s" % type(node).__name__)


def render_kernel_ast_lines(name, nodes, printer=None):
    printer = CLikeKernelASTPrinter() if printer is None else printer
    return printer.print_ast(KernelAST(name=name, nodes=tuple(nodes)))
