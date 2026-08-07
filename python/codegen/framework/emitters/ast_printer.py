from dataclasses import dataclass

from codegen.framework.ir.kernel_ast import (
    AssignmentNode,
    BufferDeclNode,
    BufferAccess,
    CallNode,
    ExpressionRef,
    GatherNode,
    Literal,
    LoopIncrementKind,
    KernelAST,
    LoopNode,
    ScatterNode,
    SymbolRef,
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
        if isinstance(node, LoopNode):
            lines = []
            if node.vectorized and self.vectorize_pragma:
                lines.append("%s%s" % (indent, self.vectorize_pragma))
            iterator_name = self.render_entity(node.iterator.symbol)
            lines.append(
                "%sfor (%s %s = %s; %s < %s; %s) {"
                % (
                    indent,
                    self.render_entity(node.iterator.index_type),
                    iterator_name,
                    self.render_entity(node.iteration_range.begin),
                    iterator_name,
                    self.render_entity(node.iteration_range.end),
                    self.render_increment(node.increment),
                )
            )
            if node.body:
                for body_node in node.body:
                    lines.extend(self.print_node(body_node, indent + self.indent_unit))
                lines.append("%s}" % indent)
            return tuple(lines)
        if isinstance(node, AssignmentNode):
            return (
                "%s%s %s %s;"
                % (
                    indent,
                    self.render_entity(node.lhs),
                    node.operator,
                    self.render_entity(node.rhs),
                ),
            )
        if isinstance(node, BufferDeclNode):
            extents = "".join("[%s]" % self.render_entity(extent) for extent in node.extents)
            initializer = (
                " = %s" % self.render_entity(node.initializer)
                if node.initializer is not None
                else ""
            )
            return (
                "%s%s %s%s%s;"
                % (
                    indent,
                    self.render_entity(node.scalar_type),
                    self.render_entity(node.name),
                    extents,
                    initializer,
                ),
            )
        if isinstance(node, CallNode):
            templates = (
                "<%s>" % ", ".join(self.render_entity(arg) for arg in node.template_arguments)
                if node.template_arguments
                else ""
            )
            return (
                "%s%s%s(%s);"
                % (
                    indent,
                    self.render_entity(node.callee),
                    templates,
                    ", ".join(self.render_entity(arg) for arg in node.arguments),
                ),
            )
        if isinstance(node, GatherNode):
            return (
                "%s%s = %s[%s];"
                % (
                    indent,
                    self.render_entity(node.target),
                    self.render_entity(node.source),
                    self.render_entity(node.index),
                ),
            )
        if isinstance(node, ScatterNode):
            lines = []
            if node.atomic and self.atomic_update_pragma:
                lines.append("%s%s" % (indent, self.atomic_update_pragma))
            lines.append(
                "%s%s %s %s;"
                % (
                    indent,
                    self.render_entity(node.target),
                    node.operator,
                    self.render_entity(node.value),
                )
            )
            return tuple(lines)
        raise TypeError("unsupported Kernel AST node %s" % type(node).__name__)

    def render_increment(self, increment):
        iterator_name = self.render_entity(increment.iterator.symbol)
        if increment.kind is LoopIncrementKind.PRE_INCREMENT:
            return "++%s" % iterator_name
        if increment.kind is LoopIncrementKind.ADD_ASSIGN:
            return "%s += %s" % (iterator_name, self.render_entity(increment.amount))
        raise ValueError("unsupported loop increment kind '%s'" % increment.kind)

    def render_entity(self, entity):
        if isinstance(entity, SymbolRef):
            return entity.name
        if isinstance(entity, ExpressionRef):
            return entity.expression
        if isinstance(entity, Literal):
            return str(entity.value)
        if isinstance(entity, BufferAccess):
            return "%s%s" % (
                self.render_entity(entity.base),
                "".join("[%s]" % self.render_entity(index) for index in entity.indices),
            )
        if hasattr(entity, "name"):
            return str(entity.name)
        return str(entity)


def render_kernel_ast_lines(name, nodes, printer=None):
    printer = CLikeKernelASTPrinter() if printer is None else printer
    return printer.print_ast(KernelAST(name=name, nodes=tuple(nodes)))
