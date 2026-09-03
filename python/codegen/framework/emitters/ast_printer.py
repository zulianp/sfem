from dataclasses import dataclass

from codegen.framework.ir.kernel_ast import (
    AssignmentNode,
    BufferDeclNode,
    BufferAccess,
    BlockNode,
    CallNode,
    ExpressionRef,
    FunctionDefNode,
    GatherNode,
    Literal,
    LoopIncrementKind,
    KernelAST,
    LoopHeaderNode,
    LoopNode,
    RawLinesNode,
    ReturnNode,
    ScatterNode,
    SymbolRef,
)


@dataclass(frozen=True)
class PrinterLayout:
    """Where the printer puts things, as opposed to what it prints.

    The emitters do not agree on layout.  The residual path closes a signature
    on its own line; the energy path's scatter helpers close it on the last
    parameter.  Some pragmas sit at the statement indent, others at column
    zero.  These are real differences in existing generated code, so a
    migration that must stay byte-identical has to reproduce whichever one its
    kernel uses.

    They live here rather than on the nodes because layout is the printer's
    job.  ``CallNode.wrap_arguments`` is the exception and should move here
    once it has company; it was added before this object existed.
    """

    close_signature_on_last_param: bool = False
    atomic_pragma_at_column_zero: bool = False


@dataclass(frozen=True)
class CLikeKernelASTPrinter:
    indent_unit: str = "    "
    vectorize_pragma: str = ""
    atomic_update_pragma: str = ""
    layout: PrinterLayout = PrinterLayout()

    def print_ast(self, ast):
        if not isinstance(ast, KernelAST):
            raise TypeError("print_ast expects KernelAST")
        lines = []
        for node in ast.nodes:
            lines.extend(self.print_node(node))
        return tuple(lines)

    def print_loop_header(self, node, indent=""):
        """The pragma and opening line of a loop, without its closing brace.

        Shared by ``LoopNode``, which closes what this opens, and
        ``LoopHeaderNode``, which leaves the brace to a caller still building
        its body as lines.
        """
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
        return tuple(lines)

    def print_node(self, node, indent=""):
        if isinstance(node, RawLinesNode):
            # Verbatim, and the indent is ignored on purpose -- see the node.
            return tuple(node.lines)
        if isinstance(node, FunctionDefNode):
            lines = []
            if node.template_params:
                lines.append(
                    "%stemplate <%s>" % (indent, ", ".join(node.template_params))
                )
            opener = " ".join(part for part in (node.qualifier, node.return_type) if part)
            lines.append("%s%s %s(" % (indent, opener, node.name))
            last = len(node.params) - 1
            close_on_last = self.layout.close_signature_on_last_param and node.params
            for position, param in enumerate(node.params):
                if position == last:
                    tail = ") {" if close_on_last else ""
                else:
                    tail = ","
                lines.append("%s        %s%s" % (indent, param, tail))
            if not close_on_last:
                lines.append("%s) {" % indent)
            for body_node in node.body:
                lines.extend(self.print_node(body_node, indent + self.indent_unit))
            lines.append("%s}" % indent)
            return tuple(lines)
        if isinstance(node, ReturnNode):
            if node.value is None:
                return ("%sreturn;" % indent,)
            return ("%sreturn %s;" % (indent, self.render_entity(node.value)),)
        if isinstance(node, BlockNode):
            lines = ["%s{" % indent]
            for body_node in node.body:
                lines.extend(self.print_node(body_node, indent + self.indent_unit))
            lines.append("%s}" % indent)
            return tuple(lines)
        if isinstance(node, LoopHeaderNode):
            return self.print_loop_header(node.loop, indent)
        if isinstance(node, LoopNode):
            lines = list(self.print_loop_header(node, indent))
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
            arguments = ", ".join(self.render_entity(arg) for arg in node.arguments)
            if node.wrap_arguments:
                return (
                    "%s%s%s(" % (indent, self.render_entity(node.callee), templates),
                    "%s%s);" % (indent + self.indent_unit * 2, arguments),
                )
            return (
                "%s%s%s(%s);"
                % (indent, self.render_entity(node.callee), templates, arguments),
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
                pragma_indent = (
                    "" if self.layout.atomic_pragma_at_column_zero else indent
                )
                lines.append("%s%s" % (pragma_indent, self.atomic_update_pragma))
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
