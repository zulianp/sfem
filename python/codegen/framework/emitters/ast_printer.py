from dataclasses import dataclass

from codegen.framework.ir.passes import VectorizationContractPass
from codegen.framework.targets import current_target

from codegen.framework.ir.kernel_ast import (
    AssignmentNode,
    BufferDeclNode,
    BufferAccess,
    BlockNode,
    CallNode,
    ExpressionRef,
    FunctionDefNode,
    IfNode,
    GatherNode,
    Literal,
    LoopIncrementKind,
    KernelAST,
    KernelASTPassPipeline,
    LoopHeaderNode,
    LoopNode,
    RawLinesNode,
    ReturnNode,
    ScatterNode,
    SymbolRef,
    LoopKind,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
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
    indent_unit: str = "  "
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
                lines.append("%s    %s%s" % (indent, param, tail))
            if not close_on_last:
                lines.append("%s) {" % indent)
            for body_node in node.body:
                lines.extend(self.print_node(body_node, indent + self.indent_unit))
            lines.append("%s}" % indent)
            return tuple(lines)
        if isinstance(node, IfNode):
            condition = self.render_entity(node.condition)
            if node.inline_body and len(node.body) == 1 and not node.orelse:
                inner = self.print_node(node.body[0], "")
                if len(inner) == 1:
                    return ("%sif (%s) %s" % (indent, condition, inner[0]),)
            lines = ["%sif (%s) {" % (indent, condition)]
            for child in node.body:
                lines.extend(self.print_node(child, indent + self.indent_unit))
            if node.orelse:
                lines.append("%s} else {" % indent)
                for child in node.orelse:
                    lines.extend(self.print_node(child, indent + self.indent_unit))
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


#: Analyses every kernel passes through on its way to text.  Kept here, at the
#: one place every AST is rendered, so no emitter can route around it.
DEFAULT_PASSES = KernelASTPassPipeline((VectorizationContractPass(),))


def render_kernel_ast_lines(name, nodes, printer=None, passes=DEFAULT_PASSES):
    """Render a kernel, after the pass pipeline has had a look at it.

    The pipeline runs before printing rather than after, so a kernel that
    breaks a contract never becomes a file.  ``passes=None`` skips it, which
    exists for tests that construct deliberately invalid trees.
    """
    printer = CLikeKernelASTPrinter() if printer is None else printer
    ast = KernelAST(name=name, nodes=tuple(nodes))
    if passes is not None:
        ast, _results = passes.apply(ast)
    return printer.print_ast(ast)


def work_item_scope_header_lines(indent="", serial=False):
    """The bound target's work-item scope, opened -- the caller closes it.

    The one text spelling of the scope, rendered from the node
    `TargetPlatform.work_item_scope_node` decides.  `lane_loop_header_lines`
    below is the same thing with the decision hardcoded, and the callers that
    still use it are the ones that have not moved.

    `serial` asks for the scope a scatter needs: two work items of one block can
    land on the same node, so the loop must not be vectorized.  That is a
    different question from how the loop is spelled, which is why the target
    answers it and this only prints the answer.
    """
    target = current_target()
    node = (
        target.serial_work_item_scope_node(())
        if serial
        else target.work_item_scope_node(())
    )
    if isinstance(node, BlockNode):
        return ("%s{" % indent,)
    pragma = target.vectorize_pragma() if node.vectorized else None
    printer = CLikeKernelASTPrinter(vectorize_pragma=pragma or "")
    return tuple(
        "%s%s" % (indent, line)
        for line in render_kernel_ast_lines(
            "work_item_scope", (LoopHeaderNode(node),), printer=printer
        )
    )


def lane_loop_header_lines(pragma, indent=""):
    """The lane loop's pragma and `for`, rendered from the IR.

    The one spelling of `for (int lane = 0; lane < ne; ++lane) {` for the three
    emitters that build their bodies as lines.  Before this there were four:
    `energy_codegen.py` spliced `source_builder.simd_lines()`,
    `residual_codegen.py` spliced `_vectorize_pragma()`,
    `inexact_apply_codegen.py` had a private `_simd_pragma`, and all three then
    wrote the `for` out by hand -- while every one of them *also* rendered this
    same loop through `LoopHeaderNode` somewhere else in the same file.

    `pragma` is the bound target's vectorize pragma or a falsy value; the caller
    fetches it, because the three reach their target differently and that is a
    separate question from how the loop is spelled.

    The pragma is indented with its loop.  It did not used to be:
    `simd_lines()` and `_vectorize_pragma()` return the bare pragma and their
    call sites spliced it unindented, so 204 of the `#pragma omp simd` lines in
    the shipped tree sat at column zero beside an indented `for`.  That was an
    artefact of four emitters spelling one loop, and it outlived them by one
    commit because straightening it moves generated bytes and that deserved to
    be its own change rather than a side effect of giving the loop one spelling.
    """
    lane = iterator("lane", "int")
    rendered = render_kernel_ast_lines(
        "lane_loop_header",
        (
            LoopHeaderNode(
                LoopNode(
                    LoopKind.SIMD,
                    lane,
                    iteration_range(0, expr_ref("ne", "tile_extent")),
                    pre_increment(lane),
                    vectorized=bool(pragma),
                )
            ),
        ),
        printer=CLikeKernelASTPrinter(vectorize_pragma=pragma or ""),
    )
    return tuple("%s%s" % (indent, line) for line in rendered)


def mesh_loop_lines(target, indent="  "):
    """The target's pass over the mesh, spelled.

    The decision -- blocked over `VS` with a tail count, or grid-strided with
    one element per thread -- is `TargetPlatform.mesh_loop_nodes`; this is the
    only place it becomes text.  The split is the one `work_item_scope_node`
    already uses: `targets` is index 4 and `ir` is index 3, so a target may
    build a node and may not spell one.

    Each successive line is one level deeper: the loop header, then whatever it
    declares inside.
    """
    return _indented_nodes(target.mesh_loop_nodes(), "mesh_loop", indent)


def element_loop_lines(target, pragma_indent="", indent="  ", reduction=None):
    """The target's scalar pass over the mesh, spelled.

    `pragma_indent` exists only because the tracked tree spells the parallel-for
    pragma at column 0 in one caller and at column 2 in another.  It reproduces
    an inconsistency faithfully rather than deciding it; settling on one column
    is a deliberate whitespace diff of its own.
    """
    return (
        *(
            "%s%s" % (pragma_indent, pragma)
            for pragma in target.parallel_element_loop_lines("static", reduction)
        ),
        *_indented_nodes(target.element_loop_nodes(), "element_loop", indent),
    )


def _indented_nodes(nodes, reason, indent):
    rendered = render_kernel_ast_lines(reason, nodes)
    return tuple("%s%s" % (indent * (1 + i), line) for i, line in enumerate(rendered))
