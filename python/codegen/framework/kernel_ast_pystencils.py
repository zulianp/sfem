from dataclasses import dataclass
from contextlib import contextmanager
import importlib.util
import os

try:
    from .kernel_ast import (
        AssignmentNode,
        BufferDeclNode,
        BufferAccess,
        ExpressionRef,
        GatherNode,
        KernelAST,
        Literal,
        LoopIncrementKind,
        LoopNode,
        ScatterNode,
        SymbolRef,
    )
except ImportError:
    from kernel_ast import (
        AssignmentNode,
        BufferDeclNode,
        BufferAccess,
        ExpressionRef,
        GatherNode,
        KernelAST,
        Literal,
        LoopIncrementKind,
        LoopNode,
        ScatterNode,
        SymbolRef,
    )


@dataclass(frozen=True)
class PystencilsAvailability:
    available: bool
    reason: str = ""

    def to_dict(self):
        return {"available": self.available, "reason": self.reason}


@dataclass(frozen=True)
class PystencilsLoweringResult:
    available: PystencilsAvailability
    ast_name: str
    lowered: object = None
    diagnostics: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def success(self):
        return self.available.available and self.lowered is not None

    def to_dict(self):
        return {
            "available": self.available.to_dict(),
            "ast_name": self.ast_name,
            "success": self.success,
            "diagnostics": list(self.diagnostics),
        }


class PystencilsKernelASTAdapter:
    """Optional lowering boundary from SFEM KernelAST to pystencils.

    The SFEM model remains the source of truth for FEM geometry, gathers,
    scatters, PA lifecycle, and parity constraints. This adapter is intentionally
    separate so pystencils can be used for backend AST/codegen where it preserves
    the requested loop/buffer structure.
    """

    config_home: str = "/private/tmp/sfem_pystencils"

    def __init__(self, default_float_type="float64"):
        self.default_float_type = str(default_float_type)

    def availability(self):
        spec = importlib.util.find_spec("pystencils")
        if spec is None:
            return PystencilsAvailability(False, "pystencils is not installed")
        return PystencilsAvailability(True, "")

    def lower(self, ast):
        if not isinstance(ast, KernelAST):
            raise TypeError("lower expects KernelAST")
        available = self.availability()
        if not available.available:
            return PystencilsLoweringResult(
                available,
                ast.name,
                lowered=None,
                diagnostics=("pystencils lowering skipped",),
            )

        with self._sandboxed_config_home():
            from pystencils.astnodes import Block

        lowered, unsupported = self._lower_nodes(ast.nodes)

        if unsupported:
            return PystencilsLoweringResult(
                available,
                ast.name,
                lowered=None,
                diagnostics=(
                    "pystencils adapter cannot preserve these SFEM node(s) yet: %s"
                    % ", ".join(sorted(set(unsupported))),
                ),
            )

        return PystencilsLoweringResult(
            available,
            ast.name,
            lowered=Block(list(lowered)),
            diagnostics=("lowered %d SFEM node(s) to pystencils AST" % len(lowered),),
        )

    def generate_c(self, ast):
        result = self.lower(ast)
        if not result.success:
            return result

        with self._sandboxed_config_home():
            from pystencils.backends.cbackend import generate_c

            source = generate_c(
                result.lowered,
                custom_backend=_SFEMPystencilsCBackend(
                    signature_only=False,
                    dialect=None,
                ),
            )

        return PystencilsLoweringResult(
            result.available,
            result.ast_name,
            lowered=source,
            diagnostics=result.diagnostics + ("generated C with pystencils C backend",),
        )

    def _lower_nodes(self, nodes):
        lowered = []
        unsupported = []
        for node in nodes:
            try:
                lowered_node = self._lower_node(node, unsupported)
            except Exception as exc:
                lowered_node = None
                unsupported.append("%s(%s)" % (type(node).__name__, type(exc).__name__))
            if lowered_node is not None:
                lowered.append(lowered_node)
        return tuple(lowered), tuple(unsupported)

    def _lower_node(self, node, unsupported):
        if isinstance(node, AssignmentNode):
            return self._lower_assignment(node, unsupported)
        if isinstance(node, BufferDeclNode):
            return self._lower_buffer_decl(node, unsupported)
        if isinstance(node, GatherNode):
            return self._lower_gather(node, unsupported)
        if isinstance(node, LoopNode):
            return self._lower_loop(node, unsupported)
        if isinstance(node, ScatterNode):
            return self._lower_scatter(node, unsupported)

        unsupported.append(type(node).__name__)
        return None

    def _lower_assignment(self, node, unsupported):
        if node.operator != "=":
            unsupported.append("%s(%s)" % (type(node).__name__, node.operator))
            return None

        with self._sandboxed_config_home():
            from pystencils.astnodes import SympyAssignment

        return self._make_assignment(
            node,
            self._to_sympy(node.lhs, declaration_lhs=True),
            self._to_sympy(node.rhs),
            unsupported,
            is_const=True,
        )

    def _lower_buffer_decl(self, node, unsupported):
        if node.extents:
            unsupported.append("%s(array)" % type(node).__name__)
            return None
        if node.initializer is None:
            unsupported.append("%s(uninitialized)" % type(node).__name__)
            return None

        return self._make_assignment(
            node,
            self._typed_symbol(
                node.name.name,
                node.scalar_type.name,
                preserve_c_name=True,
            ),
            self._to_sympy(node.initializer),
            unsupported,
            is_const=False,
        )

    def _lower_gather(self, node, unsupported):
        return self._make_assignment(
            node,
            self._to_sympy(node.target, declaration_lhs=True),
            self._indexed_source(node.source, node.index),
            unsupported,
            is_const=True,
        )

    def _lower_scatter(self, node, unsupported):
        if node.atomic:
            unsupported.append("%s(atomic)" % type(node).__name__)
            return None
        if node.operator != "=":
            unsupported.append("%s(%s)" % (type(node).__name__, node.operator))
            return None

        return self._make_assignment(
            node,
            self._to_sympy(node.target),
            self._to_sympy(node.value),
            unsupported,
            is_const=False,
        )

    def _make_assignment(self, node, lhs, rhs, unsupported, is_const):
        try:
            with self._sandboxed_config_home():
                from pystencils.astnodes import SympyAssignment

            return SympyAssignment(lhs, rhs, is_const=is_const)
        except Exception as exc:
            unsupported.append("%s(%s)" % (type(node).__name__, type(exc).__name__))
            return None

    def _lower_loop(self, node, unsupported):
        body, body_unsupported = self._lower_nodes(node.body)
        unsupported.extend(body_unsupported)
        if body_unsupported:
            return None

        with self._sandboxed_config_home():
            from pystencils.astnodes import Block, LoopOverCoordinate

        return LoopOverCoordinate(
            Block(list(body)),
            coordinate_to_loop_over=0,
            start=self._to_sympy(node.iteration_range.begin),
            stop=self._to_sympy(node.iteration_range.end),
            step=self._loop_step(node),
            custom_loop_ctr=self._typed_symbol(
                node.iterator.symbol.name,
                node.iterator.index_type.name,
                preserve_c_name=True,
            ),
        )

    def _loop_step(self, node):
        if node.increment.kind == LoopIncrementKind.PRE_INCREMENT:
            return self._to_sympy(Literal(1))
        if node.increment.kind == LoopIncrementKind.ADD_ASSIGN:
            return self._to_sympy(node.increment.amount)
        raise ValueError("unsupported loop increment kind: %s" % node.increment.kind)

    @contextmanager
    def _sandboxed_config_home(self):
        old_home = os.environ.get("HOME")
        old_xdg = os.environ.get("XDG_CONFIG_HOME")
        old_cache = os.environ.get("XDG_CACHE_HOME")
        os.makedirs(self.config_home, exist_ok=True)
        os.environ["HOME"] = self.config_home
        os.environ["XDG_CONFIG_HOME"] = self.config_home
        os.environ["XDG_CACHE_HOME"] = self.config_home
        try:
            yield
        finally:
            _restore_env("HOME", old_home)
            _restore_env("XDG_CONFIG_HOME", old_xdg)
            _restore_env("XDG_CACHE_HOME", old_cache)

    def _to_sympy(self, entity, declaration_lhs=False):
        with self._sandboxed_config_home():
            import sympy as sp

        if isinstance(entity, SymbolRef):
            if declaration_lhs:
                return self._typed_symbol(entity.name, self.default_float_type)
            return sp.Symbol(entity.name)
        if isinstance(entity, Literal):
            return sp.sympify(entity.value)
        if isinstance(entity, ExpressionRef):
            return sp.sympify(entity.expression)
        if isinstance(entity, BufferAccess):
            base = self._indexed_base(entity.base.name, len(entity.indices))
            return base[tuple(self._to_sympy(index) for index in entity.indices)]
        return sp.sympify(str(entity))

    def _indexed_source(self, source, index):
        return self._indexed_base(self._source_name(source), 1)[self._to_sympy(index)]

    def _indexed_base(self, name, rank):
        with self._sandboxed_config_home():
            import sympy as sp
            from pystencils.typing import PointerType
            from pystencils.typing import create_type

        shape = tuple(
            sp.Symbol("_sfem_extent_%s_%d" % (name, dim))
            for dim in range(max(1, rank))
        )
        symbol = self._typed_symbol(
            name,
            PointerType(create_type(self.default_float_type)),
        )
        return sp.IndexedBase(symbol, shape=shape)

    def _source_name(self, source):
        if isinstance(source, SymbolRef):
            return source.name
        if isinstance(source, ExpressionRef) and source.expression.isidentifier():
            return source.expression
        raise ValueError("unsupported indexed source %s" % source)

    def _typed_symbol(self, name, c_type, preserve_c_name=False):
        with self._sandboxed_config_home():
            from pystencils.typing import TypedSymbol

        if not preserve_c_name:
            return TypedSymbol(name, c_type)
        return TypedSymbol(name, _PystencilsCType(c_type))


class _PystencilsCType:
    def __init__(self, c_name):
        self.c_name = str(c_name)

    def __hash__(self):
        return hash(self.c_name)

    def __eq__(self, other):
        return getattr(other, "c_name", None) == self.c_name

    def __str__(self):
        return self.c_name


class _SFEMPystencilsCBackend:
    def __new__(cls, *args, **kwargs):
        from pystencils.backends.cbackend import CBackend

        class Backend(CBackend):
            def _print__PystencilsCType(self, node):
                return node.c_name

        return Backend(*args, **kwargs)


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
