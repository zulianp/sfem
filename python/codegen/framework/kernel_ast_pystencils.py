from dataclasses import dataclass
from contextlib import contextmanager
import importlib.util
import os

try:
    from .kernel_ast import (
        AssignmentNode,
        BufferAccess,
        ExpressionRef,
        KernelAST,
        Literal,
        SymbolRef,
    )
except ImportError:
    from kernel_ast import (
        AssignmentNode,
        BufferAccess,
        ExpressionRef,
        KernelAST,
        Literal,
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
            import pystencils as ps

        assignments = []
        unsupported = []
        for node in ast.nodes:
            if isinstance(node, AssignmentNode):
                assignments.append(ps.Assignment(self._to_sympy(node.lhs), self._to_sympy(node.rhs)))
            else:
                unsupported.append(type(node).__name__)

        if unsupported:
            return PystencilsLoweringResult(
                available,
                ast.name,
                lowered=tuple(assignments) if assignments else None,
                diagnostics=(
                    "pystencils adapter currently lowers assignment nodes only; unsupported nodes: %s"
                    % ", ".join(sorted(set(unsupported))),
                ),
            )

        return PystencilsLoweringResult(
            available,
            ast.name,
            lowered=tuple(assignments),
            diagnostics=("lowered %d assignment node(s) to pystencils assignments" % len(assignments),),
        )

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

    def _to_sympy(self, entity):
        with self._sandboxed_config_home():
            import sympy as sp

        if isinstance(entity, SymbolRef):
            return sp.Symbol(entity.name)
        if isinstance(entity, Literal):
            return sp.sympify(entity.value)
        if isinstance(entity, ExpressionRef):
            return sp.sympify(entity.expression)
        if isinstance(entity, BufferAccess):
            base = sp.IndexedBase(entity.base.name)
            return base[tuple(self._to_sympy(index) for index in entity.indices)]
        return sp.sympify(str(entity))


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
