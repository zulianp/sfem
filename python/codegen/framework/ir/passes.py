"""Passes over the kernel IR.

Until this module the IR was build-and-print: an emitter constructed a tree
and handed it straight to the printer, and nothing ever looked at one.
``KernelASTPass.apply`` returned its input unchanged and had no production
consumer, so the layer that is supposed to hold vectorisation, unrolling,
blocking and per-warp variants held nothing.

The first real pass is an analysis rather than a transform, deliberately.  The
vectorisation contract these kernels depend on is currently enforced by
reading the *generated text* -- ``test_gen_api`` finds a line that says
``for (int lane = 0; lane < nelems; ++lane) {``, walks backwards looking for a
pragma, then counts braces to decide where the loop body ends and scans it for
nested loops and atomics.  Every one of those steps is recovering structure
that the tree already has exactly.  Checking it on the tree is the same
contract asserted where it is actually true, and it runs during emission
rather than after it, so a kernel that violates it never reaches a file.
"""

from dataclasses import dataclass, field

from codegen.framework.ir.kernel_ast import (
    BlockNode,
    FunctionDefNode,
    IfNode,
    KernelASTPass,
    KernelASTPassResult,
    LoopHeaderNode,
    LoopNode,
    ScatterNode,
)


class VectorizationContractViolation(ValueError):
    """A vector-lane loop that could not vectorise, caught before it is printed."""


def _children(node):
    for attribute in ("body", "orelse"):
        for child in getattr(node, attribute, ()) or ():
            yield child
    loop = getattr(node, "loop", None)
    if loop is not None:
        yield loop


def _walk(node):
    yield node
    for child in _children(node):
        for descendant in _walk(child):
            yield descendant


@dataclass(frozen=True)
class VectorizationContractPass(KernelASTPass):
    """A vectorised lane loop must contain no loop and no atomic.

    Both defeat vectorisation: a nested loop stops the compiler flattening the
    lane iteration, and an atomic serialises it.  The check is structural, so
    it cannot be fooled by formatting the way a text scan can, and it names the
    kernel and the loop index rather than a file and a line number.
    """

    name: str = "vectorization-contract"
    parity_preserving: bool = True
    performance_changing: bool = False

    def apply(self, ast):
        for node in ast.nodes:
            for candidate in _walk(node):
                if isinstance(candidate, LoopNode) and candidate.vectorized:
                    self._check(ast, candidate)
        return KernelASTPassResult(self, ast, ast, ())

    def _check(self, ast, loop):
        index = getattr(loop.iterator.symbol, "name", "lane")
        for child in loop.body:
            for descendant in _walk(child):
                if isinstance(descendant, (LoopNode, LoopHeaderNode)):
                    raise VectorizationContractViolation(
                        "%s: a nested loop inside vectorised lane loop '%s' "
                        "prevents it from vectorising" % (ast.name, index)
                    )
                if isinstance(descendant, ScatterNode) and descendant.atomic:
                    raise VectorizationContractViolation(
                        "%s: an atomic update inside vectorised lane loop '%s' "
                        "serialises it" % (ast.name, index)
                    )
