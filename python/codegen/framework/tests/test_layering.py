"""Enforce the framework's layering rule: imports may only point up.

The framework is organised as a lowering pipeline.  ``README.md`` states the
order and ``LAYERING.html`` gives the target architecture; the rule both imply
is that a layer may name the artifact it *receives* -- something produced
earlier in the pipeline -- and must never name a layer that comes after it.
Reading the stack top-down as specification to generated text, legal imports
point up and illegal ones point down.

The tree started with six offending package pairs.  ``KNOWN_VIOLATIONS`` pinned
them so the graph could only improve, and it is now empty: every import in the
framework points up the lowering order.  Nothing may be added to that set
without a migration step that removes it again.

The test parses imports statically with ``ast`` -- it never imports the modules
it inspects, so it cannot be fooled by, or fail because of, import side effects.
"""

import ast
import os
import unittest


# The lowering order.  Index 0 is the specification end of the pipeline; a
# module may import any layer with a *lower* index than its own.
LAYER_ORDER = (
    "symbolic",
    "fem",
    "plans",
    "ir",
    "targets",
    "emitters",
    "backends",
)

# Packages that sit outside the lowering stack.  Frontend packages sit above the
# whole stack and may import anything; tooling and tests likewise.  ``mlir`` is
# an experimental subtree that is explicitly out of scope for the layering work.
UNRANKED_PACKAGES = (
    "materials",
    "generators",
    "mlir",
    "tests",
    "tools",
    "scripts",
    "twophaseflow",
)

# Violations present at the start of the layering work, as
# ``(importing_layer, imported_layer)``.  The list has only ever shrunk:
#
#   symbolic -> fem, plans, emitters, backends   removed by S2
#   fem      -> backends                         removed by S6
#   emitters -> backends                         removed by S6
#
# The last two were never really violations to repair.  ``backends/`` held two
# unrelated things: the target definitions, which belong below emission, and the
# orchestrators that drive the emitters.  S6 split ``targets/`` out into its own
# layer, so an emitter naming a target is now the upward edge it always was.
# ``fem/`` likewise held two emitter modules -- thirteen of eighteen functions in
# ``tensor_product_geometry`` build C text -- which moved to ``emitters/``.
#
# The graph is clean.  Nothing may be added here without a migration step that
# removes it again.
KNOWN_VIOLATIONS = frozenset()

FRAMEWORK_PACKAGE = "codegen.framework"


def _framework_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _layer_of_path(path, root):
    """The layer package a file belongs to, or None for the framework root."""
    relative = os.path.relpath(path, root)
    head = relative.split(os.sep)[0]
    if head.endswith(".py"):
        return None
    return head


def _imported_layer(module_name):
    """The layer a dotted module name refers to, or None if it is not a layer."""
    if not module_name:
        return None
    parts = module_name.split(".")
    if module_name.startswith(FRAMEWORK_PACKAGE):
        parts = parts[2:]
    if not parts:
        return None
    head = parts[0]
    return head if head in LAYER_ORDER else None


def _iter_imports(tree, module_layer):
    """Yield the layer names a parsed module imports.

    Handles absolute imports (``codegen.framework.plans.generation``), bare
    package imports (``plans.generation``, used by the fallback branches), and
    relative imports (``from ..plans import x``), which resolve against the
    importing module's own layer.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                layer = _imported_layer(alias.name)
                if layer is not None:
                    yield layer
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level >= 2:
                # 'from ..plans import x' inside a layer package.
                layer = _imported_layer(node.module) if node.module else None
                if layer is not None:
                    yield layer
            elif not node.level:
                layer = _imported_layer(node.module)
                if layer is not None:
                    yield layer


def collect_edges():
    """Every ``(importing_layer, imported_layer)`` pair in the framework.

    Self-edges and imports from unranked packages are excluded.
    """
    root = _framework_root()
    edges = {}
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            module_layer = _layer_of_path(path, root)
            if module_layer not in LAYER_ORDER:
                continue
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as error:  # pragma: no cover - defensive
                raise AssertionError("could not parse %s: %s" % (path, error))
            for imported_layer in _iter_imports(tree, module_layer):
                if imported_layer == module_layer:
                    continue
                edges.setdefault((module_layer, imported_layer), set()).add(
                    os.path.relpath(path, root)
                )
    return edges


def violations(edges):
    """Edges that point down the stack, keyed by ``(importer, imported)``."""
    rank = {layer: index for index, layer in enumerate(LAYER_ORDER)}
    return {
        edge: sorted(files)
        for edge, files in edges.items()
        if rank[edge[0]] < rank[edge[1]]
    }


class LayeringTest(unittest.TestCase):
    def test_no_new_downward_imports(self):
        """No layer may import a layer that comes after it in the pipeline."""
        found = violations(collect_edges())
        unexpected = sorted(set(found) - KNOWN_VIOLATIONS)
        if unexpected:
            lines = ["new layering violations introduced:"]
            for importer, imported in unexpected:
                lines.append("  %s -> %s" % (importer, imported))
                for path in found[(importer, imported)]:
                    lines.append("      %s" % path)
            lines.append("")
            lines.append(
                "imports may only point up the lowering order: %s" % " -> ".join(LAYER_ORDER)
            )
            self.fail("\n".join(lines))

    def test_known_violations_are_still_present(self):
        """Fail when a known violation is fixed but not removed from the list.

        This keeps the allowlist honest: it shrinks as the migration proceeds
        and never quietly protects an edge that no longer exists.
        """
        found = set(violations(collect_edges()))
        stale = sorted(KNOWN_VIOLATIONS - found)
        if stale:
            lines = ["these layering violations are fixed; remove them from KNOWN_VIOLATIONS:"]
            lines.extend("  %s -> %s" % edge for edge in stale)
            self.fail("\n".join(lines))

    def test_layer_order_covers_every_layer_package(self):
        """Every package under the framework is either a layer or unranked."""
        root = _framework_root()
        packages = sorted(
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
            and not name.startswith(("_", "."))
            and os.path.isfile(os.path.join(root, name, "__init__.py"))
        )
        known = set(LAYER_ORDER) | set(UNRANKED_PACKAGES)
        unclassified = [name for name in packages if name not in known]
        self.assertEqual(
            unclassified,
            [],
            "package(s) %s are neither in LAYER_ORDER nor UNRANKED_PACKAGES; "
            "decide which layer they belong to" % unclassified,
        )


if __name__ == "__main__":
    unittest.main()
