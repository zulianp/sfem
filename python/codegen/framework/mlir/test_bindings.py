"""Smoke test for LLVM mlir-python-bindings (not the unrelated pymlir package).

Both packages install into the ``mlir`` namespace. If ``pymlir`` is installed
alongside ``mlir-python-bindings``, imports fail with errors like::

    ImportError: cannot import name 'affine' from 'mlir.dialects.affine'

Fix: ``pip uninstall pymlir``
"""

from __future__ import annotations

import importlib.util
import sys


def _pymlir_conflict_hint() -> str:
    parser_spec = importlib.util.find_spec("mlir.parser")
    if parser_spec is None or parser_spec.origin is None:
        return ""
    if "astnodes" not in parser_spec.origin:
        return ""
    return (
        "\n\nDetected pymlir files in the mlir package namespace. "
        "pymlir conflicts with mlir-python-bindings.\n"
        "Fix: pip uninstall pymlir"
    )


def main() -> int:
    try:
        from mlir import ir  # noqa: F401
        import mlir.dialects.spirv as spirv  # noqa: F401
    except ImportError as exc:
        hint = _pymlir_conflict_hint()
        print("MLIR import failed: %s%s" % (exc, hint), file=sys.stderr)
        return 1

    print("MLIR & SPIR-V successfully loaded!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
