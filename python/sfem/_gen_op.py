"""Compatibility shim: wrapper generation now lives inside the framework.

The C ABI, ``sfem::Op`` subclass and factory-registration generators are part of
the framework's packaging layer, not runtime-library surface.  They moved to
``codegen.framework.package.op_wrappers``; this module re-exports them so
existing callers keep working unchanged.
"""

from codegen.framework.package.op_wrappers import *  # noqa: F401,F403
from codegen.framework.package import op_wrappers as _op_wrappers

globals().update(
    {
        _name: _value
        for _name, _value in vars(_op_wrappers).items()
        if not _name.startswith("__")
    }
)
